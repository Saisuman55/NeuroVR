"""
train_m5_optimized.py — M5-chip optimized training targeting 98%+

Apple M5 tuning:
  - Batch size 16  → fits MPS memory comfortably, no throttle
  - num_workers=2  → M5 efficiency cores handle prefetch without starving GPU
  - pin_memory=False → MPS doesn't support it (removes warning)
  - torch.backends.mps.enable_fallback_for_mps_  → avoids unsupported ops
  - Gradient accumulation x2 → effective batch=32 without memory spike
  - EfficientNet-B4 stays — M5 handles it fine at batch=16
  - Cosine Annealing Warm Restarts — better than step-decay on MPS
  - No Mixup in Phase B — reduces MPS memory churn
  - AdamW + weight_decay — cleaner generalization than Adam
  - Live metrics.json update every epoch for dashboard
  - Baseline: 95.18% → Target: 98%+
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (
    get_classification_transforms,
    ClassificationDataset,
    load_config,
    set_seed,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from model import get_classifier, unfreeze_top_layers

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")
METRICS_JSON  = os.path.join(BASE, "stitch_frontend", "metrics.json")
CKPT_DIR      = os.path.join(BASE, "models", "classifier")
BEST_CKPT     = os.path.join(CKPT_DIR, "brain_tumor_98plus_best.pth")
PREV_CKPT     = os.path.join(CKPT_DIR, "brain_tumor_classifier_best.pth")
LOG_FILE      = os.path.join(BASE, "outputs", "training_m5.log")

# ── M5 Tuned Hyperparameters ───────────────────────────────────────────────
M5 = dict(
    batch_size      = 16,    # safe for 16GB unified memory at 380px img
    num_workers     = 2,     # 2 efficiency cores for prefetch
    pin_memory      = False, # MPS doesn't support pin_memory
    grad_accum      = 2,     # effective batch = 32 (no extra memory)
    phase_a_epochs  = 0,     # Skipped! User requested jump to Phase B
    phase_b_epochs  = 30,    # full unfreeze, very low LR
    phase_a_lr      = 3e-5,  # gentle — we're already at 95%
    phase_b_lr      = 8e-6,  # ultra-fine polishing
    weight_decay    = 1e-4,
    label_smoothing = 0.05,  # mild smoothing
    clip_grad       = 1.0,
    patience        = 10,
    ema_decay       = 0.999,
    mixup_alpha     = 0.2,   # gentle mixup only in Phase A
)


# ── Utilities ──────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self):
        self._bak = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self._bak)


def mixup(x, y, alpha, device):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def log(msg):
    print(msg, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def save_metrics(ep, total_ep, phase, t_acc, v_acc, t_loss, v_loss, status="training"):
    os.makedirs(os.path.dirname(METRICS_JSON), exist_ok=True)
    with open(METRICS_JSON, "w") as f:
        json.dump({
            "status": status,
            "phase": phase,
            "current_epoch": ep,
            "total_epochs": total_ep,
            "best_val_acc": round(max(v_acc), 2) if v_acc else 0.0,
            "target_acc": 98.0,
            "history": {
                "epoch": ep,
                "train_acc":  [round(x, 2) for x in t_acc],
                "val_acc":    [round(x, 2) for x in v_acc],
                "train_loss": [round(x, 4) for x in t_loss],
                "val_loss":   [round(x, 4) for x in v_loss],
            }
        }, f, indent=2)


def build_loaders(config):
    """Build loaders with M5-optimised settings."""
    cfg   = config["classification"]
    aug   = cfg["augmentation"]
    size  = cfg["img_size"]
    names = cfg["class_names"]
    data  = config["paths"]["data_classification"]

    train_tf, val_tf = get_classification_transforms(size, aug)

    full = ClassificationDataset(
        os.path.join(data, "Training"), names, size, train_tf, "train"
    )
    labels = [s[1] for s in full.samples]
    tr_idx, va_idx = train_test_split(
        list(range(len(full))),
        test_size=cfg.get("val_split", 0.1),
        random_state=config["seed"],
        stratify=labels,
    )
    val_ds = ClassificationDataset(
        os.path.join(data, "Training"), names, size, val_tf, "val"
    )
    val_ds.samples = [full.samples[i] for i in va_idx]

    kw = dict(num_workers=M5["num_workers"], pin_memory=M5["pin_memory"])
    tr_loader = DataLoader(Subset(full, tr_idx),
                           batch_size=M5["batch_size"], shuffle=True, **kw)
    va_loader = DataLoader(val_ds,
                           batch_size=M5["batch_size"], shuffle=False, **kw)
    return tr_loader, va_loader


# ── Train / Validate ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_mixup=False, grad_accum=1):
    model.train()
    total_loss = correct = total = 0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(tqdm(loader, desc="  Train", leave=False), 1):
        x, y = x.to(device), y.to(device)

        if use_mixup:
            x, ya, yb, lam = mixup(x, y, M5["mixup_alpha"], device)
            out  = model(x)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
            _, pred = torch.max(out, 1)
            correct += (lam*(pred==ya).float() + (1-lam)*(pred==yb).float()).sum().item()
        else:
            out  = model(x)
            loss = criterion(out, y)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()

        (loss / grad_accum).backward()

        if step % grad_accum == 0 or step == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), M5["clip_grad"])
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * x.size(0)
        total      += x.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device, ema=None):
    if ema: ema.apply()
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="  Val  ", leave=False):
            x, y = x.to(device), y.to(device)
            out  = model(x)
            loss = criterion(out, y)
            _, pred = torch.max(out, 1)
            correct    += (pred == y).sum().item()
            total_loss += loss.item() * x.size(0)
            total      += x.size(0)
    if ema: ema.restore()
    return total_loss / total, correct / total


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    config = load_config("config.yaml")
    set_seed(config["seed"])

    # Apple M5 MPS device
    device = torch.device("mps")
    torch.backends.mps.enable_fallback_for_mps_ = True  # graceful op fallback

    log("\n" + "═"*62)
    log("  🍎  Apple M5 Optimised Training  →  Target: 98%+")
    log(f"  Batch={M5['batch_size']} | Workers={M5['num_workers']} | "
        f"GradAccum={M5['grad_accum']} (eff. batch={M5['batch_size']*M5['grad_accum']})")
    log("═"*62 + "\n")

    tr_loader, va_loader = build_loaders(config)
    criterion = nn.CrossEntropyLoss(label_smoothing=M5["label_smoothing"])

    # ── Build model ──
    model = get_classifier(
        num_classes=config["classification"]["num_classes"],
        dropout=config["classification"]["dropout"],
        freeze_backbone=False,
        model_name=config["classification"].get("model_name", "efficientnet_b4"),
    ).to(device)

    # Load best available checkpoint
    ckpt_path = BEST_CKPT if os.path.exists(BEST_CKPT) else PREV_CKPT
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        base_acc = round(float(ckpt.get("metric", 0)) * 100, 2)
        log(f"  ✅ Loaded: {os.path.basename(ckpt_path)}  (baseline {base_acc:.2f}%)\n")
    else:
        base_acc = 0.0
        log("  ⚠️  No checkpoint — starting fresh\n")

    all_t_acc, all_v_acc, all_t_loss, all_v_loss = [], [], [], []
    global_best = base_acc / 100.0
    TOTAL = M5["phase_a_epochs"] + M5["phase_b_epochs"]

    # ════════════════════════════════════════════════════════
    # PHASE A — Fine-tune top 30 backbone layers
    # ════════════════════════════════════════════════════════
    log("── Phase A: top-30 fine-tune, Mixup ON ──")
    unfreeze_top_layers(model, num_layers=30)

    opt_a = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=M5["phase_a_lr"], weight_decay=M5["weight_decay"]
    )
    sched_a = CosineAnnealingWarmRestarts(opt_a, T_0=10, T_mult=1, eta_min=1e-8)
    ema = EMA(model, M5["ema_decay"])
    patience_cnt = 0

    for ep in range(1, M5["phase_a_epochs"] + 1):
        tl, ta = train_one_epoch(model, tr_loader, criterion, opt_a, device,
                                  use_mixup=True, grad_accum=M5["grad_accum"])
        ema.update()
        vl, va = validate(model, va_loader, criterion, device, ema)
        sched_a.step()

        all_t_acc.append(ta*100); all_v_acc.append(va*100)
        all_t_loss.append(tl);    all_v_loss.append(vl)
        save_metrics(ep, TOTAL, "Phase A – Top-30 Fine-tune",
                     all_t_acc, all_v_acc, all_t_loss, all_v_loss)

        lr_now = sched_a.get_last_lr()[0]
        log(f"  [A] {ep:02}/{M5['phase_a_epochs']} | "
            f"Tr {ta*100:.2f}% | Val {va*100:.2f}% | "
            f"Best {max(all_v_acc):.2f}% | LR {lr_now:.1e}")

        if va > global_best:
            global_best = va
            patience_cnt = 0
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "metric": va}, BEST_CKPT)
            log(f"      💾 Saved best  ({va*100:.2f}%)")
        else:
            patience_cnt += 1
            if patience_cnt >= M5["patience"]:
                log("  ⏹  Early stop Phase A"); break

    # ════════════════════════════════════════════════════════
    # PHASE B — Full unfreeze, ultra-low LR, no Mixup
    # ════════════════════════════════════════════════════════
    log(f"\n── Phase B: full unfreeze, LR={M5['phase_b_lr']:.0e}, Mixup OFF ──")

    # Reload best from Phase A
    if os.path.exists(BEST_CKPT):
        ckpt = torch.load(BEST_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        log(f"  Reloaded best checkpoint ({max(all_v_acc):.2f}%)")

    for param in model.parameters():
        param.requires_grad = True

    opt_b = optim.AdamW(model.parameters(),
                        lr=M5["phase_b_lr"], weight_decay=M5["weight_decay"]//2)
    sched_b = CosineAnnealingWarmRestarts(opt_b, T_0=15, T_mult=1, eta_min=1e-9)
    ema = EMA(model, decay=0.9995)
    patience_cnt = 0
    phase_b_best = global_best

    for ep in range(1, M5["phase_b_epochs"] + 1):
        tl, ta = train_one_epoch(model, tr_loader, criterion, opt_b, device,
                                  use_mixup=False, grad_accum=M5["grad_accum"])
        ema.update()
        vl, va = validate(model, va_loader, criterion, device, ema)
        sched_b.step()

        all_t_acc.append(ta*100); all_v_acc.append(va*100)
        all_t_loss.append(tl);    all_v_loss.append(vl)
        global_ep = M5["phase_a_epochs"] + ep
        save_metrics(global_ep, TOTAL, "Phase B – Full Unfreeze",
                     all_t_acc, all_v_acc, all_t_loss, all_v_loss)

        lr_now = sched_b.get_last_lr()[0]
        log(f"  [B] {ep:02}/{M5['phase_b_epochs']} | "
            f"Tr {ta*100:.2f}% | Val {va*100:.2f}% | "
            f"Best {max(all_v_acc):.2f}% | LR {lr_now:.1e}")

        if va > phase_b_best:
            phase_b_best = va
            global_best  = va
            patience_cnt = 0
            torch.save({"epoch": global_ep, "model_state_dict": model.state_dict(),
                        "metric": va}, BEST_CKPT)
            log(f"      💾 New best  ({va*100:.2f}%)")
            if va >= 0.98:
                log(f"\n  🎯 98%+ REACHED! Final: {va*100:.2f}% — stopping early.")
                break
        else:
            patience_cnt += 1
            if patience_cnt >= M5["patience"]:
                log("  ⏹  Early stop Phase B"); break

    # ── Done ──
    final = max(all_v_acc)
    reached = final >= 98.0
    status  = "completed" if reached else "completed_below_target"
    save_metrics(len(all_t_acc), TOTAL, "Complete",
                 all_t_acc, all_v_acc, all_t_loss, all_v_loss, status)

    log("\n" + "═"*62)
    log(f"  🏁  DONE  |  Best Val Accuracy: {final:.2f}%")
    log(f"  {'✅  GOAL ACHIEVED: 98%+ !' if reached else '🔄  Close — consider one more run'}")
    log("═"*62 + "\n")


if __name__ == "__main__":
    main()

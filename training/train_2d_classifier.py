"""Train a patient-separated 2D CNN on the Cheng CE-MRI dataset.

Research/educational use only. This model is not clinically validated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

CLASSES = ["meningioma", "glioma", "pituitary"]  # source labels 1, 2, 3


def extract_archives(root: Path) -> Path:
    out = root / "mat"
    out.mkdir(parents=True, exist_ok=True)
    for archive in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    return out


def _read_scalar(group, key):
    value = np.asarray(group[key]).squeeze()
    return value.item() if value.size == 1 else value


def read_metadata(path: Path):
    with h5py.File(path, "r") as f:
        g = f["cjdata"]
        label = int(_read_scalar(g, "label")) - 1
        pid_raw = np.asarray(g["PID"]).squeeze()
        pid = "".join(chr(int(x)) for x in np.ravel(pid_raw))
    return label, pid or path.stem


def patient_split(pid: str) -> str:
    bucket = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def build_index(mat_dir: Path, index_path: Path):
    rows = []
    for path in sorted(mat_dir.rglob("*.mat")):
        label, pid = read_metadata(path)
        if 0 <= label < len(CLASSES):
            rows.append({"path": str(path), "label": label, "patient_id": pid,
                         "split": patient_split(pid)})
    index_path.write_text(json.dumps({"classes": CLASSES, "samples": rows}, indent=2))
    return rows


class ChengDataset(Dataset):
    def __init__(self, rows, train=False):
        self.rows = rows
        ops = [transforms.Resize((224, 224))]
        if train:
            ops += [transforms.RandomRotation(10), transforms.RandomHorizontalFlip(0.5)]
        ops += [transforms.Grayscale(3), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(ops)

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with h5py.File(row["path"], "r") as f:
            image = np.asarray(f["cjdata"]["image"], dtype=np.float32)
        image -= image.min()
        image /= max(float(image.max()), 1e-6)
        pil = Image.fromarray((image * 255).astype(np.uint8), mode="L")
        return self.transform(pil), int(row["label"])


def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            for truth, guess in zip(y, pred): confusion[truth, guess] += 1
            correct += int((pred == y).sum()); total += len(y)
    recalls = confusion.diag() / confusion.sum(1).clamp_min(1)
    return {"accuracy": correct / max(total, 1), "macro_recall": recalls.mean().item(),
            "confusion_matrix": confusion.tolist(), "samples": total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/training/figshare_brain_tumor"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    mat_dir = extract_archives(args.data)
    rows = build_index(mat_dir, args.data / "patient_split.json")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    loaders = {s: DataLoader(ChengDataset([r for r in rows if r["split"] == s], s == "train"),
                             batch_size=args.batch_size, shuffle=s == "train", num_workers=0)
               for s in ("train", "val", "test")}
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    out = Path("models/classification_2d"); out.mkdir(parents=True, exist_ok=True)
    best = -1.0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for x, y in loaders["train"]:
            opt.zero_grad(set_to_none=True); loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward(); opt.step(); running += float(loss) * len(y)
        metrics = evaluate(model, loaders["val"], device)
        metrics.update(epoch=epoch, train_loss=running / len(loaders["train"].dataset)); history.append(metrics)
        print(json.dumps(metrics))
        if metrics["macro_recall"] > best:
            best = metrics["macro_recall"]
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                        "architecture": "efficientnet_b0", "validation": metrics,
                        "dataset": "Cheng CE-MRI Figshare 1512427 v8"}, out / "best_model.pt")
    ckpt = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    report = {"dataset": ckpt["dataset"], "patient_separated": True,
              "best_validation": ckpt["validation"], "test": evaluate(model, loaders["test"], device),
              "history": history, "disclaimer": "Research use only; not clinically validated."}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__": main()

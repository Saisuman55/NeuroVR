"""Fine-tune the MONAI BraTS SegResNet on MSD Task01 real labeled volumes."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.losses import DiceLoss
from monai.networks.nets import SegResNet
from torch.utils.data import DataLoader, Dataset


def split_for(path: Path):
    bucket = int(hashlib.sha256(path.stem.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val"


def brats_channels(label):
    # MSD Task01 labels: 1=edema, 2=non-enhancing/necrotic core, 3=enhancing.
    return np.stack([(label == 2) | (label == 3), label > 0, label == 3]).astype(np.float32)


class PatchDataset(Dataset):
    def __init__(self, cases, patch=64, train=True): self.cases, self.patch, self.train = cases, patch, train
    def __len__(self): return len(self.cases)
    def __getitem__(self, idx):
        image_path, label_path = self.cases[idx]
        image = nib.load(image_path).get_fdata(dtype=np.float32)
        label = nib.load(label_path).get_fdata(dtype=np.float32).astype(np.uint8)
        if image.shape[-1] != 4: raise ValueError(f"Expected four channels: {image_path} {image.shape}")
        image = np.moveaxis(image, -1, 0)
        for c in range(4):
            nz = image[c] != 0
            if nz.any(): image[c, nz] = (image[c, nz] - image[c, nz].mean()) / max(image[c, nz].std(), 1e-6)
        coords = np.argwhere(label > 0)
        center = coords[random.randrange(len(coords))] if self.train and len(coords) else np.array(label.shape) // 2
        starts = [int(np.clip(center[d] - self.patch // 2, 0, max(label.shape[d] - self.patch, 0))) for d in range(3)]
        sl = tuple(slice(s, min(s + self.patch, label.shape[d])) for d, s in enumerate(starts))
        x, y = image[(slice(None),) + sl], brats_channels(label[sl])
        pads = [(0, 0)] + [(0, self.patch - x.shape[d + 1]) for d in range(3)]
        x = np.pad(x, pads); y = np.pad(y, pads)
        return torch.from_numpy(x), torch.from_numpy(y)


def dice_scores(logits, target):
    pred = torch.sigmoid(logits) > 0.5; truth = target > 0.5
    dims = tuple(range(2, pred.ndim)); inter = (pred & truth).sum(dims).float()
    return ((2 * inter + 1e-5) / (pred.sum(dims) + truth.sum(dims) + 1e-5)).mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/training/msd_task01/Task01_BrainTumour"))
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--patch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    images = sorted((args.data / "imagesTr").glob("*.nii.gz"))
    cases = [(str(x), str(args.data / "labelsTr" / x.name)) for x in images]
    if not cases: raise FileNotFoundError(f"No MSD training cases under {args.data}")
    train = [c for c in cases if split_for(Path(c[0])) == "train"]
    val = [c for c in cases if split_for(Path(c[0])) == "val"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SegResNet(blocks_down=(1,2,2,4), blocks_up=(1,1,1), init_filters=16,
                      in_channels=4, out_channels=3, dropout_prob=0.2).to(device)
    source = Path("models/monai/brats_mri_segmentation/models/model.pt")
    state = torch.load(source, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in state: state = state[key]; break
    model.load_state_dict(state, strict=True)
    loaders = {"train": DataLoader(PatchDataset(train,args.patch,True),1,shuffle=True,num_workers=0),
               "val": DataLoader(PatchDataset(val,args.patch,False),1,shuffle=False,num_workers=0)}
    loss_fn = DiceLoss(sigmoid=True, squared_pred=True); opt = torch.optim.AdamW(model.parameters(),lr=args.lr)
    out=Path("models/segmentation_3d_finetuned"); out.mkdir(parents=True,exist_ok=True); best=-1; hist=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y in loaders["train"]:
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(x.to(device)),y.to(device)); loss.backward(); opt.step(); losses.append(float(loss))
        model.eval(); scores=[]
        with torch.inference_mode():
            for x,y in loaders["val"]: scores.append(dice_scores(model(x.to(device)),y.to(device)).cpu())
        mean=torch.stack(scores).mean(0); row={"epoch":epoch,"loss":sum(losses)/len(losses),"tc_dice":float(mean[0]),"wt_dice":float(mean[1]),"et_dice":float(mean[2]),"mean_dice":float(mean.mean())}; hist.append(row); print(json.dumps(row))
        if row["mean_dice"]>best: best=row["mean_dice"]; torch.save({"state_dict":model.state_dict(),"validation":row,"dataset":"MSD Task01_BrainTumour","patient_separated":True},out/"best_model.pt")
    (out/"metrics.json").write_text(json.dumps({"history":hist,"best_mean_dice":best,"validation_cases":len(val),"disclaimer":"Research use only; not clinically validated."},indent=2))


if __name__ == "__main__": main()

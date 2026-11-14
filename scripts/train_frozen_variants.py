"""
Frozen-backbone ablation: trains baseline (original GLRA) and gated variant
with backbone frozen. Only the GLRA module is trainable.
Evaluates on test set after every epoch — no separate test job needed.

Usage:
    python train_frozen_variants.py \
        --base_ckpt /path/to/checkpoint.pt \
        --data_path ... --gt_path ... \
        --test_data_path ... --test_gt_path ...
"""
import os
import csv
import cv2
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim

import BottleCSP
from Network import RAINet
from DerainDataset import TrainDataset
from GLRA import GLRA as OriginalGLRA
from GLRA_stripfusion import GatedFusionGLRA

parser = argparse.ArgumentParser()
parser.add_argument("--base_ckpt", type=str, required=True,
                    help="Pre-trained checkpoint to freeze backbone from")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--gt_path", type=str, required=True)
parser.add_argument("--test_data_path", type=str, required=True)
parser.add_argument("--test_gt_path", type=str, required=True)
parser.add_argument("--out_root", type=str, default="./checkpoints/")
parser.add_argument("--results_root", type=str, default="./results/")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--steps_per_epoch", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--num_M", type=int, default=32)
parser.add_argument("--num_Z", type=int, default=32)
parser.add_argument("--T", type=int, default=4)
parser.add_argument("--S", type=int, default=20)
opt = parser.parse_args()

device = torch.device("cuda:0")
cudnn.benchmark = True

VARIANTS = [
    ("baseline_frozen", OriginalGLRA),
    ("gated_frozen",    GatedFusionGLRA),
]


def is_image(name):
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [INFO] Missing keys (will be randomly initialized): {len(missing)}")
    if unexpected:
        print(f"  [INFO] Unexpected keys (ignored): {len(unexpected)}")


def freeze_backbone_unfreeze_glra(model, glra_cls):
    for p in model.parameters():
        p.requires_grad = False
    trainable = 0
    for m in model.modules():
        if isinstance(m, glra_cls):
            for p in m.parameters():
                p.requires_grad = True
                trainable += p.numel()
    return trainable


def evaluate_test_set(model, test_data_path, test_gt_path):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for img_name in sorted(os.listdir(test_data_path)):
            if not is_image(img_name):
                continue
            O = cv2.imread(os.path.join(test_data_path, img_name))
            gt = cv2.imread(os.path.join(test_gt_path, img_name.replace("rain-", "norain-")))
            if O is None or gt is None:
                continue
            O_rgb = O[:, :, ::-1].copy()
            inp = torch.tensor(O_rgb.transpose(2, 0, 1)[None], dtype=torch.float32).to(device)
            outputs, _ = model(inp)
            out = torch.clamp(outputs[-1], 0., 255.)
            out_bgr = np.array(out.cpu().numpy().squeeze(), dtype=np.uint8).transpose(1, 2, 0)[:, :, ::-1].copy()
            mse = np.mean((out_bgr.astype(np.float64) - gt.astype(np.float64)) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
            psnrs.append(psnr)
            try:
                s = compare_ssim(out_bgr, gt, data_range=255, channel_axis=2)
            except TypeError:
                s = compare_ssim(out_bgr, gt, data_range=255, multichannel=True)
            ssims.append(s)
    return float(np.mean(psnrs)), float(np.mean(ssims))


def train_variant(variant_name, variant_cls):
    print(f"\n{'='*60}")
    print(f"  Training: {variant_name}  (frozen backbone)")
    print(f"{'='*60}")

    BottleCSP.GLRA = variant_cls

    ckpt_dir = os.path.join(opt.out_root, f"Rain100L_{variant_name}")
    results_dir = os.path.join(opt.results_root, f"Rain100L_{variant_name}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model = RAINet(opt).to(device)

    print(f"  Loading base checkpoint: {opt.base_ckpt}")
    load_checkpoint(model, opt.base_ckpt)

    trainable = freeze_backbone_unfreeze_glra(model, variant_cls)
    print(f"  Trainable params (GLRA only): {trainable:,}")

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=opt.lr
    )

    dataset = TrainDataset(
        opt.data_path, opt.gt_path,
        patchsize=64,
        length=opt.batch_size * opt.steps_per_epoch
    )
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                        num_workers=opt.workers, pin_memory=True)

    csv_path = os.path.join(results_dir, "metrics_per_epoch.csv")
    best_psnr, best_epoch = -1.0, -1

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "psnr", "ssim", "train_loss"])

    for epoch in range(opt.epochs):
        model.train()
        epoch_loss = 0.0
        tic = time.time()

        for im_rain, im_gt in loader:
            im_rain = im_rain.to(device, non_blocking=True)
            im_gt = im_gt.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs, _ = model(im_rain, labels=im_gt)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)

        psnr, ssim = evaluate_test_set(model, opt.test_data_path, opt.test_gt_path)

        print(f"[{variant_name}] Epoch {epoch+1}/{opt.epochs} "
              f"Loss={epoch_loss:.2e} PSNR={psnr:.4f} SSIM={ssim:.4f} "
              f"time={time.time()-tic:.1f}s")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, round(psnr, 4), round(ssim, 4),
                                     round(epoch_loss, 4)])

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
        }, os.path.join(ckpt_dir, f"DerainNet_state_{epoch+1}.pt"))

        if psnr > best_psnr:
            best_psnr = psnr
            best_epoch = epoch + 1

    print(f"\n[{variant_name}] Best PSNR: {best_psnr:.4f} at epoch {best_epoch}")
    return best_psnr, best_epoch


if __name__ == "__main__":
    summary = []
    for name, cls in VARIANTS:
        best_psnr, best_epoch = train_variant(name, cls)
        summary.append((name, best_psnr, best_epoch))

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for name, psnr, epoch in summary:
        print(f"  {name}: Best PSNR={psnr:.4f} at epoch {epoch}")
    baseline = next(r for r in summary if "baseline" in r[0])
    gated = next(r for r in summary if "gated" in r[0])
    diff = gated[1] - baseline[1]
    print(f"\n  Gated vs Baseline: {diff:+.4f} dB")

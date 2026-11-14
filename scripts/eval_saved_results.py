"""
Compute PSNR, SSIM, and LPIPS from already-saved output images.
Usage:
    python eval_saved_results.py \
        --results_dir ./results/Rain100L_additive_v2 \
        --gt_path ./data/Rain100L/test/norain \
        --out_csv ./results/Rain100L_additive_v2/metrics_per_epoch.csv
"""
import os
import csv
import argparse
import numpy as np
import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, required=True)
parser.add_argument("--gt_path", type=str, required=True)
parser.add_argument("--out_csv", type=str, required=True)
opt = parser.parse_args()

loss_fn = lpips.LPIPS(net='alex').cuda()

epoch_dirs = sorted(
    (int(d.replace("epoch_", "")), d)
    for d in os.listdir(opt.results_dir)
    if d.startswith("epoch_") and os.path.isdir(os.path.join(opt.results_dir, d))
)

rows = []
for epoch, dirname in epoch_dirs:
    epoch_dir = os.path.join(opt.results_dir, dirname)
    psnrs, ssims, lpipss = [], [], []

    for img_name in os.listdir(epoch_dir):
        pred = cv2.imread(os.path.join(epoch_dir, img_name))
        gt_name = img_name.replace('rain-', 'norain-')
        gt = cv2.imread(os.path.join(opt.gt_path, gt_name))
        if pred is None or gt is None:
            continue

        # PSNR on Y channel (YCbCr) — matches paper's evaluation protocol
        pred_y = cv2.cvtColor(pred, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
        gt_y = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
        mse = np.mean((pred_y - gt_y) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
        psnrs.append(psnr)

        # SSIM on Y channel
        ssims.append(ssim(pred_y, gt_y, data_range=255))

        # LPIPS (expects [-1, 1] RGB tensors)
        def to_lpips_tensor(img_bgr):
            rgb = img_bgr[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
            return torch.tensor(rgb.transpose(2, 0, 1)[None]).cuda()

        with torch.no_grad():
            lp = loss_fn(to_lpips_tensor(pred), to_lpips_tensor(gt)).item()
        lpipss.append(lp)

    row = {
        "epoch": epoch,
        "psnr": round(np.mean(psnrs), 4),
        "ssim": round(np.mean(ssims), 4),
        "lpips": round(np.mean(lpipss), 4),
    }
    rows.append(row)
    print(f"Epoch {epoch:3d} | PSNR {row['psnr']:.4f} | SSIM {row['ssim']:.4f} | LPIPS {row['lpips']:.4f}")

with open(opt.out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "psnr", "ssim", "lpips"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved to {opt.out_csv}")
best = max(rows, key=lambda r: r["psnr"])
print(f"Best PSNR: {best['psnr']} at epoch {best['epoch']} | SSIM {best['ssim']} | LPIPS {best['lpips']}")

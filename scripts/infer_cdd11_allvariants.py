#!/usr/bin/env python
"""
Zero-shot TEST (no training) of all attention-ablation (strip-fusion + mechanism-swap) variants
on the CDD-11 real-world test set.

Each variant's Rain100L-trained checkpoint is loaded and run on CDD-11/rain;
outputs are saved and scored (PSNR_Y / SSIM_Y / LPIPS) against CDD-11/clear.
"""
import os, csv, argparse
import numpy as np
import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as compare_ssim

import BottleCSP
from GLRA_stripfusion import (
    GLRA as MultiplicativeGLRA,
    AdditiveFusionGLRA, ConcatFusionGLRA, GatedFusionGLRA, ResidualGatedFusionGLRA,
    ChannelAttnGLRA, SelfAttnGLRA, SwinAttnGLRA,
)

CK = "./checkpoints"
# label, variant class, checkpoint path
VARIANTS = [
    ("multiplicative",  MultiplicativeGLRA,      f"{CK}/Rain100L_multiplicative_clean/best_val.pt"),
    ("additive",        AdditiveFusionGLRA,      f"{CK}/Rain100L_additive_clean/best_val.pt"),
    ("concat",          ConcatFusionGLRA,        f"{CK}/Rain100L_concat_clean/best_val.pt"),
    ("gated",           GatedFusionGLRA,         f"{CK}/Rain100L_gated_clean/best_val.pt"),
    ("residual_gated",  ResidualGatedFusionGLRA, f"{CK}/Rain100L_residual_gated_clean/best_val.pt"),
    ("channel_attn",    ChannelAttnGLRA,         f"{CK}/Rain100L_channel_attn_frozen/DerainNet_state_78.pt"),
    ("self_attn",       SelfAttnGLRA,            f"{CK}/Rain100L_self_attn_frozen/DerainNet_state_98.pt"),
    ("swin_attn",       SwinAttnGLRA,            f"{CK}/Rain100L_swin_attn_frozen/DerainNet_state_98.pt"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--rain", default="./data/CDD-11/rain")
parser.add_argument("--gt",   default="./data/CDD-11/clear")
parser.add_argument("--out_root", default="./results")
parser.add_argument("--tag", default="cdd11", help="output dir prefix: results/<tag>_<variant>/epoch_best")
parser.add_argument("--summary_csv",   default="./analysis_outputs/cdd11_allvariants_summary.csv")
parser.add_argument("--persample_csv", default="./analysis_outputs/cdd11_allvariants_persample.csv")
parser.add_argument("--num_M", type=int, default=32)
parser.add_argument("--num_Z", type=int, default=32)
parser.add_argument("--T",     type=int, default=4)
parser.add_argument("--S",     type=int, default=20)
opt = parser.parse_args()

device = torch.device("cuda:0")
loss_fn = lpips.LPIPS(net='alex').to(device)
from Network import RAINet


def is_image(n):
    return n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))


def to_lpips_tensor(img_bgr):
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
    return torch.tensor(rgb.transpose(2, 0, 1)[None]).to(device)


def psnr_ssim_y(pred_bgr, gt_bgr):
    py = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    gy = cv2.cvtColor(gt_bgr,  cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    mse = np.mean((py - gy) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
    return psnr, compare_ssim(py, gy, data_range=255)


def load_model(variant_cls, ckpt_path):
    BottleCSP.GLRA = variant_cls
    model = RAINet(opt).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    msd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    print(f"    loaded {len(filtered)}/{len(msd)} keys", flush=True)
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


names = sorted(n for n in os.listdir(opt.rain) if is_image(n))
print(f"CDD-11 test images: {len(names)}")

summary, persample = [], []
for label, cls, ckpt_path in VARIANTS:
    print(f"\n===== {label} =====", flush=True)
    if not os.path.isfile(ckpt_path):
        print(f"    [SKIP] missing {ckpt_path}"); continue
    model = load_model(cls, ckpt_path)
    out_dir = os.path.join(opt.out_root, f"{opt.tag}_{label}", "epoch_best")
    os.makedirs(out_dir, exist_ok=True)

    ps, ss, ls = [], [], []
    with torch.no_grad():
        for i, name in enumerate(names):
            O  = cv2.imread(os.path.join(opt.rain, name))
            gt = cv2.imread(os.path.join(opt.gt, name))
            if O is None:
                continue
            inp = torch.tensor(O[:, :, ::-1].copy().transpose(2, 0, 1)[None],
                               dtype=torch.float32).to(device)
            out = torch.clamp(model(inp)[0][-1], 0., 255.)
            pred = out.cpu().numpy().squeeze().astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1].copy()
            cv2.imwrite(os.path.join(out_dir, name), pred)
            if gt is not None and gt.shape == pred.shape:
                p, s = psnr_ssim_y(pred, gt)
                lp = loss_fn(to_lpips_tensor(pred), to_lpips_tensor(gt)).item()
                ps.append(p); ss.append(s); ls.append(lp)
                persample.append({"variant": label, "name": name, "psnr_out": round(p, 3)})
            del inp, out
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()

    if ps:
        row = {"variant": label, "n": len(ps),
               "psnr_Y": round(float(np.mean(ps)), 4),
               "ssim_Y": round(float(np.mean(ss)), 4),
               "lpips":  round(float(np.mean(ls)), 4)}
        summary.append(row)
        print(f"    -> PSNR_Y={row['psnr_Y']}  SSIM_Y={row['ssim_Y']}  LPIPS={row['lpips']}  "
              f"(images in {out_dir})", flush=True)

os.makedirs(os.path.dirname(opt.summary_csv), exist_ok=True)
with open(opt.summary_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["variant", "n", "psnr_Y", "ssim_Y", "lpips"]); w.writeheader(); w.writerows(summary)
with open(opt.persample_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["variant", "name", "psnr_out"]); w.writeheader(); w.writerows(persample)

print("\n" + "=" * 70)
print("  CDD-11 zero-shot results (attention-ablation variants)")
print("=" * 70)
for r in summary:
    print(f"  {r['variant']:<16} PSNR_Y={r['psnr_Y']:<9} SSIM_Y={r['ssim_Y']:<8} LPIPS={r['lpips']}")
print("=" * 70)
print(f"Saved summary: {opt.summary_csv}")
print(f"Output images: {opt.out_root}/cdd11_<variant>/epoch_best/")

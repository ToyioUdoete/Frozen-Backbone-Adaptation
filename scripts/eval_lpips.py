#!/usr/bin/env python
"""
Fill in the missing LPIPS values for the attention-ablation rows in tab:unified_summary
(Selective Strip Fusion + Attention Mechanism Swaps).

Two kinds of rows:
  (A) Images already saved on disk  -> just score them.
  (B) Only a checkpoint exists       -> rebuild the model with the right GLRA
                                        variant, load the best-epoch checkpoint,
                                        run inference on the Rain100L test set,
                                        save the outputs, then score.

All rows are scored identically:
  - PSNR / SSIM on the Y channel (YCbCr)  -> matches the paper protocol
  - LPIPS on RGB in [-1, 1] with net='alex' -> matches the existing conv-row LPIPS
so the whole LPIPS column is internally comparable.

Output: prints a summary table and writes analysis_outputs/lpips_summary.csv
"""
import os, csv, argparse
from types import SimpleNamespace
import numpy as np
import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as compare_ssim

import BottleCSP
from GLRA_stripfusion import (
    AdditiveFusionGLRA, ConcatFusionGLRA, GatedFusionGLRA,
    ChannelAttnGLRA, SelfAttnGLRA, SwinAttnGLRA,
)

CKPT = "./checkpoints"
RES  = "./results"

# label, kind, variant_class(or None), source_path, epoch, paper_psnr(for reference)
ROWS = [
    # (A) score existing saved images
    ("Add., 1500",      "images", None,               f"{RES}/Rain100L_additive_v2",     96, 36.112),
    ("Concat., 1500",   "images", None,               f"{RES}/Rain100L_concat_v1",       88, 35.475),
    ("Concat., Frozen*","images", None,               f"{RES}/Rain100L_baseline_frozen", 61, 35.995),
    ("Gated, Frozen",   "images", None,               f"{RES}/Rain100L_gated_frozen",    98, 36.2285),
    # (B) regenerate from checkpoint, then score
    ("Add., 50",        "ckpt",   AdditiveFusionGLRA, f"{CKPT}/Rain100L_additive_v3",    92, 32.864),
    ("Concat., 50",     "ckpt",   ConcatFusionGLRA,   f"{CKPT}/Rain100L_concat_v2",      95, 33.092),
    ("Gated, 50",       "ckpt",   GatedFusionGLRA,    f"{CKPT}/Rain100L_gated_v1",       91, 33.123),
    ("Channel Attn.",   "ckpt",   ChannelAttnGLRA,    f"{CKPT}/Rain100L_channel_attn_frozen", 78, 32.729),
    ("Self-Attn.",      "ckpt",   SelfAttnGLRA,       f"{CKPT}/Rain100L_self_attn_frozen",    98, 33.409),
    ("Swin-Style",      "ckpt",   SwinAttnGLRA,       f"{CKPT}/Rain100L_swin_attn_frozen",    98, 34.866),
]

parser = argparse.ArgumentParser()
parser.add_argument("--test_rain", default="./data/Rain100L/test/rain")
parser.add_argument("--test_gt",   default="./data/Rain100L/test/norain")
parser.add_argument("--out_csv",   default="./analysis_outputs/lpips_summary.csv")
parser.add_argument("--num_M", type=int, default=32)
parser.add_argument("--num_Z", type=int, default=32)
parser.add_argument("--T",     type=int, default=4)
parser.add_argument("--S",     type=int, default=20)
opt = parser.parse_args()

device = torch.device("cuda:0")
loss_fn = lpips.LPIPS(net='alex').to(device)


def is_image(n):
    return n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))


def to_lpips_tensor(img_bgr):
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
    return torch.tensor(rgb.transpose(2, 0, 1)[None]).to(device)


def score_pair(pred_bgr, gt_bgr):
    """PSNR_Y, SSIM_Y, LPIPS_RGB for one image pair."""
    pred_y = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    gt_y   = cv2.cvtColor(gt_bgr,  cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    mse = np.mean((pred_y - gt_y) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
    ssim_v = compare_ssim(pred_y, gt_y, data_range=255)
    with torch.no_grad():
        lp = loss_fn(to_lpips_tensor(pred_bgr), to_lpips_tensor(gt_bgr)).item()
    return psnr, ssim_v, lp


def score_image_dir(img_dir):
    psnrs, ssims, lps = [], [], []
    for name in sorted(os.listdir(img_dir)):
        if not is_image(name):
            continue
        pred = cv2.imread(os.path.join(img_dir, name))
        gt   = cv2.imread(os.path.join(opt.test_gt, name.replace("rain-", "norain-")))
        if pred is None or gt is None:
            continue
        p, s, l = score_pair(pred, gt)
        psnrs.append(p); ssims.append(s); lps.append(l)
    return psnrs, ssims, lps


def regenerate_and_score(variant_cls, ckpt_path, save_dir):
    from Network import RAINet                      # imported here so BottleCSP.GLRA is set first
    BottleCSP.GLRA = variant_cls
    model = RAINet(opt).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    missing = len(model_sd) - len(filtered)
    if missing:
        print(f"    [WARN] {missing} model keys not found in checkpoint (loaded {len(filtered)}/{len(model_sd)})")
    model.load_state_dict(filtered, strict=False)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    psnrs, ssims, lps = [], [], []
    with torch.no_grad():
        for name in sorted(os.listdir(opt.test_rain)):
            if not is_image(name):
                continue
            O  = cv2.imread(os.path.join(opt.test_rain, name))
            gt = cv2.imread(os.path.join(opt.test_gt, name.replace("rain-", "norain-")))
            if O is None or gt is None:
                continue
            inp = torch.tensor(O[:, :, ::-1].copy().transpose(2, 0, 1)[None],
                               dtype=torch.float32).to(device)
            out = torch.clamp(model(inp)[0][-1], 0., 255.)
            pred = out.cpu().numpy().squeeze().astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1].copy()
            cv2.imwrite(os.path.join(save_dir, name), pred)
            p, s, l = score_pair(pred, gt)
            psnrs.append(p); ssims.append(s); lps.append(l)
            del inp, out
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return psnrs, ssims, lps


results = []
for label, kind, variant_cls, src, epoch, paper_psnr in ROWS:
    print(f"\n===== {label}  ({kind}) =====")
    if kind == "images":
        img_dir = os.path.join(src, f"epoch_{epoch}")
        if not os.path.isdir(img_dir):
            print(f"    [SKIP] {img_dir} missing"); continue
        psnrs, ssims, lps = score_image_dir(img_dir)
    else:
        ckpt_path = os.path.join(src, f"DerainNet_state_{epoch}.pt")
        save_dir  = os.path.join(RES, os.path.basename(src), f"epoch_{epoch}")
        if not os.path.isfile(ckpt_path):
            print(f"    [SKIP] {ckpt_path} missing"); continue
        psnrs, ssims, lps = regenerate_and_score(variant_cls, ckpt_path, save_dir)

    if not lps:
        print("    [SKIP] no scored images"); continue
    row = {
        "row": label, "n": len(lps),
        "psnr_Y": round(float(np.mean(psnrs)), 4),
        "ssim_Y": round(float(np.mean(ssims)), 4),
        "lpips":  round(float(np.mean(lps)), 4),
        "paper_psnr": paper_psnr,
    }
    results.append(row)
    print(f"    n={row['n']}  PSNR_Y={row['psnr_Y']}  SSIM_Y={row['ssim_Y']}  "
          f"LPIPS={row['lpips']}  (paper PSNR={paper_psnr})")

os.makedirs(os.path.dirname(opt.out_csv), exist_ok=True)
with open(opt.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["row", "n", "psnr_Y", "ssim_Y", "lpips", "paper_psnr"])
    w.writeheader()
    w.writerows(results)

print("\n" + "=" * 78)
print("  SUMMARY — LPIPS to paste into tab:unified_summary (last column)")
print("=" * 78)
print(f"  {'Row':<18}{'LPIPS':>9}   {'(recomputed PSNR_Y / SSIM_Y for sanity)':<40}")
for r in results:
    print(f"  {r['row']:<18}{r['lpips']:>9.4f}   PSNR_Y={r['psnr_Y']:.3f}  SSIM_Y={r['ssim_Y']:.4f}  "
          f"[paper PSNR {r['paper_psnr']}]")
print("=" * 78)
print(f"\nSaved: {opt.out_csv}")

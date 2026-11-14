#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import torch
import time
from types import SimpleNamespace
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

import BottleCSP
from Zoomed import ZoomGLRA_v2
from Network import RAINet

def is_image(name):
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))

def test_zoom(ablation_name, ablation_class, model_path, test_dir, gt_dir, output_dir):
    print(f"\n{'='*70}")
    print(f"Testing {ablation_name} on CDD-11")
    print(f"{'='*70}\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    # Create model
    args = SimpleNamespace(num_M=32, num_Z=32, T=4, S=20)
    BottleCSP.GLRA = ablation_class

    print(f"Creating model...", flush=True)
    model = RAINet(args).to(device)

    print(f"Loading checkpoint: {model_path}", flush=True)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get("state_dict", checkpoint)

    cleaned = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned[key] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    print(f"Model ready!\n", flush=True)

    # Test on all images
    img_names = sorted([f for f in os.listdir(test_dir) if is_image(f)])
    total_count = len(img_names)

    print(f"Found {total_count} images. Starting processing...\n", flush=True)

    psnrs = []
    ssims = []
    total_time = 0.0

    with torch.no_grad():
        for idx, img_name in enumerate(img_names, 1):
            img_path = os.path.join(test_dir, img_name)
            gt_path = os.path.join(gt_dir, img_name)
            rain_bgr = cv2.imread(img_path)
            gt_bgr = cv2.imread(gt_path)

            if rain_bgr is None or gt_bgr is None:
                print(f"[{idx:3d}/{total_count}] SKIP: {img_name} (read failed)", flush=True)
                continue

            rain_rgb = cv2.cvtColor(rain_bgr, cv2.COLOR_BGR2RGB)
            inp = torch.from_numpy(rain_rgb.transpose(2, 0, 1)[None, ...]).float().to(device)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            outputs, _ = model(inp)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start
            total_time += elapsed

            out = torch.clamp(outputs[-1], 0.0, 255.0).cpu().numpy().squeeze().astype(np.uint8)
            out_rgb = out.transpose(1, 2, 0)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

            psnr = compare_psnr(out_bgr, gt_bgr, data_range=255)
            try:
                ssim = compare_ssim(out_bgr, gt_bgr, data_range=255, channel_axis=2)
            except TypeError:
                ssim = compare_ssim(out_bgr, gt_bgr, data_range=255, multichannel=True)

            psnrs.append(psnr)
            ssims.append(ssim)

            cv2.imwrite(os.path.join(output_dir, img_name), out_bgr)

            print(f"[{idx:3d}/{total_count}] {img_name:20s} | PSNR: {psnr:6.3f} | SSIM: {ssim:6.4f} | Time: {elapsed:6.3f}s", flush=True)

    avg_psnr = np.mean(psnrs) if psnrs else 0.0
    avg_ssim = np.mean(ssims) if ssims else 0.0
    avg_time = total_time / max(1, total_count)

    print(f"\n{'='*70}")
    print(f"{ablation_name} Results Summary")
    print(f"{'='*70}")
    print(f"Images processed: {len(psnrs)}")
    print(f"Avg PSNR: {avg_psnr:.4f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"Avg time/image: {avg_time:.4f}s")
    print(f"Results saved to: {output_dir}\n", flush=True)

    # Save metrics
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Ablation: {ablation_name}\n")
        f.write(f"Images: {len(psnrs)}\n")
        f.write(f"Avg PSNR: {avg_psnr:.4f}\n")
        f.write(f"Avg SSIM: {avg_ssim:.4f}\n")
        f.write(f"Avg time/image: {avg_time:.4f}s\n")

if __name__ == "__main__":
    test_zoom(
        "ZoomGLRA",
        ZoomGLRA_v2,
        "./checkpoints/zoom_glra_ablation/zoom_glra/best_zoom_glra.pt",
        "./rain",
        "/path/to/data/data/CDD-11/clear",
        "./ablation_results_cdd11/zoom"
    )

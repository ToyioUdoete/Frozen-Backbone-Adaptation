#!/usr/bin/env python
"""Quick CPU LPIPS for the 4 rows that already have saved images (no GPU/queue needed)."""
import os, numpy as np, cv2, torch, lpips
from skimage.metrics import structural_similarity as compare_ssim

torch.set_num_threads(4)
GT = "./data/Rain100L/test/norain"
ROWS = [
    ("Add., 1500",       "./results/Rain100L_additive_v2/epoch_96",     36.112),
    ("Concat., 1500",    "./results/Rain100L_concat_v1/epoch_88",       35.475),
    ("Concat., Frozen*", "./results/Rain100L_baseline_frozen/epoch_61", 35.995),
    ("Gated, Frozen",    "./results/Rain100L_gated_frozen/epoch_98",    36.2285),
]
dev = torch.device("cpu")
loss_fn = lpips.LPIPS(net='alex').to(dev)

def t(img):  # BGR -> RGB [-1,1]
    rgb = img[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
    return torch.tensor(rgb.transpose(2, 0, 1)[None]).to(dev)

for label, d, paper in ROWS:
    if not os.path.isdir(d):
        print(f"{label:<18} SKIP (missing {d})"); continue
    ps, ss, ls = [], [], []
    for n in sorted(os.listdir(d)):
        if not n.lower().endswith((".png", ".jpg")): continue
        pred = cv2.imread(os.path.join(d, n))
        gt = cv2.imread(os.path.join(GT, n.replace("rain-", "norain-")))
        if pred is None or gt is None: continue
        py = cv2.cvtColor(pred, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
        gy = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
        mse = np.mean((py - gy) ** 2)
        ps.append(20 * np.log10(255. / np.sqrt(mse)) if mse > 0 else 100.)
        ss.append(compare_ssim(py, gy, data_range=255))
        with torch.no_grad():
            ls.append(loss_fn(t(pred), t(gt)).item())
    print(f"{label:<18} LPIPS={np.mean(ls):.4f}   PSNR_Y={np.mean(ps):.3f}  SSIM_Y={np.mean(ss):.4f}  [paper PSNR {paper}]", flush=True)
print("DONE")

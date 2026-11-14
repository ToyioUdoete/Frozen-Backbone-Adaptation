"""
Run inference for key GLRA variants at their best epoch and save output images.
"""
import os, cv2
import numpy as np
import torch
import BottleCSP
from Network import RAINet
from GLRA_stripfusion import GLRA as MultiplicativeGLRA, GatedFusionGLRA

VARIANTS = [
    ("baseline_frozen", MultiplicativeGLRA, "./checkpoints/Rain100L_baseline_frozen", 61,  "./results/Rain100L_baseline_frozen"),
    ("gated_frozen",    GatedFusionGLRA,    "./checkpoints/Rain100L_gated_frozen",    98,  "./results/Rain100L_gated_frozen"),
]

TEST_RAIN = "./data/Rain100L/test/rain"
TEST_GT   = "./data/Rain100L/test/norain"
device    = torch.device("cuda:0")

class _Opt:
    num_M = 32; num_Z = 32; T = 4; S = 20

def is_image(n):
    return n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))

for name, cls, ckpt_dir, epoch, res_dir in VARIANTS:
    print(f"\n{'='*55}\n  {name}  (epoch {epoch})\n{'='*55}")
    epoch_dir = os.path.join(res_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    BottleCSP.GLRA = cls
    model = RAINet(_Opt()).to(device)
    model.eval()

    ckpt_path = os.path.join(ckpt_dir, f"DerainNet_state_{epoch}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)

    psnrs = []
    with torch.no_grad():
        for img_name in sorted(os.listdir(TEST_RAIN)):
            if not is_image(img_name):
                continue
            O  = cv2.imread(os.path.join(TEST_RAIN, img_name))
            gt = cv2.imread(os.path.join(TEST_GT, img_name.replace("rain-", "norain-")))
            if O is None or gt is None:
                continue
            inp = torch.tensor(O[:, :, ::-1].copy().transpose(2, 0, 1)[None],
                               dtype=torch.float32).to(device)
            out = torch.clamp(model(inp)[0][-1], 0., 255.)
            out_bgr = out.cpu().numpy().squeeze().astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1].copy()

            out_y = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
            gt_y  = cv2.cvtColor(gt,      cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
            mse   = np.mean((out_y - gt_y) ** 2)
            psnrs.append(20 * np.log10(255. / np.sqrt(mse)) if mse > 0 else 100.)

            cv2.imwrite(os.path.join(epoch_dir, img_name), out_bgr)

    print(f"  Avg PSNR: {np.mean(psnrs):.4f}  ({len(psnrs)} images)")
    print(f"  Saved to: {epoch_dir}")

print("\nDone.")
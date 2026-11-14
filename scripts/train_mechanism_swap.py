"""
Frozen-backbone mechanism swap ablations:
  - Channel attention (SE block)
  - Standard self-attention
  - Swin window attention

Compare against baseline_frozen results from train_frozen_variants.py.
"""
import os, csv, cv2, time, argparse
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim

import BottleCSP
from Network import RAINet
from DerainDataset import TrainDataset
from GLRA_stripfusion import ChannelAttnGLRA, SelfAttnGLRA, SwinAttnGLRA

parser = argparse.ArgumentParser()
parser.add_argument("--base_ckpt", type=str, required=True)
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
    ("channel_attn_frozen", ChannelAttnGLRA),
    ("self_attn_frozen",    SelfAttnGLRA),
    ("swin_attn_frozen",    SwinAttnGLRA),
]


def is_image(n):
    return n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    skipped = len(sd) - len(filtered)
    if skipped:
        print(f"  [INFO] Skipped {skipped} keys with shape mismatch (new module — randomly initialized)")
    model.load_state_dict(filtered, strict=False)


def freeze_backbone(model, glra_cls):
    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for m in model.modules():
        if isinstance(m, glra_cls):
            for p in m.parameters():
                p.requires_grad = True
                n += p.numel()
    return n


def eval_test(model, data_path, gt_path):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for name in sorted(os.listdir(data_path)):
            if not is_image(name):
                continue
            O = cv2.imread(os.path.join(data_path, name))
            gt = cv2.imread(os.path.join(gt_path, name.replace("rain-", "norain-")))
            if O is None or gt is None:
                continue
            inp = torch.tensor(O[:, :, ::-1].copy().transpose(2, 0, 1)[None],
                               dtype=torch.float32).to(device)
            out = torch.clamp(model(inp)[0][-1], 0., 255.)
            out_bgr = out.cpu().numpy().squeeze().astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1].copy()
            mse = np.mean((out_bgr.astype(np.float64) - gt.astype(np.float64)) ** 2)
            psnrs.append(20 * np.log10(255. / np.sqrt(mse)) if mse > 0 else 100.)
            try:
                ssims.append(compare_ssim(out_bgr, gt, data_range=255, channel_axis=2))
            except TypeError:
                ssims.append(compare_ssim(out_bgr, gt, data_range=255, multichannel=True))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def train_variant(name, cls):
    print(f"\n{'='*60}\n  {name}  (frozen backbone)\n{'='*60}")
    BottleCSP.GLRA = cls

    ckpt_dir = os.path.join(opt.out_root, f"Rain100L_{name}")
    res_dir = os.path.join(opt.results_root, f"Rain100L_{name}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    model = RAINet(opt).to(device)
    load_checkpoint(model, opt.base_ckpt)
    trainable = freeze_backbone(model, cls)
    print(f"  Trainable params: {trainable:,}")

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=opt.lr)
    loader = DataLoader(
        TrainDataset(opt.data_path, opt.gt_path, 64, opt.batch_size * opt.steps_per_epoch),
        batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True
    )

    csv_path = os.path.join(res_dir, "metrics_per_epoch.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "psnr", "ssim", "train_loss"])

    best_psnr, best_epoch = -1., -1
    for epoch in range(opt.epochs):
        model.train()
        loss_sum, tic = 0., time.time()
        for rain, gt in loader:
            rain, gt = rain.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = model(rain, labels=gt)[0][0]
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        psnr, ssim = eval_test(model, opt.test_data_path, opt.test_gt_path)
        loss_avg = loss_sum / len(loader)
        print(f"[{name}] Epoch {epoch+1}/{opt.epochs} "
              f"Loss={loss_avg:.2e} PSNR={psnr:.4f} SSIM={ssim:.4f} time={time.time()-tic:.1f}s")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, round(psnr,4), round(ssim,4), round(loss_avg,4)])

        torch.save({"model": model.state_dict(), "epoch": epoch+1},
                   os.path.join(ckpt_dir, f"DerainNet_state_{epoch+1}.pt"))

        if psnr > best_psnr:
            best_psnr, best_epoch = psnr, epoch + 1

    print(f"\n[{name}] Best PSNR: {best_psnr:.4f} at epoch {best_epoch}")
    return best_psnr, best_epoch


if __name__ == "__main__":
    summary = []
    for name, cls in VARIANTS:
        summary.append((name, *train_variant(name, cls)))

    print("\n" + "="*60)
    print("FINAL SUMMARY (compare against baseline_frozen PSNR=35.9954)")
    print("="*60)
    for name, psnr, epoch in summary:
        diff = psnr - 35.9954
        print(f"  {name}: PSNR={psnr:.4f} (epoch {epoch}) vs baseline: {diff:+.4f} dB")

#!/usr/bin/env python
"""
Train one GLRA strip-fusion variant with proper val/test split and full metrics.

Features:
  - 80/20 train/val split from training data (sorted filenames, reproducible)
  - 1500 steps/epoch with random crop sampling (matches Run A training intensity)
  - Val PSNR tracked every epoch; best-val checkpoint saved as best_val.pt
  - Checkpoint resume (--resume N) loads full optimizer/scheduler state
  - At end: test eval with PSNR+SSIM+LPIPS (Y-channel), saves derained images
  - Per-epoch CSV log with train loss, val PSNR, (test PSNR at end)

Usage:
    python train_variant.py \
        --variant gated \
        --data_path ./data/Rain100L/train/rain \
        --gt_path   ./data/Rain100L/train/norain \
        --test_rain ./data/Rain100L/test/rain \
        --test_gt   ./data/Rain100L/test/norain \
        --model_dir ./checkpoints/Rain100L_gated_v2/ \
        --results_dir ./results/Rain100L_gated_v2 \
        [--resume N] [--freeze_backbone] [--base_ckpt /path/to/ckpt.pt]

Variants: multiplicative | additive | concat | gated | residual_gated
"""

from __future__ import print_function
import argparse, os, csv, random, time
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from math import ceil
from skimage.metrics import structural_similarity as compare_ssim
import lpips as lpips_lib

import BottleCSP
from Network import RAINet
from GLRA_stripfusion import (
    GLRA               as MultiplicativeGLRA,
    AdditiveFusionGLRA,
    ConcatFusionGLRA,
    GatedFusionGLRA,
    ResidualGatedFusionGLRA,
)

VARIANT_MAP = {
    "multiplicative":  MultiplicativeGLRA,
    "additive":        AdditiveFusionGLRA,
    "concat":          ConcatFusionGLRA,
    "gated":           GatedFusionGLRA,
    "residual_gated":  ResidualGatedFusionGLRA,
}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, required=True,
                    choices=list(VARIANT_MAP.keys()),
                    help="Which GLRA fusion variant to train")

parser.add_argument("--data_path", type=str,
                    default="./data/Rain100L/train/rain")
parser.add_argument("--gt_path",   type=str,
                    default="./data/Rain100L/train/norain")
parser.add_argument("--test_rain", type=str,
                    default="./data/Rain100L/test/rain")
parser.add_argument("--test_gt",   type=str,
                    default="./data/Rain100L/test/norain")

parser.add_argument("--model_dir",   type=str, required=True,
                    help="Directory for epoch checkpoints")
parser.add_argument("--results_dir", type=str, required=True,
                    help="Directory for saved test images and CSV")

parser.add_argument("--workers",       type=int,   default=8)
parser.add_argument("--batchSize",     type=int,   default=4)
parser.add_argument("--patchSize",     type=int,   default=128)
parser.add_argument("--niter",         type=int,   default=100)
parser.add_argument("--steps_per_epoch", type=int, default=1500)
parser.add_argument("--lr",            type=float, default=3e-4)
parser.add_argument("--warmup_epochs", type=int,   default=5)
parser.add_argument("--grad_clip",     type=float, default=0.5)
parser.add_argument("--milestone",     type=int,   nargs="+", default=[25, 50, 75])
parser.add_argument("--scheduler",     type=str,   default="multistep",
                    choices=["multistep", "cosine"],
                    help="LR scheduler. cosine recommended for short runs (steps_per_epoch<=50)")
parser.add_argument("--val_split",     type=float, default=0.2,
                    help="Fraction of training images held out for validation")

parser.add_argument("--num_M", type=int, default=32)
parser.add_argument("--num_Z", type=int, default=32)
parser.add_argument("--T",     type=int, default=4)
parser.add_argument("--S",     type=int, default=20)

parser.add_argument("--resume",    type=int,  default=-1,
                    help="Resume from epoch N checkpoint in model_dir")
parser.add_argument("--base_ckpt", type=str,  default=None,
                    help="Optional pre-trained base checkpoint to init from before training")
parser.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze backbone; only train GLRA module parameters")

parser.add_argument("--manualSeed", type=int, default=6488)
parser.add_argument("--gpu_id",     type=str, default=None)

opt = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
if opt.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda:0")
cudnn.benchmark = True

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

os.makedirs(opt.model_dir,   exist_ok=True)
os.makedirs(opt.results_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Train/val split  (deterministic: sort filenames, take last val_split %)
# ---------------------------------------------------------------------------
def is_image(n):
    return n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))

all_train_names = sorted(n for n in os.listdir(opt.data_path) if is_image(n))
n_val   = max(1, int(len(all_train_names) * opt.val_split))
n_train = len(all_train_names) - n_val
train_names = all_train_names[:n_train]
val_names   = all_train_names[n_train:]
print(f"Train: {len(train_names)} images  |  Val: {len(val_names)} images")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    """Random-crop patch dataset that supports a specific file list."""
    def __init__(self, rain_dir, gt_dir, names, patch_size, steps_per_epoch, batch_size):
        self.rain_dir   = rain_dir
        self.gt_dir     = gt_dir
        self.names      = names
        self.patch_size = patch_size
        self.length     = steps_per_epoch * batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name = self.names[idx % len(self.names)]
        O = cv2.imread(os.path.join(self.rain_dir, name))
        gt_name = name.replace("rain-", "norain-")
        B = cv2.imread(os.path.join(self.gt_dir, gt_name))

        # BGR → RGB
        O = O[:, :, ::-1].copy()
        B = B[:, :, ::-1].copy()

        # Random crop
        h, w = O.shape[:2]
        r = random.randint(0, h - self.patch_size)
        c = random.randint(0, w - self.patch_size)
        O = O[r:r + self.patch_size, c:c + self.patch_size]
        B = B[r:r + self.patch_size, c:c + self.patch_size]

        # Random horizontal flip (50 %)
        if random.random() < 0.5:
            O = O[:, ::-1, :].copy()
            B = B[:, ::-1, :].copy()

        O = torch.tensor(O.transpose(2, 0, 1), dtype=torch.float32)
        B = torch.tensor(B.transpose(2, 0, 1), dtype=torch.float32)
        return O, B


# ---------------------------------------------------------------------------
# PSNR helpers  (Y-channel, matching paper protocol)
# ---------------------------------------------------------------------------
def psnr_ycbcr(pred_bgr, gt_bgr):
    pred_y = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    gt_y   = cv2.cvtColor(gt_bgr,  cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    mse = np.mean((pred_y - gt_y) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0

def ssim_ycbcr(pred_bgr, gt_bgr):
    pred_y = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    gt_y   = cv2.cvtColor(gt_bgr,  cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    return compare_ssim(pred_y, gt_y, data_range=255)

def to_lpips_tensor(img_bgr):
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
    return torch.tensor(rgb.transpose(2, 0, 1)[None]).to(device)


# ---------------------------------------------------------------------------
# LR warmup
# ---------------------------------------------------------------------------
def apply_warmup(optimizer, epoch, iteration, num_iter_epoch):
    if opt.warmup_epochs <= 0 or epoch >= opt.warmup_epochs:
        return optimizer.param_groups[0]["lr"]
    warmup_total = max(1, opt.warmup_epochs * num_iter_epoch)
    warmup_iter  = epoch * num_iter_epoch + iteration + 1
    scale = min(1.0, warmup_iter / warmup_total)
    for pg in optimizer.param_groups:
        pg["lr"] = opt.lr * scale
    return opt.lr * scale


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(path, net, optimizer, lr_scheduler, epoch, extra=None):
    ckpt = {
        "model":     net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
        "epoch":     epoch + 1,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_base_ckpt(net, path):
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd   = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model_sd  = net.state_dict()
    filtered  = {k: v for k, v in sd.items()
                 if k in model_sd and v.shape == model_sd[k].shape}
    net.load_state_dict(filtered, strict=False)
    print(f"Loaded base checkpoint: {path}  ({len(filtered)}/{len(model_sd)} keys matched)")


# ---------------------------------------------------------------------------
# Freeze backbone (only train GLRA modules)
# ---------------------------------------------------------------------------
def configure_trainable(net, variant_cls, freeze_backbone):
    if not freeze_backbone:
        for p in net.parameters():
            p.requires_grad_(True)
        n_trainable = sum(p.numel() for p in net.parameters())
    else:
        for p in net.parameters():
            p.requires_grad_(False)
        n_trainable = 0
        for m in net.modules():
            if isinstance(m, variant_cls):
                for p in m.parameters():
                    p.requires_grad_(True)
                    n_trainable += p.numel()
    print(f"Trainable params: {n_trainable:,}  (freeze_backbone={freeze_backbone})")
    return [p for p in net.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Val inference  (PSNR on Y-channel, full images, no patching)
# ---------------------------------------------------------------------------
def eval_val(net):
    net.eval()
    psnrs = []
    with torch.no_grad():
        for name in val_names:
            O  = cv2.imread(os.path.join(opt.data_path, name))
            gt_name = name.replace("rain-", "norain-")
            gt = cv2.imread(os.path.join(opt.gt_path, gt_name))
            if O is None or gt is None:
                continue
            inp = torch.tensor(
                O[:, :, ::-1].copy().transpose(2, 0, 1)[None],
                dtype=torch.float32
            ).to(device)
            out = torch.clamp(net(inp)[0][-1], 0., 255.)
            out_bgr = (out.cpu().numpy().squeeze()
                       .astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1].copy())
            psnrs.append(psnr_ycbcr(out_bgr, gt))
            del inp, out
            torch.cuda.empty_cache()
    net.train()
    return float(np.mean(psnrs)) if psnrs else 0.0


# ---------------------------------------------------------------------------
# Full test eval  (PSNR+SSIM+LPIPS on Y-channel, saves images)
# ---------------------------------------------------------------------------
def eval_test(net, save_dir, lpips_fn):
    net.eval()
    os.makedirs(save_dir, exist_ok=True)
    psnrs, ssims, lps = [], [], []
    test_names = sorted(n for n in os.listdir(opt.test_rain) if is_image(n))
    with torch.no_grad():
        for name in test_names:
            O  = cv2.imread(os.path.join(opt.test_rain, name))
            gt_name = name.replace("rain-", "norain-")
            gt = cv2.imread(os.path.join(opt.test_gt, gt_name))
            if O is None or gt is None:
                continue
            inp = torch.tensor(
                O[:, :, ::-1].copy().transpose(2, 0, 1)[None],
                dtype=torch.float32
            ).to(device)
            out = torch.clamp(net(inp)[0][-1], 0., 255.)
            out_bgr = (out.cpu().numpy().squeeze()
                       .astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1].copy())
            cv2.imwrite(os.path.join(save_dir, name), out_bgr)
            psnrs.append(psnr_ycbcr(out_bgr, gt))
            ssims.append(ssim_ycbcr(out_bgr, gt))
            with torch.no_grad():
                lps.append(float(lpips_fn(
                    to_lpips_tensor(out_bgr), to_lpips_tensor(gt)
                ).item()))
    return (float(np.mean(psnrs))  if psnrs else 0.,
            float(np.mean(ssims))  if ssims else 0.,
            float(np.mean(lps))    if lps   else 0.)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
variant_cls = VARIANT_MAP[opt.variant]
BottleCSP.GLRA = variant_cls

net = RAINet(opt).to(device)
if opt.base_ckpt:
    load_base_ckpt(net, opt.base_ckpt)

trainable_params = configure_trainable(net, variant_cls, opt.freeze_backbone)
optimizer  = optim.Adam(trainable_params, lr=opt.lr)
if opt.scheduler == "cosine":
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=1e-6)
else:
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)

# Resume from epoch checkpoint
if opt.resume >= 0:
    ckpt_path = os.path.join(opt.model_dir, f"DerainNet_state_{opt.resume}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        net.load_state_dict(ckpt)
        for _ in range(opt.resume):
            scheduler.step()
    print(f"Resumed from: {ckpt_path}  lr={optimizer.param_groups[0]['lr']:.2e}")

# DataLoader
train_set = PatchDataset(
    opt.data_path, opt.gt_path, train_names,
    opt.patchSize, opt.steps_per_epoch, opt.batchSize
)
loader = DataLoader(
    train_set,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=opt.workers,
    pin_memory=True,
    persistent_workers=(opt.workers > 0),
)

num_iter_epoch = ceil(len(train_set) / opt.batchSize)

# CSV log
csv_path = os.path.join(opt.results_dir, "metrics_per_epoch.csv")
csv_exists = os.path.isfile(csv_path)
csv_fh = open(csv_path, "a", newline="")
csv_writer = csv.DictWriter(csv_fh, fieldnames=["epoch", "train_loss", "val_psnr"])
if not csv_exists:
    csv_writer.writeheader()

# Best-val tracking
best_val_psnr = -1.0
best_val_ckpt = os.path.join(opt.model_dir, "best_val.pt")

print(f"\n{'='*65}")
print(f"  Variant: {opt.variant}   Epochs: {opt.resume+1}→{opt.niter}")
print(f"  Steps/epoch: {opt.steps_per_epoch}  Batch: {opt.batchSize}  Patch: {opt.patchSize}")
sched_info = f"cosine T_max={opt.niter}" if opt.scheduler=="cosine" else f"multistep milestones={opt.milestone}"
print(f"  Val images: {len(val_names)}   LR: {opt.lr}  Scheduler: {sched_info}")
print(f"{'='*65}\n")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for epoch in range(max(opt.resume, 0), opt.niter):
    net.train()
    epoch_loss = 0.0
    tic = time.time()

    for ii, (im_rain, im_gt) in enumerate(loader):
        if opt.scheduler == "cosine":
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = apply_warmup(optimizer, epoch, ii, num_iter_epoch)
        im_rain = im_rain.cuda(non_blocking=True)
        im_gt   = im_gt.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs, _ = net(im_rain, labels=im_gt)
        loss = outputs[0]
        out  = outputs[-1]

        if not torch.isfinite(loss):
            fail_path = os.path.join(opt.model_dir,
                f"failure_e{epoch+1}_i{ii+1}.pt")
            save_checkpoint(fail_path, net, optimizer, scheduler, epoch,
                            extra={"failed_iter": ii+1, "failed_loss": float(loss)})
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch+1}, iter {ii+1}: {loss.item()}")

        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=opt.grad_clip, error_if_nonfinite=True)
        except RuntimeError as e:
            fail_path = os.path.join(opt.model_dir,
                f"failure_e{epoch+1}_i{ii+1}.pt")
            save_checkpoint(fail_path, net, optimizer, scheduler, epoch)
            raise FloatingPointError(
                f"Non-finite grad at epoch {epoch+1}, iter {ii+1}") from e
        optimizer.step()
        epoch_loss += loss.item()

        if ii % 300 == 0:
            print(f"  [E{epoch+1}/{opt.niter}] {ii}/{num_iter_epoch} "
                  f"loss={loss.item():.2e} lr={lr:.2e}")

    avg_loss = epoch_loss / (ii + 1)
    scheduler.step()

    # Save epoch checkpoint
    save_checkpoint(
        os.path.join(opt.model_dir, f"DerainNet_state_{epoch+1}.pt"),
        net, optimizer, scheduler, epoch)

    # Val PSNR
    val_psnr = eval_val(net)
    if val_psnr > best_val_psnr:
        best_val_psnr = val_psnr
        save_checkpoint(best_val_ckpt, net, optimizer, scheduler, epoch,
                        extra={"best_val_psnr": val_psnr, "best_val_epoch": epoch+1})

    csv_writer.writerow({"epoch": epoch+1,
                         "train_loss": round(avg_loss, 6),
                         "val_psnr":   round(val_psnr, 4)})
    csv_fh.flush()

    print(f"Epoch {epoch+1:3d} | loss={avg_loss:.2e} | "
          f"val_psnr={val_psnr:.4f} | best_val={best_val_psnr:.4f} | "
          f"time={time.time()-tic:.0f}s")
    print("-" * 65)

csv_fh.close()

# ---------------------------------------------------------------------------
# Final test evaluation on best-val checkpoint
# ---------------------------------------------------------------------------
print(f"\nLoading best-val checkpoint ({best_val_ckpt}) for test eval...")
best_ckpt_data = torch.load(best_val_ckpt, map_location=device)
net.load_state_dict(best_ckpt_data["model"])

best_epoch = best_ckpt_data.get("best_val_epoch", "?")
test_save_dir = os.path.join(opt.results_dir, f"epoch_{best_epoch}")

print("Computing LPIPS (loading AlexNet)...")
lpips_fn = lpips_lib.LPIPS(net="alex").to(device)

test_psnr, test_ssim, test_lpips = eval_test(net, test_save_dir, lpips_fn)

print(f"\n{'='*65}")
print(f"  Variant : {opt.variant}")
print(f"  Best val epoch : {best_epoch}  (val PSNR {best_val_psnr:.4f})")
print(f"  Test PSNR  : {test_psnr:.4f}")
print(f"  Test SSIM  : {test_ssim:.4f}")
print(f"  Test LPIPS : {test_lpips:.4f}")
print(f"  Images saved to: {test_save_dir}")
print(f"{'='*65}\n")

# Append test results to CSV
with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([])
    w.writerow(["# TEST RESULTS (best val epoch)"])
    w.writerow(["epoch", "test_psnr", "test_ssim", "test_lpips"])
    w.writerow([best_epoch, round(test_psnr, 4), round(test_ssim, 4), round(test_lpips, 4)])

print("Done. All results written to:", opt.results_dir)

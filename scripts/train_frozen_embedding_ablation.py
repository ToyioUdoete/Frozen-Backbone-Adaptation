import os
import csv
import cv2
import time
import argparse
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from Network import RAINet
import BottleCSP
from DCMLP_Ablations import (
    DirectionalDilationDCMLP,
    DeformableConvDCMLP,
    DynamicMultiDilationDCMLP,
)


try:
    import lpips
except Exception:
    lpips = None


def is_image(name):
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


class PatchPairDataset(Dataset):
    def __init__(self, data_dir, gt_dir, file_names, patch_size=64):
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.file_names = file_names
        self.patch_size = patch_size
        self.rand = random.Random(66)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        rain = cv2.imread(os.path.join(self.data_dir, name))
        gt = cv2.imread(os.path.join(self.gt_dir, name))

        if rain is None or gt is None:
            raise RuntimeError(f"Missing pair for {name}")

        rain = cv2.cvtColor(rain, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        h, w, _ = rain.shape
        p = min(self.patch_size, h, w)
        if h == p:
            top = 0
        else:
            top = self.rand.randint(0, h - p)
        if w == p:
            left = 0
        else:
            left = self.rand.randint(0, w - p)

        rain = rain[top:top + p, left:left + p]
        gt = gt[top:top + p, left:left + p]

        if self.rand.random() < 0.5:
            rain = rain[:, ::-1, :]
            gt = gt[:, ::-1, :]

        rain = rain.astype(np.float32).transpose(2, 0, 1)
        gt = gt.astype(np.float32).transpose(2, 0, 1)

        return torch.tensor(rain), torch.tensor(gt)


def split_train_val(names, val_ratio=0.2, seed=66):
    names = sorted(names)
    rnd = random.Random(seed)
    rnd.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    val_names = names[:n_val]
    train_names = names[n_val:]
    return train_names, val_names


def build_model_args(opt):
    return SimpleNamespace(
        num_M=opt.num_M,
        num_Z=opt.num_Z,
        T=opt.T,
        S=opt.S,
    )


def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    cleaned = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.clone()
        key = k[7:] if k.startswith("module.") else k
        cleaned[key] = v

    model.load_state_dict(cleaned, strict=False)


def freeze_backbone_unfreeze_embedding(model, embedding_cls):
    for p in model.parameters():
        p.requires_grad = False

    trainable_count = 0
    for m in model.modules():
        if isinstance(m, embedding_cls):
            for p in m.parameters():
                p.requires_grad = True
                trainable_count += p.numel()

    return trainable_count


def set_trainable_params(model, embedding_cls, freeze_backbone=True):
    if freeze_backbone:
        trainable_count = freeze_backbone_unfreeze_embedding(model, embedding_cls)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        for p in model.parameters():
            p.requires_grad = True
        trainable_params = list(model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
    return trainable_count, trainable_params


def evaluate_full_images(model, names, data_dir, gt_dir, device, save_dir=None, lpips_metric=None):
    model.eval()
    psnrs = []
    ssims = []
    lpips_scores = []
    total_t = 0.0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for name in names:
            rain_bgr = cv2.imread(os.path.join(data_dir, name))
            gt_bgr = cv2.imread(os.path.join(gt_dir, name))
            if rain_bgr is None or gt_bgr is None:
                continue

            rain_rgb = cv2.cvtColor(rain_bgr, cv2.COLOR_BGR2RGB)
            inp = torch.from_numpy(rain_rgb.transpose(2, 0, 1)[None, ...]).float().to(device)

            start = time.time()
            outputs, _ = model(inp)
            total_t += (time.time() - start)

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

            if lpips_metric is not None:
                out_lp = torch.from_numpy(out_rgb.transpose(2, 0, 1)[None, ...]).float().to(device) / 127.5 - 1.0
                gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
                gt_lp = torch.from_numpy(gt_rgb.transpose(2, 0, 1)[None, ...]).float().to(device) / 127.5 - 1.0
                score = float(lpips_metric(out_lp, gt_lp).item())
                lpips_scores.append(score)

            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, name), out_bgr)

    avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    avg_ssim = float(np.mean(ssims)) if ssims else 0.0
    avg_lpips = float(np.mean(lpips_scores)) if lpips_scores else -1.0
    avg_time = total_t / max(1, len(psnrs))

    return avg_psnr, avg_ssim, avg_lpips, avg_time


def train_variant(opt, variant_name, variant_cls, train_names, val_names, test_names, lpips_metric):
    BottleCSP.DCMLP = variant_cls
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = build_model_args(opt)
    model = RAINet(model_args).to(device)
    load_checkpoint(model, opt.base_ckpt, device)

    trainable_count, trainable_params = set_trainable_params(
        model,
        variant_cls,
        freeze_backbone=opt.freeze_backbone,
    )
    if not trainable_params:
        raise RuntimeError(f"No trainable params found for {variant_name}")

    optimizer = optim.Adam(trainable_params, lr=opt.lr)

    train_set = PatchPairDataset(opt.data_path, opt.gt_path, train_names, patch_size=opt.patch_size)
    loader = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    if len(loader) == 0:
        raise RuntimeError(
            "Training loader has zero batches. Increase dataset size or reduce batch_size."
        )

    variant_dir = os.path.join(opt.out_root, variant_name)
    os.makedirs(variant_dir, exist_ok=True)
    best_ckpt = os.path.join(variant_dir, "best_embedding_only.pt")

    best_val_psnr = -1.0

    print(f"\n[{variant_name}] freeze_backbone={opt.freeze_backbone} Trainable params: {trainable_count}")
    for epoch in range(1, opt.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for rain, gt in loader:
            rain = rain.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs, _ = model(rain, labels=gt)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        epoch_loss /= max(1, len(loader))

        val_psnr, val_ssim, val_lpips, val_t = evaluate_full_images(
            model,
            val_names,
            opt.data_path,
            opt.gt_path,
            device,
            save_dir=None,
            lpips_metric=lpips_metric,
        )

        print(
            f"[{variant_name}] Epoch {epoch}/{opt.epochs} "
            f"Loss={epoch_loss:.4e} ValPSNR={val_psnr:.4f} ValSSIM={val_ssim:.4f} "
            f"ValLPIPS={val_lpips:.4f} Time={val_t:.4f}s"
        )

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), best_ckpt)

    model.load_state_dict(torch.load(best_ckpt, map_location=device), strict=False)
    test_save = os.path.join(variant_dir, "test_outputs")
    test_psnr, test_ssim, test_lpips, test_t = evaluate_full_images(
        model,
        test_names,
        opt.test_data_path,
        opt.test_gt_path,
        device,
        save_dir=test_save,
        lpips_metric=lpips_metric,
    )

    metrics_file = os.path.join(variant_dir, "metrics.txt")
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(f"variant={variant_name}\n")
        f.write(f"freeze_backbone={str(opt.freeze_backbone).lower()}\n")
        f.write(f"trainable_params={trainable_count}\n")
        f.write(f"best_val_psnr={best_val_psnr:.6f}\n")
        f.write(f"test_psnr={test_psnr:.6f}\n")
        f.write(f"test_ssim={test_ssim:.6f}\n")
        f.write(f"test_lpips={test_lpips:.6f}\n")
        f.write(f"test_time_per_image={test_t:.6f}\n")

    return {
        "variant": variant_name,
        "freeze_backbone": str(opt.freeze_backbone).lower(),
        "trainable_params": trainable_count,
        "best_val_psnr": best_val_psnr,
        "test_psnr": test_psnr,
        "test_ssim": test_ssim,
        "test_lpips": test_lpips,
        "test_time_per_image": test_t,
        "metrics_file": metrics_file,
    }


def parse_variants(raw):
    if raw.strip().lower() == "all":
        return ["directional", "deformable", "dynamic"]
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Frozen-backbone embedding ablation training")
    parser.add_argument("--data_path", type=str, required=True, help="train/val rainy images")
    parser.add_argument("--gt_path", type=str, required=True, help="train/val gt images")
    parser.add_argument("--test_data_path", type=str, default=None, help="test rainy images")
    parser.add_argument("--test_gt_path", type=str, default=None, help="test gt images")
    parser.add_argument("--base_ckpt", type=str, default="./checkpoints/Rain100L_single/DerainNet_state_80.pt")
    parser.add_argument("--out_root", type=str, default="./ablation_train_runs")

    parser.add_argument("--variants", type=str, default="all", help="all or comma list: directional,deformable,dynamic")
    parser.add_argument(
        "--freeze_backbone",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="whether to freeze restoration backbone and train only embedding module",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--num_M", type=int, default=32)
    parser.add_argument("--num_Z", type=int, default=32)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--S", type=int, default=20)

    opt = parser.parse_args()

    if opt.test_data_path is None:
        opt.test_data_path = opt.data_path
    if opt.test_gt_path is None:
        opt.test_gt_path = opt.gt_path

    os.makedirs(opt.out_root, exist_ok=True)

    all_names = [n for n in os.listdir(opt.data_path) if is_image(n)]
    all_names = [n for n in all_names if os.path.exists(os.path.join(opt.gt_path, n))]
    if len(all_names) < 5:
        raise RuntimeError("Need at least 5 paired images for train/val split.")

    train_names, val_names = split_train_val(all_names, val_ratio=opt.val_ratio, seed=66)
    test_names = [n for n in os.listdir(opt.test_data_path) if is_image(n)]
    test_names = [n for n in test_names if os.path.exists(os.path.join(opt.test_gt_path, n))]

    lpips_metric = None
    if lpips is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_metric = lpips.LPIPS(net="alex").to(device)
        lpips_metric.eval()
    else:
        print("[WARN] lpips package not available. LPIPS will be reported as -1.")

    variant_map = {
        "directional": DirectionalDilationDCMLP,
        "deformable": DeformableConvDCMLP,
        "dynamic": DynamicMultiDilationDCMLP,
    }

    selected = parse_variants(opt.variants)
    unknown = [v for v in selected if v not in variant_map]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    summary_rows = []
    for name in selected:
        row = train_variant(
            opt,
            variant_name=name,
            variant_cls=variant_map[name],
            train_names=train_names,
            val_names=val_names,
            test_names=test_names,
            lpips_metric=lpips_metric,
        )
        summary_rows.append(row)

    summary_csv = os.path.join(opt.out_root, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "freeze_backbone",
                "trainable_params",
                "best_val_psnr",
                "test_psnr",
                "test_ssim",
                "test_lpips",
                "test_time_per_image",
                "metrics_file",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\nAll variants finished.")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()

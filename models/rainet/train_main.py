#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import argparse
import os
import cv2
import random
import time
import numpy as np
from math import ceil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from Network import RAINet
from DerainDataset import TrainDataset

from ptflops import get_model_complexity_info
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


# ----------------------
# Arguments
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True,
                    help="Path to rainy training images")
parser.add_argument("--gt_path", type=str, required=True,
                    help="Path to ground-truth (clean) training images")

parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batchSize', type=int, default=12)
parser.add_argument('--patchSize', type=int, default=64)
parser.add_argument('--niter', type=int, default=100)

parser.add_argument('--num_M', type=int, default=32)
parser.add_argument('--num_Z', type=int, default=32)
parser.add_argument('--T', type=int, default=4)
parser.add_argument('--S', type=int, default=20)

parser.add_argument('--resume', type=int, default=-1)
parser.add_argument('--milestone', type=int, nargs='+', default=[25, 50, 75])
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--steps_per_epoch', type=int, default=1500)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--grad_clip', type=float, default=0.5)

parser.add_argument('--gpu_id', type=str, default=None,
                    help='Comma-separated GPU IDs, e.g. "0,1,2,3". Defaults to all PBS-allocated GPUs.')

parser.add_argument('--log_dir', default='./checkpoints/Rain100L_single/')
parser.add_argument('--model_dir', default='./checkpoints/Rain100L_single/')

parser.add_argument('--manualSeed', type=int, default=6488)

opt = parser.parse_args()


# ----------------------
# Setup
# ----------------------
if opt.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda:0")

torch.cuda.empty_cache()

os.makedirs(opt.model_dir, exist_ok=True)
os.makedirs(opt.log_dir, exist_ok=True)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

print("Random Seed:", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True


# ----------------------
# Training loop
# ----------------------
def apply_warmup(optimizer, epoch, iteration, num_iter_epoch):
    if opt.warmup_epochs <= 0 or epoch >= opt.warmup_epochs:
        return optimizer.param_groups[0]['lr']

    warmup_total_iters = max(1, opt.warmup_epochs * num_iter_epoch)
    warmup_iter = epoch * num_iter_epoch + iteration + 1
    warmup_scale = min(1.0, warmup_iter / warmup_total_iters)
    warmup_lr = opt.lr * warmup_scale
    for param_group in optimizer.param_groups:
        param_group['lr'] = warmup_lr
    return warmup_lr


def save_checkpoint(path, net, optimizer, lr_scheduler, epoch, extra=None):
    checkpoint = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict(),
        'epoch': epoch + 1,
    }
    if extra is not None:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def save_failure_checkpoint(net, optimizer, lr_scheduler, epoch, iteration, loss):
    failure_path = os.path.join(
        opt.model_dir, f"DerainNet_state_{epoch+1}_iter_{iteration+1}_failure.pt"
    )
    save_checkpoint(
        failure_path,
        net,
        optimizer,
        lr_scheduler,
        epoch,
        extra={
            'failed_iter': iteration + 1,
            'failed_loss': float(loss) if loss is not None else None,
        }
    )
    print(f"Saved failure checkpoint to {failure_path}")


def train_model(net, optimizer, lr_scheduler, dataset):
    data_loader = DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )

    writer = SummaryWriter(opt.log_dir)
    step = 0
    num_data = len(dataset)
    num_iter_epoch = ceil(num_data / opt.batchSize)

    for epoch in range(max(opt.resume, 0), opt.niter):
        mse_per_epoch = 0
        tic = time.time()

        for ii, data in enumerate(data_loader):
            lr = apply_warmup(optimizer, epoch, ii, num_iter_epoch)
            im_rain, im_gt = [x.cuda(non_blocking=True) for x in data]

            net.train()
            optimizer.zero_grad(set_to_none=True)

            outputs, _ = net(im_rain, labels=im_gt)
            loss = outputs[0]
            out = outputs[-1]

            if not torch.isfinite(loss):
                save_failure_checkpoint(net, optimizer, lr_scheduler, epoch, ii, loss)
                raise FloatingPointError(
                    f"Non-finite loss detected at epoch {epoch+1}, iter {ii+1}: {loss.item()}"
                )
            if not torch.isfinite(out).all():
                save_failure_checkpoint(net, optimizer, lr_scheduler, epoch, ii, loss)
                raise FloatingPointError(
                    f"Non-finite output detected at epoch {epoch+1}, iter {ii+1}"
                )

            loss.backward()
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    max_norm=opt.grad_clip,
                    error_if_nonfinite=True
                )
            except RuntimeError as exc:
                save_failure_checkpoint(net, optimizer, lr_scheduler, epoch, ii, loss)
                raise FloatingPointError(
                    f"Non-finite gradient detected at epoch {epoch+1}, iter {ii+1}"
                ) from exc
            optimizer.step()

            mse_iter = loss.item()
            mse_per_epoch += mse_iter

            if ii % 300 == 0:
                out = torch.clamp(out, 0., 255.)
                pre_out = out.detach().cpu().float().numpy().astype(np.float32)
                gt_np = im_gt.detach().cpu().float().numpy().astype(np.float32)

                psnr = []
                for i in range(len(pre_out)):
                    img1 = pre_out[i].transpose(1, 2, 0).astype(np.float64)
                    img2 = gt_np[i].transpose(1, 2, 0).astype(np.float64)
                    mse = np.mean((img1 - img2) ** 2)
                    psnr.append(20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0)

                avg_psnr = sum(psnr) / len(psnr)

                print(f"[Epoch {epoch+1}/{opt.niter}] "
                      f"{ii}/{num_iter_epoch} "
                      f"Loss={mse_iter:.2e} "
                      f"PSNR={avg_psnr:.2f} "
                      f"Grad={float(grad_norm):.2e} "
                      f"lr={lr:.2e}")

                writer.add_scalar('train/Loss', mse_iter, step)
                writer.add_scalar('train/PSNR', avg_psnr, step)
                writer.add_scalar('train/GradNorm', float(grad_norm), step)
                writer.add_scalar('train/LR', lr, step)

            step += 1

        mse_per_epoch /= (ii + 1)
        print(f"Epoch {epoch+1} Avg Loss: {mse_per_epoch:.2e}")

        lr_scheduler.step()

        save_path = os.path.join(
            opt.model_dir, f"DerainNet_state_{epoch+1}.pt"
        )
        save_checkpoint(save_path, net, optimizer, lr_scheduler, epoch)

        print(f"Epoch time: {time.time() - tic:.2f}s")
        print("-" * 80)

    writer.close()
    print("Training finished.")


# ----------------------
# Main
# ----------------------
if __name__ == '__main__':
    netDerain = RAINet(opt).cuda()

    macs, params = get_model_complexity_info(
        netDerain, (3, opt.patchSize, opt.patchSize),
        as_strings=True,
        flops_units="GMac",
        param_units="M",
        print_per_layer_stat=False
    )
    print(f"Computational complexity: {macs}")
    print(f"Number of parameters: {params}")


    optimizerDerain = optim.Adam(netDerain.parameters(), lr=opt.lr)
    schedulerDerain = MultiStepLR(
        optimizerDerain, milestones=opt.milestone, gamma=0.2
    )

    if opt.resume >= 0:
        checkpoint_path = os.path.join(opt.model_dir, f"DerainNet_state_{opt.resume}.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            netDerain.load_state_dict(checkpoint['model'])
            optimizerDerain.load_state_dict(checkpoint['optimizer'])
            schedulerDerain.load_state_dict(checkpoint['scheduler'])
        else:
            # Legacy: file is a bare model state_dict — restore weights and
            # advance the scheduler to the correct LR for this epoch.
            netDerain.load_state_dict(checkpoint)
            for _ in range(opt.resume):
                schedulerDerain.step()
        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"  LR = {optimizerDerain.param_groups[0]['lr']:.2e}")

    train_dataset = TrainDataset(
        opt.data_path,
        opt.gt_path,
        opt.patchSize,
        opt.batchSize * opt.steps_per_epoch
    )

    train_model(netDerain, optimizerDerain, schedulerDerain, train_dataset)

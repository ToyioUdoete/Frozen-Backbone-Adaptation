import csv
import cv2
import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import time

from utils import *
from Network import RAINet
from ptflops import get_model_complexity_info
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


# --------------------------------------------------
# Arguments
# --------------------------------------------------
parser = argparse.ArgumentParser(description="RAINet Test (Single GPU)")

parser.add_argument("--model_dir", type=str, default="./checkpoints/Rain100L_single",
                    help="path to model files")
parser.add_argument("--data_path", type=str,
                    default="/home/user/rainnet/dataset/Rain100L/rain",
                    help="path to testing data")
parser.add_argument("--gt_path", type=str,
                    default="/home/user/rainnet/dataset/Rain100L/norain",
                    help="path to testing gt data")

parser.add_argument("--num_M", type=int, default=32)
parser.add_argument("--num_Z", type=int, default=32)
parser.add_argument("--T", type=int, default=4)
parser.add_argument("--S", type=int, default=20)

parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--save_path", type=str,
                    default="./results/Rain100L_single",
                    help="path to save results")
parser.add_argument("--epoch", type=int, default=None,
                    help="evaluate a single checkpoint epoch (default: evaluate all)")

opt = parser.parse_args()


# --------------------------------------------------
# Setup
# --------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
device = torch.device("cuda:0")

torch.cuda.empty_cache()
os.makedirs(opt.save_path, exist_ok=True)


# --------------------------------------------------
# Utils
# --------------------------------------------------
def print_network(net):
    num_params = sum(p.numel() for p in net.parameters())
    print("Total number of parameters:", num_params)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Loading model...")

    model = RAINet(opt).to(device)
    model.eval()

    macs, params = get_model_complexity_info(
        model, (3, 64, 64),
        as_strings=True,
        flops_units="GMac",
        param_units="M",
        print_per_layer_stat=False
    )

    print(f"Computational complexity: {macs}")
    print(f"Number of parameters: {params}")
    print_network(model)

    psnrs_all_epochs = []

    all_checkpoints = sorted(
        int(f.replace("DerainNet_state_", "").replace(".pt", ""))
        for f in os.listdir(opt.model_dir) if f.endswith(".pt")
    )
    checkpoints = [opt.epoch] if opt.epoch is not None else all_checkpoints
    for epoch in checkpoints:
        torch.cuda.empty_cache()

        epoch_save_dir = os.path.join(opt.save_path, f"epoch_{epoch}")
        os.makedirs(epoch_save_dir, exist_ok=True)

        ckpt_path = os.path.join(opt.model_dir, f"DerainNet_state_{epoch}.pt")
        print(f"\nLoading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)

        psnrs = []
        total_time = 0
        count = 0

        with torch.no_grad():
            for img_name in os.listdir(opt.data_path):
                if not is_image(img_name):
                    continue

                img_path = os.path.join(opt.data_path, img_name)
                gt_path = os.path.join(opt.gt_path, img_name.replace('rain-', 'norain-'))

                O = cv2.imread(img_path)
                gt = cv2.imread(gt_path)

                if O is None or gt is None:
                    print(f"[WARN] Skipping {img_name} (missing file)")
                    continue

                # BGR -> RGB via numpy slice (avoids cv2.split numpy 2.x issue)
                O_rgb = O[:, :, ::-1].copy()
                O_tensor = torch.tensor(O_rgb.transpose(2, 0, 1)[None, ...], dtype=torch.float32).to(device)

                torch.cuda.synchronize()
                start = time.time()
                outputs, _ = model(O_tensor)
                torch.cuda.synchronize()
                total_time += time.time() - start

                out = torch.clamp(outputs[-1], 0., 255.)
                # CHW RGB -> HWC BGR
                out_bgr = np.array(out.cpu().numpy().squeeze(), dtype=np.uint8).transpose(1, 2, 0)[:, :, ::-1].copy()

                # PSNR on Y channel (YCbCr) — matches paper's evaluation protocol
                out_y = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
                gt_y = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
                mse = np.mean((out_y - gt_y) ** 2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
                psnrs.append(psnr)

                cv2.imwrite(os.path.join(epoch_save_dir, img_name), out_bgr)
                count += 1

        avg_psnr = sum(psnrs) / len(psnrs)
        avg_time = total_time / max(count, 1)

        print(f"Epoch {epoch} | Avg PSNR: {avg_psnr:.4f} | Avg Time: {avg_time:.4f}s")
        psnrs_all_epochs.append(avg_psnr)

    print("\nMAX PSNR:", max(psnrs_all_epochs),
          "at epoch index", np.argmax(psnrs_all_epochs))


if __name__ == "__main__":
    main()

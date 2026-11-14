import os,cv2,numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
RESULTS_DIR="./results/Rain100L"
TEST_NORAIN="./data/Rain100L/test/norain"
epoch_dirs=sorted([d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR,d)) and d.startswith("epoch_")],key=lambda x:int(x.split("_")[1]))
best_psnr,best_ssim,best_epoch=0,0,None
for epoch_dir in epoch_dirs:
    result_path=os.path.join(RESULTS_DIR,epoch_dir)
    pred_files=sorted([f for f in os.listdir(result_path) if f.endswith((".png",".jpg"))])
    if not pred_files: continue
    psnrs,ssims=[],[]
    for fname in pred_files:
        pred=cv2.imread(os.path.join(result_path,fname))
        gt=cv2.imread(os.path.join(TEST_NORAIN,fname))
        if pred is None or gt is None: continue
        pred_y=cv2.cvtColor(pred,cv2.COLOR_BGR2YCrCb)[:,:,0]
        gt_y=cv2.cvtColor(gt,cv2.COLOR_BGR2YCrCb)[:,:,0]
        psnrs.append(compare_psnr(gt_y,pred_y,data_range=255))
        ssims.append(compare_ssim(gt_y,pred_y,data_range=255))
    if not psnrs: continue
    avg_psnr=sum(psnrs)/len(psnrs)
    avg_ssim=sum(ssims)/len(ssims)
    print(f"{epoch_dir} -> PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
    if avg_psnr>best_psnr:
        best_psnr,best_ssim,best_epoch=avg_psnr,avg_ssim,epoch_dir
print("="*50)
print(f"BEST: {best_epoch} | PSNR: {best_psnr:.2f} dB | SSIM: {best_ssim:.4f}")
print("Paper Rain100L: PSNR ~40.93 dB | SSIM ~0.989")

"""
Build side-by-side visual comparisons: Rainy | Derained | Ground Truth
Outputs one grid image per variant showing N sample images.

Usage:
    python make_visual_comparisons.py
"""
import os
import cv2
import numpy as np

RAIN_DIR   = "./data/Rain100L/test/rain"
GT_DIR     = "./data/Rain100L/test/norain"
OUT_DIR    = "./visual_comparisons"

# (label, results_dir, best_epoch)
VARIANTS = [
    ("Additive (epoch 96)",                "./results/Rain100L_additive_v2",    96),
    ("Concat (epoch 88)",                  "./results/Rain100L_concat_v1",      88),
    ("Multiplicative frozen (epoch 61)",   "./results/Rain100L_baseline_frozen", 61),
    ("Gated frozen (epoch 98)",            "./results/Rain100L_gated_frozen",    98),
]

# Which images to show — pick a spread across the 100 test images
SAMPLE_NAMES = [
    "rain-010.png",
    "rain-025.png",
    "rain-050.png",
    "rain-075.png",
    "rain-090.png",
]

THUMB_W = 300   # width of each panel
LABEL_H = 30    # height of text header row
FONT    = cv2.FONT_HERSHEY_SIMPLEX
os.makedirs(OUT_DIR, exist_ok=True)


def load_resize(path, w):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h))


def add_label(img, text, color=(255, 255, 255)):
    bar = np.zeros((LABEL_H, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (4, 20), FONT, 0.55, color, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


for label, res_dir, epoch in VARIANTS:
    epoch_dir = os.path.join(res_dir, f"epoch_{epoch}")
    rows = []

    for img_name in SAMPLE_NAMES:
        rain_path   = os.path.join(RAIN_DIR, img_name)
        derain_path = os.path.join(epoch_dir, img_name)
        gt_path     = os.path.join(GT_DIR, img_name.replace("rain-", "norain-"))

        try:
            rain   = load_resize(rain_path,   THUMB_W)
            derain = load_resize(derain_path, THUMB_W)
            gt     = load_resize(gt_path,     THUMB_W)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue

        rain   = add_label(rain,   "Rainy Input",   (200, 200, 200))
        derain = add_label(derain, "Derained",      (100, 255, 100))
        gt     = add_label(gt,     "Ground Truth",  (100, 200, 255))

        # pad all three to same height before hstack
        max_h = max(rain.shape[0], derain.shape[0], gt.shape[0])
        def pad_h(im, h):
            if im.shape[0] < h:
                pad = np.zeros((h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
                return np.vstack([im, pad])
            return im
        row = np.hstack([pad_h(rain, max_h), pad_h(derain, max_h), pad_h(gt, max_h)])
        rows.append(row)

    if not rows:
        print(f"[WARN] No images found for {label}")
        continue

    # add a title bar at the top
    grid_w = rows[0].shape[1]
    title_bar = np.zeros((40, grid_w, 3), dtype=np.uint8)
    cv2.putText(title_bar, label, (8, 28), FONT, 0.8, (255, 255, 100), 2, cv2.LINE_AA)

    # thin separator between rows
    sep = np.full((4, grid_w, 3), 60, dtype=np.uint8)
    grid = title_bar
    for i, row in enumerate(rows):
        grid = np.vstack([grid, row])
        if i < len(rows) - 1:
            grid = np.vstack([grid, sep])

    out_name = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-") + ".png"
    out_path = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out_path, grid)
    print(f"Saved: {out_path}  ({grid.shape[1]}x{grid.shape[0]})")

print("\nDone. All comparisons in ./visual_comparisons/")
# evaluate_test.py
# Compute Dice and IoU on test set (data/test/image & data/test/mask).
# Saves overlays to outputs_test/ and metrics.csv.

import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
import csv
import argparse

from main import UNet  # reuse model definition

MODEL_PATH = Path("models/best_model.pth")
OUT_DIR = Path("outputs_test")
OUT_DIR.mkdir(exist_ok=True)

def dice_np(pred, target, eps=1e-6):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = (pred & target).sum()
    return 2*inter / (pred.sum() + target.sum() + eps)

def iou_np(pred, target, eps=1e-6):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = (pred & target).sum()
    union = (pred | target).sum()
    return inter / (union + eps)

def main(device="cpu"):
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Train first.")
    model = UNet(in_ch=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    img_paths = sorted([p for p in glob(os.path.join("data", "test", "image", "*.*")) if p.lower().endswith((".jpg",".png",".jpeg"))])
    mask_paths = sorted([p for p in glob(os.path.join("data", "test", "mask", "*.*")) if p.lower().endswith((".jpg",".png",".jpeg"))])

    if len(img_paths) != len(mask_paths):
        raise RuntimeError("Test image/mask count mismatch")

    results = []
    for ip, mp in tqdm(list(zip(img_paths, mask_paths))):
        img = np.array(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        mask = (np.array(Image.open(mp).convert("L")) > 127).astype(np.uint8)
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).cpu().numpy()[0,0]
        pred = (prob >= 0.5).astype(np.uint8)
        d = dice_np(pred, mask)
        j = iou_np(pred, mask)
        results.append((os.path.basename(ip), float(d), float(j)))
        # save overlay RGB
        img_rgb = np.stack([img*255, img*255, img*255], axis=-1).astype(np.uint8)
        mask_rgb = np.zeros_like(img_rgb)
        mask_rgb[...,0] = pred*255
        overlay = (0.6*img_rgb + 0.4*mask_rgb).astype(np.uint8)
        out_path = OUT_DIR / f"overlay_{Path(ip).stem}.png"
        Image.fromarray(overlay).save(out_path)

    dices = [r[1] for r in results]
    ious = [r[2] for r in results]
    print(f"Test cases: {len(results)}  Mean Dice: {np.mean(dices):.4f}  Mean IoU: {np.mean(ious):.4f}")

    with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename","dice","iou"])
        w.writerows(results)
    print("Saved overlays and outputs_test/metrics.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(device=args.device)

# compute_thresholds.py
# Derive area thresholds (pixels) from training masks and save to models/thresholds.json

import os, json
from glob import glob
import numpy as np
from PIL import Image
from pathlib import Path

def main(data_dir="data"):
    mask_glob = os.path.join(data_dir, "train", "mask", "*.*")
    mask_paths = sorted([p for p in glob(mask_glob) if p.lower().endswith((".jpg",".png",".jpeg"))])
    if not mask_paths:
        raise RuntimeError("No training masks found in data/train/mask")
    areas = []
    for p in mask_paths:
        m = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        areas.append(int((m > 127).sum()))
    areas = np.array(areas)
    t1 = int(np.percentile(areas, 33))
    t2 = int(np.percentile(areas, 66))
    out = {"t1_px": t1, "t2_px": t2}
    Path("models").mkdir(exist_ok=True)
    with open("models/thresholds.json","w") as f:
        json.dump(out, f)
    print("Saved thresholds:", out)

if __name__ == "__main__":
    main()

import os, glob

def show(split):
    img_dir = os.path.join("data", split, "images")
    mask_dir = os.path.join("data", split, "masks")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*")))
    print(f"\n=== {split.upper()} ===")
    print("Image dir:", img_dir)
    print("Mask dir :", mask_dir)
    print("Images found:", len(imgs))
    print("Masks found :", len(masks))
    if imgs:
        print("Example image:", imgs[0])
    if masks:
        print("Example mask :", masks[0])

for s in ["train", "val"]:
    show(s)

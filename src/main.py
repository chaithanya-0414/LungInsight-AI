# main.py — 2D UNet for lung slice segmentation (JPG/PNG dataset)
# Updated: uses torch.amp current API, safe checkpointing, resumable on interrupt

import os
import argparse
import time
from pathlib import Path
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class LungSliceDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
        if len(self.img_paths) == 0:
            self.img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.jpg")))
        if len(self.mask_paths) == 0:
            self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        if len(self.mask_paths) == 0:
            raise RuntimeError(f"No masks found in {mask_dir}")

        if len(self.img_paths) != len(self.mask_paths):
            raise RuntimeError("Image/mask count mismatch")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 0.5).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask


# --------------------------------------------------
# UNet Model
# --------------------------------------------------
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)          # (B, base, H, W)
        p1 = self.pool(c1)         # (B, base, H/2, W/2)
        c2 = self.enc2(p1)         # (B, base*2, H/2, W/2)
        p2 = self.pool(c2)         # (B, base*2, H/4, W/4)
        c3 = self.enc3(p2)         # (B, base*4, H/4, W/4)
        p3 = self.pool(c3)         # (B, base*4, H/8, W/8)
        c4 = self.enc4(p3)         # (B, base*8, H/8, W/8)

        # Decoder with safe resizing to match skip shapes
        u3 = self.up3(c4)  # may be slightly different shape than c3
        if u3.shape[2:] != c3.shape[2:]:
            u3 = F.interpolate(u3, size=c3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(d3)
        if u2.shape[2:] != c2.shape[2:]:
            u2 = F.interpolate(u2, size=c2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[2:] != c1.shape[2:]:
            u1 = F.interpolate(u1, size=c1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))

        return self.outc(d1)


# --------------------------------------------------
# Dice Loss
# --------------------------------------------------
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return 1 - num / den


# --------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------
def save_checkpoint(path, model, optimizer, scaler, epoch, batch):
    ck = {
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "batch": batch
    }
    torch.save(ck, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state"])
    if optimizer is not None and "opt_state" in ck and ck["opt_state"] is not None:
        optimizer.load_state_dict(ck["opt_state"])
    if scaler is not None and ck.get("scaler_state") is not None:
        scaler.load_state_dict(ck["scaler_state"])
    return ck.get("epoch", 0), ck.get("batch", 0)


# --------------------------------------------------
# TRAIN FUNCTION
# --------------------------------------------------
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    train_dataset = LungSliceDataset(
        os.path.join(args.data_dir, "train", "image"),
        os.path.join(args.data_dir, "train", "mask")
    )
    val_dataset = LungSliceDataset(
        os.path.join(args.data_dir, "valid", "image"),
        os.path.join(args.data_dir, "valid", "mask")
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    # Use new API: specify device type for scaler when using CUDA
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)
    save_every_n_batches = args.save_every_n_batches

    best_dice = -1.0
    start_epoch = 1
    start_batch = 1

    # optionally resume
    if args.resume and os.path.exists(args.resume):
        start_epoch, start_batch = load_checkpoint(args.resume, model, optimizer, scaler, device=device)
        start_epoch = int(start_epoch)
        start_batch = int(start_batch)
        print(f"Resuming from {args.resume} at epoch {start_epoch}, batch {start_batch}")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            t0 = time.time()
            for batch_idx, (img, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), start=1):
                # if resuming mid-epoch skip already processed batches
                if epoch == start_epoch and batch_idx < start_batch:
                    continue

                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                # new autocast usage
                if device.type == "cuda":
                    autocast_ctx = torch.amp.autocast(device_type="cuda")
                else:
                    autocast_ctx = torch.amp.autocast(device_type="cpu")

                with autocast_ctx:
                    pred = model(img)
                    loss = bce(pred, mask) + dice_loss(pred, mask)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

                # periodic checkpoint
                if save_every_n_batches > 0 and (batch_idx % save_every_n_batches) == 0:
                    ckpt_path = checkpoint_dir / f"checkpoint_e{epoch}_b{batch_idx}.pth"
                    save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, batch_idx)
                    print(f"Saved checkpoint: {ckpt_path}")

            avg_train_loss = running_loss / max(1, len(train_loader))

            # validation
            model.eval()
            val_dice = 0.0
            with torch.no_grad():
                for img, mask in val_loader:
                    img = img.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    # use same autocast type for validation
                    if device.type == "cuda":
                        val_autocast = torch.amp.autocast(device_type="cuda")
                    else:
                        val_autocast = torch.amp.autocast(device_type="cpu")
                    with val_autocast:
                        pred = model(img)
                    val_dice += (1 - dice_loss(pred, mask).item())

            val_dice /= max(1, len(val_loader))
            epoch_time = time.time() - t0
            print(f"Epoch {epoch}/{args.epochs} | TrainLoss={avg_train_loss:.4f} | ValDice={val_dice:.4f} | Time={epoch_time:.1f}s")

            # save best model
            if val_dice > best_dice:
                best_dice = val_dice
                best_path = checkpoint_dir / "best_model.pth"
                torch.save(model.state_dict(), best_path)
                print(f"Saved best model: {best_path}")

            # reset start_batch after first resumed epoch
            start_batch = 1

    except KeyboardInterrupt:
        # save checkpoint on interrupt
        ckpt_path = checkpoint_dir / f"interrupt_e{epoch}_b{batch_idx}.pth"
        save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, batch_idx)
        print(f"\nTraining interrupted. Checkpoint saved to {ckpt_path}. You can resume with --resume {ckpt_path}")
        raise

# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers (set 0 on Windows if issues)")
    parser.add_argument("--save_every_n_batches", type=int, default=500, help="Intermediate checkpoint frequency (0 to disable)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(args)

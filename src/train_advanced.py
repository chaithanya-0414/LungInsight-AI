# train_advanced.py
# Enhanced training script using advanced loss functions and metrics

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

# Import advanced modules
from models.losses import ComboLoss, FocalTverskyLoss, get_loss_function
from models.metrics import SegmentationMetrics
from models.attention_unet import AttentionUNet

# Import existing UNet
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double Convolution block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Standard UNet"""
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
        c1 = self.enc1(x)
        p1 = self.pool(c1)
        c2 = self.enc2(p1)
        p2 = self.pool(c2)
        c3 = self.enc3(p2)
        p3 = self.pool(c3)
        c4 = self.enc4(p3)

        u3 = self.up3(c4)
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


class LungSliceDataset(Dataset):
    """Dataset for lung slice segmentation"""
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


def train(args):
    """Training function with advanced features"""
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Load datasets
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

    # Create model
    if args.architecture == 'unet':
        model = UNet(in_ch=1, base=args.base_channels).to(device)
        print(f"Using standard UNet with base={args.base_channels}")
    elif args.architecture == 'attention_unet':
        model = AttentionUNet(in_ch=1, out_ch=1, base=args.base_channels).to(device)
        print(f"Using Attention UNet with base={args.base_channels}")
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create loss function
    if args.loss == 'combo':
        criterion = ComboLoss(bce_weight=1.0, dice_weight=1.0, focal_weight=0.5)
        print("Using Combo Loss (BCE + Dice + Focal)")
    elif args.loss == 'focal_tversky':
        criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5)
        print("Using Focal Tversky Loss")
    else:
        criterion = get_loss_function(args.loss)
        print(f"Using {args.loss} loss")

    # Training loop
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_dice = -1.0

    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        
        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Forward pass
            pred = model(img)
            loss = criterion(pred, mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation with comprehensive metrics
        model.eval()
        val_metrics = {
            'dice': 0.0,
            'iou': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0
        }
        
        with torch.no_grad():
            for img, mask in val_loader:
                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                
                pred = model(img)
                pred_prob = torch.sigmoid(pred)
                
                # Compute metrics
                pred_np = (pred_prob > 0.5).cpu().numpy()
                mask_np = mask.cpu().numpy()
                
                batch_metrics = SegmentationMetrics.compute_all_metrics(pred_np, mask_np)
                
                for key in val_metrics.keys():
                    val_metrics[key] += batch_metrics[key]

        # Average metrics
        for key in val_metrics.keys():
            val_metrics[key] /= len(val_loader)

        epoch_time = time.time() - t0
        
        print(f"\nEpoch {epoch}/{args.epochs} | Time={epoch_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Dice:   {val_metrics['dice']:.4f}")
        print(f"  Val IoU:    {val_metrics['iou']:.4f}")
        print(f"  Val Sens:   {val_metrics['sensitivity']:.4f}")
        print(f"  Val Spec:   {val_metrics['specificity']:.4f}")

        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_path = checkpoint_dir / f"best_model_{args.architecture}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved best model: {best_path}")

    print(f"\n✅ Training completed! Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced training with new features")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--architecture", type=str, default="unet", 
                       choices=["unet", "attention_unet"],
                       help="Model architecture to use")
    parser.add_argument("--loss", type=str, default="combo",
                       choices=["bce", "dice", "focal", "tversky", "focal_tversky", "combo"],
                       help="Loss function to use")
    parser.add_argument("--base_channels", type=int, default=32,
                       help="Base number of channels in UNet")
    
    args = parser.parse_args()
    train(args)

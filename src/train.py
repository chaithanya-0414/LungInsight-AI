import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import glob

# -------------------------
# Model Definition (Same as app.py)
# -------------------------
class DoubleConv(nn.Module):
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

# -------------------------
# Dataset
# -------------------------
class LungDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            # Apply same transform to image and mask
            # Note: This is tricky with standard torchvision transforms if random
            # For simplicity, we'll do basic resizing and tensor conversion here
            # In a real pipeline, use albumentations or functional transforms
            pass
            
        # Resize
        image = image.resize((256, 256), resample=Image.BILINEAR)
        mask = mask.resize((256, 256), resample=Image.NEAREST)
        
        img_np = np.array(image, dtype=np.float32) / 255.0
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
        
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        return img_tensor, mask_tensor

# -------------------------
# Training Loop
# -------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy data if dirs don't exist for demo purposes
    if not os.path.exists(args.img_dir):
        print(f"Creating dummy data in {args.img_dir}")
        os.makedirs(args.img_dir, exist_ok=True)
        os.makedirs(args.mask_dir, exist_ok=True)
        # Create a few dummy images
        for i in range(5):
            img = Image.new('L', (256, 256), color=i*20)
            mask = Image.new('L', (256, 256), color=0)
            # Draw a circle in mask
            from PIL import ImageDraw
            d = ImageDraw.Draw(mask)
            d.ellipse([100, 100, 150, 150], fill=255)
            img.save(os.path.join(args.img_dir, f"img_{i}.png"))
            mask.save(os.path.join(args.mask_dir, f"img_{i}.png"))

    dataset = LungDataset(args.img_dir, args.mask_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = UNet(in_ch=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss() # Combine with Dice in real app
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            for imgs, masks in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1}/{args.epochs} Loss: {epoch_loss/len(dataloader):.4f}")
        
    # Save model
    os.makedirs("models", exist_ok=True)
    save_path = "models/best_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="data/images", help="Path to training images")
    parser.add_argument("--mask_dir", type=str, default="data/masks", help="Path to training masks")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    train(args)

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets as tv_datasets
from PIL import Image
import numpy as np
import kagglehub
import cv2

def get_tumor_roi_mask(image_np):
    """
    Simulates a tumor mask for training images that don't have ground truth.
    Uses intensity thresholding to pick up bright nodules/tumors in CT.
    Returns: binary mask (same shape as image)
    """
    # CT images: tumors are typically bright (-400 to 200 HU, but here 0-255)
    # Simple thresholding on high intensity regions
    _, mask = cv2.threshold(image_np, 180, 255, cv2.THRESH_BINARY)
    
    # Clean up with morphology
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask / 255.0

class CroppedTwoChannelDataset(Dataset):
    def __init__(self, image_folder_ds, transform=None):
        self.base_ds = image_folder_ds
        self.transform = transform
        self.classes = image_folder_ds.classes
        self.class_to_idx = image_folder_ds.class_to_idx

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img_path, label = self.base_ds.samples[idx]
        
        # Load grayscale for ROI detection
        pil_img = Image.open(img_path).convert("L")
        img_np = np.array(pil_img)
        
        # 1. Generate Pseudo-Mask
        mask_np = get_tumor_roi_mask(img_np)
        
        # 2. Extract Bounding Box and Crop
        coords = np.argwhere(mask_np > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            # Add padding
            pad = 10
            y_min, x_min = max(0, y_min-pad), max(0, x_min-pad)
            y_max, x_max = min(img_np.shape[0], y_max+pad), min(img_np.shape[1], x_max+pad)
            
            crop_img = img_np[y_min:y_max, x_min:x_max]
            crop_mask = mask_np[y_min:y_max, x_min:x_max]
        else:
            # Fallback to center crop if no ROI detected
            crop_img = img_np
            crop_mask = mask_np

        # 3. Convert back to PIL for standardized transforms
        crop_img_pil = Image.fromarray(crop_img).resize((224, 224), Image.BILINEAR)
        crop_mask_pil = Image.fromarray((crop_mask * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)
        
        # 4. Transform to Tensors
        # Normalize Image (Channel 0)
        img_t = transforms.ToTensor()(crop_img_pil)
        img_t = transforms.Normalize(mean=[0.485], std=[0.229])(img_t)
        
        # Mask (Channel 1)
        mask_t = transforms.ToTensor()(crop_mask_pil)
        
        # Concatenate to (2, 224, 224)
        two_channel_tensor = torch.cat([img_t, mask_t], dim=0)
        
        return two_channel_tensor, label

def get_subtype_dataloaders(data_dir=None, batch_size=32, num_workers=2):
    if data_dir is None:
        data_dir = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")
            
    train_dir = os.path.join(data_dir, "Data", "train")
    if not os.path.exists(train_dir): train_dir = data_dir
            
    valid_dir = os.path.join(data_dir, "Data", "valid")
    if not os.path.exists(valid_dir): valid_dir = data_dir

    print(f"Building 2-Channel Cropped Dataloaders from: {train_dir}")
    
    # Load base ImageFolder (for class logic)
    train_base = tv_datasets.ImageFolder(root=train_dir)
    valid_base = tv_datasets.ImageFolder(root=valid_dir)
    
    # Print Mapping
    print(f"Label Mapping: {train_base.class_to_idx}")
    
    # Wrap in our 2-channel cropper
    train_ds = CroppedTwoChannelDataset(train_base)
    val_ds = CroppedTwoChannelDataset(valid_base)
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

if __name__ == "__main__":
    loader, _ = get_subtype_dataloaders()
    if loader:
        imgs, labels = next(iter(loader))
        print(f"Sample 2-Channel Batch Shape: {imgs.shape}")
        print(f"Sample Labels: {labels}")

if __name__ == "__main__":
    loader, _ = get_subtype_dataloaders()
    if loader:
        imgs, labels = next(iter(loader))
        print(f"Sample Batch Shape: {imgs.shape}")
        print(f"Sample Labels: {labels}")

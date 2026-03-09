import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard.streamlit_app import load_model as load_unet
from dashboard.streamlit_app import preprocess_pil, postprocess_prob, overlay_rgb, device
from torchvision import transforms

def crop_tumor(image_array, mask_array):
    """
    Finds the bounding box of the positive mask and crops the original image tightly around the tumor.
    Expects grayscale inputs of (H, W) mapped to floats.
    """
    if mask_array.sum() == 0:
        return None # No tumor found

    # skimage regionprops or simple numpy argwhere
    coords = np.argwhere(mask_array == 1)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add a small padding (e.g., 5 pixels)
    pad = 5
    y_min = max(0, y_min - pad)
    y_max = min(mask_array.shape[0], y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(mask_array.shape[1], x_max + pad)
    
    cropped_img = image_array[y_min:y_max, x_min:x_max]
    return cropped_img

def load_classifier(model_path="models/classifier_densenet121_v2.pth"):
    print(f"DEBUG: load_classifier called with {model_path}")
    from src.train_classifier import build_classifier
    
    # Fallback to V1 if V2 not found
    if not os.path.exists(model_path):
        model_path = "models/classifier_densenet121.pth"
        if not os.path.exists(model_path):
            return None, [], 0
        
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get('classes', ['adenocarcinoma', 'large_cell', 'normal', 'squamous_cell'])
    input_channels = checkpoint.get('input_channels', 3 if "v2" not in model_path else 2)
    num_classes = len(classes)
    
    # build_classifier in train_classifier supports input_channels now
    model = build_classifier(num_classes=num_classes, in_channels=input_channels, device=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"DEBUG: load_classifier returning 3 values: model={model is not None}, classes={len(classes)}, in_channels={input_channels}")
    return model, classes, input_channels

def run_full_pipeline(image_path):
    print("--- Running Full 2-Stage Medical AI Pipeline (Phase 2) ---")
    
    # Standardize image
    pil_img = Image.open(image_path).convert("L")
    tensor, arr = preprocess_pil(pil_img)
    tensor = tensor.to(device)
    
    # ---------------------------------------------------------
    # STAGE 1: UNet Segmentation (Detect Tumor)
    # ---------------------------------------------------------
    print("Stage 1: Running UNet Segmentation...")
    unet = load_unet(device=device)
    if unet is None:
        print("Error: UNet model not found. Cannot proceed to Pipeline.")
        return None
        
    with torch.no_grad():
        prob = torch.sigmoid(unet(tensor)).cpu().numpy()[0,0]
    
    mask = postprocess_prob(prob, 0.5)
    tumor_area_px = int(mask.sum())
    
    if tumor_area_px == 0:
         print("Result: No Tumor Detected (Healthy Lung). Exiting early.")
         return {
             "tumor_found": False,
             "subtype": "normal",
             "confidence": 1.0,
             "tumor_area_px": 0
         }
         
    print(f"Tumor Detected! Size: {tumor_area_px} pixels.")
    
    # ---------------------------------------------------------
    # STAGE 2: Tumor Subtype Classification (ROI Crop & Analyze)
    # ---------------------------------------------------------
    print("Stage 2: Running Subtype Classification...")
    classifier, classes, in_channels = load_classifier()
    
    if classifier is None:
        print("Warning: Classifier model not found. Returning Stage 1 results only.")
        return {
             "tumor_found": True,
             "subtype": "Unknown (Model Not Trained)",
             "confidence": 0.0,
             "tumor_area_px": tumor_area_px
        }
        
    # Crop logic based on ROI
    coords = np.argwhere(mask == 1)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    pad = 10
    y_min, x_min = max(0, y_min-pad), max(0, x_min-pad)
    y_max, x_max = min(arr.shape[0], y_max+pad), min(arr.shape[1], x_max+pad)
    
    crop_img = arr[y_min:y_max, x_min:x_max]
    crop_mask = mask[y_min:y_max, x_min:x_max]
    
    # Resize to 224x224
    crop_img_pil = Image.fromarray((crop_img * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    crop_mask_pil = Image.fromarray((crop_mask * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)
    
    if in_channels == 2:
        # 2-Channel Mode (Image + Mask)
        img_t = transforms.ToTensor()(crop_img_pil)
        img_t = transforms.Normalize(mean=[0.485], std=[0.229])(img_t)
        mask_t = transforms.ToTensor()(crop_mask_pil)
        clf_tensor = torch.cat([img_t, mask_t], dim=0).unsqueeze(0).to(device)
    else:
        # Legacy/Full Scan Mode (RGB)
        clf_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        clf_tensor = clf_transform(crop_img_pil.convert("RGB")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = classifier(clf_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    confidence_score, predicted_idx = torch.max(probabilities, 0)
    subtype = classes[predicted_idx.item()]
    
    print(f"Result: Diagnosed as {subtype.upper()} (Confidence: {confidence_score.item()*100:.1f}%)")
    
    return {
         "tumor_found": True,
         "subtype": subtype,
         "confidence": float(confidence_score.item()),
         "tumor_area_px": tumor_area_px,
         "analyzed_img": crop_img_pil
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    
    run_full_pipeline(args.image_path)

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Add repo structure to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard.streamlit_app import load_model, preprocess_pil, postprocess_prob, overlay_rgb, device
from models.grad_cam import GradCAM
from models.uncertainty import MCDropout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--out_dir", type=str, default="assets", help="Directory to save generated samples")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading model...")
    model = load_model(device=device)
    if model is None:
        print("Model not found! Run training first.")
        return
        
    print(f"Processing image: {args.image_path}")
    pil_img = Image.open(args.image_path).convert("L")
    tensor, arr = preprocess_pil(pil_img)
    tensor = tensor.to(device)
    
    # 1. Standard Prediction & Overlay
    print("Running inference...")
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).cpu().numpy()[0,0]
    mask = postprocess_prob(prob, 0.5)
    
    overlay = overlay_rgb(arr, mask, color=(255, 0, 0)) # Red
    Image.fromarray(overlay).save(os.path.join(args.out_dir, "demo_segmentation.png"))
    
    # 2. Grad-CAM
    print("Generating Grad-CAM...")
    grad_cam = GradCAM(model, target_layer=model.dec1)
    cam = grad_cam.generate_cam(tensor)
    overlay_gradcam = grad_cam.overlay_heatmap(arr, cam, alpha=0.5)
    Image.fromarray(overlay_gradcam).save(os.path.join(args.out_dir, "demo_gradcam.png"))
    
    # 3. Uncertainty
    print("Generating Uncertainty Map...")
    mc_model = MCDropout(model, n_samples=10)
    _, uncertainty, _ = mc_model.predict_with_uncertainty(tensor)
    uncertainty_np = uncertainty.cpu().numpy()[0, 0]
    
    # Colorize uncertainty
    uncertainty_colored = cv2.applyColorMap((uncertainty_np * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    uncertainty_colored = cv2.cvtColor(uncertainty_colored, cv2.COLOR_BGR2RGB)
    img_rgb = (np.stack([arr, arr, arr], axis=-1) * 255).astype(np.uint8)
    overlay_unc = cv2.addWeighted(img_rgb, 0.6, uncertainty_colored, 0.4, 0)
    
    Image.fromarray(overlay_unc).save(os.path.join(args.out_dir, "demo_uncertainty.png"))
    
    print(f"Success! Saved 3 demo images to {args.out_dir}/")

if __name__ == "__main__":
    main()

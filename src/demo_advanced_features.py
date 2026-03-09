# demo_advanced_features.py
# Demonstration script showcasing all new advanced features

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Import all new modules
from models.grad_cam import GradCAM, visualize_gradcam_comparison
from models.metrics import SegmentationMetrics, compute_roc_curve, compute_pr_curve
from models.losses import ComboLoss, FocalLoss, TverskyLoss
from models.uncertainty import MCDropout, visualize_uncertainty
from models.ensemble import ModelEnsemble
from models.attention_unet import AttentionUNet
from models.radiomics_extractor import compute_radiomics_features, format_radiomics_report

# Import existing UNet
import sys
sys.path.append('.')
from app import UNet


def demo_gradcam(model, input_tensor, device='cpu'):
    """Demonstrate Grad-CAM visualization"""
    print("\n" + "="*60)
    print("DEMO 1: Grad-CAM Visualization")
    print("="*60)
    
    model.eval()
    model.to(device)
    
    # Create Grad-CAM instance (target the last decoder layer)
    grad_cam = GradCAM(model, target_layer=model.dec1)
    
    # Generate CAM
    cam = grad_cam.generate_cam(input_tensor.to(device))
    
    print(f"✓ Generated Grad-CAM heatmap: {cam.shape}")
    print(f"  - Min value: {cam.min():.4f}")
    print(f"  - Max value: {cam.max():.4f}")
    print(f"  - Mean value: {cam.mean():.4f}")
    
    return cam


def demo_comprehensive_metrics(pred, target):
    """Demonstrate comprehensive metrics"""
    print("\n" + "="*60)
    print("DEMO 2: Comprehensive Clinical Metrics")
    print("="*60)
    
    # Compute all metrics
    metrics = SegmentationMetrics.compute_all_metrics(pred, target)
    
    print("\n📊 Segmentation Metrics:")
    print(f"  Dice Coefficient:      {metrics['dice']:.4f}")
    print(f"  IoU (Jaccard):         {metrics['iou']:.4f}")
    print(f"  Sensitivity (Recall):  {metrics['sensitivity']:.4f}")
    print(f"  Specificity:           {metrics['specificity']:.4f}")
    print(f"  Precision:             {metrics['precision']:.4f}")
    print(f"  F1 Score:              {metrics['f1_score']:.4f}")
    print(f"  Hausdorff Distance:    {metrics['hausdorff_95']:.2f} pixels")
    print(f"  Avg Surface Distance:  {metrics['avg_surface_distance']:.2f} pixels")
    print(f"  Volumetric Similarity: {metrics['volumetric_similarity']:.4f}")
    
    return metrics


def demo_advanced_losses(pred, target):
    """Demonstrate advanced loss functions"""
    print("\n" + "="*60)
    print("DEMO 3: Advanced Loss Functions")
    print("="*60)
    
    # Convert to tensors
    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
    target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float()
    
    # Compute different losses
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
    combo_loss = ComboLoss(bce_weight=1.0, dice_weight=1.0, focal_weight=0.5)
    
    # Note: pred should be logits, so we need to convert back
    pred_logits = torch.logit(torch.clamp(pred_tensor, 1e-7, 1-1e-7))
    
    focal_val = focal_loss(pred_logits, target_tensor)
    tversky_val = tversky_loss(pred_logits, target_tensor)
    combo_val = combo_loss(pred_logits, target_tensor)
    
    print("\n📉 Loss Values:")
    print(f"  Focal Loss:    {focal_val.item():.4f}")
    print(f"  Tversky Loss:  {tversky_val.item():.4f}")
    print(f"  Combo Loss:    {combo_val.item():.4f}")
    
    return focal_val, tversky_val, combo_val


def demo_uncertainty(model, input_tensor, device='cpu'):
    """Demonstrate uncertainty quantification"""
    print("\n" + "="*60)
    print("DEMO 4: Uncertainty Quantification")
    print("="*60)
    
    # Create MC Dropout wrapper
    mc_model = MCDropout(model, n_samples=10)
    
    # Get prediction with uncertainty
    mean_pred, uncertainty, samples = mc_model.predict_with_uncertainty(input_tensor.to(device))
    
    print(f"\n🎲 Monte Carlo Dropout (10 samples):")
    print(f"  Mean prediction shape: {mean_pred.shape}")
    print(f"  Uncertainty shape:     {uncertainty.shape}")
    print(f"  Uncertainty range:     [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
    print(f"  Mean uncertainty:      {uncertainty.mean():.4f}")
    
    # Find high uncertainty regions
    high_uncertainty_threshold = np.percentile(uncertainty.cpu().numpy(), 90)
    high_uncertainty_pixels = (uncertainty > high_uncertainty_threshold).sum().item()
    total_pixels = uncertainty.numel()
    
    print(f"  High uncertainty pixels: {high_uncertainty_pixels}/{total_pixels} ({100*high_uncertainty_pixels/total_pixels:.1f}%)")
    
    return mean_pred, uncertainty


def demo_ensemble(models, input_tensor, device='cpu'):
    """Demonstrate model ensemble"""
    print("\n" + "="*60)
    print("DEMO 5: Model Ensemble")
    print("="*60)
    
    # Create ensemble
    ensemble = ModelEnsemble(models, strategy='average')
    
    # Get prediction with confidence
    ensemble_pred, confidence = ensemble.predict_with_confidence(input_tensor, device=device)
    
    print(f"\n🤝 Ensemble of {len(models)} models:")
    print(f"  Ensemble prediction shape: {ensemble_pred.shape}")
    print(f"  Confidence shape:          {confidence.shape}")
    print(f"  Mean confidence:           {confidence.mean():.4f}")
    print(f"  Min confidence:            {confidence.min():.4f}")
    print(f"  Max confidence:            {confidence.max():.4f}")
    
    return ensemble_pred, confidence


def demo_attention_unet(input_tensor, device='cpu'):
    """Demonstrate Attention UNet"""
    print("\n" + "="*60)
    print("DEMO 6: Attention UNet Architecture")
    print("="*60)
    
    # Create Attention UNet
    att_unet = AttentionUNet(in_ch=1, out_ch=1, base=32)
    att_unet.to(device)
    att_unet.eval()
    
    # Forward pass with attention maps
    with torch.no_grad():
        output, attention_maps = att_unet.forward_with_attention(input_tensor.to(device))
    
    print(f"\n🎯 Attention UNet:")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of attention levels: {len(attention_maps)}")
    
    for level, att_map in attention_maps.items():
        print(f"  {level}: {att_map.shape}")
    
    return output, attention_maps


def demo_radiomics(image, mask):
    """Demonstrate radiomics feature extraction"""
    print("\n" + "="*60)
    print("DEMO 7: Radiomics Feature Extraction")
    print("="*60)
    
    # Extract features
    features = compute_radiomics_features(image, mask)
    
    print(f"\n📐 Extracted {len(features)} radiomics features:")
    print(f"\n  Shape Features:")
    print(f"    - Area: {features['area_pixels']:.2f} pixels")
    print(f"    - Compactness: {features['compactness']:.4f}")
    print(f"    - Eccentricity: {features['eccentricity']:.4f}")
    
    print(f"\n  Intensity Features:")
    print(f"    - Mean: {features['mean_intensity']:.4f}")
    print(f"    - Std: {features['std_intensity']:.4f}")
    print(f"    - Skewness: {features['skewness']:.4f}")
    
    print(f"\n  Texture Features:")
    print(f"    - GLCM Contrast: {features['glcm_contrast']:.4f}")
    print(f"    - GLCM Homogeneity: {features['glcm_homogeneity']:.4f}")
    
    # Generate full report
    report = format_radiomics_report(features)
    
    return features, report


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("Medical Imaging Project Enhancement")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create dummy data for demonstration
    print("\n📦 Creating dummy data...")
    batch_size = 1
    height, width = 256, 256
    
    # Random input image
    input_tensor = torch.randn(batch_size, 1, height, width)
    
    # Random prediction and target (for metrics demo)
    pred = np.random.rand(height, width)
    target = (np.random.rand(height, width) > 0.5).astype(np.float32)
    
    # Create a simple model
    print("📦 Loading model...")
    model = UNet(in_ch=1, base=32)
    model.eval()
    
    # Run all demos
    try:
        # Demo 1: Grad-CAM
        cam = demo_gradcam(model, input_tensor, device)
        
        # Demo 2: Comprehensive Metrics
        metrics = demo_comprehensive_metrics(pred, target)
        
        # Demo 3: Advanced Losses
        losses = demo_advanced_losses(pred, target)
        
        # Demo 4: Uncertainty
        mean_pred, uncertainty = demo_uncertainty(model, input_tensor, device)
        
        # Demo 5: Ensemble (using same model 3 times for demo)
        models = [model, model, model]
        ensemble_pred, confidence = demo_ensemble(models, input_tensor, device)
        
        # Demo 6: Attention UNet
        output, attention_maps = demo_attention_unet(input_tensor, device)
        
        # Demo 7: Radiomics
        features, report = demo_radiomics(pred, target)
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n📝 Summary:")
        print("  ✓ Grad-CAM visualization")
        print("  ✓ 9 comprehensive metrics")
        print("  ✓ 3 advanced loss functions")
        print("  ✓ Uncertainty quantification")
        print("  ✓ Model ensemble")
        print("  ✓ Attention UNet")
        print("  ✓ 25+ radiomics features")
        
        print("\n🎉 Your project now has state-of-the-art capabilities!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

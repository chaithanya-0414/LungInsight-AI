# app_enhanced.py
# Enhanced Streamlit UI with advanced features: Grad-CAM, Comprehensive Metrics, Uncertainty, Radiomics
# Run: streamlit run app_enhanced.py

import io
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import streamlit as st
import torch
import json
import cv2

# Import existing modules
from app import (UNet, load_model, preprocess_pil, postprocess_prob, overlay_rgb,
                 MODEL_PATH, THR_PATH, device, apply_denoising, detect_foreign_objects)

# Import new advanced modules
from models.grad_cam import GradCAM
from models.metrics import SegmentationMetrics
from models.uncertainty import MCDropout, visualize_uncertainty
from models.radiomics_extractor import compute_radiomics_features, format_radiomics_report

# Import report generator
from models.report_generator import generate_full_report

# Page config
st.set_page_config(layout="wide", page_title="Advanced Lung Segmentation")
st.title("🫁 Advanced Lung Segmentation Analysis")
st.markdown("*Powered by State-of-the-Art Deep Learning*")

# Initialize session state for report data
if 'report_data' not in st.session_state:
    st.session_state['report_data'] = {}

# Load model
# Load model
model = load_model(device=device)

# If cached loading fails, try direct loading (bypassing cache)
if model is None:
    st.warning("Cached load_model returned None. Attempting direct load...")
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location=device)
            model = UNet(in_ch=1)
            if isinstance(state, dict) and any(k.startswith("model_state") or k == "model_state" for k in state.keys()):
                if "model_state" in state:
                    model.load_state_dict(state["model_state"])
                elif "state_dict" in state:
                    model.load_state_dict(state["state_dict"])
                else:
                    model.load_state_dict(state)
            else:
                model.load_state_dict(state)
            model.to(device)
            model.eval()
            st.success("Successfully loaded model directly!")
        except Exception as e:
            st.error(f"Direct load failed: {e}")
            model = None

if model is None:
    st.error(f"Model not found at {MODEL_PATH}. Train first.")
    # Debugging for deployment
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Model Path: {MODEL_PATH}")
    st.write(f"Model Path Exists: {MODEL_PATH.exists()}")
    if os.path.exists("models"):
        st.write(f"Contents of models directory: {os.listdir('models')}")
    else:
        st.write("models directory does not exist!")
    st.stop()

# Load thresholds
if THR_PATH.exists():
    with open(THR_PATH, "r") as f:
        thr = json.load(f)
    t1_px = thr.get("t1_px")
    t2_px = thr.get("t2_px")
else:
    t1_px = t2_px = None

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.write(f"**Device:** {device}")

# Advanced features toggles
st.sidebar.subheader("🚀 Advanced Features")
enable_gradcam = st.sidebar.checkbox("Enable Grad-CAM Visualization", value=True)
enable_uncertainty = st.sidebar.checkbox("Enable Uncertainty Quantification", value=True)
enable_radiomics = st.sidebar.checkbox("Enable Radiomics Features", value=True)
enable_comprehensive_metrics = st.sidebar.checkbox("Show Comprehensive Metrics", value=True)

# Basic settings
threshold = st.sidebar.slider("Probability Threshold", 0.1, 0.9, 0.5, 0.05)
use_denoising = st.sidebar.checkbox("Advanced Denoising (NL-Means)")
detect_foreign = st.sidebar.checkbox("Detect Foreign Objects")
remove_foreign = False
inpaint_foreign = False

if detect_foreign:
    st.sidebar.markdown("---")
    st.sidebar.caption("Foreign Object Handling")
    remove_foreign = st.sidebar.checkbox("Exclude from Tumor Mask", value=True, help="Subtract foreign object pixels from the final tumor prediction.")
    inpaint_foreign = st.sidebar.checkbox("Inpaint (Remove from Image)", value=False, help="Fill foreign object area with surrounding texture before analysis.")

# MC Dropout settings (if uncertainty enabled)
if enable_uncertainty:
    mc_samples = st.sidebar.slider("Uncertainty Samples", 5, 20, 10)

# File upload
uploaded = st.file_uploader("📁 Upload Medical Image", type=["jpg", "png", "jpeg"])

if uploaded:
    # Clear report data on new upload
    if 'last_uploaded' not in st.session_state or st.session_state['last_uploaded'] != uploaded.name:
        st.session_state['last_uploaded'] = uploaded.name
        st.session_state['report_data'] = {}

    # Process image
    pil = Image.open(io.BytesIO(uploaded.read()))
    tensor, arr = preprocess_pil(pil)
    
    # Preprocessing
    if use_denoising:
        with st.spinner("Applying advanced denoising..."):
            arr = apply_denoising(arr)
            # Update tensor if denoised
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    
    foreign_mask = None
    if detect_foreign:
        foreign_mask = detect_foreign_objects(arr)
        if foreign_mask.sum() > 0:
            st.warning(f"⚠️ Foreign Object Detected! ({foreign_mask.sum()} pixels)")
            
            # Strategy 2: Inpainting (Pre-processing)
            if inpaint_foreign:
                with st.spinner("Inpainting foreign objects..."):
                    # Convert to uint8 for cv2
                    img_u8 = (arr * 255).astype(np.uint8)
                    mask_u8 = (foreign_mask * 255).astype(np.uint8)
                    # Inpaint
                    inpainted = cv2.inpaint(img_u8, mask_u8, 3, cv2.INPAINT_TELEA)
                    # Update arr and tensor
                    arr = inpainted.astype(np.float32) / 255.0
                    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
                    st.success("✨ Foreign objects inpainted (removed) from analysis image.")
    
    # Create tabs for different analyses
    tabs = st.tabs(["📊 Main Analysis", "🔍 Grad-CAM", "📈 Comprehensive Metrics", "🎲 Uncertainty", "🧬 Radiomics"])
    
    # ==================== TAB 1: MAIN ANALYSIS ====================
    with tabs[0]:
        st.header("Main Segmentation Analysis")
        
        # Get prediction
        tensor = tensor.to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).cpu().numpy()[0, 0]
        mask = postprocess_prob(prob, threshold)

        # Strategy 1: Mask Subtraction (Post-processing)
        if detect_foreign and remove_foreign and foreign_mask is not None:
            # Only remove if we didn't already inpaint (if we inpainted, they shouldn't be there, but safe to check)
            # If inpainting worked perfectly, model shouldn't predict them. 
            # But subtraction is a hard guarantee.
            overlap = (mask == 1) & (foreign_mask == 1)
            if overlap.sum() > 0:
                mask[foreign_mask == 1] = 0
                st.info(f"🛡️ Excluded {overlap.sum()} pixels overlapping with foreign objects from tumor mask.")
        
        # Calculate metrics
        area_px = int(mask.sum())
        total_px = mask.size
        coverage_ratio = area_px / total_px
        cancer_area_pct = coverage_ratio * 100
        
        # Determine stage
        stage_color = (255, 0, 0)
        if area_px == 0:
            stage = "Healthy Lung"
            stage_color = (0, 255, 0)
        elif t1_px and t2_px:
            if area_px <= t1_px:
                stage = "Initial"
                stage_color = (0, 255, 0)
            elif area_px <= t2_px:
                stage = "Mid"
                stage_color = (0, 0, 255)
            else:
                stage = "Final"
                stage_color = (255, 0, 0)
        else:
            stage = "Unknown"
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔴 Tumor Pixels", f"{area_px:,}")
        col2.metric("📏 Total Pixels", f"{total_px:,}")
        col3.metric("📊 Coverage", f"{cancer_area_pct:.2f}%")
        col4.metric("🏥 Stage", stage)
        
        # Progress bar
        st.progress(min(1.0, coverage_ratio))
        
        # Display images
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(arr, caption="Original Image", use_column_width=True)
        with col_img2:
            overlay = overlay_rgb(arr, mask, color=stage_color)
            st.image(overlay, caption="Segmentation Overlay", use_column_width=True)
        
        # Optional: Show probability map
        if st.checkbox("Show Probability Map"):
            st.image(prob, caption="Probability Map", clamp=True, use_column_width=True)

        # Save to session state for report
        st.session_state['report_data']['main'] = {
            'area_px': area_px,
            'coverage_pct': cancer_area_pct,
            'stage': stage,
            'overlay_img': overlay
        }
    
    # ==================== TAB 2: GRAD-CAM ====================
    with tabs[1]:
        if enable_gradcam:
            st.header("🔍 Grad-CAM Visualization")
            st.markdown("*Visualize which regions the model focuses on during prediction*")
            
            try:
                # Create Grad-CAM
                grad_cam = GradCAM(model, target_layer=model.dec1)
                
                with st.spinner("Generating Grad-CAM..."):
                    cam = grad_cam.generate_cam(tensor.to(device))
                    overlay_gradcam = grad_cam.overlay_heatmap(arr, cam, alpha=0.5)
                
                # Display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(arr, caption="Original", use_column_width=True)
                with col2:
                    st.image(cam, caption="Grad-CAM Heatmap", use_column_width=True, clamp=True)
                with col3:
                    st.image(overlay_gradcam, caption="Grad-CAM Overlay", use_column_width=True)
                
                st.success("✅ Grad-CAM shows the regions the model pays attention to")
                
                # Statistics
                st.subheader("Grad-CAM Statistics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Min Activation", f"{cam.min():.4f}")
                col2.metric("Max Activation", f"{cam.max():.4f}")
                col3.metric("Mean Activation", f"{cam.mean():.4f}")

                # Save to session state
                st.session_state['report_data']['gradcam'] = {
                    'max_act': float(cam.max()),
                    'mean_act': float(cam.mean()),
                    'overlay_img': overlay_gradcam
                }
                
            except Exception as e:
                st.error(f"Error generating Grad-CAM: {e}")
        else:
            st.info("Enable Grad-CAM in the sidebar to see visualization")
    
    # ==================== TAB 3: COMPREHENSIVE METRICS ====================
    with tabs[2]:
        if enable_comprehensive_metrics:
            st.header("📈 Comprehensive Clinical Metrics")
            st.markdown("""
            **What is Ground Truth?**
            To calculate accuracy metrics (like Dice Score, Sensitivity), we need a "Ground Truth" mask - 
            this is the *correct* segmentation usually drawn by a radiologist.
            """)
            
            # Need ground truth for comparison
            gt_col1, gt_col2 = st.columns([2, 1])
            with gt_col1:
                gt_upload = st.file_uploader("Upload Ground Truth Mask (PNG/JPG)", type=["png", "jpg"], key="gt_metrics")
            
            with gt_col2:
                st.write("") # Spacer
                st.write("") # Spacer
                if st.button("🎲 Generate Dummy GT"):
                    # Create a dummy GT by slightly perturbing the prediction
                    # This is JUST FOR DEMO purposes
                    dummy_gt = mask.copy()
                    # Erode and dilate to make it slightly different
                    kernel = np.ones((5,5), np.uint8)
                    if np.random.rand() > 0.5:
                        dummy_gt = cv2.dilate(dummy_gt, kernel, iterations=1)
                    else:
                        dummy_gt = cv2.erode(dummy_gt, kernel, iterations=1)
                    
                    # Save to a temporary file so it can be "uploaded" or used
                    dummy_pil = Image.fromarray(dummy_gt * 255)
                    st.session_state['dummy_gt'] = dummy_pil
                    st.success("Generated dummy ground truth (for demo only)")

            # Use uploaded or dummy GT
            gt_pil = None
            if gt_upload:
                gt_pil = Image.open(gt_upload).convert("L").resize((256, 256))
            elif 'dummy_gt' in st.session_state:
                gt_pil = st.session_state['dummy_gt']
            
            if gt_pil:
                gt_arr = (np.array(gt_pil) > 128).astype(np.float32)
                
                with st.spinner("Computing comprehensive metrics..."):
                    metrics = SegmentationMetrics.compute_all_metrics(mask.astype(np.float32), gt_arr)
                
                st.success("✅ Metrics computed successfully!")
                
                # Display metrics in organized layout
                st.subheader("Overlap Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Dice Coefficient", f"{metrics['dice']:.4f}")
                col2.metric("IoU (Jaccard)", f"{metrics['iou']:.4f}")
                col3.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                
                st.subheader("Classification Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Sensitivity (Recall)", f"{metrics['sensitivity']:.4f}")
                col2.metric("Specificity", f"{metrics['specificity']:.4f}")
                col3.metric("Precision", f"{metrics['precision']:.4f}")
                
                st.subheader("Distance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Hausdorff Distance (95%)", f"{metrics['hausdorff_95']:.2f} px")
                col2.metric("Avg Surface Distance", f"{metrics['avg_surface_distance']:.2f} px")
                col3.metric("Volumetric Similarity", f"{metrics['volumetric_similarity']:.4f}")
                
                # Visualization
                st.subheader("Visual Comparison")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(gt_arr, caption="Ground Truth", use_column_width=True)
                with col2:
                    st.image(mask, caption="Prediction", use_column_width=True)
                with col3:
                    # Overlay comparison
                    comparison = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    comparison[gt_arr == 1] = [0, 255, 0]  # Green: GT
                    comparison[mask == 1] = [255, 0, 0]    # Red: Pred
                    comparison[(gt_arr == 1) & (mask == 1)] = [255, 255, 0]  # Yellow: Overlap
                    st.image(comparison, caption="Comparison (Green=GT, Red=Pred, Yellow=Overlap)", use_column_width=True)
                
                # Save to session state
                st.session_state['report_data']['metrics'] = metrics
                st.session_state['report_data']['metrics']['comparison_img'] = comparison

            else:
                st.info("📤 Upload a ground truth mask (or generate a dummy one) to see metrics.")
        else:
            st.info("Enable Comprehensive Metrics in the sidebar")
    
    # ==================== TAB 4: UNCERTAINTY ====================
    with tabs[3]:
        if enable_uncertainty:
            st.header("🎲 Uncertainty Quantification")
            st.markdown("*Estimate prediction confidence using Monte Carlo Dropout*")
            
            try:
                mc_model = MCDropout(model, n_samples=mc_samples)
                
                with st.spinner(f"Running {mc_samples} forward passes..."):
                    mean_pred, uncertainty, samples = mc_model.predict_with_uncertainty(tensor.to(device))
                
                # Convert to numpy
                mean_pred_np = mean_pred.cpu().numpy()[0, 0]
                uncertainty_np = uncertainty.cpu().numpy()[0, 0]
                
                # Display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(mean_pred_np, caption="Mean Prediction", use_column_width=True, clamp=True)
                with col2:
                    st.image(uncertainty_np, caption="Uncertainty Map", use_column_width=True, clamp=True)
                with col3:
                    # Overlay uncertainty on image
                    uncertainty_colored = cv2.applyColorMap((uncertainty_np * 255).astype(np.uint8), cv2.COLORMAP_HOT)
                    uncertainty_colored = cv2.cvtColor(uncertainty_colored, cv2.COLOR_BGR2RGB)
                    img_rgb = (np.stack([arr, arr, arr], axis=-1) * 255).astype(np.uint8)
                    overlay_unc = cv2.addWeighted(img_rgb, 0.6, uncertainty_colored, 0.4, 0)
                    st.image(overlay_unc, caption="Uncertainty Overlay", use_column_width=True)
                
                # Statistics
                st.subheader("Uncertainty Statistics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Uncertainty", f"{uncertainty_np.mean():.4f}")
                col2.metric("Max Uncertainty", f"{uncertainty_np.max():.4f}")
                col3.metric("Min Uncertainty", f"{uncertainty_np.min():.4f}")
                
                # High uncertainty regions
                high_unc_threshold = np.percentile(uncertainty_np, 90)
                high_unc_pixels = (uncertainty_np > high_unc_threshold).sum()
                col4.metric("High Uncertainty Pixels", f"{high_unc_pixels:,}")
                
                st.info(f"ℹ️ High uncertainty regions (top 10%) may require expert review")

                # Save to session state
                st.session_state['report_data']['uncertainty'] = {
                    'mean_unc': float(uncertainty_np.mean()),
                    'max_unc': float(uncertainty_np.max()),
                    'high_unc_px': int(high_unc_pixels),
                    'map_img': overlay_unc
                }
                
            except Exception as e:
                st.error(f"Error computing uncertainty: {e}")
        else:
            st.info("Enable Uncertainty Quantification in the sidebar")
    
    # ==================== TAB 5: RADIOMICS ====================
    with tabs[4]:
        if enable_radiomics:
            st.header("🧬 Radiomics Feature Extraction")
            st.markdown("*Quantitative analysis of segmented regions*")
            
            try:
                with st.spinner("Extracting radiomics features..."):
                    features = compute_radiomics_features(arr, mask)
                
                st.success(f"✅ Extracted {len(features)} radiomics features!")
                
                # Display features in organized tabs
                feature_tabs = st.tabs(["Shape Features", "Intensity Features", "Texture Features", "Full Report"])
                
                with feature_tabs[0]:
                    st.subheader("Shape Features")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Area", f"{features['area_pixels']:.0f} px")
                    col2.metric("Perimeter", f"{features['perimeter']:.2f} px")
                    col3.metric("Compactness", f"{features['compactness']:.4f}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Eccentricity", f"{features['eccentricity']:.4f}")
                    col2.metric("Solidity", f"{features['solidity']:.4f}")
                    col3.metric("Aspect Ratio", f"{features['aspect_ratio']:.4f}")
                
                with feature_tabs[1]:
                    st.subheader("Intensity Features")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Intensity", f"{features['mean_intensity']:.4f}")
                    col2.metric("Std Intensity", f"{features['std_intensity']:.4f}")
                    col3.metric("Intensity Range", f"{features['intensity_range']:.4f}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Skewness", f"{features['skewness']:.4f}")
                    col2.metric("Kurtosis", f"{features['kurtosis']:.4f}")
                    col3.metric("Entropy", f"{features['entropy']:.4f}")
                
                with feature_tabs[2]:
                    st.subheader("Texture Features (GLCM)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Contrast", f"{features['glcm_contrast']:.4f}")
                    col2.metric("Homogeneity", f"{features['glcm_homogeneity']:.4f}")
                    col3.metric("Energy", f"{features['glcm_energy']:.4f}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Correlation", f"{features['glcm_correlation']:.4f}")
                    col2.metric("Dissimilarity", f"{features['glcm_dissimilarity']:.4f}")
                    col3.metric("ASM", f"{features['glcm_asm']:.4f}")
                
                with feature_tabs[3]:
                    st.subheader("Complete Radiomics Report")
                    report = format_radiomics_report(features)
                    st.text(report)
                    
                    # Download button
                    st.download_button(
                        "📥 Download Text Report",
                        report,
                        file_name=f"radiomics_report_{uploaded.name}.txt",
                        mime="text/plain"
                    )
                
                # Save to session state
                st.session_state['report_data']['radiomics'] = features
                
            except Exception as e:
                st.error(f"Error extracting radiomics features: {e}")
        else:
            st.info("Enable Radiomics Features in the sidebar")

    # ==================== REPORT GENERATION ====================
    st.sidebar.markdown("---")
    st.sidebar.header("📄 Report Generation")
    
    if st.sidebar.button("Generate Comprehensive PDF Report"):
        if 'report_data' in st.session_state and st.session_state['report_data']:
            with st.spinner("Generating PDF Report..."):
                try:
                    pdf_bytes = generate_full_report(uploaded.name, st.session_state['report_data'])
                    st.sidebar.download_button(
                        label="📥 Download Full PDF Report",
                        data=pdf_bytes,
                        file_name=f"comprehensive_report_{uploaded.name}.pdf",
                        mime="application/pdf"
                    )
                    st.sidebar.success("Report Ready!")
                except Exception as e:
                    st.sidebar.error(f"Error generating PDF: {e}")
        else:
            st.sidebar.warning("No analysis data available. Please upload an image and run analysis first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Advanced Medical Imaging Analysis System</strong></p>
    <p>Features: Grad-CAM • Comprehensive Metrics • Uncertainty Quantification • Radiomics</p>
</div>
""", unsafe_allow_html=True)

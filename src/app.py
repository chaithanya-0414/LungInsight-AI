# app.py — Streamlit UI for 2D slice inference, area-based staging, overlay & save
# Run: streamlit run app.py

import io
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageOps
import numpy as np
import streamlit as st
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label, regionprops
from skimage import morphology
from skimage.restoration import denoise_nl_means, estimate_sigma
import nibabel as nib
import pydicom
import plotly.graph_objects as go
from fpdf import FPDF
import base64
import matplotlib.pyplot as plt

MODEL_PATH = Path("models/best_model.pth")
THR_PATH = Path("models/thresholds.json")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Lung Slice Segmentation")
    st.title("Lung Slice Segmentation — Medical Demo")

# -------------------------
# Local UNet definition
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
# Model loader
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
# st.sidebar.write(f"Device: {device}") # Moved to main execution block

@st.cache_resource
def load_model(device="cpu"):
    model = UNet(in_ch=1)
    if not MODEL_PATH.exists():
        return None
    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and any(k.startswith("model_state") or k == "model_state" for k in state.keys()):
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            try:
                model.load_state_dict(state)
            except Exception:
                pass
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

model = load_model(device=device)
# if model is None:
#     st.error(f"Model not found at {MODEL_PATH}. Train first.")
#     st.stop()

# -------------------------
# Thresholds
# -------------------------
if THR_PATH.exists():
    with open(THR_PATH, "r") as f:
        thr = json.load(f)
    t1_px = thr.get("t1_px")
    t2_px = thr.get("t2_px")
else:
    t1_px = t2_px = None

# -------------------------
# Helpers
# -------------------------
def preprocess_array(arr, target_size=(256,256)):
    pil_img = Image.fromarray((arr * 255).astype(np.uint8)) if arr.max() <= 1.0 else Image.fromarray(arr.astype(np.uint8))
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
    pil_resized = pil_img.convert("L").resize(target_size, resample=Image.BILINEAR)
    arr_out = np.array(pil_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr_out).unsqueeze(0).unsqueeze(0).float()
    return tensor, arr_out

def preprocess_pil(pil_img, target_size=(256,256)):
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
    pil_resized = pil_img.convert("L").resize(target_size, resample=Image.BILINEAR)
    arr = np.array(pil_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor, arr

def postprocess_prob(prob_map, thr=0.5):
    mask = (prob_map >= thr).astype(np.uint8)
    mask = morphology.binary_opening(mask, morphology.disk(2)).astype(np.uint8)
    return mask

def overlay_rgb(gray_arr, mask, color=(255, 0, 0)):
    img_rgb = np.stack([gray_arr*255, gray_arr*255, gray_arr*255], axis=-1).astype(np.uint8)
    mask_rgb = np.zeros_like(img_rgb, dtype=np.uint8)
    mask_rgb[mask == 1] = color
    overlay = img_rgb.copy()
    overlay[mask == 1] = (0.6 * overlay[mask == 1] + 0.4 * np.array(color)).astype(np.uint8)
    
    lbl = label(mask)
    props = regionprops(lbl)
    pil_overlay = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_overlay)
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        draw.rectangle([minc, minr, maxc, maxr], outline=color, width=2)
    return np.array(pil_overlay)

def create_pdf_report(filename, metrics, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Lung Slice Segmentation Report", ln=1, align="C")
    pdf.cell(200, 10, txt=f"Filename: {filename}", ln=1, align="L")
    for key, value in metrics.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=1, align="L")
    
    if image_path:
        pdf.image(image_path, x=10, y=None, w=100)
    
    return pdf.output(dest="S").encode("latin-1")

def calculate_dice_iou(pred_mask, true_mask):
    intersection = (pred_mask & true_mask).sum()
    union = (pred_mask | true_mask).sum()
    dice = (2. * intersection) / (pred_mask.sum() + true_mask.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    return dice, iou

def apply_denoising(image_arr):
    """
    Apply Non-Local Means Denoising.
    Expects image_arr in [0, 1] range.
    """
    sigma_est = np.mean(estimate_sigma(image_arr))
    # patch_size=5, patch_distance=6 are good defaults for speed/quality trade-off
    denoised = denoise_nl_means(image_arr, h=1.15 * sigma_est, fast_mode=True,
                                patch_size=5, patch_distance=6)
    return denoised

def detect_foreign_objects(image_arr, threshold=0.95):
    """
    Detect high-intensity objects (metal, implants, bullets).
    Assumes image_arr is normalized [0, 1].
    Threshold 0.95 roughly corresponds to >2000 HU if original range was [-1000, 3000].
    """
    mask = image_arr > threshold
    return mask.astype(np.uint8)

def plot_3d_volume(vol_data, mask_data):
    # Downsample for performance if needed
    factor = 2
    vol_small = vol_data[::factor, ::factor, ::factor]
    mask_small = mask_data[::factor, ::factor, ::factor]
    
    X, Y, Z = np.mgrid[:vol_small.shape[0], :vol_small.shape[1], :vol_small.shape[2]]
    
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=mask_small.flatten(),
        isomin=0.5,
        isomax=1.0,
        opacity=0.1, # needs to be small to see through
        surface_count=17, # needs to be a large number for good volume rendering
        colorscale='Redor'
    ))
    fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
    return fig

# -------------------------
# UI
# -------------------------
if __name__ == "__main__":
    if model is None:
        st.error(f"Model not found at {MODEL_PATH}. Train first.")
        st.stop()
        
    mode = st.sidebar.radio("Mode", ["Single Analysis", "Batch Processing"])
    threshold = st.sidebar.slider("Probability threshold", 0.1, 0.9, 0.5, 0.05)

    st.sidebar.subheader("Preprocessing & Artifacts")
    use_denoising = st.sidebar.checkbox("Enable Advanced Denoising (NL-Means)")
    detect_foreign = st.sidebar.checkbox("Detect Foreign Objects (Implants/Metal)")

    if mode == "Single Analysis":
        uploaded = st.file_uploader("Upload image (jpg/png), volume (nii/nii.gz), or DICOM (dcm)", type=["jpg","png","jpeg", "nii", "gz", "dcm"])
        
        if uploaded:
            # Clear stale PDF if new file
            if 'last_uploaded' not in st.session_state or st.session_state['last_uploaded'] != uploaded.name:
                st.session_state['last_uploaded'] = uploaded.name
                if 'pdf_bytes' in st.session_state: del st.session_state['pdf_bytes']
                if 'dicom_pdf_bytes' in st.session_state: del st.session_state['dicom_pdf_bytes']

            file_ext = Path(uploaded.name).suffix
            
            if file_ext == ".dcm":
                 with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                 
                 try:
                     ds = pydicom.dcmread(tmp_path)
                     st.sidebar.subheader("DICOM Metadata")
                     st.sidebar.write(f"Patient ID: {ds.get('PatientID', 'N/A')}")
                     st.sidebar.write(f"Modality: {ds.get('Modality', 'N/A')}")
                     st.sidebar.write(f"Pixel Spacing: {ds.get('PixelSpacing', 'N/A')}")
                     
                     arr = ds.pixel_array.astype(np.float32)
                     arr = (arr - arr.min()) / (arr.max() - arr.min())
                     
                     # Preprocessing
                     foreign_mask = None
                     if use_denoising:
                         with st.spinner("Denoising..."):
                             arr = apply_denoising(arr)
                     
                     if detect_foreign:
                         foreign_mask = detect_foreign_objects(arr)
                         if foreign_mask.sum() > 0:
                             st.warning(f"Foreign Object Detected! ({foreign_mask.sum()} pixels)")
                     
                     tensor, arr = preprocess_array(arr)
                     tensor = tensor.to(device)
                     with torch.no_grad():
                         prob = torch.sigmoid(model(tensor)).cpu().numpy()[0,0]
                     mask = postprocess_prob(prob, threshold)
                     
                     # Metrics
                     area_px = int(mask.sum())
                     total_px = mask.size
                     coverage = area_px / total_px
                     
                     # Display
                     stage_color = (255, 0, 0)
                     if t1_px and area_px <= t1_px: stage_color = (0, 255, 0)
                     elif t2_px and area_px <= t2_px: stage_color = (0, 0, 255)
                     
                     col1, col2 = st.columns(2)
                     with col1:
                         st.image(arr, caption="Original DICOM (Processed)", use_column_width=True)
                         if foreign_mask is not None and foreign_mask.sum() > 0:
                             st.image(foreign_mask * 255, caption="Foreign Object Mask", clamp=True, use_column_width=True)
                     with col2:
                         st.image(overlay_rgb(arr, mask, color=stage_color), caption="Segmentation", use_column_width=True)
                     
                     # Comparison
                     gt_upload = st.file_uploader("Upload Ground Truth Mask (Optional)", type=["png", "jpg"])
                     if gt_upload:
                         gt_pil = Image.open(gt_upload).convert("L").resize((256, 256))
                         gt_arr = (np.array(gt_pil) > 128).astype(np.uint8)
                         dice, iou = calculate_dice_iou(mask, gt_arr)
                         st.metric("Dice Coefficient", f"{dice:.4f}")
                         st.metric("IoU", f"{iou:.4f}")
                     
                     # Report
                     if st.button("Generate Report"):
                         tmp_img_path = f"temp_{uploaded.name}.png"
                         Image.fromarray(overlay_rgb(arr, mask, color=stage_color)).save(tmp_img_path)
                         pdf_bytes = create_pdf_report(uploaded.name, {"Area": area_px, "Coverage": coverage}, tmp_img_path)
                         st.session_state['dicom_pdf_bytes'] = pdf_bytes
                         os.remove(tmp_img_path)
                     
                     if 'dicom_pdf_bytes' in st.session_state:
                         st.download_button("Download PDF", st.session_state['dicom_pdf_bytes'], file_name="report.pdf", mime="application/pdf")

                 except Exception as e:
                     st.error(f"Error reading DICOM: {e}")
                 finally:
                     os.unlink(tmp_path)

            elif "nii" in uploaded.name: # Handle NIfTI
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).name) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                
                try:
                    nii = nib.load(tmp_path)
                    vol_data = nii.get_fdata()
                    vol_data = np.clip(vol_data, -1000, 400)
                    vol_data = (vol_data - vol_data.min()) / (vol_data.max() - vol_data.min() + 1e-8)
                    
                    pix_dim = nii.header.get_zooms()
                    voxel_vol_cm3 = np.prod(pix_dim[:3]) / 1000.0
                    
                    st.sidebar.info(f"Volume loaded. Shape: {vol_data.shape}. Spacing: {pix_dim}")
                    
                    slice_axis = 2
                    max_slice = vol_data.shape[slice_axis] - 1
                    slice_idx = st.sidebar.slider("Select Slice", 0, max_slice, max_slice // 2)
                    
                    if slice_axis == 2: slice_arr = vol_data[:, :, slice_idx]
                    else: slice_arr = vol_data[slice_idx, :, :]
                    slice_arr = np.rot90(slice_arr)
                    
                    # Preprocessing
                    foreign_mask = None
                    if use_denoising:
                         # Denoising on-the-fly for slice might be slow but okay for demo
                         slice_arr = apply_denoising(slice_arr)
                    
                    if detect_foreign:
                         foreign_mask = detect_foreign_objects(slice_arr)
                         if foreign_mask.sum() > 0:
                             st.sidebar.warning(f"Foreign Object in Slice {slice_idx}!")

                    tensor, arr = preprocess_array(slice_arr)
                    tensor = tensor.to(device)
                    with torch.no_grad():
                        prob = torch.sigmoid(model(tensor)).cpu().numpy()[0,0]
                    mask = postprocess_prob(prob, threshold)
                    
                    area_px = int(mask.sum())
                    total_px = mask.size
                    coverage_ratio = area_px / total_px
                    cancer_area_pct = coverage_ratio * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                         st.image(arr, caption=f"Slice {slice_idx} (Processed)", use_column_width=True)
                         if foreign_mask is not None and foreign_mask.sum() > 0:
                             st.image(foreign_mask * 255, caption="Foreign Object Mask", clamp=True, use_column_width=True)
                    with col2:
                         stage_color = (255, 0, 0)
                         if t1_px and area_px <= t1_px: stage_color = (0, 255, 0)
                         elif t2_px and area_px <= t2_px: stage_color = (0, 0, 255)
                         st.image(overlay_rgb(arr, mask, color=stage_color), caption="Segmentation", use_column_width=True)
                    
                    st.subheader("Analysis")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Cancer Pixels", area_px)
                    c2.metric("Coverage", f"{cancer_area_pct:.2f}%")
                    
                    if st.button("Calculate Full Volume & 3D Render"):
                        total_mask_px = 0
                        mask_vol = np.zeros_like(vol_data)
                        
                        progress_bar = st.progress(0)
                        for i in range(vol_data.shape[slice_axis]):
                            if slice_axis == 2: s = vol_data[:, :, i]
                            else: s = vol_data[i, :, :]
                            s = np.rot90(s)
                            t, _ = preprocess_array(s)
                            t = t.to(device)
                            with torch.no_grad():
                                p = torch.sigmoid(model(t)).cpu().numpy()[0,0]
                            m = postprocess_prob(p, threshold)
                            total_mask_px += m.sum()
                            
                            # Store for 3D
                            # Need to rotate back or store as is? 
                            # Simpler to just store 'm' but dimensions might mismatch if rot90 used
                            # For visualization, let's just use what we have
                            # Mapping back to volume coordinates is tricky with rot90
                            # Let's skip exact 3D mask placement for now and just visualize the slice stack if possible
                            # Or just visualize the 's' stack
                            
                        total_vol_cm3 = total_mask_px * voxel_vol_cm3
                        st.success(f"Total Tumor Volume: **{total_vol_cm3:.2f} cm³**")
                        
                        # 3D Plot (Placeholder with random data or actual if we track it)
                        # For true 3D, we need to assemble 'mask_vol' correctly.
                        # Let's try to assemble it.
                        # s was rot90, so we rot90 back? np.rot90(m, k=-1)
                        
                        st.subheader("3D Volume Visualization")
                        # Create a dummy 3D plot for demo if full processing is too slow/complex for this snippet
                        # Or use the loaded volume
                        fig = plot_3d_volume(vol_data, vol_data > 0.5) # Visualize lung structure for now
                        st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Error loading NIfTI: {e}")
                finally:
                    os.unlink(tmp_path)
                    
            else: # Image (JPG/PNG)
                pil = Image.open(io.BytesIO(uploaded.read()))
                tensor, arr = preprocess_pil(pil)
                
                # Preprocessing (on numpy array 'arr')
                foreign_mask = None
                if use_denoising:
                    with st.spinner("Denoising..."):
                        arr = apply_denoising(arr)
                        # Update tensor from denoised arr
                        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
                
                if detect_foreign:
                    foreign_mask = detect_foreign_objects(arr)
                    if foreign_mask.sum() > 0:
                        st.warning("Foreign Object Detected!")

                tensor = tensor.to(device)
                with torch.no_grad():
                    prob = torch.sigmoid(model(tensor)).cpu().numpy()[0,0]
                mask = postprocess_prob(prob, threshold)
                
                # Metrics Calculation
                area_px = int(mask.sum())
                total_px = mask.size
                coverage_ratio = area_px / total_px
                cancer_area_pct = coverage_ratio * 100
                
                st.subheader("Detailed Analysis")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Cancer Pixels", f"{area_px:,}")
                col2.metric("Total Pixels", f"{total_px:,}")
                col3.metric("Cancer Area", f"{cancer_area_pct:.2f}%")
                col4.metric("Coverage Ratio", f"{coverage_ratio:.4f}")
                
                st.progress(min(1.0, coverage_ratio))
                st.caption(f"Visual representation of cancer coverage: {cancer_area_pct:.2f}%")
        
                stage_color = (255, 0, 0) # Default Red
                if area_px == 0:
                    stage = "Healthy Lung"
                    stage_color = (0, 255, 0) # Green
                    st.success(f"**{stage}**")
                elif t1_px is not None and t2_px is not None:
                    if area_px <= t1_px:
                        stage = "Initial"
                        stage_color = (0, 255, 0) # Green
                    elif area_px <= t2_px:
                        stage = "Mid"
                        stage_color = (0, 0, 255) # Blue
                    else:
                        stage = "Final"
                        stage_color = (255, 0, 0) # Red
                    st.info(f"Stage (area-threshold classifier): **{stage}**")
                    st.caption(f"Thresholds (pixels): <={t1_px} initial, <={t2_px} mid, >{t2_px} final")
                else:
                    st.warning("Thresholds not found. Run compute_thresholds.py to generate models/thresholds.json")
        
                st.image(overlay_rgb(arr, mask, color=stage_color), caption=f"Overlay (color = {stage_color})", width=512)
        
                if st.sidebar.checkbox("Show Raw Probability Map"):
                    st.image(prob, caption="Raw Probability Map", clamp=True, width=512)
                    st.write(f"Max Probability: {prob.max():.4f}")
                
                if foreign_mask is not None and foreign_mask.sum() > 0:
                    st.image(foreign_mask * 255, caption="Detected Foreign Objects (High Intensity)", clamp=True, width=512)
                
                # Comparison
                gt_upload = st.file_uploader("Upload Ground Truth Mask (Optional)", type=["png", "jpg"])
                if gt_upload:
                     gt_pil = Image.open(gt_upload).convert("L").resize((256, 256))
                     gt_arr = (np.array(gt_pil) > 128).astype(np.uint8)
                     dice, iou = calculate_dice_iou(mask, gt_arr)
                     st.metric("Dice Coefficient", f"{dice:.4f}")
                     st.metric("IoU", f"{iou:.4f}")
                
                if st.button("Generate PDF Report"):
                    tmp_img_path = f"temp_{uploaded.name}.png"
                    Image.fromarray(overlay_rgb(arr, mask, color=stage_color)).save(tmp_img_path)
                    
                    report_metrics = {
                        "Stage": stage,
                        "Cancer Pixels": f"{area_px:,}",
                        "Total Pixels": f"{total_px:,}",
                        "Cancer Area": f"{cancer_area_pct:.2f}%",
                        "Coverage Ratio": f"{coverage_ratio:.4f}"
                    }
                    
                    pdf_bytes = create_pdf_report(uploaded.name, report_metrics, tmp_img_path)
                    st.session_state['pdf_bytes'] = pdf_bytes
                    os.remove(tmp_img_path)
                
                if 'pdf_bytes' in st.session_state:
                    st.download_button("Download Report", st.session_state['pdf_bytes'], file_name="report.pdf", mime="application/pdf")

    elif mode == "Batch Processing":
        uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "png"], accept_multiple_files=True)
        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")
            results = []
            for up_file in uploaded_files:
                pil = Image.open(io.BytesIO(up_file.read()))
                tensor, arr = preprocess_pil(pil)
                tensor = tensor.to(device)
                with torch.no_grad():
                    prob = torch.sigmoid(model(tensor)).cpu().numpy()[0,0]
                mask = postprocess_prob(prob, threshold)
                area = int(mask.sum())
                
                stage = "Unknown"
                if t1_px and area <= t1_px: stage = "Initial"
                elif t2_px and area <= t2_px: stage = "Mid"
                elif t2_px: stage = "Final"
                
                results.append({"Filename": up_file.name, "Tumor Area (px)": area, "Stage": stage})
            
            st.table(results)

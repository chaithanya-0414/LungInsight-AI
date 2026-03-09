from fpdf import FPDF
import os
import tempfile
from PIL import Image
import numpy as np

class ComprehensiveReportGenerator(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_font("Arial", "", "arial.ttf", uni=True) if os.path.exists("arial.ttf") else None

    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Advanced Medical Imaging Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', fill=True)
        self.ln(5)

    def add_image(self, image_data, caption, width=100):
        """
        Adds an image to the PDF.
        image_data: can be a file path, or a numpy array (which will be saved temporarily)
        """
        temp_path = None
        if isinstance(image_data, np.ndarray):
            # Normalize if needed
            if image_data.max() <= 1.0:
                image_data = (image_data * 255).astype(np.uint8)
            
            # Handle different channel formats
            if len(image_data.shape) == 2: # Grayscale
                img = Image.fromarray(image_data)
            elif len(image_data.shape) == 3:
                if image_data.shape[0] == 3: # CHW -> HWC
                    image_data = image_data.transpose(1, 2, 0)
                # RGB or RGBA
                img = Image.fromarray(image_data)
            else:
                return # Unsupported format

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                img.save(tmp.name)
                temp_path = tmp.name
            
            image_path = temp_path
        else:
            image_path = image_data

        try:
            # Center the image
            x_centered = (210 - width) / 2
            self.image(image_path, x=x_centered, w=width)
            self.ln(2)
            self.set_font('Arial', 'I', 10)
            self.cell(0, 5, caption, 0, 1, 'C')
            self.ln(5)
        except Exception as e:
            print(f"Error adding image: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def add_key_value(self, key, value):
        self.set_font('Arial', 'B', 10)
        self.cell(50, 8, f"{key}:", 0, 0)
        self.set_font('Arial', '', 10)
        self.cell(0, 8, f"{value}", 0, 1)

def generate_full_report(filename, data):
    """
    Generates a comprehensive PDF report.
    data: Dictionary containing all analysis results
    """
    pdf = ComprehensiveReportGenerator()
    
    # --- Page 1: Main Analysis ---
    pdf.add_page()
    pdf.add_section_title("1. Main Segmentation Analysis")
    pdf.add_key_value("Filename", filename)
    
    if 'main' in data:
        m = data['main']
        pdf.add_key_value("Tumor Pixels", m.get('area_px', 'N/A'))
        pdf.add_key_value("Coverage", f"{m.get('coverage_pct', 0):.2f}%")
        pdf.add_key_value("Stage", m.get('stage', 'N/A'))
        
        if 'overlay_img' in m:
            pdf.add_image(m['overlay_img'], "Segmentation Result", width=120)

    # --- Page 2: Grad-CAM ---
    if 'gradcam' in data:
        pdf.add_page()
        pdf.add_section_title("2. Model Interpretability (Grad-CAM)")
        gc = data['gradcam']
        
        if 'overlay_img' in gc:
            pdf.add_image(gc['overlay_img'], "Grad-CAM Attention Map", width=120)
            
        pdf.add_key_value("Max Activation", f"{gc.get('max_act', 0):.4f}")
        pdf.add_key_value("Mean Activation", f"{gc.get('mean_act', 0):.4f}")
        pdf.multi_cell(0, 5, "The heatmap above shows regions that most influenced the model's prediction. Red/Yellow areas indicate high importance.")

    # --- Page 3: Comprehensive Metrics ---
    if 'metrics' in data:
        pdf.add_page()
        pdf.add_section_title("3. Clinical Metrics Evaluation")
        met = data['metrics']
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, "Overlap Metrics:", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(60, 8, f"Dice Coefficient: {met.get('dice', 0):.4f}", 1)
        pdf.cell(60, 8, f"IoU: {met.get('iou', 0):.4f}", 1)
        pdf.cell(60, 8, f"F1 Score: {met.get('f1_score', 0):.4f}", 1)
        pdf.ln(10)
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, "Classification Metrics:", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(60, 8, f"Sensitivity: {met.get('sensitivity', 0):.4f}", 1)
        pdf.cell(60, 8, f"Specificity: {met.get('specificity', 0):.4f}", 1)
        pdf.cell(60, 8, f"Precision: {met.get('precision', 0):.4f}", 1)
        pdf.ln(10)
        
        if 'comparison_img' in met:
            pdf.add_image(met['comparison_img'], "Ground Truth vs Prediction Comparison", width=120)

    # --- Page 4: Uncertainty ---
    if 'uncertainty' in data:
        pdf.add_page()
        pdf.add_section_title("4. Uncertainty Quantification")
        unc = data['uncertainty']
        
        if 'map_img' in unc:
            pdf.add_image(unc['map_img'], "Uncertainty Map (Monte Carlo Dropout)", width=120)
            
        pdf.add_key_value("Mean Uncertainty", f"{unc.get('mean_unc', 0):.4f}")
        pdf.add_key_value("Max Uncertainty", f"{unc.get('max_unc', 0):.4f}")
        pdf.add_key_value("High Uncertainty Pixels", f"{unc.get('high_unc_px', 0)}")

    # --- Page 5: Radiomics ---
    if 'radiomics' in data:
        pdf.add_page()
        pdf.add_section_title("5. Radiomics Features")
        rad = data['radiomics']
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, "Shape Features:", 0, 1)
        pdf.set_font('Arial', '', 9)
        pdf.multi_cell(0, 5, f"Area: {rad.get('area_pixels', 0):.0f}\nPerimeter: {rad.get('perimeter', 0):.2f}\nCompactness: {rad.get('compactness', 0):.4f}\nEccentricity: {rad.get('eccentricity', 0):.4f}")
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, "Intensity Features:", 0, 1)
        pdf.set_font('Arial', '', 9)
        pdf.multi_cell(0, 5, f"Mean: {rad.get('mean_intensity', 0):.4f}\nStd Dev: {rad.get('std_intensity', 0):.4f}\nSkewness: {rad.get('skewness', 0):.4f}\nKurtosis: {rad.get('kurtosis', 0):.4f}")
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, "Texture Features (GLCM):", 0, 1)
        pdf.set_font('Arial', '', 9)
        pdf.multi_cell(0, 5, f"Contrast: {rad.get('glcm_contrast', 0):.4f}\nHomogeneity: {rad.get('glcm_homogeneity', 0):.4f}\nEnergy: {rad.get('glcm_energy', 0):.4f}\nCorrelation: {rad.get('glcm_correlation', 0):.4f}")

    return pdf.output(dest="S").encode("latin-1")

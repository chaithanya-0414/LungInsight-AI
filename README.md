# LungInsight AI: Advanced Lung Cancer Analysis Platform 🫁

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview
**LungInsight AI** is a state-of-the-art medical imaging platform designed for automated lung tumor analysis. It goes beyond simple segmentation by providing histological subtype classification and clinical staging, assisted by explainable AI (Grad-CAM) and uncertainty quantification.

## 🎯 Key Features
- **Precision Segmentation**: Pixel-level tumor identification using UNet and Attention-UNet architectures.
- **Histological Subtyping**: ROI-focused classification into **Adenocarcinoma**, **Squamous Cell Carcinoma**, and **Large Cell Carcinoma**.
- **Clinical Staging (The Roots)**: Automated staging (Initial, Mid, Final) based on tumor area thresholds.
- **Explainable AI (XAI)**: Visual interpretation of model decisions through Grad-CAM heatmaps.
- **Uncertainty Quantification**: Monte Carlo Dropout-based confidence mapping for clinical reliability.
- **Radiomics Extraction**: Automated calculation of 50+ specialized features (Shape, Intensity, Texture).
- **Interactive Dashboard**: A comprehensive Streamlit-based workspace for radiologists.

## 🏗️ Technical Architecture
The system employs a dual-stage deep learning pipeline for maximum accuracy:

1. **Segmentation Stage**: A UNet model extracts the tumor perimeter from raw CT slices.
2. **Classification Stage**: A DenseNet121 model analyzes a 2-channel ROI (Image + Mask) to determine the tumor subtype.

```mermaid
graph TD
    A[Raw CT scan] --> B[UNet Segmentation]
    B --> C[Tumor Mask]
    C --> D[ROI Extraction & Cropping]
    D --> E[DenseNet121 Classifier]
    E --> F[Subtype Diagnosis]
    C --> G[Staging Logic]
    G --> H[Initial/Mid/Final Stage]
```

## � Example Outputs

### 1. Main Segmentation & Subtyping
The dashboard provides clear overlays and classification metrics.
![Main Analysis](https://raw.githubusercontent.com/your-username/lung-cancer-detection-ai/main/docs/assets/main_analysis.png)
*(Example: Adenocarcinoma detected with 94.2% confidence)*

### 2. Staging Review (Roots)
A dedicated tab for reviewing the area-based staging logic.
![Staging Review](https://raw.githubusercontent.com/your-username/lung-cancer-detection-ai/main/docs/assets/staging_review.png)
*(Example: Stage FINAL indicated by red color coding)*

### 3. Model Interpretability (Grad-CAM)
Understand where the model "looks" to make its prediction.
![Grad-CAM Output](https://raw.githubusercontent.com/your-username/lung-cancer-detection-ai/main/docs/assets/gradcam_sample.png)

## 🚀 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LungInsight-AI.git
   cd LungInsight-AI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard
```bash
streamlit run dashboard/app_enhanced.py
```

## 🧬 Dataset
The system is trained on diverse lung CT datasets. For more details on the data structure and volume, see [docs/dataset.md](docs/dataset.md).

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

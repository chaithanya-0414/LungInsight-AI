# Dataset & Training Specifications

## 📊 Dataset Overview
The **LungInsight AI** platform is trained and validated on standard medical imaging datasets (DICOM, NIfTI, JPG/PNG) to ensure robustness across different scanner types.

- **Type**: Axial CT scan slices.
- **Volume**: 
  - **Segmentation**: ~4,000 annotated slices.
  - **Classification**: High-resolution ROI crops across 4 classes (Adenocarcinoma, Squamous, Large Cell, Normal).
- **Format**: Supports 16-bit DICOM and 8-bit standard image formats.

## 🏗️ Data Structure
For development and re-training, the data is organized as follows:
```text
data/
├── train/
│   ├── images/  (Raw CT slices)
│   └── masks/   (Binary ground truth)
└── val/
    ├── images/
    └── masks/
```

## ⚙️ Training Hardware & Benchmarks
- **Hardware**: Optimized for NVIDIA CUDA-enabled GPUs.
- **Learning Rate**: 1e-4 with AdamW Optimizer.
- **Batch Size**: 16 (Segmentation), 32 (Classification).
- **Validation Metrics**: Dice Score (Segmentation), F1-Score (Classification).

For implementation details of the training loop, refer to [src/train_classifier.py](../src/train_classifier.py).

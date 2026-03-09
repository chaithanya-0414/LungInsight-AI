# Project Architecture

1. **Data Prep**: CT scans are pre-processed and normalized.
2. **Model**: Uses a UNet/Attention UNet architecture for lung tumor segmentation.
3. **Inference**: Flask/FastAPI based API or Streamlit dashboard for real-time inference.
4. **Outputs**: Dice coefficient metrics, uncertainty estimations, and PDF clinical reports.

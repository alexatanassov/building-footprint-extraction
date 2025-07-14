# Building Footprint Extraction from Aerial Imagery

A modular machine learning pipeline for extracting **building footprints** from **high-resolution satellite or drone imagery**, designed for scalable geospatial analysis and deployment.

Applicable to real-world domains including:
- Urban planning and land use mapping  
- Post-disaster response and damage assessment  
- Real estate and property analytics  
- Automated drone mapping workflows  
- Insurance and risk underwriting

---

## Overview

This project integrates **semantic segmentation**, **self-supervised pretraining**, and **geospatial post-processing** to extract accurate building footprints from raw RGB aerial imagery.

The complete pipeline includes:
- SimCLR-based self-supervised pretraining for encoder generalization
- Multiple segmentation models: U-Net, ResNet-UNet, Swin-Unet, and SegFormer
- Post-processing for polygon extraction (GeoJSON/Shapefile)
- Quantization-aware training (QAT) and ONNX export for optimized inference
- Batch inference pipeline and FastAPI server
- Docker containerization for scalable deployment

---

## Project Structure

```bash
building-footprint-extraction/
├── data/              # Raw, tiled, and processed imagery
├── datasets/          # Custom PyTorch datasets and loaders
├── models/            # U-Net, ResNet-UNet, Swin-Unet, SegFormer
├── scripts/
│   ├── train_unet.py                  # U-Net training
│   ├── train_segformer.py            # SegFormer training with binary head
│   ├── inference_segformer.py        # Model evaluation and visualization
│   ├── qat_train_unet.py             # QAT training pipeline for U-Net
│   ├── export_quantize_segformer.py  # ONNX + dynamic quantization for SegFormer
│   └── postprocess_polygons.py       # Raster-to-polygon conversion
├── notebooks/         # EDA, results analysis, architecture diagrams
├── checkpoints/       # Saved model weights
├── logs/              # Metrics logs and prediction visualizations
├── api/               # FastAPI inference server
├── requirements.txt   # Python dependencies
└── README.md

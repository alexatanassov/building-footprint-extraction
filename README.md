# 🏠 Building Footprint Extraction from Aerial Imagery

An end-to-end machine learning pipeline for automatically extracting **building footprints** from **high-resolution satellite or drone imagery**.

Built for real-world geospatial use cases including:
- Urban planning & zoning
- Post-disaster damage assessment
- Real estate analytics
- Drone mapping automation
- Insurance underwriting

---

## 🔍 Overview

This project combines **self-supervised learning** with **semantic segmentation** and **geospatial post-processing** to generate accurate, vectorized building footprints from raw RGB aerial tiles.

The full system includes:
- Self-supervised encoder (SimCLR / MAE)
- Segmentation head (U-Net / DeepLabV3)
- Post-processing to output polygons (GeoJSON/Shapefile)
- Batch inference & FastAPI server
- Docker containerization for deployment

---

## 📁 Project Structure

```bash
building-footprint-extraction/
├── data/            # Raw, tiled, and processed images
├── src/             # Models, datasets, loss functions, utilities
├── scripts/         # Training, inference, post-processing
├── notebooks/       # EDA, results, math explanations
├── api/             # FastAPI inference server
├── configs/         # YAML configs for training/inference
├── docker/          # Dockerfiles for training/inference environments
├── requirements.txt # Python dependencies
├── README.md
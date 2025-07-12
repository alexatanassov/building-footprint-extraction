import sys
import os

# Add project root to path explicitly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from datasets.building_dataset import get_dataloaders

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- Load Model ----------
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
).to(device)
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

# ---------- Metrics ----------
def compute_dice(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + eps)

def compute_iou(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + eps) / (union + eps)

# ---------- RGB Preprocessing ----------
def tensor_to_rgb_image(tensor):
    rgb = tensor[:3].cpu().numpy()  # Take first 3 channels
    rgb = np.transpose(rgb, (1, 2, 0))  # C,H,W -> H,W,C
    rgb = (rgb * 255).astype(np.uint8)
    return Image.fromarray(rgb)

# ---------- Data ----------
IMAGE_DIR = "/content/data/tiles_subset_5000/images"
LABEL_DIR = "/content/data/tiles_subset_5000/labels"
_, _, test_loader = get_dataloaders(IMAGE_DIR, LABEL_DIR, batch_size=1)

# ---------- Evaluation ----------
dice_scores = []
iou_scores = []

model.eval()
with torch.no_grad():
    for img_tensor, mask_tensor in tqdm(test_loader, desc="Evaluating SegFormer"):
        img_tensor = img_tensor.to(device)
        mask_tensor = mask_tensor.to(device).unsqueeze(1)

        # Convert to RGB image and preprocess
        rgb_image = tensor_to_rgb_image(img_tensor[0])
        inputs = processor(images=rgb_image, return_tensors="pt").to(device)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # [1, num_classes, H, W]

        # Convert to binary mask: class 1 = building
        probs = torch.softmax(logits, dim=1)
        pred_mask = probs[:, 1:2]  # Assuming class 1 is 'building'

        # Resize to match original mask size
        pred_mask = torch.nn.functional.interpolate(
            pred_mask, size=mask_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        dice = compute_dice(pred_mask, mask_tensor).item()
        iou = compute_iou(pred_mask, mask_tensor).item()

        dice_scores.append(dice)
        iou_scores.append(iou)

# ---------- Results ----------
avg_dice = sum(dice_scores) / len(dice_scores)
avg_iou = sum(iou_scores) / len(iou_scores)

print(f"SegFormer Results on Test Set — Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
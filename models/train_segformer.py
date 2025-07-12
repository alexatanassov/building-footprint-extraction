import sys
import os

# Add project root to sys.path so `datasets/` and `models/` can be found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import csv
from tqdm import tqdm
from PIL import Image
from datasets.building_dataset import get_dataloaders
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- Hyperparameters ----------
EPOCHS = 10
BATCH_SIZE = 8  # lower for memory constraints
LR = 5e-5

# ---------- Data Paths ----------
IMAGE_DIR = "/content/data/tiles_subset_5000/images"
LABEL_DIR = "/content/data/tiles_subset_5000/labels"

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

# ---------- Preprocessing ----------
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

def preprocess(img_tensor, mask_tensor):
    rgb = img_tensor[:, :3, :, :]  # [B, 3, H, W]
    rgb = rgb.clone().detach()

    # Normalize and convert to uint8 (for processor compatibility)
    rgb = rgb / 6.0  # since you had values up to ~7
    rgb = torch.clamp(rgb, 0, 1)
    rgb = (rgb * 255).byte()

    # Convert to numpy
    rgb_imgs = [x.permute(1, 2, 0).cpu().numpy() for x in rgb]  # HWC

    processed = processor(images=rgb_imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in processed.items()}
    masks = mask_tensor.unsqueeze(1).to(device).float()
    return inputs, masks

# ---------- Load Data ----------
train_loader, val_loader, _ = get_dataloaders(IMAGE_DIR, LABEL_DIR, batch_size=BATCH_SIZE)

# ---------- Model ----------
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=1,  # binary
    ignore_mismatched_sizes=True  # override ADE20K head
).to(device)

model.config.num_labels = 1
model.config.id2label = {0: "background", 1: "building"}
model.config.label2id = {"background": 0, "building": 1}

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------- Training ----------
def train():
    model.train()
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/segformer_metrics.csv"

    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Dice", "Val IoU"])

        for epoch in range(EPOCHS):
            total_loss = 0
            loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}", leave=False)

            for img_batch, mask_batch in loop:
                inputs, masks = preprocess(img_batch, mask_batch)
                outputs = model(**inputs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)

            # ---------- Validation ----------
            model.eval()
            dice_scores = []
            iou_scores = []

            with torch.no_grad():
                for img_batch, mask_batch in val_loader:
                    inputs, masks = preprocess(img_batch, mask_batch)
                    outputs = model(**inputs)
                    preds = torch.sigmoid(outputs.logits)

                    dice = compute_dice(preds, masks).item()
                    iou = compute_iou(preds, masks).item()
                    dice_scores.append(dice)
                    iou_scores.append(iou)

            val_dice = sum(dice_scores) / len(dice_scores)
            val_iou = sum(iou_scores) / len(iou_scores)

            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f} | Val Dice = {val_dice:.4f} | Val IoU = {val_iou:.4f}", flush=True)
            writer.writerow([epoch + 1, avg_loss, val_dice, val_iou])
            model.train()

    torch.save(model.state_dict(), "checkpoints/segformer.pth")
    print("Model saved to checkpoints/segformer.pth")

if __name__ == "__main__":
    train()
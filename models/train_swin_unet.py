import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import csv
from tqdm import tqdm
from datasets.building_dataset import get_dataloaders
from models.swin_unet import SwinUNet

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- Hyperparameters ----------
EPOCHS = 10
BATCH_SIZE = 8
LR = 5e-5

# ---------- Paths ----------
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

# ---------- Data ----------
train_loader, val_loader, _ = get_dataloaders(IMAGE_DIR, LABEL_DIR, batch_size=BATCH_SIZE)

# ---------- Model ----------
model = SwinUNet(img_size=256, in_channels=3, num_classes=1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ---------- Training ----------
def train():
    model.train()
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/swinunet_metrics.csv"

    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Dice", "Val IoU"])

        for epoch in range(EPOCHS):
            total_loss = 0
            loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}", leave=False)

            for imgs, masks in loop:
                imgs = imgs[:, :3, :, :].to(device)  # Use first 3 channels (RGB)
                masks = masks.unsqueeze(1).to(device).float()

                preds = model(imgs)
                preds = torch.nn.functional.interpolate(preds, size=masks.shape[2:], mode="bilinear", align_corners=False)

                loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, masks)

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
                for imgs, masks in val_loader:
                    imgs = imgs[:, :3, :, :].to(device)
                    masks = masks.unsqueeze(1).to(device).float()

                    preds = model(imgs)
                    preds = torch.nn.functional.interpolate(preds, size=masks.shape[2:], mode="bilinear", align_corners=False)
                    preds = torch.sigmoid(preds)

                    dice_scores.append(compute_dice(preds, masks).item())
                    iou_scores.append(compute_iou(preds, masks).item())

            val_dice = sum(dice_scores) / len(dice_scores)
            val_iou = sum(iou_scores) / len(iou_scores)

            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f} | Val Dice = {val_dice:.4f} | Val IoU = {val_iou:.4f}", flush=True)
            writer.writerow([epoch + 1, avg_loss, val_dice, val_iou])
            model.train()

    model.eval()
    torch.save(model.state_dict(), "checkpoints/swin_unet.pth")
    print("Model saved to checkpoints/swin_unet.pth")

if __name__ == "__main__":
    train()
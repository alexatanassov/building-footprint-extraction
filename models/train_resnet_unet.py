import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets.building_dataset import get_dataloaders
from models.resnet_unet import ResNetUNet
from models.losses import DiceBCELoss

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

# ---------- Device ----------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------- Hyperparameters ----------
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4

# ---------- Data Paths ----------
IMAGE_DIR = "/content/data/tiles_subset_5000/images"
LABEL_DIR = "/content/data/tiles_subset_5000/labels"

# ---------- Data ----------
train_loader, val_loader, test_loader = get_dataloaders(IMAGE_DIR, LABEL_DIR, batch_size=BATCH_SIZE)

# ---------- Model ----------
model = ResNetUNet(n_channels=8, n_classes=1).to(device)
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------- Training ----------
def train():
    model.train()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_file = "logs/resnet_unet_metrics.csv"
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Dice", "Val IoU"])  # header

        for epoch in range(EPOCHS):
            total_loss = 0
            loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}", leave=False)

            for imgs, masks in loop:
                imgs, masks = imgs.to(device), masks.to(device).unsqueeze(1)

                preds = model(imgs)
                loss = criterion(preds, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)

            # ---------- Validation ----------
            model.eval()
            dice_scores, iou_scores = [], []

            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device).unsqueeze(1)
                    preds = model(imgs)
                    dice_scores.append(compute_dice(preds, masks).item())
                    iou_scores.append(compute_iou(preds, masks).item())

            val_dice = sum(dice_scores) / len(dice_scores)
            val_iou = sum(iou_scores) / len(iou_scores)

            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f} | Val Dice = {val_dice:.4f} | Val IoU = {val_iou:.4f}", flush=True)
            writer.writerow([epoch + 1, avg_loss, val_dice, val_iou])
            model.train()

    torch.save(model.state_dict(), "checkpoints/resnet_unet.pth")
    print("Model saved to checkpoints/resnet_unet.pth")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
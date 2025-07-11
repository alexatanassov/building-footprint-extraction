import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets.building_dataset import get_dataloaders
from models.unet import UNet
from models.losses import DiceBCELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4

# Paths to compressed tile data
IMAGE_DIR = "/content/data/tiles_subset_5000/images"
LABEL_DIR = "/content/data/tiles_subset_5000/labels"

print("Found images:", len(os.listdir(IMAGE_DIR)))
print("Found labels:", len(os.listdir(LABEL_DIR)))

# Load data
train_loader, val_loader, test_loader = get_dataloaders(IMAGE_DIR, LABEL_DIR, batch_size=BATCH_SIZE)

# Model, loss, optimizer
model = UNet(in_channels=8, out_channels=1).to(device)
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
def train():
    model.train()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_file = "logs/unet_training_log.csv"
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss"])  # CSV header

        for epoch in range(EPOCHS):
            total_loss = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

            for imgs, masks in loop:
                imgs = imgs.to(device)
                masks = masks.to(device).unsqueeze(1)  # Add channel dim

                preds = model(imgs)
                loss = criterion(preds, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}", flush=True)
            writer.writerow([epoch + 1, avg_loss])

    torch.save(model.state_dict(), "checkpoints/unet.pth")
    print("Model saved to checkpoints/unet.pth", flush=True)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets.building_dataset import get_dataloaders
from models.unet import UNet
from models.losses import DiceBCELoss

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Hyperparameters
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-4

# Paths to compressed tile data
IMAGE_DIR = "data/tiles_npz/images"
LABEL_DIR = "data/tiles_npz/labels"

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
        writer.writerow(["Epoch", "Loss"])  # Header

        for epoch in range(EPOCHS):
            total_loss = 0
            for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                imgs, masks = imgs.to(device), masks.to(device).unsqueeze(1)  # Add channel dim to mask

                preds = model(imgs)
                loss = criterion(preds, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            writer.writerow([epoch + 1, avg_loss])

    torch.save(model.state_dict(), "checkpoints/unet.pth")
    print("Model saved to checkpoints/unet.pth")

if __name__ == "__main__":
    train()
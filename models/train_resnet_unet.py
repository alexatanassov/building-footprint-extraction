import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets.building_dataset import get_dataloaders
from models.resnet_unet import ResNetUNet
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
model = ResNetUNet(n_channels=8, n_classes=1).to(device)
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
def train():
    model.train()
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

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/unet.pth")
    print("Model saved to checkpoints/unet.pth")

if __name__ == "__main__":
    train()
import sys
import os
import matplotlib.pyplot as plt
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pprint
pprint.pprint(sys.path)
from datasets.building_dataset import get_dataloaders 

# === Parameters ===
IMAGE_DIR = "data/tiles_npz/images"
LABEL_DIR = "data/tiles_npz/labels"
BATCH_SIZE = 8

# === Load dataloaders ===
train_loader, val_loader, test_loader = get_dataloaders(IMAGE_DIR, LABEL_DIR, batch_size=BATCH_SIZE)

# === Get a batch from the training loader ===
batch = next(iter(train_loader))
images, masks = batch  # images: (B, C, H, W), masks: (B, H, W)

# === Plotting ===
num_samples = min(BATCH_SIZE, 8)
fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))

for i in range(num_samples):
    img = images[i][:3].permute(1, 2, 0).numpy()  # Only RGB bands
    
    mask = masks[i].numpy()

    # Denormalize image for visualization
    img = img * 0.5 + 0.5  # (mean=0.5, std=0.5)
    img = (img * 255).astype("uint8")

    axes[i, 0].imshow(img)
    axes[i, 0].set_title("Image")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("Mask")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig("visualized_dataloader_samples.png", dpi=200)
plt.show()
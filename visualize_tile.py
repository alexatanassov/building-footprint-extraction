import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Settings
IMAGE_DIR = "data/tiles/images"
LABEL_DIR = "data/tiles/labels"
NUM_SAMPLES = 12  # Total tiles to show (must be even for paired layout)
TILE_PAIRS = NUM_SAMPLES // 2  # Each row will show image + label

# Get available tiles
tiles = os.listdir(IMAGE_DIR)
random.shuffle(tiles)
tiles = tiles[:TILE_PAIRS]  # Pick a few random ones

# Setup matplotlib grid
fig, axes = plt.subplots(nrows=TILE_PAIRS, ncols=2, figsize=(8, 2 * TILE_PAIRS))
fig.suptitle("Random Image Tiles and Their Label Masks", fontsize=16)

for i, tile in enumerate(tiles):
    tile_name = tile.replace(".npy", "")
    
    # Load image and label
    img_path = os.path.join(IMAGE_DIR, tile_name + ".npy")
    label_path = os.path.join(LABEL_DIR, tile_name + ".npy")
    
    if not os.path.exists(label_path):
        continue  # skip if no label

    img = np.load(img_path)
    label = np.load(label_path)

    # Prepare RGB
    rgb = np.moveaxis(img[:3], 0, -1)
    rgb = rgb / (rgb.max() + 1e-8)  # normalize

    # Plot image
    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title(f"Image: {tile_name}")
    axes[i, 0].axis("off")

    # Plot label
    axes[i, 1].imshow(label, cmap="gray")
    axes[i, 1].set_title("Label Mask")
    axes[i, 1].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for suptitle
plt.show()

plt.savefig("sample_grid.png", dpi=200)
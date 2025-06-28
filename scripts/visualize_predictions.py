import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datasets.building_dataset import get_dataloaders
from models.unet import UNet

# Load test data set
_, _, test_loader = get_dataloaders(
    image_dir="data/tiles_npz/images",
    label_dir="data/tiles_npz/labels",
    batch_size=8
)

# Load trained model
model = UNet(in_channels=8, out_channels=1)
model.load_state_dict(torch.load("checkpoints/unet.pth", map_location=torch.device("mps")))
model.eval()
model.to("mps")

# Inference
images, masks = next(iter(test_loader))
images = images.to("mps")

with torch.no_grad():
    outputs = model(images)
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float().cpu()

images = images.cpu()
masks = masks.cpu()

# Plot predictions
num_samples = min(8, images.shape[0])
fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))

for i in range(num_samples):
    # Show image (first 3 bands as RGB)
    img = images[i][:3].permute(1, 2, 0).numpy()  # CHW -> HWC
    img = img * 0.5 + 0.5  # Denormalize
    img = (img * 255).astype(np.uint8)

    gt_mask = masks[i].numpy()
    pred_mask = preds[i].squeeze().numpy()

    axes[i, 0].imshow(img)
    axes[i, 0].set_title("Input Image")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(gt_mask, cmap="gray")
    axes[i, 1].set_title("Ground Truth")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(img)
    axes[i, 2].imshow(pred_mask, cmap="Reds", alpha=0.5)
    axes[i, 2].set_title("Predicted Mask")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig("prediction_samples.png", dpi=200)
plt.show()
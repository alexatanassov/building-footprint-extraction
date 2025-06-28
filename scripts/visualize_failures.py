import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datasets.building_dataset import get_dataloaders
from models.unet import UNet


def compute_iou(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum((1, 2))
    union = ((pred + target) >= 1).float().sum((1, 2))
    return (intersection / (union + 1e-6)).cpu().numpy()


def visualize_failures():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    _, _, test_loader = get_dataloaders("data/tiles_npz/images", "data/tiles_npz/labels", batch_size=1)

    model = UNet(in_channels=8, out_channels=1)
    model.load_state_dict(torch.load("checkpoints/unet.pth", map_location=device))
    model = model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for idx, (image, mask) in enumerate(test_loader):
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)
            pred = torch.sigmoid(output).squeeze(1)
            mask = mask.squeeze(1)

            iou = compute_iou(pred, mask)[0]

            results.append({
                "image": image.cpu().squeeze().numpy(),
                "mask": mask.cpu().squeeze().numpy(),
                "pred": pred.cpu().squeeze().numpy(),
                "iou": iou,
                "index": idx
            })

    # Sort by IoU
    results = sorted(results, key=lambda x: x["iou"])
    worst_cases = results[:20]  # bottom 20

    os.makedirs("results/failures", exist_ok=True)

    for i, case in enumerate(worst_cases):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # RGB for first 3 bands
        img = case["image"][:3] if case["image"].ndim == 3 else np.zeros((3, 256, 256))
        img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        axs[0].imshow(img)
        axs[0].set_title("RGB Tile")
        axs[0].axis("off")

        axs[1].imshow(case["mask"], cmap="gray")
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(case["pred"], cmap="gray")
        axs[2].set_title(f"Prediction (IoU: {case['iou']:.2f})")
        axs[2].axis("off")

        plt.tight_layout()
        plt.savefig(f"results/failures/failure_{i:02d}_iou_{case['iou']:.2f}.png")
        plt.close()

    print("Saved bottom 20 IoU failure examples to results/failures/")


if __name__ == "__main__":
    visualize_failures()
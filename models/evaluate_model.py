import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.building_dataset import get_dataloaders
from models.unet import UNet
import numpy as np
import json
import os


def compute_iou(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    targets = (targets > 0.5).float()

    intersection = (preds.bool() & targets.bool()).float().sum((1, 2))
    union = (preds.bool() | targets.bool()).float().sum((1, 2))

    iou = intersection / (union + 1e-6)
    return iou.mean().item()


def compute_dice(preds, targets):
    intersection = (preds * targets).float().sum((1, 2))
    dice = (2. * intersection + 1e-6) / (preds.sum((1, 2)) + targets.sum((1, 2)) + 1e-6)
    return dice.mean().item()

def compute_pixel_accuracy(preds, targets):
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    return (correct / total).item()

def evaluate_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    _, _, test_loader = get_dataloaders("data/tiles_npz/images", "data/tiles_npz/labels", batch_size=8)

    model = UNet(in_channels=8, out_channels=1)
    model.load_state_dict(torch.load("checkpoints/unet.pth", map_location=device))
    model = model.to(device)
    model.eval()

    iou_scores = []
    dice_scores = []
    acc_scores = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.squeeze(1)
            masks = masks.squeeze(1)

            iou = compute_iou(preds, masks)
            dice = compute_dice(preds, masks)
            acc = compute_pixel_accuracy(preds, masks)

            iou_scores.append(iou)
            dice_scores.append(dice)
            acc_scores.append(acc)

    print(f"\nEvaluation Results:")
    print(f"Mean IoU:   {np.mean(iou_scores):.4f}")
    print(f"Mean Dice:  {np.mean(dice_scores):.4f}")
    print(f"Accuracy:   {np.mean(acc_scores):.4f}")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Store metrics
    metrics = {
    "Mean IoU": float(np.mean(iou_scores)),
    "Mean Dice": float(np.mean(dice_scores)),
    "Accuracy": float(np.mean(acc_scores))
    }   

    # Save to JSON
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved metrics to results/metrics.json")


if __name__ == "__main__":
    evaluate_model()
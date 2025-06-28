import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BuildingDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.load(self.image_paths[idx], allow_pickle=True)['arr_0']
        mask = np.load(self.label_paths[idx], allow_pickle=True)['arr_0']

        # Normalize if needed
        img = img.astype(np.float32) / 255.0 if img.max() > 1 else img
        mask = mask.astype(np.float32)

        # Convert to HWC for Albumentations
        if self.transform:
            augmented = self.transform(image=img.transpose(1, 2, 0), mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img.clone().detach(), mask.clone().detach()


def get_dataset_paths(image_dir, label_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.npz")))
    label_paths = sorted(glob.glob(os.path.join(label_dir, "*.npz")))
    return image_paths, label_paths


def get_dataloaders(image_dir, label_dir, batch_size=8, seed=42, max_samples=5000):
    image_paths, label_paths = get_dataset_paths(image_dir, label_dir)

    # Subsample the dataset
    image_paths = image_paths[:max_samples]
    label_paths = label_paths[:max_samples]

    train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
        image_paths, label_paths, test_size=0.2, random_state=seed)

    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
        test_imgs, test_lbls, test_size=0.5, random_state=seed)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2()
    ])

    train_ds = BuildingDataset(train_imgs, train_lbls, transform=transform)
    val_ds = BuildingDataset(val_imgs, val_lbls, transform=transform)
    test_ds = BuildingDataset(test_imgs, test_lbls, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
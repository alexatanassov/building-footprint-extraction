import torch
from models.resnet_unet import ResNetUNet  # relative import since you're inside models/

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Simulate input: batch size 2, 8 channels, 256x256 tiles
    x = torch.randn(2, 8, 256, 256).to(device)

    model = ResNetUNet(n_channels=8, n_classes=1).to(device)

    # Forward pass
    with torch.no_grad():
        y = model(x)

    print(f"Forward pass succeeded!")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

if __name__ == "__main__":
    main()
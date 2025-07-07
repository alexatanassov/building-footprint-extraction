import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class UpBlock(nn.Module):
    def __init__(self, in_channels, bridge_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + bridge_channels, out_channels)

    def forward(self, x, bridge):
        x = self.up(x)
        # Align shape if needed due to rounding in encoder
        diffY = bridge.size()[2] - x.size()[2]
        diffX = bridge.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, bridge], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, n_channels=8, n_classes=1):
        super().__init__()

        # Load ResNet34 encoder with pretrained ImageNet weights
        self.encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Replace input layer to accept multi-channel input (e.g. 8-channel satellite imagery)
        self.encoder.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Encoder stage definitions
        self.input_block = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
        )
        self.pool1 = self.encoder.maxpool
        self.encoder1 = self.encoder.layer1  # 64
        self.encoder2 = self.encoder.layer2  # 128
        self.encoder3 = self.encoder.layer3  # 256
        self.encoder4 = self.encoder.layer4  # 512

        # Decoder blocks with skip connections from encoder
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)

        # Final 1×1 convolution for segmentation mask output
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

        # Final upsampling to match input resolution
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        # Encoder forward pass
        x0 = self.input_block(x)        # Initial conv
        x1 = self.pool1(x0)             # Downsample
        x2 = self.encoder1(x1)          # Stage 1
        x3 = self.encoder2(x2)          # Stage 2
        x4 = self.encoder3(x3)          # Stage 3
        x5 = self.encoder4(x4)          # Stage 4

        # Decoder with skip connections
        d4 = self.up4(x5, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x0)

        return self.upsample(self.final(d1))
import torch
import torch.nn as nn
from transformers import SwinModel, SwinConfig


class SwinUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_channels=3, num_classes=1):
        super(SwinUNet, self).__init__()

        # Swin Transformer Configuration
        config = SwinConfig(
            image_size=img_size,
            patch_size=patch_size,
            num_channels=in_channels,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            use_absolute_positions=False,
        )

        self.encoder = SwinModel(config)

        # Swin final output: [B, L, C] where L = H//32 * W//32, C = 768
        self.up1 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        self.final_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)  # to match 256x256
        self.final = nn.Conv2d(96, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        outputs = self.encoder(x)
        x = outputs.last_hidden_state  # [B, L, C]

        # Reshape from [B, L, C] → [B, C, H', W']
        L = x.shape[1]
        C = x.shape[2]
        H_feat = W_feat = int(L ** 0.5)  # assumes square input
        x = x.transpose(1, 2).reshape(B, C, H_feat, W_feat)

        x = self.up1(x)
        x = self.conv1(x)

        x = self.up2(x)
        x = self.conv2(x)

        x = self.up3(x)
        x = self.conv3(x)

        x = self.final_up(x)  # upscale to 256×256
        return self.final(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Compute standard binary cross-entropy loss
        bce_loss = self.bce(logits, targets)

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Compute Dice loss
        smooth = 1.0 # for stability
        intersection = (probs * targets).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice_loss = 1 - ((2. * intersection + smooth) / (union + smooth))
        dice_loss = dice_loss.mean()

        # Combine BCE and Dice losses
        return bce_loss + dice_loss
import torch.nn as nn
import segmentation_models_pytorch as smp

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.dice = smp.losses.DiceLoss(
            mode="multilabel",
            from_logits=True
        )

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        dice = self.dice(pred, target)
        bce = self.bce(pred, target)
        return dice, bce
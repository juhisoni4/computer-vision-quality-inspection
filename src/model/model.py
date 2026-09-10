import segmentation_models_pytorch as smp
import torch
from src.model.dice_loss import DiceBCELoss

class SegModel(torch.nn.Module):
    def __init__(self, config):
        super(SegModel, self).__init__()
        self.config = config

        if self.config.model.model_name == "segformer":        
            self.model = smp.Segformer(                # UnetPlusPlus Unet Segformer FPN DeepLabV3Plus
                encoder_name=self.config.model.encoder_name,
                encoder_weights=self.config.model.encoder_weights,
                in_channels=self.config.model.in_channels,
                classes=self.config.model.num_classes,
                activation=None
            )
        elif self.config.model.model_name == "unet++":
            self.model = smp.UnetPlusPlus(
                encoder_name=self.config.model.encoder_name,
                encoder_weights=self.config.model.encoder_weights,
                in_channels=self.config.model.in_channels,
                classes=self.config.model.num_classes,
                activation=None
            )
        elif self.config.model.model_name == "unet":
            self.model = smp.Unet(
                encoder_name=self.config.model.encoder_name,
                encoder_weights=self.config.model.encoder_weights,
                in_channels=self.config.model.in_channels,
                classes=self.config.model.num_classes,
                activation=None
            )
        elif self.config.model.model_name == "fpn":
            self.model = smp.FPN(
                encoder_name=self.config.model.encoder_name,
                encoder_weights=self.config.model.encoder_weights,
                in_channels=self.config.model.in_channels,
                classes=self.config.model.num_classes,
                activation=None
            )
        else:
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.config.model.encoder_name,
                encoder_weights=self.config.model.encoder_weights,
                in_channels=self.config.model.in_channels,
                classes=self.config.model.num_classes,
                activation=None
            )

        self.criterion = DiceBCELoss()

    def forward(self, x, y):
        pred_x = self.model(x)

        dice_loss, bce_loss = self.criterion(pred_x, y)

        loss = dice_loss + bce_loss
        
        loss_metrics = {
            "loss": loss.detach().item(),
            "dice_loss": dice_loss.detach().item(),
            "bce_loss": bce_loss.detach().item()
        }
        return loss, loss_metrics
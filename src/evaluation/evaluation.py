from pathlib import Path

import mlflow
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.config.logger_config import setup_logger
logger = setup_logger("eval_module")


class Evaluation:
    def __init__(self, config):
        super(Evaluation, self).__init__()
        self.config = config
        self.results_dir = Path(__file__).resolve().parent.parent / "results"
        self.colors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]),
        ]

    def visualize_data(self, image_data, mask_data, model, device="cpu", threshold=0.5, epoch=0):

        model.eval()
        pred_mask = None

        B, C, H, W = image_data.shape

        if model is not None:
            fig, axes = plt.subplots(
                        B, 5,
                        figsize=(30, 4 * B),
                        constrained_layout=True
                    )
        else:
            fig, axes = plt.subplots(
                B, 3,
                figsize=(30, 4 * B),
                constrained_layout=True
            )        

        for index in range(B):

            image = image_data[index]
            mask = mask_data[index]

            if model is not None:   
                with torch.no_grad():
                    image_tensor = image.unsqueeze(0).to(device)
                    pred_mask = model(image_tensor)
                    pred_mask = torch.sigmoid(pred_mask)
                    pred_mask = (pred_mask > threshold).float().squeeze(0).cpu().numpy()

            # Undo ImageNet normalization
            if image.min() < 0:
                mean = torch.tensor(
                    [0.485, 0.456, 0.406],
                    device=image.device
                ).view(3, 1, 1)

                std = torch.tensor(
                    [0.229, 0.224, 0.225],
                    device=image.device
                ).view(3, 1, 1)

                image = image * std + mean
                image = image.clamp(0, 1)

            image = image.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # Image should become H,W,3
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))

            image = image.astype(np.float32)

            mask = mask.astype(np.float32)

            if mask.ndim != 3:
                raise ValueError(
                    f"Expected 3D mask, got {mask.shape}"
                )

            if mask.shape[0] == 4:
                pass

            elif mask.shape[-1] == 4:
                mask = np.transpose(mask, (2, 0, 1))

            elif mask.shape[1] == 4:
                mask = np.transpose(mask, (1, 0, 2))

            else:
                raise ValueError(
                    f"Cannot identify 4 mask channels: {mask.shape}"
                )

            if image.shape[:2] != mask.shape[1:]:
                raise ValueError(
                    f"Image and mask spatial sizes don't match: "
                    f"image={image.shape}, mask={mask.shape}"
                )

            combined_mask = np.max(mask, axis=0)

            if pred_mask is not None:
                combined_pred_mask = np.max(pred_mask, axis=0)
                pred_overlay = image.copy()

            overlay = image.copy()

            for class_id in range(4):

                class_mask = mask[class_id] > 0

                overlay[class_mask] = (
                    overlay[class_mask] * 0.45
                    + self.colors[class_id] * 0.55
                )

                if pred_mask is not None:
                    class_pred_mask = pred_mask[class_id] > 0

                    pred_overlay[class_pred_mask] = (
                        pred_overlay[class_pred_mask] * 0.45
                        + self.colors[class_id] * 0.55
                    )

            overlay = np.clip(overlay, 0, 1)
            
            axes[index, 0].imshow(image)
            axes[index, 0].set_title("Original Image")
            axes[index, 0].set_aspect("auto")

            axes[index, 1].imshow(
                combined_mask,
                cmap="gray",
                vmin=0,
                vmax=1
            )
            axes[index, 1].set_title("Combined Mask")
            axes[index, 1].set_aspect("auto")

            axes[index, 2].imshow(overlay)
            axes[index, 2].set_title("Defect Overlay")
            axes[index, 2].set_aspect("auto")

            if pred_mask is not None:
                axes[index, 3].imshow(
                    combined_pred_mask,
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )
                axes[index, 3].set_title("Predicted Combined Mask")
                axes[index, 3].set_aspect("auto")

                axes[index, 4].imshow(pred_overlay)
                axes[index, 4].set_title("Predicted Defect Overlay")
                axes[index, 4].set_aspect("auto")

        fig.savefig(f"{self.results_dir}/visualize_data_{epoch}.png")
        mlflow.log_artifact(f"{self.results_dir}/visualize_data_{epoch}.png", artifact_path=f"epochs/epoch_{epoch}")    
    
    
    def dice_score(self, y_true, y_pred, eps=1e-7):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)

        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()

        return (2 * tp + eps) / (2 * tp + fp + fn + eps)

    def dice_score_per_class(self, y_true, y_pred, epoch=0, eps=1e-7):

        dice_scores = []

        for c in range(4):
            dice = self.dice_score(
                y_true[:, c],
                y_pred[:, c],
                eps=eps
            )

            dice_scores.append(dice)

        dice_score = np.array(dice_scores).mean()

        return dice_score

    def iou_score(self, y_true, y_pred, eps=1e-7):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)

        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()

        return (intersection + eps) / (union + eps)

    def iou_score_per_class(self, y_true, y_pred, epoch=0, eps=1e-7):

        iou_scores = []

        for c in range(4):
            iou = self.iou_score(
                y_true[:, c],
                y_pred[:, c],
                eps=eps
            )

            iou_scores.append(iou)

        iou_scores = np.array(iou_scores).mean()
        
        return iou_scores
        

    def precision_recall(self, y_true, y_pred, epoch=0, eps=1e-7):

        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)

        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        return precision, recall

    def precision_recall_per_class(self, y_true, y_pred, epoch=0, eps=1e-7):
    
        precision_c = {}
        recall_c = {}

        for c in range(4):
            precision, recall = self.precision_recall(
                y_true[:, c],
                y_pred[:, c],
                epoch=epoch,
                eps=eps
            )

            precision_c[c] = precision
            recall_c[c] = recall
            
        return precision_c, recall_c
        
    def complete_evaluation(self, model, device, valid_loader, epoch):
        dice_scores = []
        iou_scores = []
        
        precision_c1 = []
        precision_c2 = []
        precision_c3 = []
        precision_c4 = []
        
        recall_c1 = []
        recall_c2 = []
        recall_c3 = []
        recall_c4 = []
        
        with torch.no_grad():
        
            for i, (data, target) in enumerate(valid_loader): 
                  image_data = data.to(device, non_blocking=True)
                  mask_data = target.to(device, non_blocking=True)
            
                  logits = model(image_data.to(device))
                  
                  y_pred = torch.sigmoid(logits)
                  
                  pred_data = (y_pred >= 0.5).float()
                  
                  dice_score = self.dice_score_per_class(mask_data.cpu().detach().numpy(), pred_data.cpu().detach().numpy(), epoch)
                  dice_scores.append(dice_score)
      
                  iou_score = self.iou_score_per_class(mask_data.cpu().detach().numpy(), pred_data.cpu().detach().numpy(), epoch)
                  iou_scores.append(iou_score)
      
                  precision_c, recall_c = self.precision_recall_per_class(mask_data.cpu().detach().numpy(), pred_data.cpu().detach().numpy(), epoch)
                     
                  precision_c1.append(precision_c[0])
                  precision_c2.append(precision_c[1])
                  precision_c3.append(precision_c[2])
                  precision_c4.append(precision_c[3])                  
                  
                  recall_c1.append(recall_c[0])
                  recall_c2.append(recall_c[1])
                  recall_c3.append(recall_c[2])
                  recall_c4.append(recall_c[3])            

            image_data, mask_data = next(iter(valid_loader))
            image_data = image_data.to(device, non_blocking=True)
            mask_data = mask_data.to(device, non_blocking=True)
            
            self.visualize_data(image_data, mask_data, model, device=device, threshold=0.5, epoch=epoch)
            
            mlflow.log_metric("DICE_MEAN", np.array(dice_scores).mean(), step=epoch)
            
            mlflow.log_metric("IOU_MEAN", np.array(iou_scores).mean(), step=epoch)
            
            mlflow.log_metric(f"PRECISION_CLASS_1", np.array(precision_c1).mean(), step=epoch)
            mlflow.log_metric(f"PRECISION_CLASS_2", np.array(precision_c2).mean(), step=epoch)
            mlflow.log_metric(f"PRECISION_CLASS_3", np.array(precision_c3).mean(), step=epoch)
            mlflow.log_metric(f"PRECISION_CLASS_4", np.array(precision_c4).mean(), step=epoch)
            
            mlflow.log_metric(f"RECALL_CLASS_1", np.array(recall_c1).mean(), step=epoch)
            mlflow.log_metric(f"RECALL_CLASS_2", np.array(recall_c2).mean(), step=epoch)
            mlflow.log_metric(f"RECALL_CLASS_3", np.array(recall_c3).mean(), step=epoch)
            mlflow.log_metric(f"RECALL_CLASS_4", np.array(recall_c4).mean(), step=epoch)  
            
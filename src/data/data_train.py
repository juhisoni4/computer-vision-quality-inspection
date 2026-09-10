import torch
from tqdm import tqdm
import mlflow
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from src.evaluation.evaluation import Evaluation
from src.config.logger_config import setup_logger
logger = setup_logger("data_loading_module")
import torch.distributed as dist


class DataTrain:
    def __init__(self, config, train_loader, valid_loader, train_sampler, valid_sampler):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler
        self.config = config        

    def train_model(self, model, dataloader, optimizer):        

        # Place model to device
        model.to(self.config.device)

        # Enable training mode
        model.train()

        # Initialize variables to keep track of recon loss
        running_loss = 0.0

        train_loss_metrics = {}

        running_loss_metrics = {
            "loss": 0.0,
            "dice_loss": 0.0,
            "bce_loss": 0.0
        }

        total = 0
        
        torch.manual_seed(42)
        for i, (data, target) in tqdm(enumerate(dataloader)):
            data = data.to(self.config.device, non_blocking=True)
            target = target.to(self.config.device, non_blocking=True)

            total += data.size(0)

            optimizer.zero_grad()

            loss, loss_metrics = model(data, target)

            running_loss += loss.item()

            running_loss_metrics["loss"] += loss_metrics["loss"]
            running_loss_metrics["dice_loss"] += loss_metrics["dice_loss"]
            running_loss_metrics["bce_loss"] += loss_metrics["bce_loss"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

        stats = torch.tensor(
            [
                running_loss,    
                running_loss_metrics["loss"],
                running_loss_metrics["dice_loss"],
                running_loss_metrics["bce_loss"],
                total
            ],
            dtype=torch.float64,
            device=self.config.device,
        )

        dist.all_reduce(
            stats,
            op=dist.ReduceOp.SUM,
        )

        train_loss = stats[0] / stats[4]

        train_loss_metrics["loss"] = stats[1] / stats[4]
        train_loss_metrics["dice_loss"] = stats[2] / stats[4]
        train_loss_metrics["bce_loss"] = stats[3] / stats[4]

        return train_loss, train_loss_metrics

    def validation_model(self, model, dataloader):
        # Place model to device
        model.to(self.config.device)

        # Enable training mode
        model.eval()

        # Initialize variables to keep track of recon loss
        running_loss = 0.0

        running_loss_metrics = {
            "loss": 0.0,
            "dice_loss": 0.0,
            "bce_loss": 0.0
        }

        val_loss_metrics = {}

        total = 0      
        
        torch.manual_seed(42)
        for i, (data, target) in tqdm(enumerate(dataloader)):
            data = data.to(self.config.device, non_blocking=True)
            target = target.to(self.config.device, non_blocking=True)

            total += data.size(0)

            loss, loss_metrics = model(data, target)

            running_loss += loss.item()

            running_loss_metrics["loss"] += loss_metrics["loss"]
            running_loss_metrics["dice_loss"] += loss_metrics["dice_loss"]
            running_loss_metrics["bce_loss"] += loss_metrics["bce_loss"]


        stats = torch.tensor(
            [
                running_loss,    
                running_loss_metrics["loss"],
                running_loss_metrics["dice_loss"],
                running_loss_metrics["bce_loss"],
                total
            ],
            dtype=torch.float64,
            device=self.config.device,
        )

        dist.all_reduce(
            stats,
            op=dist.ReduceOp.SUM,
        )

        validation_loss = stats[0] / stats[4]

        val_loss_metrics["loss"] = stats[1] / stats[4]
        val_loss_metrics["dice_loss"] = stats[2] / stats[4]
        val_loss_metrics["bce_loss"] = stats[3] / stats[4]

        return validation_loss, val_loss_metrics

    def train_epoch(self, model, optimizer, rank):
        train_history = {}

        with mlflow.start_run(run_name=f"trial_{self.config.model.trail}_{self.config.model.model_name}_{self.config.model.encoder_name}"):

            no_epochs = self.config.training.epochs

            mlflow.log_text(str(model), "model_architecture.txt")

            mlflow.log_params({"learning_rate": self.config.model.learning_rate, "batch_size": self.config.model.batch_size, "epochs": no_epochs})

            for epoch in range(1, no_epochs + 1):
                self.train_sampler.set_epoch(epoch)
                self.valid_sampler.set_epoch(epoch)

                train_loss, train_loss_metrics = self.train_model(
                    model=model, dataloader=self.train_loader, optimizer=optimizer
                )

                val_loss, val_loss_metrics = self.validation_model(
                    model=model, dataloader=self.valid_loader
                )

                # Store model performance metrics in Python3 dict
                train_history[epoch] = {
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }

                if rank == 0 and epoch % 5 == 0:
                    model.eval()
                    torch.manual_seed(42)
                    
                    evaluation = Evaluation(config=self.config.model)

                    evaluation.complete_evaluation(model.module.model, self.config.device, self.valid_loader, epoch)

                    model.train()

                    torch.save((model.module.state_dict(), optimizer.state_dict()),
                            (
                                f"{self.config.model.trail}-{self.config.model.model_name}-{self.config.model.encoder_name}-{self.config.model.encoder_weights}"
                                f"-{epoch}-{self.config.model.learning_rate}-{self.config.model.batch_size}.pt"))
                                
                    mlflow.log_artifact(f"{self.config.model.trail}-{self.config.model.model_name}-{self.config.model.encoder_name}-{self.config.model.encoder_weights}"
                                f"-{epoch}-{self.config.model.learning_rate}-{self.config.model.batch_size}.pt", artifact_path="model_checkpoints")

                
                logger.info(f"Dice Loss: {train_loss_metrics['dice_loss']}, BCE Loss: {train_loss_metrics['bce_loss']}")
                logger.info(f"Dice Loss: {val_loss_metrics['dice_loss']}, BCE Loss: {val_loss_metrics['bce_loss']}")
                

                logger.info(
                    f"Epoch = {epoch}; train loss = {train_loss} ")
                logger.info(
                    f"Epoch = {epoch}; val loss = {val_loss}")

                mlflow.log_metric("TrainLoss", train_loss, step=epoch)
                mlflow.log_metric("ValidationLoss", val_loss, step=epoch)

                mlflow.log_metric("TrainDiceLoss", train_loss_metrics["dice_loss"], step=epoch)
                mlflow.log_metric("TrainBCELoss", train_loss_metrics["bce_loss"], step=epoch)
                mlflow.log_metric("ValidationDiceLoss", val_loss_metrics["dice_loss"], step=epoch)
                mlflow.log_metric("ValidationBCELoss", val_loss_metrics["bce_loss"], step=epoch)
        
        return train_history
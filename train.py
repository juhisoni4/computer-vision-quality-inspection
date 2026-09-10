import gc
import os
import random

import numpy as np
import torch
from src.data.cv_dataset import CVDataset
from src.config.config import load_config
from src.data.data_train import DataTrain
from src.data.data_preprocessing import DataPreprocessing
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from src.model.model import SegModel
from torch.utils.data import DataLoader
import mlflow
import numpy as np
import torch.distributed as dist
from src.config.logger_config import setup_logger
logger = setup_logger("train_module")


def setup():    

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return local_rank

def cleanup():
    dist.destroy_process_group()

def train(data_config, local_rank, rank, world_size):
    logger.info(f"Starting training on rank {rank}/{world_size} using GPU {local_rank}")

    model = SegModel(data_config).to(data_config.device)   
    # model.load_state_dict(torch.load("5-efficientnet-b4-imagenet-30-0.0001-16.pt")[0])

    data_preprocessing = DataPreprocessing(config=data_config)

    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True
    )        
        
    train_dataset = CVDataset(
        data_preprocessing.train_df,
        data_preprocessing.train_ids,
        data_config.data.train_img_dir,            
        data_config
    )

    valid_dataset = CVDataset(
        data_preprocessing.valid_df,
        data_preprocessing.valid_ids,
        data_config.data.train_img_dir,            
        data_config
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=data_config.training.seed
    )

    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=data_config.training.seed
    )

    train_loader = DataLoader(
        train_dataset,  # Adjust the slice as needed
        batch_size=data_config.model.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=data_config.model.num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,  # Adjust the slice as needed
        batch_size=data_config.model.batch_size,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=data_config.model.num_workers
    )

    data_train = DataTrain(config=data_config, train_loader=train_loader, valid_loader=valid_loader, train_sampler=train_sampler, valid_sampler=valid_sampler)

    logger.info(f"Model created on device: {data_config.device}")

    learning_rate = data_config.model.learning_rate
    l_2_regularization = data_config.model.weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l_2_regularization)

    train_history = data_train.train_epoch(model, optimizer, rank)

    cleanup()

    logger.info(f"Training Done. Final training loss: {train_history[data_config.training.epochs - 1]['train_loss']}")
    logger.info(f"Training Done. Final validation loss: {train_history[data_config.training.epochs - 1]['val_loss']}")
        


if __name__ == "__main__":
    config = load_config("src/config/config.yaml", "src/config/unet_resnet.yaml")

    local_rank = setup()

    device = torch.device(f"cuda:{local_rank}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.info(
        f"Rank {rank}/{world_size} using GPU {local_rank}"
    )

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    gc.collect()

    torch.cuda.empty_cache()

    torch.manual_seed(config.training.seed)
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)

    mlflow.enable_system_metrics_logging()
    mlflow.set_tracking_uri("http://localhost:5050")
    mlflow.set_experiment("MLflow CV Exp")
    train(config, local_rank, rank, world_size)
    
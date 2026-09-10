import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
from src.config.logger_config import setup_logger
logger = setup_logger("data_preprocessing_module")

class DataPreprocessing:
    def __init__(self, config):
        self._time_dataset = None
        self.df = pd.read_csv(config.data.data_path)

        self.df["ClassId"] = self.df["ClassId"].astype(int)

        self.df["Label"] = self.df["EncodedPixels"].notnull().astype(int)

        self.labels = (
            self.df.pivot(index="ImageId",
                        columns="ClassId",
                        values="Label")
            .fillna(0)
            .astype(int)
        )

        self.labels.columns = ["c1","c2","c3","c4"]

        X = self.labels.index.values
        y = self.labels.values

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=config.training.seed
        )
        
        train_idx, test_idx = next(
            msss.split(X, y)
        )
        
        self.train_ids = X[train_idx]
        self.valid_ids = X[test_idx]
        
        self.train_df = self.df[
            self.df["ImageId"].isin(self.train_ids)
        ].copy()
        
        self.valid_df = self.df[
            self.df["ImageId"].isin(self.valid_ids)
        ].copy()        
        
        logger.info(f"Train images: {len(self.train_ids)}")
        logger.info(f"Validation images: {len(self.valid_ids)}")
        
        logger.info(f"Train rows: {len(self.train_df)}")
        logger.info(f"Validation rows: {len(self.valid_df)}")
        
        train_labels = self.labels.loc[
            self.train_ids, ["c1", "c2", "c3", "c4"]
        ]
        
        valid_labels = self.labels.loc[
            self.valid_ids, ["c1", "c2", "c3", "c4"]
        ]
        
        logger.info(f"\nClass distribution:")
        
        for c in ["c1", "c2", "c3", "c4"]:
        
            logger.info(
                f"{c}: "
                f"Train = {train_labels[c].sum()} "
                f"({train_labels[c].mean():.4f}), "
                f"Valid = {valid_labels[c].sum()} "
                f"({valid_labels[c].mean():.4f})"
            )
import os
from PIL import Image
import cv2
from torch.utils.data import Dataset
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CVDataset(Dataset):

    def __init__(self, df, image_ids, img_dir, config, is_train):

        self.df = df
        self.transform = None
        self.image_ids = image_ids # df['ImageId'].unique()
        self.config = config
        self.img_dir = img_dir
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            A.Resize(256, 512),
            ToTensorV2()
        ])
        self.valid_transform = val_transform = A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            A.Resize(256, 1024),
            ToTensorV2()
        ])
        
        if is_train:
            self.transform = self.train_transform
        else:
            self.transform = self.valid_transform
        
        self.mask_dict = {}
        for img_id in self.image_ids:
            self.mask_dict[img_id] = {}
            for class_id in range(1, self.config.model.num_classes+1):
                row = df[(df['ImageId'] == img_id) & (df['ClassId'] == class_id)]
                if len(row) > 0:
                    rle = row['EncodedPixels'].values[0]
                    mask = self.rle_decode(rle, (256, 1600))
                else:
                    mask = np.zeros((self.config.model.img_height, self.config.model.img_width), dtype=np.uint8)
                self.mask_dict[img_id][class_id] = mask
                
        self.class2_indices = []

        for idx, img_id in enumerate(self.image_ids):
            class2_mask = self.mask_dict[img_id][2]
        
            if np.any(class2_mask > 0):
                self.class2_indices.append(idx)
        
        class2_set = set(self.class2_indices)

        normal_indices = [
            i for i in range(len(self.image_ids))
            if i not in class2_set
        ]
        
        self.sample_indices = (
            normal_indices +
            self.class2_indices * 4
        ) 

    def __getitem__(self, idx):
        # img_id = self.image_ids[idx] # Debugging line to check which image is being loaded
        real_idx = self.sample_indices[idx]
        img_id = self.image_ids[real_idx]
        
        # Load image (grayscale)
        img_path = os.path.join(self.img_dir, img_id)
        img_path = img_path.replace("\\", "/")  # Ensure compatibility across OS
        image = np.array(
            Image.open(img_path).convert("RGB")
        )
        # image = np.array(image, dtype=np.float32) / 255.0
        # Stack masks for all classes -> (C, H, W)
        masks = []
        for c in range(1, self.config.model.num_classes+1):
            mask = self.mask_dict[img_id][c]
            mask = cv2.resize(
                mask,
                (1600, self.config.model.img_height),
                interpolation=cv2.INTER_NEAREST
            )
            masks.append(mask)
        masks = np.stack(masks, axis=0)
        masks = masks.transpose(1, 2, 0)
        # masks = torch.from_numpy(masks).float()   # shape (4, H, W)
        if self.transform:
            augmented = self.transform(image=image, mask=masks)            
            image = augmented["image"]
            masks = augmented["mask"]
        # Convert to torch tensors
        # image = torch.from_numpy(image)  # (1, H, W)
               # (4, H, W)
               
        if isinstance(masks, np.ndarray):
          masks = torch.from_numpy(masks)

        masks = masks.permute(2, 0, 1).float()
        
        return image, masks

    def __len__(self):
        return len(self.sample_indices) # len(self.image_ids)

    def rle_decode(self, rle, shape=(256, 1600)):
        """Severstal RLE -> (H, W) binary mask."""
        
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        if not isinstance(rle, str) or rle.strip() == "":
            return mask.reshape(shape, order="F")

        values = np.asarray(rle.split(), dtype=np.int64)

        starts = values[0::2] - 1
        lengths = values[1::2]

        for start, length in zip(starts, lengths):
            mask[start:start + length] = 1

        return mask.reshape(shape, order="F")
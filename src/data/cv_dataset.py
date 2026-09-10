import os
from PIL import Image
import cv2
from torch.utils.data import Dataset
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CVDataset(Dataset):

    def __init__(self, df, image_ids, img_dir, config, transform=None):

        self.df = df
        self.transform = transform
        self.image_ids = image_ids
        self.config = config
        self.img_dir = img_dir
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
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

    def __getitem__(self, idx):
        img_id = self.image_ids[idx] 
        img_path = os.path.join(self.img_dir, img_id)
        img_path = img_path.replace("\\", "/")  # Ensure compatibility across OS
        image = np.array(
            Image.open(img_path).convert("RGB")
        )
        # Stack masks for all classes -> (C, H, W)
        masks = []
        for c in range(1, self.config.model.num_classes+1):
            mask = self.mask_dict[img_id][c]
            mask = cv2.resize(
                mask,
                (self.config.model.img_width, self.config.model.img_height),
                interpolation=cv2.INTER_NEAREST
            )
            masks.append(mask)
        masks = np.stack(masks, axis=0)   # shape (4, H, W)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        # Convert to torch tensors
        masks = torch.from_numpy(masks).float()       # (4, H, W)
        return image, masks

    def __len__(self):
        return len(self.image_ids)

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
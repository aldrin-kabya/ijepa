# bingrgb.py
import os
import numpy as np
from logging import getLogger
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

logger = getLogger()

class BingRGBDataset(Dataset):
    def __init__(self, image_file, label_file, transform=None):
        self.image = Image.open(image_file)
        self.label = Image.open(label_file)
        self.transform = transform

    def __len__(self):
        return self.image.size[0] * self.image.size[1]

    def __getitem__(self, idx):
        row = idx // self.image.size[0]
        col = idx % self.image.size[0]
        img_patch = self.image.crop((col, row, col + 224, row + 224)) # Assuming 224x224 patches
        label_patch = self.label.crop((col, row, col + 224, row + 224))

        if self.transform:
            img_patch = self.transform(img_patch)
            label_patch = self.transform(label_patch)

        return img_patch, label_patch

def make_bingrgb(image_file, label_file, transform, batch_size, num_workers=8, pin_mem=True, drop_last=True):
    dataset = BingRGBDataset(image_file=image_file, label_file=label_file, transform=transform)
    logger.info('BingRGB dataset created')
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers
    )
    logger.info('BingRGB data loader created')
    return dataset, data_loader

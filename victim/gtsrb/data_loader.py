from __future__ import print_function

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB

data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image brightness
data_jitter_brightness = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(brightness=-5),
    transforms.ColorJitter(brightness=5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(saturation=5),
    transforms.ColorJitter(saturation=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(contrast=5),
    transforms.ColorJitter(contrast=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(hue=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image horizontally and vertically
data_hvflip = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image horizontally
data_hflip = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image vertically
data_vflip = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and shear image
data_shear = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees=15, shear=2),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and translate image
data_translate = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and crop image 
data_center = transforms.Compose([
    transforms.Resize((36, 36)),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and convert image to grayscale
data_grayscale = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])


class GTSRBData(pl.LightningDataModule):
    def __init__(self, data_dir, num_workers, batch_size, download):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.download = download

    def train_dataloader(self):
        dataset = GTSRB(root=self.data_dir, split="train", transform=data_transforms, download=self.download)
        dataloader = DataLoader(
            torch.utils.data.ConcatDataset([
                GTSRB(root=self.data_dir, split="train", transform=data_transforms, download=self.download),
                GTSRB(root=self.data_dir, split="train", transform=data_jitter_brightness, download=self.download),
                GTSRB(root=self.data_dir, split="train", transform=data_jitter_contrast, download=self.download),
                GTSRB(root=self.data_dir, split="train", transform=data_translate, download=self.download),
                GTSRB(root=self.data_dir, split="train", transform=data_hvflip, download=self.download),
                GTSRB(root=self.data_dir, split="train", transform=data_shear, download=self.download)

            ]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = GTSRB(root=self.data_dir, split="test", transform=data_transforms, download=self.download)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

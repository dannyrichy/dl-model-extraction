import os
from victim.cifar100_models.models import CIFAR100Module
from utils import DEVICE
from victim import RESNET50, VGG19_BN

import zipfile

import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR100
from tqdm import tqdm


class CIFAR100Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR100(root='.', train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR100(root='.', train=False, transform=transform, download=True)
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

    
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


def main(args):
    seed_everything(0)
    checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False)

    trainer = Trainer(
        fast_dev_run=False,
        gpus=-1,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args['max_epochs'],
        checkpoint_callback=checkpoint,
        precision=32,
    )

    data = CIFAR100Data(args)
    model = CIFAR100Module(args, len(data.train_dataloader()))
    
    model.model.to(DEVICE)
#     if bool(args.pretrained):
#         state_dict = os.path.join(
#             "cifar10_models", "state_dicts", args.classifier + ".pt"
#         )
#         model.model.load_state_dict(torch.load(state_dict))
    trainer.fit(model, data.train_dataloader(), data.test_dataloader())
#     print(trainer.callback_metrics["val_acc"])
    return trainer, model, data

trainer, model, data = main({
            "model_name": VGG19_BN,
            "batch_size": 256,
            "max_epochs": 100,
            "num_workers": 2,
            "learning_rate": 1e-2,
            "weight_decay": 1e-2
        })

import numpy as np
li = []
for x, y in data.test_dataloader():
    y_pred = torch.argmax(model.model(x), dim=1)
    li.append(torch.sum(y_pred == y).cpu().detach().numpy())
print(np.sum(li)/len(data.test_data_loader()))
torch.save(model, "vgg19bn_cifar100.pt")

# tmp = torch.load("resnet50_cifar100.pt")
# for x,y in data.test_dataloader():
#     y_pred = torch.argmax(tmp.model(x), dim=1)
#     break

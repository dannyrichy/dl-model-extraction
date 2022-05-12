import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from victim.cifar10_models.data_loader import CIFAR10Data
from victim.cifar10_models.models import CIFAR10Module


def _helper(args):
    checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False)

    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        # logger=logger if not bool(args.dev + args.test_phase) else None,
        auto_select_gpus=True,
        deterministic=True,
        enable_model_summary=False,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        enable_checkpointing=checkpoint,
        precision=args.precision,
        accelerator='gpu'
    )

    model = CIFAR10Module(args)
    # data = CIFAR10Data(data_dir=os.path.join(os.getcwd(), "data"), num_workers=1, batch_size=args.batch_size, download=True)
    state_dict = os.path.join(
        os.getcwd(), "model_weights", "state_dicts", args.classifier + ".pt"
    )
    model.model.load_state_dict(torch.load(state_dict))
    # trainer.test(model, data.test_dataloader())

    return model


if __name__ == '__main__':
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet50")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()

    _helper(args)

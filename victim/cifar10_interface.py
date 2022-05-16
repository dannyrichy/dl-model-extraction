import os

import torch

from victim.cifar10_models.models import CIFAR10Module


def _helper(arg_dict):
    """

    :param arg_dict:
    :type arg_dict: dict
    :return:
    :rtype:
    """
    # checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False)

    # trainer = Trainer(
    #     fast_dev_run=bool(arg_dict["dev"]),
    #     # logger=logger if not bool(args.dev + args.test_phase) else None,
    #     auto_select_gpus=True,
    #     deterministic=True,
    #     enable_model_summary=False,
    #     log_every_n_steps=1,
    #     max_epochs=arg_dict["max_epochs"],
    #     enable_checkpointing=checkpoint,
    #     precision=arg_dict["precision"],
    #     accelerator='gpu'
    # )

    model = CIFAR10Module(arg_dict)
    # data = CIFAR10Data(data_dir=os.path.join(os.getcwd(), "data"), num_workers=1, batch_size=args.batch_size, download=True)
    state_dict = os.path.join(
        os.getcwd(), "model_weights", "state_dicts", arg_dict["model_name"] + ".pt"
    )
    model.model.load_state_dict(torch.load(state_dict))
    # trainer.test(model, data.test_dataloader())

    return model

import os

import torch

from victim.cifar10_models.models import CIFAR10Module


def model_builder(arg_dict):
    """

    :param arg_dict:
    :type arg_dict: dict
    :return:
    :rtype:
    """

    model = CIFAR10Module(arg_dict)
    state_dict = os.path.join(
        os.getcwd(), "model_weights", "state_dicts", arg_dict["model_name"] + ".pt"
    )
    model.model.load_state_dict(torch.load(state_dict))
    return model

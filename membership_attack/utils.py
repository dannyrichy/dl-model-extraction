import os

import torch


def _indices_loader(indices_file):
    return torch.load(indices_file)


def _path_generator(model_file):
    return os.path.join(os.getcwd(), model_file)

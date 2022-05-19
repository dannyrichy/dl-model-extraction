import os

import torch
from torch.distributions import Categorical


def _indices_loader(indices_file):
    return torch.load(indices_file)


def _path_generator(model_file):
    return os.path.join(os.getcwd(), model_file)


def _entropy_calculate(log_its):
    return Categorical(torch.nn.Softmax(log_its)).entropy()

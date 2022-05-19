import os

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader


def _fetch_dataset(_):
    pass


def _indices_generator(q_type, size, percentage):
    if q_type == 'random':
        return np.random.default_rng().choice(
            size,
            size=size * percentage,
            replace=False
        )

    elif q_type in ['coreset', 'coreset_cross']:
        return None


def _path_generator(model_file):
    return os.path.join(os.getcwd(), model_file)


def generate_results(model_file):
    model = torch.load(_path_generator(model_file))

    # TODO: Fetch the query_percentage and query_type
    _, attacker, victim, q_type, perc = tuple(model_file.split("_"))
    indices = _indices_generator(q_type, dataloader.size(), perc)
    dataset = Subset(_fetch_dataset(_), indices)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

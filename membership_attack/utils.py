import os
import pickle

import numpy as np

from victim import CIFAR_10


def _fetch_dataset(_):
    pass


def _indices_generator(q_type, data_size, size, data_type):
    if q_type == 'random':
        return np.random.default_rng().choice(
            data_size,
            size=size,
            replace=False
        )

    elif q_type in ['coreset', 'coreset_cross']:
        file_name = 'cifar{}_{}_index_19'.format(10 if data_type == CIFAR_10 else 100,
                                                 'cross_entropy' if q_type == 'coreset_cross' else 'entropy')
        with open(os.path.join('attacker', 'coreset', file_name), 'rb') as f:
            return pickle.load(f)[:size]


def _path_generator(model_file):
    return os.path.join(os.getcwd(), model_file)

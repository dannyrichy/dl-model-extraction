import torch

from victim import *

config = {
    "batch_size": 500,
    "learning_rate": 0.008,
    "epochs": 80,
    "query_size":10000,
    "query_type": 'random',
    "victim": { "data": CIFAR_10,  "model_name": RESNET50 },
    "attacker": RESNET34,
    "klogits": 0
}

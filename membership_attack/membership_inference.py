import torch

from attacker.utils import fetch_model
from membership_attack.utils import _path_generator, _indices_generator


def _attacker(attacker, q_type, data_size model_path):
    # TODO: Change output to account for both CIFAR10, CIFAR100
    model = fetch_model(attacker, 10)
    model.load_state_dict(torch.load(_path_generator(model_path)))
    indices = _indices_generator(q_type, data_size=, perc)
    dataset = Subset(_fetch_dataset(_), indices)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

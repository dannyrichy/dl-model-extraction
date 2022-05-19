import torch
from torch.utils.data import DataLoader, Subset, default_collate

from attacker.utils import get_model, get_dataset
from membership_attack.utils import _path_generator, _indices_loader, _entropy_calculate
from utils import DEVICE
from victim import OOD
from victim.interface import fetch_logits


def _attacker(attacker, train_set, test_set, ood_dataset, indices_path, model_path):
    model = get_model(attacker, 10)
    model.load_state_dict(torch.load(_path_generator(model_path)))
    model = model.to(DEVICE)
    model.eval()
    indices = _indices_loader(indices_path)
    indices = [i for i in range(len(train_set)) if i not in indices]
    train_set = Subset(train_set, indices)

    return {
        "in_sample": [
            _entropy_calculate(model(x)[0])
            for d_set in [train_set, test_set]
            for x, _ in DataLoader(d_set, batch_size=1, shuffle=False,
                                   collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)))
        ],
        "ood": [_entropy_calculate(model(x)[0]) for x, _ in DataLoader(ood_dataset, batch_size=1, shuffle=False,
                                                                       collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)))],
    }


def _victim(victim_args, train_set, test_set, ood_dataset):
    return {
        "in_sample": [
            _entropy_calculate(fetch_logits(victim_args)[0])
            for d_set in [train_set, test_set]
            for x, _ in DataLoader(d_set, batch_size=1, shuffle=False,
                                   collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)))
        ],
        "ood": [_entropy_calculate(fetch_logits(victim_args, x)[0])
                for x, _ in DataLoader(ood_dataset, batch_size=1, shuffle=False,
                                       collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)))],
    }


def generate_results(victim_args, attacker, indices_path, model_path, data_type):
    train_set, test_set, outputs = get_dataset(data_type)
    ood_dataset = get_dataset(OOD)
    attacker_result = _attacker(attacker, train_set, test_set, ood_dataset, indices_path, model_path)
    victim_result = _victim(victim_args, train_set, test_set, ood_dataset)

    return attacker_result, victim_result

import torch
from torch.utils.data import DataLoader, Subset

from attacker.utils import get_model, get_dataset
from membership_attack.utils import _path_generator, _indices_loader


def _entropy_calculate(log_its):
    return torch.distributions.Categorical(torch.nn.Softmax(log_its)).entropy()


def _attacker(attacker, data_type, indices_path, model_path):
    model = get_model(attacker, 10)
    model.load_state_dict(torch.load(_path_generator(model_path)))

    indices = _indices_loader(indices_path)
    full_dataset = get_dataset(data_type)
    indices = [i for i in range(len(full_dataset)) if i not in indices]

    dataset = Subset(full_dataset, indices)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    return [_entropy_calculate(model(x)) for x, _ in dataloader]

import torch


# model = torch.hub.load("pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
from victim.pytorch_cifar_models import resnet


def test_resnet(dataset, model_name, x):
    num_classes = 10 if dataset == "cifar10" else 100
    model = getattr(resnet, f"{dataset}_{model_name}")()
    # x = torch.empty((1, 3, 32, 32))
    assert x.shape == (1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, num_classes)
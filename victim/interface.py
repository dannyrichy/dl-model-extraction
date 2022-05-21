import victim.cifar100_interface as cifar100
import victim.cifar10_interface as cifar10
from victim import *
from utils import DEVICE


def fetch_logits(args=None, query_img=None):
    """

    :param args:
    :type args:
    :param query_img:
    :type query_img: list
    :return:
    :rtype:
    """
    if args is None:
        args = {
            "data": CIFAR_10,
            "model_name": RESNET50,
            "batch_size": 256,
            "max_epochs": 100,
            "num_workers": 8,
            "learning_rate": 1e-2,
            "weight_decay": 1e-2
        }
    else:
        args["batch_size"] = 256
        args["max_epochs"] = 100
        args["num_workers"] = 8
        args["learning_rate"] = 1e-2
        args["weight_decay"] = 1e-2
    if args["data"] in[CIFAR_10, OOD]:
        model = cifar10.model_builder(args)
        model.to(DEVICE)
        model.model.eval()
        return model.model(query_img)
    elif args["data"] == CIFAR_100:
        model = cifar100.model_builder(args)
        model.to(DEVICE)
        model.eval()
        return model(query_img)


def fetch_victim_model(args=None):
    """

    :param args:
    :type args:
    :return:
    :rtype:
    """
    if args is None:
        args = {
            "data": CIFAR_10,
            "model_name": RESNET50,
            "batch_size": 256,
            "max_epochs": 100,
            "num_workers": 8,
            "learning_rate": 1e-2,
            "weight_decay": 1e-2
        }
    else:
        args["batch_size"] = 256
        args["max_epochs"] = 100
        args["num_workers"] = 8
        args["learning_rate"] = 1e-2
        args["weight_decay"] = 1e-2
    if args["data"] in[CIFAR_10, OOD]:
        model = cifar10.model_builder(args)
        model.to(DEVICE)
        model.model.eval()
        return model.model
    elif args["data"] == CIFAR_100:
        model = cifar100.model_builder(args)
        model.to(DEVICE)
        model.eval()
        return model



import victim.cifar100_interface as cifar100
import victim.cifar10_interface as cifar10
from utils import DEVICE
from victim import *


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
    if args["data"] == CIFAR_10:
        model = cifar10.model_builder(args)
        model.to(DEVICE)
        # model.model.eval()
        return model.model(query_img)
    elif args["data"] == CIFAR_100:
        model = cifar100.model_builder(args)
        model.to(DEVICE)
        return model(query_img)

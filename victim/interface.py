import victim.cifar100_interface as cifar100
import victim.cifar10_interface as cifar10
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
        # parser = ArgumentParser()
        # parser.add_argument("--data", type=str, default="cifar10", choices=["cifar10", "cifar100", "gtsrb"])
        # # PROGRAM level args
        # parser.add_argument("--test_phase", type=int, default=1, choices=[0, 1])
        # parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
        # parser.add_argument(
        #     "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
        # )
        #
        # # TRAINER args
        # parser.add_argument("--classifier", type=str, default="resnet50")
        # parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])
        #
        # parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
        # parser.add_argument("--batch_size", type=int, default=256)
        # parser.add_argument("--max_epochs", type=int, default=100)
        # parser.add_argument("--num_workers", type=int, default=8)
        # parser.add_argument("--gpu_id", type=str, default="0")
        #
        # parser.add_argument("--learning_rate", type=float, default=1e-2)
        # parser.add_argument("--weight_decay", type=float, default=1e-2)
        #
        # args = parser.parse_args()
    if args["data"] == CIFAR_10:
        model = cifar10._helper(args)
        return model.model(query_img)
    elif args["data"] == CIFAR_100:
        model = cifar100._helper(args)
        return model(query_img)

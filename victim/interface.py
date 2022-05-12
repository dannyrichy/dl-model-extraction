from argparse import ArgumentParser

from victim.cifar10_interface import _helper


def get_weights(model):
    """

    :param model: Model
    :type model: torch.nn.Module
    :return:
    :rtype:
    """
    return {
        name: params
        for name, params in model.named_parameters()
    }


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
        parser = ArgumentParser()

        # PROGRAM level args
        parser.add_argument("--test_phase", type=int, default=1, choices=[0, 1])
        parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
        parser.add_argument(
            "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
        )

        # TRAINER args
        parser.add_argument("--classifier", type=str, default="resnet50")
        parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

        parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--max_epochs", type=int, default=100)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--gpu_id", type=str, default="0")

        parser.add_argument("--learning_rate", type=float, default=1e-2)
        parser.add_argument("--weight_decay", type=float, default=1e-2)

        args = parser.parse_args()
    model = _helper(args)
    return [model(x) for x in query_img]

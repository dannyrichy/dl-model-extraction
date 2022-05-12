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


def fetch_logits(args, query_img=None):
    """

    :param args:
    :type args:
    :param query_img:
    :type query_img:
    :return:
    :rtype:
    """
    # TODO: Add transformers for images
    return _helper(args)

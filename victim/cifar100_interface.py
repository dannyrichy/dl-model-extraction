from victim.pytorch_cifar_models import resnet, vgg, repvgg, shufflenetv2, mobilenetv2, vit


def model_builder(args_dict):
    if args_dict["model_name"].startswith("resnet"):
        return getattr(resnet, f"{args_dict['data']}_{args_dict['model_name']}")()
    elif args_dict["model_name"].startswith("repvgg"):
        return getattr(repvgg, f"{args_dict['data']}_{args_dict['model_name']}")()
    elif args_dict["model_name"].startswith("mobilenetv2"):
        return getattr(mobilenetv2, f"{args_dict['data']}_{args_dict['model_name']}")()
    elif args_dict["model_name"].startswith("vgg"):
        return getattr(vgg, f"{args_dict['data']}_{args_dict['model_name']}")()
    elif args_dict["model_name"].startswith("shufflenetv2"):
        return getattr(shufflenetv2, f"{args_dict['data']}_{args_dict['model_name']}")()
    elif args_dict["model_name"].startswith("vit"):
        return getattr(vit, f"{args_dict['data']}_{args_dict['model_name']}")()

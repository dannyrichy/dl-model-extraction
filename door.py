import os
import zipfile
from argparse import ArgumentParser

import requests
from tqdm import tqdm

from victim.interface import fetch_logits


def download_cifar10_model():
    url = (
        "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
    )

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2 ** 20  # Mebibyte
    t = tqdm(total=total_size, unit="MiB", unit_scale=True)

    with open(os.path.join("model_weights", "state_dicts.zip"), "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        raise Exception("Error, something went wrong")

    print("Download successful. Unzipping file...")
    path_to_zip_file = os.path.join(os.getcwd(), "model_weights", "state_dicts.zip")
    directory_to_extract_to = os.path.join(os.getcwd(), "model_weights")
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        print("Unzip file successful!")


if __name__ == '__main__':
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
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
    parser.add_argument("--gpu_id", type=str, default="3")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()
    fetch_logits(args)

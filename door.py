from data_free.extract import run_dfme
from victim import CIFAR_10, RESNET50
from victim.interface import fetch_victim_model

if __name__ == '__main__':
    tmp = run_dfme(fetch_victim_model(args={
        "data": CIFAR_10,
        "model_name": RESNET50}))
    print(tmp)

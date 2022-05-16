import torch
config = {
    "batch_size": 500,
    "learning_rate": 0.008,
    "epochs": 100,
    "query_size":1000,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}
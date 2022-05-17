import torch
from victim.__init__ import *

config = {
    "batch_size": 500,
    "learning_rate": 0.008,
    "epochs": 80,
    "query_size":10000,
    "query_type": 'coreset_cross',
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "victim": { "data": CIFAR_100,  "model_name": RESNET_56 }
}

# config = {
#     "batch_size": 50,
#     "learning_rate": 0.001,
#     "epochs": 10,
#     "query_size":500,
#     "query_type": 'coreset',
#     "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#     "victim": { "data": CIFAR_10,  "model_name": VGG11_BN }
# }

parameters = {
        "query_size": [10000, 20000, 30000, 40000],
        "query_type": ['coreset', 'coreset_cross'],
        "victim":[{ "data": CIFAR_10, "model_name": RESNET50}]
}

# parameters = {
#         "query_size": [10000, 20000, 30000, 40000],
#         "query_type": ['random', 'coreset', 'coreset_cross'],
#         "victim":[{ "data": CIFAR_100,  "model_name": RESNET_56},
#                   {"data": CIFAR_10,  "model_name": VGG11_BN},
#                   {"data": CIFAR_100,  "model_name": VGG11_BN}]
# }

# parameters = {
#         "query_size": [1000],
#         "query_type": ['coreset_cross'],
#         "victim":[{ "data": CIFAR_10,  "model_name": RESNET50}]
# }

# # Parameter Runs

# ## Run1

# parameters = {
#         "query_size": [10000, 20000, 30000, 40000, 50000],
#         "query_type": ['random', 'coreset', 'coreset_cross'],
#         "victim":[{ "data": CIFAR_10,  "model_name": RESNET50}]
# }

# config = {
#     "batch_size": 500,
#     "learning_rate": 0.008,
#     "epochs": 100,
#     "query_size":5000,
#     "query_type": 'coreset',
#     "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#     "victim": { "data": CIFAR_100,  "model_name": RESNET_56 }
# }

# ## Run2

#

#

import torch
from victim.__init__ import *

config = {
    "batch_size": 500,
    "learning_rate": 0.008,
    "epochs": 80,
    "query_size":10000,
    "query_type": 'random',
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "victim": { "data": CIFAR_10,  "model_name": RESNET50 },
    "attacker": RESNET50
}

parameters = {
        "query_size": [20000],
        "query_type": ['random','coreset', 'coreset_cross'],
        "victim":[{ "data": CIFAR_10, "model_name": RESNET50}],
        "attacker": [RESNET34]
}

# # Parameter Runs

# ## Best Query Type, Best Query sizes on 2 datasets

# ### Run1 - CIFAR10 - A_RESNET34 - V_RESNET50

# config = {
#     "batch_size": 500,
#     "learning_rate": 0.008,
#     "epochs": 100,
#     "query_size":5000,
#     "query_type": 'coreset',
#     "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#     "victim": { "data": CIFAR_100,  "model_name": RESNET_56 }
# }

# parameters = {
#         "query_size": [10000, 20000, 30000, 40000, 50000],
#         "query_type": ['random', 'coreset', 'coreset_cross'],
#         "victim":[{ "data": CIFAR_10,  "model_name": RESNET50}]
# }

# ### Run2 - CIFAR10 - A_RESNET34 - V_RESNET50 - (running currently)

# config = {
#     "batch_size": 500,
#     "learning_rate": 0.008,
#     "epochs": 80,
#     "query_size":10000,
#     "query_type": 'coreset',
#     "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#     "victim": { "data": CIFAR_10,  "model_name": RESNET50 },
# }

# parameters = {
#         "query_size": [10000, 20000, 30000, 40000],
#         "query_type": ['coreset', 'coreset_cross'],
#         "victim":[{ "data": CIFAR_10, "model_name": RESNET50}]
# }

# ### Run3 - CIFAR10 - A_RESNET34 - V_VGG19_BN - (tbr)

# config = {
#     "batch_size": 500,
#     "learning_rate": 0.008,
#     "epochs": 80,
#     "query_size":10000,
#     "query_type": 'random',
#     "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#     "victim": { "data": CIFAR_10,  "model_name": RESNET34 },
#     "attacker": RESNET34
# }

# parameters = {
#         "query_size": [10000, 20000, 30000, 40000, 10000],
#         "query_type": ['random','coreset', 'coreset_cross'],
#         "victim":[{ "data": CIFAR_10, "model_name": VGG19_BN}],
#         "attacker":[RESNET34]
# }

# ### Run3 - CIFAR100 - A_RESNET34 (tbr)

# config = {
#     "batch_size": 500,
#     "learning_rate": 0.008,
#     "epochs": 80,
#     "query_size":10000,
#     "query_type": 'random',
#     "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#     "victim": { "data": CIFAR_100,  "model_name": RESNET_52 },
#     "attacker": RESNET34
# }

# parameters = {
#         "query_size": [10000, 20000, 30000, 40000, 10000],
#         "query_type": ['random','coreset', 'coreset_cross'],
#         "victim":[{ "data": CIFAR_100, "model_name": VGG19_BN}, {"data": CIFAR_100, "model_name": RESNET_56 } ],
#         "attacker":[RESNET34]
# }

# ## Best Attacker, Worst Victim

# ### Run1 - CIFAR10 - Best Query

#

#

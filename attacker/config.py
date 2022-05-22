from victim import *

# Configuration for attacker trainning

config = {"batch_size": 500,
          "learning_rate": 0.008,
          "lr_cycles": 1,
          "lr_steps": 1000,
          "base_lr": 1e-5,
          "max_lr": 1e-1,
          "test_trial": {"query_size": [20000],
                           "query_type": ['coreset'],
                           "victim":[{ "data": CIFAR_10, "model_name": RESNET50}],
                           "attacker":[RESNET34],
                           "k_logits": [None]}
}

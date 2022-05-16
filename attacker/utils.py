import torch
import numpy as np
import pickle

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)

def LoadCoreset(filename):
    with open('attacker/coreset/'+filename, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

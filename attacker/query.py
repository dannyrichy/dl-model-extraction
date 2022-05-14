import torch
import numpy as np
from attacker.config import *
from attacker.utils import *
from victim.interface import fetch_logits


def QueryVictim(victim_type, trainloader, query_size, sampling=None):
  # create sampleset from trainloader
  dataset = trainloader.dataset
  if(sampling=='random'):
    indices = np.random.default_rng().choice(len(dataset), size=query_size, replace=False)
  elif(sampling=='coreset'):
    indices = LoadCoreset('cifar10_entropy_index_19')
  elif(sampling=='coreset_cross'):
    indices = LoadCoreset('cifar10_cross_entropy_index_19')
  else:
    indices = np.arange(query_size)
  dataset = torch.utils.data.Subset(dataset, indices)
  dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

  # query victim
  X, _ = next(iter(dataloader))
  # if torch.cuda.is_available():
  #   X = X.type(torch.cuda.FloatTensor)
  Y = fetch_logits(query_img=X)
  Y = torch.max(Y.data, 1)[1]
  if torch.cuda.is_available():
    Y = Y.type(torch.cuda.LongTensor)

  # create queryset
  querydataset = torch.utils.data.TensorDataset(X, Y)
  queryloader = torch.utils.data.DataLoader(querydataset, batch_size=config['batch_size'], shuffle=True)
  return queryloader


def QueryVictimModel(victim, trainloader, query_size, sampling=None):
  # create sampleset from trainloader
  dataset = trainloader.dataset
  if(sampling=='random'):
    indices = np.random.default_rng().choice(len(dataset), size=query_size, replace=False)
  else:
    indices = np.arange(query_size)
  dataset = torch.utils.data.Subset(dataset, indices)
  dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

  # query victim
  X, _ = next(iter(dataloader))
  if torch.cuda.is_available():
    xList = X.type(torch.cuda.FloatTensor)
    victim.to(torch.device(config['device']))
  # Y = victim(xList) 
  Y = torch.max(Y.data, 1)[1]
  if torch.cuda.is_available():
    Y = Y.type(torch.cuda.LongTensor)

  # create queryset
  querydataset = torch.utils.data.TensorDataset(X, Y)
  queryloader = torch.utils.data.DataLoader(querydataset, batch_size=config['batch_size'], shuffle=True)
  return queryloader



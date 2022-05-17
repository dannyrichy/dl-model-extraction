import torch
import numpy as np
from attacker.config import *
from attacker.utils import *
from victim.interface import fetch_logits


def QueryVictim(victim, trainloader, query_size, query_type=None):
    # create sampleset from trainloader
    dataset = trainloader.dataset
    if(query_type=='random'):
        indices = np.random.default_rng().choice(len(dataset), size=query_size, replace=False)
    elif(query_type=='coreset' or query_type=='coreset_cross'):
        indices = LoadCoreset(victim["data"], query_type)
        indices = indices[:query_size]
    else:
        indices = np.arange(query_size)
    dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    assert len(dataloader.dataset) == query_size,"Sampled dataset not equal to query size"
    assert len(dataloader.dataset[0]) == len(trainloader.dataset[0]), "Sampled dimenions don't match input"

    # query victim
    print(f'Query {victim["model_name"]} victim on {victim["data"]} dataset with query size {query_size}')
    X = []
    Y = []
    cntr = 0
    for (xList, _) in dataloader:
        yList = fetch_logits(args=victim, query_img=xList)
        yList = torch.max(yList.data, 1)[1]
        X.append(xList)
        Y.append(yList)
        cntr+=len(xList)
        print('\r %d ...' % cntr, end='')

    # create queryset   
    querydataset = torch.utils.data.TensorDataset(torch.cat(X), torch.cat(Y))
    queryloader = torch.utils.data.DataLoader(querydataset, batch_size=config['batch_size'], shuffle=True)
    assert len(queryloader.dataset) == query_size,"Queried dataloader not equal to query size"
    assert len(queryloader.dataset[0]) == len(trainloader.dataset[0]),"Queried dataloader dimension are wrong"
    print(f'\r    - input:{len(trainloader.dataset)} queried:{len(queryloader.dataset), len(queryloader.dataset[0])}')
    return queryloader

# # archived: query victim model directly
#     X, _ = next(iter(dataloader))
#     if torch.cuda.is_available():
#         xList = X.type(torch.cuda.FloatTensor)
#     victim.to(torch.device(config['device']))
#     Y = victim(xList) 
#     Y = torch.max(Y.data, 1)[1]
#     if torch.cuda.is_available():
#         Y = Y.type(torch.cuda.LongTensor)
#
#     # create queryset
#     querydataset = torch.utils.data.TensorDataset(X, Y)

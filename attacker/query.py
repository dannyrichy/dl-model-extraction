import os
import torch
import numpy as np
from attacker.config import *
from attacker.utils import *
from victim.interface import fetch_logits


def QueryVictim(victim, outputs, trainloader, query_size, query_type=None):
    # update filename
    filename = f'attacker/queried/query_data_{victim["data"]}_{victim["model_name"]}.pt'
    # load data if file exists:
    if(os.path.exists(os.path.join(os.getcwd(),filename))):
        print(f'Loading queried {victim["data"]} dataset with {victim["model_name"]} victim')
        queryloader = torch.load(filename)
    else:
        # query and save data
        queryloader = QueryVictimDataset(victim, trainloader)
        torch.save(queryloader, filename)
    
    # sample data
    dataloader = QueryType(victim, outputs, queryloader, query_size, query_type=query_type)
    return dataloader


# Query Victim on given dataset
def QueryVictimDataset(victim, trainloader):
    # query victim
    print(f'Query {victim["model_name"]} victim on {victim["data"]} dataset')
    X = []
    Y = []
    cntr = 0
    for (xList, _) in trainloader:
        if torch.cuda.is_available():
            xList = xList.type(torch.cuda.FloatTensor)
        yList = fetch_logits(args=victim, query_img=xList)
        yList = torch.max(yList.data, 1)[1]
        X.append(xList)
        Y.append(yList)
        cntr+=len(xList)
        print('\r %d ...' % cntr, end='')

    # create queryset   
    querydataset = torch.utils.data.TensorDataset(torch.cat(X), torch.cat(Y))
    queryloader = torch.utils.data.DataLoader(querydataset, batch_size=config['batch_size'], shuffle=False)
    assert len(queryloader.dataset) == len(trainloader.dataset),"Queried dataloader not equal to query size"
    assert len(queryloader.dataset[0]) == len(trainloader.dataset[0]),"Queried dataloader dimension are wrong"
    print(f'\r    - input:{len(trainloader.dataset)} queried:{len(queryloader.dataset)}')
    return queryloader


# Sample dataset using query-type and size
def QueryType(victim, outputs, queryloader, query_size, query_type=None):
    # create sampleset from queryloader
    print(f'Sample using {query_type} with query size {query_size}')
    dataset = queryloader.dataset
    if(query_type=='random'):
        indices = np.random.default_rng().choice(len(dataset), size=query_size, replace=False)
    elif(query_type=='coreset' or query_type=='coreset_cross'):
        ind_dict = LoadCoreset(victim["data"], query_type)
        class_query_size = int(query_size/outputs);
        indices = []
        for label in ind_dict.values():
            indices.extend(label[:class_query_size])
    else:
        indices = np.arange(query_size)
    dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    assert len(dataloader.dataset) == query_size,"Sampled dataset not equal to query size"
    assert len(dataloader.dataset[0]) == len(queryloader.dataset[0]), "Sampled dimenions don't match input"
    print(f'\r    - input:{len(queryloader.dataset)} sampled:{len(dataloader.dataset)}')
    return dataloader

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

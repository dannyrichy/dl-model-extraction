import os
from torch.utils.data import TensorDataset
from attacker.config import config
from attacker.utils import *
from victim.interface import  fetch_victim_model


def query_victim(victim, outputs, train_loader, query_size,q_type=None, train=True):
    # update filename
    if train:
        filename = f'queried/query_{victim["data"]}_{victim["model_name"]}_traindata'
    else:
        filename = f'queried/query_{victim["data"]}_{victim["model_name"]}_testdata'
    # load data if file exists:
    if os.path.exists(os.path.join(os.getcwd(), filename) + '.pt'):
        print(f'Loading queried {victim["data"]} dataset with {victim["model_name"]} victim')
        query_loader = torch.load(filename + '.pt')
        print(f'    - input:{len(train_loader.dataset)} queried:{len(query_loader.dataset)}')
    else:
        # query and save data
        query_loader = query_victim_dataset(victim, train_loader)
        torch.save(query_loader, filename + '.pt')

    # sample data
    dataloader = query_type(victim, outputs, query_loader, query_size, filename, query_type=q_type)
    return dataloader


# Query Victim on given dataset
def query_victim_dataset(victim, train_loader):
    # query victim
    print(f'Query {victim["model_name"]} victim on {victim["data"]} dataset')
    X = []
    Y = []
    cntr = 0
    victim_model = fetch_victim_model(args=victim)
    for (xList, _) in train_loader:
        if torch.cuda.is_available():
            xList = xList.type(torch.cuda.FloatTensor)
        yList = victim_model(xList)
        yList = torch.max(yList.data, 1)[1]
            
        X.append(xList)
        Y.append(yList)
        cntr += len(xList)
        print('\r %d ...' % cntr, end='')

    # create queryset   
    query_dataset = TensorDataset(torch.cat(X), torch.cat(Y))
    query_loader = DataLoader(query_dataset, batch_size=config['batch_size'], shuffle=False)
    assert len(query_loader.dataset) == len(train_loader.dataset), "Queried dataloader not equal to query size"
    assert len(query_loader.dataset[0]) == len(train_loader.dataset[0]), "Queried dataloader dimension are wrong"
    print(f'\r    - input:{len(train_loader.dataset)} queried:{len(query_loader.dataset)}')
    return query_loader


# Sample dataset using query-type and size
def query_type(victim, outputs, queryloader, query_size, filename, query_type=None):
    # create sampleset from queryloader
    print(f'Sample using {query_type} with query size {query_size}')
    dataset = queryloader.dataset
    if query_type == 'random':
        indices = np.random.default_rng().choice(len(dataset), size=query_size, replace=False)
    elif query_type == 'coreset' or query_type == 'coreset_cross':
        ind_dict = load_coreset(victim["data"], query_type)
        class_query_size = int(query_size / outputs)
        indices = []
        for label in ind_dict.values():
            indices.extend(label[:class_query_size])
    else:
        indices = np.arange(query_size)

    # Saving the indices
    torch.save(indices, filename + f'{query_type}_{query_size}_indices.pt')

    dataset = torch.utils.data.Subset(dataset, indices)
    dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    assert len(dataloader.dataset) == query_size, "Sampled dataset not equal to query size"
    assert len(dataloader.dataset[0]) == len(queryloader.dataset[0]), "Sampled dimenions don't match input"
    print(f'    - input:{len(queryloader.dataset)} sampled:{len(dataloader.dataset)}')
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

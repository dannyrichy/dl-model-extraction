import torch, torchvision
import numpy as np
import pickle
import matplotlib.pyplot as plt
from attacker.config import *
from victim.__init__ import *

# set seed for all packages

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)

# Download and load datasets from Pytorch

# Download and use dataset
def getDataset(dataset, batch_size):
    # Normalizing transform
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # select dataset
    if(dataset==CIFAR_10):
        trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
    elif(dataset==CIFAR_100):
        trainset = torchvision.datasets.CIFAR100(root='./cifar100', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transform)
    else:
        raise Exception("Dataset not configured!")

    # uplaod to data loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
     
    return trainloader, testloader


# Load Indices from saved corset algorithms

def LoadCoreset(dataset, query_type):
    # select filename
    if(query_type=='coreset'):
        if(dataset==CIFAR_10):
            filename = 'cifar10_entropy_index_19'
        elif(dataset==CIFAR_100):
            filename = 'cifar100_entropy_index_19'
        else:
            raise Exception("Dataset is not configured for coreset!")
    elif(query_type=='coreset_cross'):
        if(dataset==CIFAR_10):
            filename = 'cifar10_cross_entropy_index_19'
        elif(dataset==CIFAR_100):
            filename = 'cifar100_cross_entropy_index_19'
        else:
            raise Exception("Dataset is not configured for coreset cross!")
    else:
        raise Exception("Query Type is not coreset!")
    
    # Open file and load data
    with open('attacker/coreset/'+filename, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


# Save and Visualize all results

# visualization
def SaveVisualize(model, result, title):
    (train_loss, train_acc, test_loss, test_acc) = result
    
    # save everything
    torch.save(model.state_dict(), f'attacker/results/{title}')
    torch.save(result,f'attacker/results/{title}_result')
    #torch.save(train_loss,f'attacker/results/{title}_TrainLoss')
    #torch.save(train_acc,f'attacker/results/{title}_TrainAcc')
    #torch.save(test_loss,f'attacker/results/{title}_TestLoss' )
    #torch.save(test_acc,f'attacker/results/{title}_TestAcc' )
    
    # plot
    plt.plot(range(config["epochs"]), train_loss, 'b-', label='Train_loss')
    plt.plot(range(config["epochs"]), test_loss, 'g-', label='Test_loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'attacker/results/{title}_Loss')
    plt.show()
    plt.plot(range(config["epochs"]), train_acc, 'b-', label='Train_accuracy')
    plt.plot(range(config["epochs"]), test_acc, 'g-', label='Test_accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'attacker/results/{title}_Accuracy')
    plt.show()
    
    # return in percentages
    train_l = np.round_(train_loss[len(train_loss)-1].cpu().numpy(),4)
    train_a = np.round_(train_acc[len(train_acc)-1].cpu().numpy()*100,2)
    test_l  = np.round_(test_loss[len(test_loss)-1].cpu().numpy(),4)
    test_a  = np.round_(test_acc[len(test_acc)-1].cpu().numpy()*100,2)
    
    return (train_l, train_a, test_l,test_a)

    

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
def getDataset(dataset):
    # Normalizing transform
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # select dataset
    if(dataset==CIFAR_10):
        trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
        outputs=10
    elif(dataset==CIFAR_100):
        trainset = torchvision.datasets.CIFAR100(root='./cifar100', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transform)
        outputs=100
    else:
        raise Exception("Dataset not configured!")

    # uplaod to data loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False)
     
    return trainloader, testloader, outputs


# Load Indices from saved corset algorithms

def LoadCoreset(dataset, query_type):
    # select filename
    if(query_type=='coreset'):
        if(dataset==CIFAR_10):
            filename = 'cifar10_entropy_dict'
        elif(dataset==CIFAR_100):
            filename = 'cifar100_entropy_dict'
        else:
            raise Exception("Dataset is not configured for coreset!")
    elif(query_type=='coreset_cross'):
        if(dataset==CIFAR_10):
            filename = 'cifar10_cross_entropy_dict'
        elif(dataset==CIFAR_100):
            filename = 'cifar100_cross_entropy_dict'
        else:
            raise Exception("Dataset is not configured for coreset cross!")
    else:
        raise Exception("Query Type is not coreset!")
    
    # Open file and load data
    with open('attacker/coreset/'+filename, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


# Get one of the attacker models

# select Model
def getModel(model_name, outputs):
    if(model_name==RESNET34):
        model = ResNet34(3, ResBlock, outputs=outputs)
    elif(model_name==RESNET50):
        model = torchvision.models.resnet50(num_classes=outputs)
    elif(model_name==VGG19_BN):
        model = torchvision.models.vgg19_bn(num_classes=outputs)
    else:
        raise Exception("Model is not configured!")
    return model


# Train the models

def Training(model, train_loader, test_loader, input_shape, epochs, optimizer, loss):
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    for epoch in range(epochs):
        num_train = 0
        num_correct_train = 0
        print("\repoch", epoch+1)
        # Train Data
        for (xList, yList) in train_loader:
            xList, yList = torch.autograd.Variable(
                xList), torch.autograd.Variable(yList)
            optimizer.zero_grad()

            if torch.cuda.is_available():
                xList = xList.type(torch.cuda.FloatTensor)
                yList = yList.type(torch.cuda.LongTensor)
                device = torch.device(config['device'])
                model.to(device)

            outputs = model(xList)
            train_loss_func = loss(outputs, yList)
            train_loss_func.backward()
            optimizer.step()

            num_train += len(yList)  # i.e., add bath size

            predicts = torch.max(outputs.data, 1)[1]
            num_correct_train += (predicts == yList).float().sum()
            
            print('\r %d ...' % num_train, end='')
        
        train_acc.append(num_correct_train / num_train)
        train_loss.append(train_loss_func.data)
        print("\r    - train_acc %.5f train_loss %.5f" %
                  (train_acc[-1], train_loss[-1]))
        #if(epoch%(int(epochs/4))==0): print(model.L0[0].weight)

        # Test Data
        num_test = 0
        num_correct_test = 0
        for (xList, yList) in test_loader:
            if torch.cuda.is_available():
                xList = xList.type(torch.cuda.FloatTensor)
                yList = yList.type(torch.cuda.LongTensor)
                device = torch.device(config['device'])
                model.to(device)
            
            outputs = model(xList)
            test_loss_func = loss(outputs, yList)

            num_test += len(yList)
            predicts = torch.max(outputs.data, 1)[1]
            num_correct_test += (predicts == yList).float().sum()

        test_acc.append(num_correct_test / num_test)
        test_loss.append(test_loss_func.data)
        print("\r    - test_acc  %.5f test_loss  %.5f" %
                (test_acc[-1], test_loss[-1]))
    return train_loss, train_acc, test_loss, test_acc


# Save and Visualize all results

# visualization
def SaveVisualize(model, result, title):
    (train_loss, train_acc, test_loss, test_acc) = result
    
    # save everything
    torch.save(model.state_dict(), f'results/{title}')
    torch.save(result,f'results/{title}_result')
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
    plt.savefig(f'results/{title}_Loss')
    plt.show()
    plt.plot(range(config["epochs"]), train_acc, 'b-', label='Train_accuracy')
    plt.plot(range(config["epochs"]), test_acc, 'g-', label='Test_accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'results/{title}_Accuracy')
    plt.show()
    
    # return in percentages
    train_l = np.round_(train_loss[len(train_loss)-1].cpu().numpy(),4)
    train_a = np.round_(train_acc[len(train_acc)-1].cpu().numpy()*100,2)
    test_l  = np.round_(test_loss[len(test_loss)-1].cpu().numpy(),4)
    test_a  = np.round_(test_acc[len(test_acc)-1].cpu().numpy()*100,2)
    
    return (train_l, train_a, test_l,test_a)

    

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader, random_split

from attacker.ResNet34 import *
from attacker.config import *
from ood.dataset import OODDataset
from victim import *
from utils import DEVICE


# set seed for all packages

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# transforms for normalizing and data augmentation

transform_normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))

transform_data_augment = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                                     torchvision.transforms.RandomHorizontalFlip(),
                                                     transform_normalize])


# Download and load datasets from Pytorch

# Download and use dataset
def get_dataset(data_type):
    """

    :param data_type:
    :type data_type: str
    :return: torch.utils.data.DataLoader, torch.utils.data.DataLoader, int
    :rtype:
    """
    # Normalizing transform
    transform = torchvision.transforms.ToTensor()

    # select dataset
    if data_type == CIFAR_10:
        train_set = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
        outputs = 10
    elif data_type == CIFAR_100:
        train_set = torchvision.datasets.CIFAR100(root='./cifar100', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transform)
        outputs = 100
    elif data_type == OOD:
        data_type = OODDataset()
        train_size = int(0.75 * len(data_type))
        train_set, test_set = random_split(data_type, [train_size, len(data_type) - train_size])
        outputs = 10

    else:
        raise Exception("Dataset not configured!")

    return train_set, test_set, outputs


def get_dataloader(data_type):
    train_set, test_set, outputs = get_dataset(data_type)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    return train_loader, test_loader, outputs


# Load Indices from saved corset algorithms

def load_coreset(dataset, query_type):
    import pickle
    # select filename
    if query_type == 'coreset':
        if dataset == CIFAR_10:
            filename = 'cifar10_entropy_dict'
        elif dataset == CIFAR_100:
            filename = 'cifar100_entropy_dict'
        else:
            raise Exception("Dataset is not configured for coreset!")
    elif query_type == 'coreset_cross':
        if dataset == CIFAR_10:
            filename = 'cifar10_cross_entropy_dict'
        elif dataset == CIFAR_100:
            filename = 'cifar100_cross_entropy_dict'
        else:
            raise Exception("Dataset is not configured for coreset cross!")
    else:
        raise Exception("Query Type is not coreset!")

    # Open file and load data
    with open('attacker/coreset/' + filename, 'rb') as fo:
        indices = pickle.load(fo)
    return indices


# Get one of the attacker models

# select Model
def get_model(model_name, outputs):
    if model_name == RESNET34:
        model = ResNet34(3, outputs=outputs)
    elif model_name == RESNET50:
        model = torchvision.models.resnet50(num_classes=outputs)
    elif model_name == VGG19_BN:
        model = torchvision.models.vgg19_bn(num_classes=outputs)
    else:
        raise Exception("Model is not configured!")
    return model


# Train the models

def training(model, train_loader, test_loader):
    # initialize score lists
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    train_loss_func = None
    
    # select optimiser and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["base_lr"], max_lr=config["max_lr"],
                                                  step_size_up=config["lr_steps"], mode='triangular2', cycle_momentum=False)
    
    # select loss
    loss = torch.nn.CrossEntropyLoss()
    
    # train for epochs
    epochs = int(2*config["lr_cycles"]*config["lr_steps"]/(len(train_loader.dataset)/config["batch_size"]))
    print(f'Total epochs: {epochs}')
    for epoch in range(epochs):
        print("\repoch", epoch + 1)
        
        # train on Train Data
        num_train = 0
        num_correct_train = 0
        model.train() 
        for (xList, yList) in train_loader:
            xList, yList = torch.autograd.Variable(xList), torch.autograd.Variable(yList)
            optimizer.zero_grad()

            if torch.cuda.is_available():
                xList = xList.type(torch.cuda.FloatTensor)
                yList = yList.type(torch.cuda.LongTensor)
                device = torch.device(DEVICE)
                model.to(device)
            
            # get outputs and train model
            with torch.set_grad_enabled(True):
                outputs = model(xList)
                train_loss_func = loss(outputs, yList)
                train_loss_func.backward()
                optimizer.step()
                scheduler.step()

            # get correct predictions count
            num_train += len(yList)
            predicts = torch.max(outputs.data, 1)[1]
            num_correct_train += (predicts == yList).float().sum()

            print('\r %d ...' % num_train, end='')

        train_acc.append(num_correct_train / num_train)
        train_loss.append(train_loss_func.data)
        print("\r    - train_acc %.5f train_loss %.5f" % (train_acc[-1], train_loss[-1]))

        # evaluate on Test Data
        num_test = 0
        num_correct_test = 0
        model.eval()
        for (xList, yList) in test_loader:
            if torch.cuda.is_available():
                xList = xList.type(torch.cuda.FloatTensor)
                yList = yList.type(torch.cuda.LongTensor)
                device = torch.device(DEVICE)
                model.to(device)
            
            #  get outputs
            with torch.set_grad_enabled(False):
                outputs = model(xList)
                test_loss_func = loss(outputs, yList)
            
            # get correct predictions count
            num_test += len(yList)
            predicts = torch.max(outputs.data, 1)[1]
            num_correct_test += (predicts == yList).float().sum()

        test_acc.append(num_correct_test / num_test)
        test_loss.append(test_loss_func.data)
        print("\r    - test_acc  %.5f test_loss  %.5f" %(test_acc[-1], test_loss[-1]))
    return train_loss, train_acc, test_loss, test_acc


# Train model with queried logits

def attacker_training_w_logits(victim_model, attacker_model, sampledtrainloader, querytestloader, k):
    # initialize score lists
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    train_loss_func = None
    
    # select optimiser and scheduler
    optimizer = torch.optim.Adam(attacker_model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["base_lr"], max_lr=config["max_lr"],
                                                  step_size_up=config["lr_steps"], mode='triangular2', cycle_momentum=False)

    # select loss
    loss = torch.nn.CrossEntropyLoss()
       
    # train for epochs
    epochs = int(2*config["lr_cycles"]*config["lr_steps"]/(len(sampledtrainloader.dataset)/config["batch_size"]))
    print(f'Total epochs: {epochs}')
    for epoch in range(epochs):
        print("\repoch", epoch + 1)
        
        # train on Train Data
        num_train = 0
        num_correct_train = 0
        attacker_model.train() 
        for (xList, _) in sampledtrainloader:
            xList = torch.autograd.Variable(xList)
            optimizer.zero_grad()

            if torch.cuda.is_available():
                xList = xList.type(torch.cuda.FloatTensor)
                device = torch.device(DEVICE)
                attacker_model.to(device)
            
            # query victim for logits --- K_LOGITS_LOGIC
            yList = victim_model(xList)
            val, ind = torch.topk(yList, k, dim=1)
            ones = (torch.ones(yList.shape)*float('-inf'))
            if torch.cuda.is_available():
                ones = ones.type(torch.cuda.FloatTensor)
            yList = ones.scatter_(1, ind, val)
            yList = torch.nn.functional.softmax(yList, dim=1)
            
            # perform data augmentation on inputs  --- K_LOGITS_LOGIC
            xList = transform_data_augment(xList)
            
            # get outputs and train model
            with torch.set_grad_enabled(True):
                outputs = attacker_model(xList)
                train_loss_func = loss(outputs, yList)
                train_loss_func.backward()
                optimizer.step()
                scheduler.step()

            # change k-logits to labels for accuracy calucaltion --- K_LOGITS_LOGIC
            yList = torch.max(yList.data, 1)[1]
                
            # get correct predictions count
            num_train += len(yList)
            predicts = torch.max(outputs.data, 1)[1]
            num_correct_train += (predicts == yList).float().sum()

            print('\r %d ...' % num_train, end='')

        train_acc.append(num_correct_train / num_train)
        train_loss.append(train_loss_func.data)
        print("\r    - train_acc %.5f train_loss %.5f" % (train_acc[-1], train_loss[-1]))

        # evaluate on Test Data
        num_test = 0
        num_correct_test = 0
        attacker_model.eval()
        for (xList, yList) in queriedtestloader:
            if torch.cuda.is_available():
                xList = xList.type(torch.cuda.FloatTensor)
                yList = yList.type(torch.cuda.LongTensor)
                device = torch.device(DEVICE)
                attacker_model.to(device)
            
            #  get outputs
            with torch.set_grad_enabled(False):
                outputs = attacker_model(xList)
                test_loss_func = loss(outputs, yList)
            
            # get correct predictions count
            num_test += len(yList)
            predicts = torch.max(outputs.data, 1)[1]
            num_correct_test += (predicts == yList).float().sum()

        test_acc.append(num_correct_test / num_test)
        test_loss.append(test_loss_func.data)
        print("\r    - test_acc  %.5f test_loss  %.5f" %(test_acc[-1], test_loss[-1]))
    return train_loss, train_acc, test_loss, test_acc


# Save and Visualize all results

# visualization
def save_visualize(model, result, title):
    (train_loss, train_acc, test_loss, test_acc) = result

    # save everything
    torch.save(model.state_dict(), f'results/{title}')
    torch.save(result, f'results/{title}_result')

    # plot
    plt.plot(train_loss, 'b-', label='Train_loss')
    plt.plot(test_loss, 'g-', label='Test_loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/{title}_Loss')
    plt.show()
    plt.plot(train_acc, 'b-', label='Train_accuracy')
    plt.plot(test_acc, 'g-', label='Test_accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'results/{title}_Accuracy')
    plt.show()

    # return in percentages
    train_l = np.round_(train_loss[len(train_loss) - 1].cpu().numpy(), 4)
    train_a = np.round_(train_acc[len(train_acc) - 1].cpu().numpy() * 100, 2)
    test_l = np.round_(test_loss[len(test_loss) - 1].cpu().numpy(), 4)
    test_a = np.round_(test_acc[len(test_acc) - 1].cpu().numpy() * 100, 2)

    return train_l, train_a, test_l, test_a

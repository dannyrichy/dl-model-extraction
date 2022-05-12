import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# set the dataset
name = "CIFAR10"
# name = "CIFAR100"
# name = "GTSRB"

# relevant constants
n_batch = 64
n_epoch = 20

if name=="CIFAR10":
    n_classes = 10
elif name=="CIFAR100":
    n_classes = 100
elif name=="GTSRB":
    n_classes = 43
    
# load data
if name=="CIFAR10":
    dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', 
                                           train=True, 
                                           download=True, 
                                           transform=transforms.ToTensor())
    
    test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', 
                                           train=False, 
                                           download=True, 
                                           transform=transforms.ToTensor())

if name=="CIFAR100":
    dataset = torchvision.datasets.CIFAR100(root='./CIFAR100', 
                                           train=True, 
                                           download = True, 
                                           transform=transforms.ToTensor())
    
    test_dataset = torchvision.datasets.CIFAR100(root='./CIFAR100', 
                                           train=False, 
                                           download=True, 
                                           transform=transforms.ToTensor())
    
if name=="GTSRB":
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Resize([32,32])])
    
    dataset = torchvision.datasets.GTSRB(root='./GTSRB', 
                                        split='train', 
                                        download = True, 
                                        transform=transform)
    test_dataset = torchvision.datasets.GTSRB(root='./GTSRB', 
                                        split='test', 
                                        download = True, 
                                        transform=transform)

train_dataloader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=n_batch, 
                                               num_workers=2)

test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=n_batch, 
                                               num_workers=2)

# convnet to obtain coreset
class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.Flatten(), 
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes))
        
    def forward(self, x):
        return self.network(x)
    
# create model
model = ConvNet(n_classes)
model = model.to(device)

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()
cross_entropy = nn.CrossEntropyLoss(reduction='none')

# Set optimizer with optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  

# training
entropy_ind_epoch   = {}
cross_entropy_ind_epoch = {}

for epoch in range(n_epoch):
    entropy_list = []
    cross_entropy_list = []
    
    total        = 0
    correct      = 0
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        entropy     = torch.distributions.Categorical(logits=outputs).entropy()
        c_entropy   = cross_entropy(outputs, labels)
        entropy_list.append(entropy)
        cross_entropy_list.append(c_entropy)
        
        _, predicted = torch.max(outputs.data, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    entropy_list = torch.cat(entropy_list)
    cross_entropy_list = torch.cat(cross_entropy_list)
    
    values, ind  = torch.sort(entropy_list, descending=True)
    entropy_ind_epoch[epoch] = ind
    values, ind  = torch.sort(cross_entropy_list, descending=True)
    cross_entropy_ind_epoch[epoch] = ind
        
    print(f'Epoch [{epoch+1}/{n_epoch}], Loss: {loss.item()}, Train acc: {100*correct/total}')

# save model
torch.save(model.state_dict(), 'Results/cifar10_convnet_model')

# save the indices
import pickle

f = open('cifar10_entropy_index_9', 'wb')
pickle.dump(np.array(entropy_ind_epoch[9].cpu()), f)
f.close()

f = open('cifar10_entropy_index_19', 'wb')
pickle.dump(np.array(entropy_ind_epoch[19].cpu()), f)
f.close()

f = open('cifar10_cross_entropy_index_9', 'wb')
pickle.dump(np.array(cross_entropy_ind_epoch[9].cpu()), f)
f.close()

f = open('cifar10_cross_entropy_index_19', 'wb')
pickle.dump(np.array(cross_entropy_ind_epoch[19].cpu()), f)
f.close()
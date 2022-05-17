import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

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
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if name=="CIFAR10":
    dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', 
                                           train=True, 
                                           download=True, 
                                           transform=transform)
    
    test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', 
                                           train=False, 
                                           download=True, 
                                           transform=transform)

if name=="CIFAR100":
    dataset = torchvision.datasets.CIFAR100(root='./CIFAR100', 
                                           train=True, 
                                           download = True, 
                                           transform=transform)
    
    test_dataset = torchvision.datasets.CIFAR100(root='./CIFAR100', 
                                           train=False, 
                                           download=True, 
                                           transform=transform)
    
if name=="GTSRB":
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Resize([32,32]), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
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

# load model
model    = models.resnet18()
model.fc = nn.Linear(512, n_classes)

model = model.to(device)

# Set Loss function with criterion
criterion     = nn.CrossEntropyLoss()
cross_entropy = nn.CrossEntropyLoss(reduction='none')

# Set optimizer with optimizer
optimizer     = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# training
ent_idx_dict  = {}
cent_idx_dict = {}
for k in range(n_classes):
    ent_idx_dict[k]  = []
    cent_idx_dict[k] = []

entropy_list  = []
centropy_list = []
labels_list   = []

for epoch in range(n_epoch):    
    total        = 0
    correct      = 0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if epoch == n_epoch-1:
            entropy  = torch.distributions.Categorical(logits=outputs).entropy()
            centropy = cross_entropy(outputs, labels)
            
            entropy_list.append(entropy)
            centropy_list.append(centropy)
            labels_list.append(labels)
    
    if epoch == n_epoch-1:
        entropy_list  = torch.cat(entropy_list)
        centropy_list = torch.cat(centropy_list)
        labels_list   = torch.cat(labels_list)
    
        values, ind  = torch.sort(entropy_list, descending=True)
        for i in ind:
            ent_idx_dict[labels_list[i].item()].append(i.item())
        values, ind  = torch.sort(centropy_list, descending=True)
        for i in ind:
            cent_idx_dict[labels_list[i].item()].append(i.item())
        
    print(f'Epoch [{epoch+1}/{n_epoch}], Loss: {loss.item()}, Train acc: {100*correct/total}')

# save model
torch.save(model.state_dict(), 'Results/cifar10_resnet18')

# save the indices
import pickle

f = open('cifar100_entropy_dict', 'wb')
pickle.dump(ent_idx_dict, f)
f.close()

f = open('cifar100_cross_entropy_dict', 'wb')
pickle.dump(cent_idx_dict, f)
f.close()
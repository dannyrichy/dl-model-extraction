import torch
import numpy as np
from torch import nn
from torchsummary import summary
import matplotlib.pyplot as plt
from attacker.config import *

# # ResBlock:
# conv2 without maxpooling 
# conv2_1, conv2_2: (64,64,F)
# conv3_1: (64,128,T)

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, downsample):
    super().__init__()
    if downsample:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=2), 
          nn.BatchNorm2d(out_channels))
    else:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
      self.shortcut = nn.Sequential()

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)
  
  def forward(self, input):
    shortcut = self.shortcut(input)
    input = nn.ReLU()(self.bn1(self.conv1(input)))
    input = nn.ReLU()(self.bn1(self.conv2(input)))
    input = input + shortcut
    return nn.ReLU()(input)

# # ResNet 34
#
#

class ResNet34(nn.Module):
  def __init__(self, in_channels, ResBlock, outputs=1000):
    super().__init__()

    self.L0 = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    
    self.L1 = nn.Sequential(
        ResBlock(64, 64, downsample=False),
        ResBlock(64, 64, downsample=False),
        ResBlock(64, 64, downsample=False))
    
    self.L2 = nn.Sequential(
        ResBlock(64, 128, downsample=True),
        ResBlock(128, 128, downsample=False),
        ResBlock(128, 128, downsample=False),
        ResBlock(128, 128, downsample=False))
    
    self.L3 = nn.Sequential(
        ResBlock(128, 256, downsample=True),
        ResBlock(256, 256, downsample=False),
        ResBlock(256, 256, downsample=False),
        ResBlock(256, 256, downsample=False),
        ResBlock(256, 256, downsample=False),
        ResBlock(256, 256, downsample=False))
    
    self.L4 = nn.Sequential(
        ResBlock(256, 512, downsample=True),
        ResBlock(512, 512, downsample=False),
        ResBlock(512, 512, downsample=False))
    
    self.pool = torch.nn.AdaptiveAvgPool2d(1)
    self.fc = torch.nn.Linear(512, outputs)

  def forward(self,input):
    input = self.L0(input)
    input = self.L1(input)
    input = self.L2(input)
    input = self.L3(input)
    input = self.L4(input)
    input = self.pool(input)
    input = torch.flatten(input,1)
    input = self.fc(input)
    return input

# check_model = ResNet34(3, ResBlock, outputs=1000)
# check_model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(check_model, (3,224,224))

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


# visualization
def plots(result, model_name):
  (train_loss, train_acc, test_loss, test_acc) = result
  plt.plot(range(config["epochs"]), train_loss, 'b-', label='Train_loss')
  plt.plot(range(config["epochs"]), test_loss, 'g-', label='Test_loss')
  plt.title(f'{model_name} - Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig(f'{model_name}_Loss')
  plt.show()
  plt.plot(range(config["epochs"]), train_acc, 'b-', label='Train_accuracy')
  plt.plot(range(config["epochs"]), test_acc, 'g-', label='Test_accuracy')
  plt.title(f'{model_name} - Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig(f'{model_name}_Accuracy')
  plt.show()

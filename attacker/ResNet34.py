# +
import torch
from torch import nn


# REF: https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
# -

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


class ResNet34(nn.Module):
  def __init__(self, in_channels, outputs=1000):
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

import argparse
import os

import torch
import torch.optim as optim

from victim.gtsrb.data_loader import GTSRBData

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

use_gpu = torch.cuda.is_available()

# FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
# Tensor = FloatTensor

# Data Initialization and Loading

data = GTSRBData(download=True, data_dir=os.path.join(os.getcwd(), "data"), num_workers=1, batch_size=args.batch_size)
train_loader = data.train_dataloader()
val_loader = data.val_dataloader()

# Neural Network and Optimizer
from model import Net

model = Net()

if use_gpu:
    model.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

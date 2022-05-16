import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from attacker.config import *

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)

def LoadCoreset(filename):
    with open('attacker/coreset/'+filename, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


# visualization
def SaveVisualize(model, result, title):
    (train_loss, train_acc, test_loss, test_acc) = result
    # save everything
    torch.save(model.state_dict(), f'attacker/results/{title}')
    torch.save(train_loss,f'attacker/results/{title}_TrainLoss')
    torch.save(train_acc,f'attacker/results/{title}_TrainAcc')
    torch.save(test_loss,f'attacker/results/{title}_TestLoss' )
    torch.save(test_acc,f'attacker/results/{title}_TestAcc' )
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

    

from utils import DEVICE
from attacker.utils import transform_normalize, transform_victim_C10, set_seed, get_model, get_dataset
from attacker.config import *
from victim.interface import  fetch_victim_model

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# REF: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html


def adverserial_transfer(attacker, victim, testloader, epsilon):
    # initialize results
    result_attacker_unperturbed = []
    result_attacker_perturbed = []
    label_attacker_perturbed = []
    
    result_victim_unperturbed = []
    result_victim_perturbed = []
    label_victim_perturbed = []
    
    # test loader to have batch size 1 and unnormalized tensors
    for image, label in tqdm(testloader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        
        # enable gradient computation for input
        image.requires_grad = True
        
        # query attacker
        output = attacker(transform_normalize(image))
        prediction = output.max(1, keepdim=True)[1]
        # store
        result_attacker_unperturbed.append(prediction.item()==label.item())
        
        # get perturbed image using fgsm
        loss = torch.nn.functional.cross_entropy(output, label)
        attacker.zero_grad()
        loss.backward()
        image_grad = image.grad.data
        
        sign_grad = image_grad.sign()
        perturbed_image = image + epsilon*sign_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        # query perturbed image
        output = attacker(transform_normalize(perturbed_image))
        prediction = output.max(1, keepdim=True)[1]
        
        # store
        result_attacker_perturbed.append(prediction.item()==label.item())
        label_attacker_perturbed.append(prediction.item())
        
        # query unperturbed image on victim
        output = victim(transform_victim_C10(image))
        prediction = output.max(1, keepdim=True)[1]
        
        # store
        result_victim_unperturbed.append(prediction.item()==label.item())
        
        # query perturbed image on victim
        output = victim(transform_victim_C10(perturbed_image))
        prediction = output.max(1, keepdim=True)[1]
        
        result_victim_perturbed.append(prediction.item()==label.item())
        label_victim_perturbed.append(prediction.item())
    
    # save results
    results = {}
    results["result_attacker_unperturbed"] = result_attacker_unperturbed
    results["result_attacker_perturbed"] = result_attacker_perturbed
    results["label_attacker_perturbed"] = label_attacker_perturbed
    results["result_victim_unperturbed"] = result_victim_unperturbed
    results["result_victim_perturbed"] = result_victim_perturbed
    results["label_victim_perturbed"] = label_victim_perturbed
    return results


def analyse_results(results):
    total = len(results["result_attacker_unperturbed"])
    print(f'total images: {total}')
    print(f'images correctly classified by attacker: {np.sum(results["result_attacker_unperturbed"])}')
    
    adverserial_for_attacker = []
    for i in range(total):
        if (results["result_attacker_unperturbed"][i]==True) and (results["result_attacker_perturbed"][i]==False):
            adverserial_for_attacker.append(i)
    print(f'images adverserial for attacker: {len(adverserial_for_attacker)}')
    
    adverserial_for_victim = []
    for i in adverserial_for_attacker:
        if (results["result_victim_perturbed"][i] == False):
            adverserial_for_victim.append(i)
    print(f'images also adverserial for victim: {len(adverserial_for_victim)}')
    
    adversery_generalised = []
    for i in adverserial_for_victim:
        if (results["label_attacker_perturbed"][i] == results["label_victim_perturbed"][i]):
            adversery_generalised.append(i)
    print(f'images with same adverserial label: {len(adversery_generalised)}')
    
    return adversery_generalised


# REWORK: TRANSFORMS NOT CONSIDERED, DONT USE UNTIL FIXED
def visualize_adversary(image, label, model, epsilon):
    image = image.to(DEVICE)
    label = label.to(DEVICE)

    # for adversial example generation
    image.requires_grad = True

    # query attacker
    output = model(image)

    # get perturbed image
    loss = torch.nn.functional.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    image_grad = image.grad.data

    sign_grad = image_grad.sign()
    perturbed_image = image + epsilon*sign_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    print('noise:')
    plt.imshow((torch.moveaxis((img-perturbed_img).squeeze(), 0, 2).cpu().detach().numpy())*(0.5/epsilon) + 0.5)
    plt.show()
    print('unperturbed:')
    plt.imshow(torch.moveaxis(img.squeeze(), 0, 2).cpu().detach().numpy())
    plt.show()
    print('perturbed:')
    plt.imshow(torch.moveaxis(perturbed_img.squeeze(), 0, 2).cpu().detach().numpy())
    plt.show()


def adverserial_experiment(parameters, epsilon, seed=None):
    
    # set seed before anything else:
    set_seed(seed)

    # get parameters from input
    victim_type = parameters["victim"][0]
    attacker_type = parameters["attacker"][0]
    querytype = parameters["query_type"][0]
    size = parameters["query_size"][0]
    k = parameters["k_logits"][0]
    print('-----------------------------------------------------------------------------')
    print(f'\tDataset: {victim_type["data"]}')
    print(f'\tVictim: {victim_type["model_name"]}\tAttacker: {attacker_type}')
    print(f'\tQuery Type: {querytype}\tQuery Size: {size}\tLogits: {k}')
    print('-----------------------------------------------------------------------------')
    
    # get dataset in dataloaders
    _, testset, outputs = get_dataset(victim_type["data"])
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    # get trained attacker
    attacker_path = f'results/A_{attacker_type}_{victim_type["model_name"]}_{victim_type["data"]}_{querytype}_{size}_k{k}'
    attacker = get_model(attacker_type, outputs)
    attacker.load_state_dict(torch.load(attacker_path))
    attacker.to(DEVICE)
    attacker.eval()
    
    # get victim model
    victim = fetch_victim_model(args=victim_type)
    
    # check if adverserial transfers from victim to attacker    
    results = adverserial_transfer(attacker, victim, testloader, epsilon)
    
    # display results
    adversery_generalised = analyse_results(results)
    
    
    # get adversial example
    #image, label = testset[adversery_generalised[0]]
    
    #visualize_adversary(image, label, model, epsilon)
    
    return adversery_generalised, results


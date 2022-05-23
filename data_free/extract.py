import logging
import random

import numpy as np
import torch
from torch import optim
from torch.nn.functional import log_softmax, l1_loss, kl_div, softmax, cross_entropy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attacker.utils import get_model
from data_free.generator import GeneratorA
from data_free.utils import NUM_CLASSES, load_params
from utils import DEVICE
from victim import RESNET34

logging.basicConfig(level=logging.DEBUG)


class DataFreeModelExtraction:
    def __init__(self, models,
                 epoch_itrs,
                 num_gen_iter,
                 num_att_iter,
                 batch_size,
                 nz,
                 loss_type,
                 vic_logit_correction,
                 approx_grad,
                 grad_epsilon,
                 grad_m,
                 forward_differences,
                 no_logits,
                 query_budget, cost_per_iteration):
        self.victim_model, self.attacker_model, self.generator_model = models
        self.epoch_itrs = epoch_itrs
        self.num_gen_iter = num_gen_iter
        self.num_att_iter = num_att_iter
        self.batch_size = batch_size
        self.nz = nz
        self.loss_type = loss_type
        self.vic_logit_correction = vic_logit_correction
        self.approx_grad = approx_grad
        self.grad_m = grad_m
        self.grad_epsilon = grad_epsilon
        self.forward_differences = forward_differences
        self.no_logits = no_logits
        self.query_budget = query_budget
        self.cost_per_iteration = cost_per_iteration

    def _attacker_loss_calc(self, attacker_logits, vic_logits):
        """Kl/ L1 Loss for student"""
        if self.loss_type == "l1":
            loss_fn = l1_loss
            return loss_fn(attacker_logits, vic_logits.detach())
        elif self.loss_type == "kl":
            loss_fn = kl_div
            attacker_logits = log_softmax(attacker_logits, dim=1)
            vic_logits = softmax(vic_logits, dim=1)
            return loss_fn(attacker_logits, vic_logits.detach(), reduction="batchmean")
        else:
            raise ValueError(self.loss_type)

    def _train_generator(self, optimiser):
        for _ in range(self.num_gen_iter):
            # Sample Random Noise
            z = torch.randn((self.batch_size, self.nz)).to(DEVICE)
            optimiser.zero_grad()
            self.generator_model.train()
            # Get fake image from generator
            fake = self.generator_model(z, pre_x=self.approx_grad)
            print(torch.min(fake), torch.max(fake))
            approx_grad_wrt_x, gradient_loss = self.estimate_gradient_objective(fake,
                                                                                pre_x=True)

            fake.backward(approx_grad_wrt_x)
            optimiser.step()

    def _train_attacker(self, optimiser):
        for _ in range(self.num_att_iter):
            z = torch.randn((self.batch_size, self.nz)).to(DEVICE)
            fake = self.generator_model(z).detach()
            optimiser.zero_grad()

            with torch.no_grad():
                vic_logits = self.victim_model(fake)

            if self.loss_type == "l1" and self.no_logits:
                vic_logits = log_softmax(vic_logits, dim=1).detach()
                if self.vic_logit_correction == 'min':
                    vic_logits -= vic_logits.min(dim=1).values.view(-1, 1).detach()
                elif self.vic_logit_correction == 'mean':
                    vic_logits -= vic_logits.mean(dim=1).view(-1, 1).detach()

            attacker_logits = self.attacker_model(fake)

            attacker_loss = self._attacker_loss_calc(attacker_logits, vic_logits)
            attacker_loss.backward()
            optimiser.step()

    def train(self, optimizer):
        """
        Trains the entire architecture

        :return:
        :rtype:
        """
        logging.info("Training started")
        self.victim_model.eval()
        self.attacker_model.train()
        attacker_optimiser, generator_optimiser = optimizer
        start_ix = 0
        while start_ix < self.epoch_itrs and self.query_budget >= self.cost_per_iteration:
            logging.info(f"Iteration: {start_ix} Generator training")
            self._train_generator(generator_optimiser)
            logging.info(f"Iteration: {start_ix} Attacker training")
            self._train_attacker(attacker_optimiser)
            start_ix += 1
            self.query_budget -= self.cost_per_iteration

    def test(self, test_loader=None):
        self.attacker_model.eval()
        self.generator_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.attacker_model(data)

                test_loss += cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            accuracy))
        acc = correct / len(test_loader.dataset)
        return acc

    def estimate_gradient_objective(self, x,
                                    generator_activation=torch.tanh,
                                    pre_x=False):
        # Sampling from unit sphere is the method 3 from this website:
        #  http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        # x = torch.Tensor(np.arange(2*1*7*7).reshape(-1, 1, 7, 7))

        if pre_x and generator_activation is None:
            raise ValueError(generator_activation)

        self.attacker_model.eval()
        self.victim_model.eval()

        with torch.no_grad():
            # Sample unit noise vector
            N = x.size(0)
            C = x.size(1)
            S = x.size(2)
            dim = S ** 2 * C

            u = np.random.randn(N * self.grad_m * dim).reshape(-1, self.grad_m, dim)  # generate random points from normal distribution

            d = np.sqrt(np.sum(u ** 2, axis=2)).reshape(-1, self.grad_m, 1)  # map to a uniform distribution on a unit sphere
            u = torch.Tensor(u / d).view(-1, self.grad_m, C, S, S)
            u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim=1)  # Shape N, m + 1, S^2

            u = u.view(-1, self.grad_m + 1, C, S, S)

            evaluation_points = (x.view(-1, 1, C, S, S).cpu() + self.grad_epsilon * u).view(-1, C, S, S)
            if pre_x:
                evaluation_points = generator_activation(evaluation_points)  # Apply args.G_activation function

            # Compute the approximation sequentially to allow large values of m
            victim_prediction = []
            attacker_prediction = []
            max_number_points = 32 * 156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

            for i in (range(N * self.grad_m // max_number_points + 1)):
                pts = evaluation_points[i * max_number_points: (i + 1) * max_number_points]
                pts = pts.to(DEVICE)

                pred_victim_pts = self.victim_model(pts).detach()
                pred_clone_pts = self.attacker_model(pts)

                victim_prediction.append(pred_victim_pts)
                attacker_prediction.append(pred_clone_pts)

            victim_prediction = torch.cat(victim_prediction, dim=0).to(DEVICE)
            attacker_prediction = torch.cat(attacker_prediction, dim=0).to(DEVICE)

            u = u.to(DEVICE)

            if self.loss_type == "l1":
                loss_fn = l1_loss
                if self.no_logits:
                    victim_prediction = log_softmax(victim_prediction, dim=1).detach()
                    if self.vic_logit_correction == 'min':
                        victim_prediction -= victim_prediction.min(dim=1).values.view(-1, 1).detach()
                    elif self.vic_logit_correction == 'mean':
                        victim_prediction -= victim_prediction.mean(dim=1).view(-1, 1).detach()
                loss_values = - loss_fn(attacker_prediction, victim_prediction, reduction='none').mean(dim=1).view(-1, self.grad_m + 1)

            elif self.loss_type == "kl":
                loss_fn = kl_div
                attacker_prediction = log_softmax(attacker_prediction, dim=1)
                victim_prediction = softmax(victim_prediction.detach(), dim=1)
                loss_values = - loss_fn(attacker_prediction, victim_prediction, reduction='none').sum(dim=1).view(-1, self.grad_m + 1)

            else:
                raise ValueError(self.loss_type)

            # Compute difference following each direction
            differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
            differences = differences.view(-1, self.grad_m, 1, 1, 1)

            # Formula for Forward Finite Differences
            gradient_estimates = 1.0 / self.grad_epsilon * differences * u[:, :-1]
            if self.forward_differences:
                gradient_estimates *= dim

            if self.loss_type == "kl":
                gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, S, S)
            else:
                gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, S, S) / (NUM_CLASSES * N)

            self.attacker_model.train()
            generator_loss = loss_values[:, -1].mean()
            return gradient_estimates.detach(), generator_loss


def run_dfme(victim_model,
             seed=random.randint(0, 100000),
             attacker_model_name=RESNET34):
    """

    :param victim_model:
    :type victim_model: torch.nn.Module

    :param seed: Random seed
    :type seed: int

    :param attacker_model_name:
    :type attacker_model_name: str

    :return:
    :rtype:
    """
    # Setting reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyper-params setting
    params = load_params()

    # Model declaration
    victim_model.eval()
    victim_model.to(DEVICE)

    attacker_model = get_model(model_name=attacker_model_name, outputs=10)
    attacker_model.to(DEVICE)

    generator_model = GeneratorA(nz=params['nz'], nc=3, img_size=32, activation=torch.tanh)
    generator_model.to(DEVICE)

    cost_per_iteration = params['batch_size'] * (
            params['num_gen_iter'] * (params['grad_m'] + 1) + params['num_att_iter']
    )

    number_epochs = params['query_budget'] // (cost_per_iteration * params['epoch_itrs']) + 1

    att_opti = optim.SGD(attacker_model.parameters(), lr=params['lr_att'],
                         weight_decay=params['lr_weight_decay'], momentum=0.9)
    gen_opti = optim.Adam(generator_model.parameters(), lr=params['lr_gen'])

    steps = sorted([int(step * number_epochs) for step in params['steps']])
    if params['scheduler'] == "multistep":
        att_scheduler = optim.lr_scheduler.MultiStepLR(att_opti, steps, params['scale'])
        gen_scheduler = optim.lr_scheduler.MultiStepLR(gen_opti, steps, params['scale'])
    # elif params['scheduler'] == "cosine":
    else:
        att_scheduler = optim.lr_scheduler.CosineAnnealingLR(att_opti, number_epochs)
        gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(gen_opti, number_epochs)

    dfme_instance = DataFreeModelExtraction(
        models=(victim_model, attacker_model, generator_model),
        epoch_itrs=params['epoch_itrs'],
        num_gen_iter=params['num_gen_iter'],
        num_att_iter=params['num_att_iter'],
        batch_size=params['batch_size'],
        nz=params['nz'],
        loss_type=params['loss_type'],
        vic_logit_correction=params['vic_logit_correction'],
        grad_epsilon=params['grad_epsilon'],
        grad_m=params['grad_m'],
        forward_differences=params['forward_differences'],
        approx_grad=params['approx_grad'],
        no_logits=params['no_logits'],
        query_budget=params['query_budget'],
        cost_per_iteration=cost_per_iteration
    )

    test_loader = DataLoader(
        datasets.CIFAR10(".", train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                         ])),
        batch_size=params['batch_size'], shuffle=True, num_workers=2)

    best_acc = 0
    acc_list = []
    for epoch in range(1, number_epochs + 1):
        logging.info(f"Master epoch: {epoch} starting {dfme_instance.query_budget}, {dfme_instance.cost_per_iteration}")
        if params['scheduler'] != "none":
            att_scheduler.step()
            gen_scheduler.step()

        dfme_instance.train(optimizer=[att_opti, gen_opti])
        # Test
        acc = dfme_instance.test(test_loader=test_loader)
        acc_list.append(acc)
        if acc > best_acc:
            best_acc = acc
            name = victim_model.__class__.__name__
            torch.save(dfme_instance.attacker_model.state_dict(), f"checkpoint/attacker_{attacker_model_name}/cifar10-{name}.pt")
            torch.save(dfme_instance.generator_model.state_dict(), f"checkpoint/attacker_{attacker_model_name}/cifar10-{name}-generator.pt")

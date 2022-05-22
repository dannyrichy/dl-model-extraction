# Repurposing code from https://github.com/cake-lab/datafree-model-extraction

import numpy as np
from torch.nn.functional import kl_div, log_softmax, softmax, l1_loss

NUM_CLASSES = 10


def compute_grad_norms(generator_model, victim_model):
    G_grad = [
        p.grad.norm().to("cpu")
        for n, p in generator_model.named_parameters()
        if "weight" in n
    ]

    S_grad = [
        p.grad.norm().to("cpu")
        for n, p in victim_model.named_parameters()
        if "weight" in n
    ]

    return np.mean(G_grad), np.mean(S_grad)


def compute_gradient(args, victim_model, attacker_model, x, pre_x=False, device="cpu"):
    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    attacker_model.eval()
    N = x.size(0)
    x_copy = x.clone().detach().requires_grad_(True)
    x_ = x_copy.to(device)

    if pre_x:
        x_ = args.G_activation(x_)

    victim_pred = victim_model(x_)
    attacker_pred = attacker_model(x_)

    if args.loss == "l1":
        loss_fn = l1_loss
        if args.no_logits:
            pred_victim_no_logits = log_softmax(victim_pred, dim=1)
            if args.logit_correction == 'min':
                victim_pred = pred_victim_no_logits - pred_victim_no_logits.min(dim=1).values.view(-1, 1)
            elif args.logit_correction == 'mean':
                victim_pred = pred_victim_no_logits - pred_victim_no_logits.mean(dim=1).view(-1, 1)
            else:
                victim_pred = pred_victim_no_logits

    elif args.loss == "kl":
        loss_fn = kl_div
        attacker_pred = log_softmax(attacker_pred, dim=1)
        victim_pred = softmax(victim_pred, dim=1)

    else:
        raise ValueError(args.loss)

    loss_values = -loss_fn(attacker_pred, victim_pred, reduction='mean')
    loss_values.backward()
    attacker_model.train()

    return x_copy.grad, loss_values


def measure_true_grad_norm(args, x):
    # Compute true gradient of loss wrt x
    true_grad, _ = compute_gradient(args, args.teacher, args.student, x, pre_x=True, device=args.device)
    true_grad = true_grad.view(-1, 3072)

    # Compute norm of gradients
    norm_grad = true_grad.norm(2, dim=1).mean().cpu()

    return norm_grad


def load_params():
    # with open(os.path.join(os.getcwd(), "dfm_config.json"), "rb") as f:
    #     params = json.load(f)
    params = {
        "batch_size": 256,
        "query_budget": 20000000,
        "epoch_itrs": 50,
        "num_gen_iter": 1,
        "num_att_iter": 5,
        "lr_att": 0.1,
        "lr_gen": 1e-4,
        "scheduler": "multistep",
        "steps": [0.1, 0.3, 0.5],
        "scale": 3e-1,
        "lr_weight_decay": 5e-4,
        "approx_grad": 1,
        "grad_m": 1,
        "grad_epsilon": 1e-3,
        "forward_differences": 1,
        "loss_type": "l1",
        "no_logits": 1,
        "vic_logit_correction": "mean",
        "nz": 256
    }
    return params

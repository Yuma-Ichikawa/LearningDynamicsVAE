import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from main import utils
from main import data
from main import vae


def dynamics_reproduce(x_dim, z_dim, num_seed=5, device="cpu", rho=1, eta=1, num_epoch=10000, lr=0.01, beta=1., reg_param=0, check_interval=20000):
    W_init =torch.randn(x_dim, z_dim, device=device)
    tW_init = torch.randn(z_dim, x_dim, device=device)
    W0=torch.ones(x_dim, z_dim, device=device)
    history_seed = {"Ms": [], "tMs": [], "Qs": [], "tQs":[], "Rs": [], "vs":[], "egs" :[]}
    for seed in range(num_seed):
        print(f"SEED={seed}")
        model = copy.deepcopy(vae.LinearVAE(input_dim=x_dim, latent_dim=z_dim, W_init=W_init, tW_init=tW_init).to(device))
        ob_state = utils.calc_observable(model, W0=W0, device=device)
        history_each, _ = vae.onepassfit_linearvae(W0, model, num_epoch=num_epoch, lr=lr, beta=beta, reg_param=reg_param, 
                                                check_interval=check_interval, device=device)

        history_seed["Ms"].append(history_each["M"])
        history_seed["tMs"].append(history_each["tM"])
        history_seed["Qs"].append(history_each["Q"])
        history_seed["tQs"].append(history_each["tQ"])
        history_seed["Rs"].append(history_each["R"])
        history_seed["vs"].append(history_each["v"])
        history_seed["egs"].append(history_each["eg"])

    return history_seed
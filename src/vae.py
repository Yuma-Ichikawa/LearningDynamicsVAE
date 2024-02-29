import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from main import utils
from main import data

class LinearVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, W_init=None, tW_init=None):
        super(LinearVAE, self).__init__()
        self.dec = nn.Linear(latent_dim, input_dim, bias=False)
        self.mu = nn.Linear(input_dim, latent_dim, bias=False)
        self.var = nn.Parameter(torch.ones(latent_dim))

        if W_init==None:
            self.dec.weight = nn.Parameter(torch.randn(input_dim, latent_dim))
            self.mu.weight = nn.Parameter(torch.randn(latent_dim, input_dim))
        else:
            self.dec.weight = nn.Parameter(W_init)
            self.mu.weight = nn.Parameter(tW_init)

        self.N = torch.tensor(input_dim)
        self.M = torch.tensor(latent_dim)

    def encode(self, x):
        mu = self.mu(x)/torch.sqrt(self.N)
        var = self.var
        return mu, var

    def decode(self, z):
        hat_x = self.dec(z)/torch.sqrt(self.N)
        return hat_x

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, var = self.encode(x.view(-1, self.N))
        hat_x = self.decode(mu)
        return hat_x, mu, var


def criterion(model, hat_x, x, mu, var, beta=1.0, reg_param=1.0):
    first_term = torch.sum(x**2, dim=1)
    second_term = -2*torch.sum(hat_x*x, dim=1)
    third_term = torch.diag(mu @ model.dec.weight.T @ model.dec.weight @ mu.T)/x.size(1)
    forth_term = torch.sum(torch.diag(model.dec.weight.T @ model.dec.weight)*var)/x.size(1)
    each_data_recon_loss = 0.5*(second_term+third_term+forth_term)
    recon_loss=each_data_recon_loss
    each_data_KLD  = 0.5*torch.sum((-torch.log(var+1e-16) + mu.pow(2) + var), dim=1)
    KLD=each_data_KLD
    reg_term_decoder = 0.5*reg_param*torch.sum(torch.diag(model.dec.weight.T @ model.dec.weight))/x.size(1)
    reg_term_encoder = 0.5*reg_param*torch.sum(torch.diag(model.mu.weight @ model.mu.weight.T))/x.size(1)
    return beta*KLD+recon_loss+reg_term_decoder+reg_term_encoder, recon_loss, KLD


def onepassfit_linearvae(W0, model, num_epoch=100, lr=0.001, beta=1.0, reg_param=1.0, check_interval=1000, device="cpu"):

    input_dim = model.dec.weight.size(0)
    latent_dim = model.dec.weight.size(1)

    history = {"elbo":[], "kl":[], "rate":[],
            "M": [], "tM": [], "Q": [], "tQ":[], "R": [], "v":[], "eg" :[]}

    xs = data.generate_data_form_SCM(N=input_dim, M=1, W0=W0, eta=1, rho=1, device=device)
    recon, mu, var = model(xs)
    loss, recon_loss, kl = criterion(model, recon, xs, mu, var, beta=beta, reg_param=reg_param)
    ob_state = utils.calc_observable(model, W0=W0, device=device)
    history["elbo"].append(loss.item())
    history["kl"].append(kl.item())
    history["rate"].append(recon_loss.item())
    history["M"].append(ob_state[0])
    history["tM"].append(ob_state[1])
    history["Q"].append(ob_state[2])
    history["tQ"].append(ob_state[3])
    history["R"].append(ob_state[4])
    history["v"].append(ob_state[5])
    history["eg"].append(ob_state[6])
    print("【INIT】 M=", ob_state[0], "tM=", ob_state[1],
        "Q=", ob_state[2], "tQ=", ob_state[3], "R=", ob_state[4])

    optimizer = torch.optim.SGD([
                {'params': model.dec.parameters(), 'lr': lr},
                {'params': model.mu.parameters(), 'lr': lr},
                {'params': model.var, 'lr': lr/input_dim}])

    model.train()
    for epoch in range(num_epoch):
        xs = data.generate_data_form_SCM(N=input_dim, M=1, W0=W0, eta=1, rho=1, device=device)
        recon, mu, var = model(xs)
        loss, recon_loss, kl = criterion(model, recon, xs, mu, var, beta=beta, reg_param=reg_param)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ob_state = utils.calc_observable(model, W0=W0, device=device)

        history["elbo"].append(loss.item())
        history["kl"].append(kl.item())
        history["rate"].append(recon_loss.item())
        history["M"].append(ob_state[0])
        history["tM"].append(ob_state[1])
        history["Q"].append(ob_state[2])
        history["tQ"].append(ob_state[3])
        history["R"].append(ob_state[4])
        history["v"].append(ob_state[5])
        history["eg"].append(ob_state[6])

        if epoch%check_interval==0:
            print(f'Time: {epoch/input_dim:.1f}, eg: {ob_state[6]:.4f}, (elbo, recon, kl)=({loss.item():.4f}, {recon_loss.item():.4f}, {kl.item():.4f})')

    return history, model
import torch

def generate_data_form_SCM(N, M, W0=None, eta=0.5, rho=1.0, device="cpu"):

    if W0==None:
        W0 = torch.ones((N, M)).to(device)
    c = torch.randn(M, 1, device=device)
    n = torch.randn(N, 1, device=device)

    X = (W0@c)/(torch.sqrt(torch.tensor(N/rho))) + torch.sqrt(torch.tensor(eta))*n
    return X.T

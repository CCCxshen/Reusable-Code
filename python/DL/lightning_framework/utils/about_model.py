from torch.optim.lr_scheduler import *
from torch import optim

def obtain_optim(opt, parameters):
    if opt["about_lt_model"]["optimizer"] == "Adam":
        optimizer = optim.Adam(
            parameters, 
            lr = opt["about_lt_model"]["lr"], 
            betas = opt["about_lt_model"]["optimizer_betas"], 
            weight_decay = opt["about_lt_model"]["optimizer_weight_decay"]
        )
    return optimizer

def obtain_scheduler(opt, optimizer):
    if opt["about_lt_model"]["lr_scheduler"] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
    return scheduler
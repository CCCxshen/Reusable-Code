import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from networks.unet import *
from utils.utils import *

def create_model(config):
    models = []
    for model_idx, model_name in enumerate(config["model"]):
        if model_name == "UNet2D":
            model = UNet_2d(config["input_channels"], config["class_nums"])
        model = initialize_model(model)
        models.append(model.to(config["device"]))
    return models
        
def create_optimizer(config, models):
    optimizers = []
    for optimizer_idx, optimizer_name in enumerate(config["optimizer"]):
        if optimizer_name == "Adam":
            optimizer = Adam(
                params = models[config["optimizer_corresponding_model_parameters"][optimizer_idx]].parameters(),
                betas = (config["beta1"], config["beta2"]),
                weight_decay = config["weight_decay"],
                lr = config["learning_rate"][config["lr_corresponding_optimizer"][optimizer_idx]]
            )
        optimizers.append(optimizer)
        
    return optimizers

def create_scheduler(config, optimizers):
    schedulers = []
    for scheduler_idx, scheduler_name in enumerate(config["lr_scheduler"]):
        if scheduler_name == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer = optimizers[config["lr_scheduler_corresponding_optimizer"][scheduler_idx]],
                T_max = config["T_max"],
                eta_min = config["min_learning_rate"]
            )
        schedulers.append(scheduler)
        
    return schedulers

def initialize_model(model):
    
    for m in model.modules():
        # 处理2D和3D卷积层
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # 处理BN层
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        # 处理线性层
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    return model

def optimizers_zero_grad(optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()

def optimizers_step(optimizers):
    for optimizer in optimizers:
        optimizer.step()
    
def schedulers_step(schedulers):
    for scheduler in schedulers:
        scheduler.step()

def save_model(models, optimizers = None, epoch = 0, indicator = {}, config = None, save_name = None):
    models_state_dict = [model.state_dict() for model in models]
    if optimizers == None: optimizers_state_dict = None
    else: optimizers_state_dict = [optimizer.state_dict() for optimizer in optimizers]
    
    checkpoint = {
        "models": models_state_dict,
        "optimizers": optimizers_state_dict,
        "epoch": epoch,
        "indicator": indicator
    }
    if save_name == None: save_name = f"model_epoch{epoch}_{config['monitoring_indicators']}={indicator[config['monitoring_indicators']]:.4f}.pth"
    save_path = os.path.join(config["root_dir"], "log", config["method_name"], "checkpoint", save_name)
    
    torch.save(checkpoint, save_path)
    
    return f"The model has been saved to {save_path}"
    
def load_model(models = None, optimizers = None, config = None):
    if config["ckpt"] == None: load_path = os.path.join(config["root_dir"], "log", config["method_name"], "checkpoint", "best_model.pth")
    else: load_path = config["ckpt"]
    
    state_dict = torch.load(load_path)
    epoch = state_dict["epoch"]
    indicator = state_dict["indicator"]
    
    if models != None:
        for model_idx in range(len(models)):
            models[model_idx].load_state_dict(state_dict["models"][model_idx])
    if optimizers != None: 
        for optimizer_idx in range(len(optimizers)):
            optimizers[optimizer_idx].load_state_dict(state_dict["optimizers"][optimizer_idx])
    
    return models, optimizers, epoch, indicator, f"Load the model from {load_path}"
    
    
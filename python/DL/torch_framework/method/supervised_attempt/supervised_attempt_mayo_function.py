import torch
import time
import random
import numpy as np
import torch.nn.functional as F

from utils.IO_utils import *
from utils.utils import *
from utils.data_utils import *
from indicator.metrics import *
from utils.matplotlib_util import *
from method.supervised_attempt.dog import *

def prep_somethings(config):
    config["dog"] = F2N_edge(config["device"])
    return config

def train_dataset_step(data_path, is_train, config):
    noised_input = np.expand_dims(np.load(data_path[0]), axis = 0)
    target_data  = np.expand_dims(np.load(data_path[1]), axis = 0)
    
    mean_bosong = 300
    possion_noise = np.random.poisson(lam = mean_bosong, size = noised_input.shape).astype(np.float64)
    poisson_noised_input = noised_input + possion_noise
    
    noised_input = normalize_(noised_input, -1024, 1000)
    poisson_noised_input = normalize_(poisson_noised_input, -1024, 1000)
    target_data = normalize_(target_data, -1024, 1000)
    
    return {
        "a": torch.from_numpy(noised_input).float().to(config["device"]),
        "b": torch.from_numpy(poisson_noised_input).float().to(config["device"]),
        "c": torch.from_numpy(target_data).float().to(config["device"]),
    }

def val_dataset_step(data_path, is_train, config):
    noised_input = np.expand_dims(np.load(data_path[0]), axis = 0)
    target_data  = np.expand_dims(np.load(data_path[1]), axis = 0)
    
    noised_input = normalize_(noised_input, -1024, 1000)
    target_data = normalize_(target_data, -1024, 1000)
    
    return {
        "a": torch.from_numpy(noised_input).float().to(config["device"]),
        "b": torch.from_numpy(target_data).float().to(config["device"]),
    }

test_dataset_step = val_dataset_step

def train_step(models, optimizers, data, step, epoch_idx, config):
    
    model = models[0]
    model.train()
    optimizer = optimizers[0]
    
    noised_input = data['a'].to(config["device"])
    poisson_noised_input = data['b'].to(config["device"])
    target_data = data['c'].to(config["device"])
    
    pred = model(poisson_noised_input)
    # loss = F.l1_loss(pred, noised_input) + (((config["epochs"] - epoch_idx) / config["epochs"])**(1/2)) * config["dog"](noised_input - pred, poisson_noised_input)
    loss = F.l1_loss(pred, noised_input) + 1 / config["dog"](noised_input - pred, poisson_noised_input)
    
    loss.backward()
    optimizer.step()
    
    return {"train_loss": loss}

def val_step(models, data, step, epoch_idx, config):
    model = models[0]
    model.eval()
    
    with torch.no_grad():
        input_data = data['a'].to(config["device"])
        target_data = data['b'].to(config["device"])
        
        pred = model(input_data).detach()
        loss = F.l1_loss(pred, target_data)
        indicator_cache = config["indicator"].compute_measure(input_data, target_data, torch.clamp(pred, 0, 1), is_train = True)

        if step % 5 == 0:
            input_data = denormalize_(input_data[0][0].detach().cpu().numpy(), -1024, 1000)
            pred = denormalize_(pred[0][0].detach().cpu().numpy(), -1024, 1000)
            target_data = denormalize_(target_data[0][0].detach().cpu().numpy(), -1024, 1000)
            show_2Dimages(
                images = [
                    input_data, pred, target_data,
                    np.clip(input_data, -160, 240), np.clip(pred, -160, 240), input_data - pred, 
                ],
                names = [
                    "I[-1024, 1000]", "P[-1024, 1000]", "T[-1024, 1000]", "I[-160, 240]", "P[-160, 240]", "I - T [-1024, 1000]"
                ],
                title = f"step={step}, HU[-1024, 1000] \n I&T: {dict2str(indicator_cache[0])} \n P&T: {dict2str(indicator_cache[1])}",
                save_path = os.path.join(config["root_dir"], "log", config["method_name"], "val"),
                save_name = f"epoch={epoch_idx}_step={step}",
                shape = (2, 3),
            )
        
    return {
        "l1_loss": loss
    }, {
        "rmse": indicator_cache[1]["rmse"],
        "psnr": indicator_cache[1]["psnr"],
        "ssim": indicator_cache[1]["ssim"],
        }
    
    
def test_step(models, data, step, config):
    
    model = models[0]
    model.eval()
    
    with torch.no_grad():
        input_data = data['a'].to(config["device"])
        target_data = data['b'].to(config["device"])
        
        pred= model(input_data)
        # loss = F.l1_loss(pred, target_data).detach()
        indicator_cache = config["indicator"].compute_measure(input_data, target_data, torch.clamp(pred, 0, 1), is_train = False)

        input_data = denormalize_(input_data[0][0].detach().cpu().numpy(), -1024, 1000)
        pred = denormalize_(pred[0][0].detach().cpu().numpy(), -1024, 1000)
        target_data = denormalize_(target_data[0][0].detach().cpu().numpy(), -1024, 1000)
        show_2Dimages(
            images = [
                input_data, pred, target_data,
                np.clip(input_data, -160, 240), np.clip(pred, -160, 240), input_data - pred, 
            ],
            names = [
                "I[-1024, 1000]", "P[-1024, 1000]", "T[-1024, 1000]", "I[-160, 240]", "P[-160, 240]", "I - T [-1024, 1000]"
            ],
            title = f"step={step}, HU[-1024, 1000] \n I&T: {dict2str(indicator_cache[0])} \n P&T: {dict2str(indicator_cache[1])}",
            save_path = os.path.join(config["root_dir"], "log", config["method_name"], "test"),
            save_name=f"step-{step}-HU[-1024,1000]",
            shape = (2, 3),
        )
    
    return {
        "I&T-rmse": indicator_cache[0]["rmse"],
        "I&T-psnr": indicator_cache[0]["psnr"],
        "I&T-ssim": indicator_cache[0]["ssim"],
        "I&T-lpips":indicator_cache[0]["lpips"],
        "I&T-vif":  indicator_cache[0]["vif"],
        "I&T-nqm":  indicator_cache[0]["nqm"],
        
        "P&T-rmse": indicator_cache[1]["rmse"],
        "P&T-psnr": indicator_cache[1]["psnr"],
        "P&T-ssim": indicator_cache[1]["ssim"],
        "P&T-lpips":indicator_cache[1]["lpips"],
        "P&T-vif":  indicator_cache[1]["vif"],
        "P&T-nqm":  indicator_cache[1]["nqm"]
        }




   
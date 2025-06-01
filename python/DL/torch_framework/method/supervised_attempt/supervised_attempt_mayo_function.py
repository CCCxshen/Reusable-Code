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


def train_dataset_step(data_path, is_train, config):

    
    return {
        "a": torch.from_numpy(input_data).float().to(config["device"]),
        "b": torch.from_numpy(target_data).float().to(config["device"])
    }

val_dataset_step = train_dataset_step
test_dataset_step = train_dataset_step

def train_step(models, optimizers, data, step, epoch_idx, config):
    
    model = models[0]
    model.train()
    optimizer = optimizers[0]
    
    input_data = data['a'].to(config["device"])
    target_data = data['b'].to(config["device"])
    
    pred = model(input_data)
    loss = F.l1_loss(input_data - pred, input_data - target_data)
    
    loss.backward()
    optimizer.step()
    
    return {"l1_loss": loss}

def val_step(models, data, step, epoch_idx, config):
    model = models[0]
    model.eval()
    
    with torch.no_grad():
        input_data = data['a'].to(config["device"])
        target_data = data['b'].to(config["device"])
        
        pred = model(input_data)
        loss = F.l1_loss(input_data - pred, input_data - target_data)
        indicator_cache = MetricsCalculator(-1024, 1000, config["device"]).compute_measure(input_data, target_data, torch.clamp(pred, 0, 1), is_train = True)

        if step % 5 == 0:
            input_data = normalize_(input_data[0][0].detach().cpu().numpy(), -1024, 1000)
            pred = normalize_(pred[0][0].detach().cpu().numpy(), -1024, 1000)
            target_data = normalize_(target_data[0][0].detach().cpu().numpy(), -1024, 1000)
            show_2Dimages(
                images = [
                    input_data, pred, target_data,
                    input_data - pred, input_data - target_data
                ],
                names = [
                    "I", "P", "T", "I - P", "I - T"
                ],
                title = f"step={step}, HU[-1024, 1000] \n I&T: {dict2str(indicator_cache[0])} \n P&T: {dict2str(indicator_cache[1])}",
                save_path = os.path.join(config["root_dir"], "log", config["method_name"], "val"),
                save_name = f"epoch={epoch_idx}_step={step}_HU[-1024,1000]",
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
        loss = F.l1_loss(input_data - pred, input_data - target_data).detach()
        indicator_cache = MetricsCalculator(-1024, 1000, config["device"]).compute_measure(input_data, target_data, torch.clamp(pred, 0, 1), is_train = False)

        input_data = normalize_(input_data[0][0].detach().cpu().numpy(), -1024, 1000)
        pred = normalize_(pred[0][0].detach().cpu().numpy(), -1024, 1000)
        target_data = normalize_(target_data[0][0].detach().cpu().numpy(), -1024, 1000)
        show_2Dimages(
            images = [
                input_data, pred, target_data,
                input_data - pred, input_data - target_data
            ],
            names = [
                "I", "P", "T", "I - P", "I - T"
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




   
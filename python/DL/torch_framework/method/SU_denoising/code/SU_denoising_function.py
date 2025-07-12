import torch
import time
import random
import numpy as np
import torch.nn.functional as F

from utils.IO_utils import *
from utils.utils import *
from utils.mha_utils import *
from utils.data_utils import *
from indicator.metrics import *
from utils.matplotlib_util import *
from method.SRAD25_task1.code.dog import *


def add_possion_noise(x):
    scale = np.random.randint(1, 10)
    noise = np.random.poisson(x * scale, x.shape) / scale
    noise_x = np.clip(x + noise, 0, 1)
    return noise_x

def add_gaussion_noise(x):
    scale = np.random.randint(1, 10) * 0.05
    noise = np.random.normal(0, 1, x.shape) * scale
    noise_x = np.clip(x + noise, 0, 1)
    return noise_x

def random_add_noise(x):
    select = [np.random.randint(1, 3) for i in range(2)]
    
    # 添加符合噪声(先加泊松，再加高斯)
    if select[0] == 1:
        noise_x = add_gaussion_noise(add_possion_noise(x))
    # 添加泊松噪声
    elif select[1] == 1:
        noise_x = add_possion_noise(x)
    # 添加高斯噪声
    elif select[1] == 2:
        noise_x = add_gaussion_noise(x)
        
def prep_somethings(config):
    config["dog"] = F2N_edge_dog(config["device"])
    return config

def train_dataset_step(data_path, is_train, config):
    I  = normalize_(np.load(data_path[0]).astype(np.float32), config["min_val"], config["max_val"])
    # target_data = normalize_(np.load(data_path[1]).astype(np.float32), config["min_val"], config["max_val"])
    
    noise_I = random_add_noise(I)

    Z, R, C = I.shape

    if R > config["train_patch_size"]:
        row = np.random.randint(0, R - config["train_patch_size"])
        col = np.random.randint(0, C - config["train_patch_size"])
        I  = I[:, row : row + config["train_patch_size"], col : col + config["train_patch_size"]]
        noise_I  = noise_I[:, row : row + config["train_patch_size"], col : col + config["train_patch_size"]]
        # target_data = target_data[:, row : row + config["train_patch_size"], col : col + config["train_patch_size"]]

    return {
        "a": torch.from_numpy(noise_I).float().to(config["device"]),
        "b": torch.from_numpy(I).float().to(config["device"]),
        "c": '_'.join(data_path[0].split('/')[-1].split('_')[:2])
    }
    

def val_dataset_step(data_path, is_train, config):
    input_data  = normalize_(np.load(data_path[0]).astype(np.float32), config["min_val"], config["max_val"])
    target_data = normalize_(np.load(data_path[1]).astype(np.float32), config["min_val"], config["max_val"])
    
    return {
        "a": torch.from_numpy(input_data).float().to(config["device"]),
        "b": torch.from_numpy(target_data).float().to(config["device"]),
        "c": '_'.join(data_path[0].split('/')[-1].split('_')[:2])
    }

test_dataset_step = val_dataset_step


def train_step(models, optimizers, data, step, epoch_idx, config):
    
    model = models[0]
    model.train()
    optimizer = optimizers[0]
    
    noise_ldct  = data['a'].to(config["device"])
    ldct  = data['b'].to(config["device"])
    
    skips_noise_ldct, encoder_noise_ldct = model.forward_encoder(noise_ldct)
    skips_ldct , encoder_ldct       = model.forward_encoder(ldct)
    
    decoder_noise_ldct = model.forward_decoder([x.clone().detach() for x in skips_noise_ldct], encoder_noise_ldct.clone().detach())
    decoder_ldct       = model.forward_decoder([x.clone().detach() for x in skips_ldct], encoder_ldct.clone().detach())
    
    encoder_loss = F.l1_loss(encoder_noise_ldct, encoder_ldct) + config["dog"](encoder_noise_ldct, encoder_ldct)
    decoder_loss = F.l1_loss(decoder_noise_ldct, decoder_ldct) + config["dog"](decoder_noise_ldct, decoder_ldct)
    
    loss = encoder_loss + decoder_loss
    
    loss.backward()
    optimizer.step()
    
    return {"train_loss": loss.detach(), "encoder_loss": encoder_loss.detach(), "decoder_loss": decoder_loss.detach()}

def val_step(models, data, step, epoch_idx, config):
    model = models[0]
    model.eval()
    
    with torch.no_grad():
        
        input_data  = data['a'].to(config["device"])
        target_data = data['b'].to(config["device"])
        
        pred = model(input_data)
        # pred = torch.clamp(pred, 0, 1)
        
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
        "l1_loss": loss.detach()
    }, {
        "mae": indicator_cache["mae"],
        "psnr": indicator_cache["psnr"],
        "ssim": indicator_cache["ssim"],
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
   

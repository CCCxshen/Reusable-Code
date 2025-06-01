import torch
import time
import random
import numpy as np
from utils.data_utils import *

def train_dataset_step(data_path, is_train, config):
    input_data = np.expand_dims(normalize_(np.load(data_path[0]), -1024, 1000), axis = 0)
    target_data = np.expand_dims(normalize_(np.load(data_path[1]), -1024, 1000), axis = 0)
    
    return {
        "a": torch.from_numpy(input_data).float().to(config["device"]),
        "b": torch.from_numpy(target_data).float().to(config["device"])
    }

val_dataset_step = train_dataset_step
test_dataset_step = train_dataset_step


def train_step(models, optimizers, data, step, epoch_idx, config):
    optimizer = optimizers[0]
    optimizer.step()
    time.sleep(0.5)
    return {"l1_loss": random.randint(0, 100) / 100}

def val_step(models, data, step, epoch_idx, config):
    time.sleep(0.5)
    return {
        "l1_loss": random.randint(0, 100) / 100
    }, {
        "mse": random.randint(0, 100) / 100, 
        "psnr": random.randint(0, 100),
        "mae": random.randint(0, 100) / 100,
        "vif": random.randint(0, 100) / 100,
        "gen": random.randint(0, 100) / 100
        }
def test_step(models, data, step, config):
    time.sleep(0.5)
    return {
        "mse": random.randint(0, 100) / 100, 
        "psnr": random.randint(0, 100),
        "mae": random.randint(0, 100) / 100,
        "vif": random.randint(0, 100) / 100,
        "gen": random.randint(0, 100) / 100
        }




   
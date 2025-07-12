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

# 单张MR归一化还是整个体数据归一化？

def prep_somethings(config):
    config['dog'] = F2N_edge_dog(config["device"])
    return config

def train_dataset_step(data_path, is_train, config):
    # input_data  = np.clip(np.load(data_path[0]).astype(np.float32), 0, 2000) / 2000
    input_data  = np.load(data_path[0]).astype(np.float32)
    input_data  = normalize_(input_data, input_data.min(), input_data.max())
    target_data = normalize_(np.load(data_path[1]).astype(np.float32), -1024, 3072)
    mask_data   = np.load(data_path[2])
    
    Z, R, C = input_data.shape

    if R > config["train_patch_size"]:
        row = np.random.randint(0, R - config["train_patch_size"])
        col = np.random.randint(0, C - config["train_patch_size"])
        input_data  = input_data[:, row : row + config["train_patch_size"], col : col + config["train_patch_size"]]
        target_data = target_data[:, row : row + config["train_patch_size"], col : col + config["train_patch_size"]]
        mask_data   = mask_data[:, row : row + config["train_patch_size"], col : col + config["train_patch_size"]]

    return {
        "a": torch.from_numpy(input_data).float().to(config["device"]),
        "b": torch.from_numpy(target_data).float().to(config["device"]),
        "c": torch.from_numpy(mask_data).to(config["device"]),
        "d": '_'.join(data_path[0].split('/')[-1].split('_')[:2])
    }

def val_dataset_step(data_path, is_train, config):
    # input_data  = np.clip(np.load(data_path[0]).astype(np.float32), 0, 2000) / 2000
    input_data  = np.load(data_path[0]).astype(np.float32)
    input_data  = normalize_(input_data, input_data.min(), input_data.max())
    target_data = normalize_(np.load(data_path[1]).astype(np.float32), -1024, 3072)
    mask_data   = np.load(data_path[2])
    
    return {
        "a": torch.from_numpy(input_data).float().to(config["device"]),
        "b": torch.from_numpy(target_data).float().to(config["device"]),
        "c": torch.from_numpy(mask_data).to(config["device"]),
        "d": '_'.join(data_path[0].split('/')[-1].split('_')[:2])
    }

def test_dataset_step(data_path, is_train, config):
    mr, spacing, origin, direction = MhaDataRead(os.path.join(data_path[0], "mr.mha"))
    mask, _, _, _ = MhaDataRead(os.path.join(data_path[0], "mask.mha"))
    person_id = data_path[0].split(os.path.sep)[-1]
    
    mr_norm = normalize_(mr, mr.min(), mr.max())
    
    return {
        'mr': mr_norm,
        'mask': mask,
        'person_id': person_id,
        'spacing': spacing,
        'origin': origin,
        'direction': direction
    }
    

def train_step(models, optimizers, data, step, epoch_idx, config):
    
    model = models[0]
    model.train()
    optimizer = optimizers[0]
    
    input_data  = data['a'].to(config["device"])
    target_data = data['b'].to(config["device"])
    mask_data   = data['c'].to(config["device"])
    
    pred = model(input_data)
    # pred = torch.clamp(pred, 0, 1)
    loss = F.l1_loss(pred * mask_data, target_data) + config['dog'](pred * mask_data, target_data)
    # loss = F.l1_loss(pred * mask_data, target_data)
    
    loss.backward()
    optimizer.step()
    
    return {"train_loss": loss.detach()}

def val_step(models, data, step, epoch_idx, config):
    model = models[0]
    model.eval()
    
    with torch.no_grad():
        
        input_data  = data['a'].to(config["device"])
        target_data = data['b'].to(config["device"])
        mask_data   = data['c'].to(config["device"])
        
        pred = model(input_data)
        # pred = torch.clamp(pred, 0, 1)
        
        loss = F.l1_loss(pred * mask_data, target_data)
        
        indicator_cache = config["indicator"].score_patient(
            denormalize_(target_data, -1024, 3072)[0][0].detach().cpu().numpy(),
            denormalize_(pred,  -1024, 3072)[0][0].detach().cpu().numpy(),
            mask_data[0][0].detach().cpu().numpy()
        )

        if step % 5 == 0:
            
            show_2Dimages(
                images = [
                    denormalize_(pred * mask_data, -1024, 3072).detach().cpu().numpy()[0][0],
                    denormalize_(target_data, -1024, 3072).detach().cpu().numpy()[0][0],
                    # (input_data.detach().cpu().numpy() * 2000)[0][0],
                    input_data.detach().cpu().numpy()[0][0],
                    np.clip(denormalize_(pred * mask_data, -1024, 3072).detach().cpu().numpy()[0][0], -160, 240),
                    np.clip(denormalize_(target_data, -1024, 3072).detach().cpu().numpy()[0][0], -160, 240)
                ],
                names = [
                    "sct[-1024, 3072]", "ct[-1024, 3072]", "mr", "sct[-160, 240]", "ct[-160, 240]"
                ],
                title = f"step={step} {data['d'][0]}\n {dict2str(indicator_cache)}",
                save_path = os.path.join(config["root_dir"], "log", config["method_name"], "val"),
                save_name = f"epoch={epoch_idx}_step={step}_{data['d'][0]}",
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
        input_data  = data['a'].to(config["device"])
        target_data = data['b'].to(config["device"])
        mask_data   = data['c'].to(config["device"])
        
        pred = model(input_data)
        # pred = torch.clamp(pred, 0, 1)
        
        
        indicator_cache = config["indicator"].score_patient(
            denormalize_(target_data, -1024, 3072)[0][0].detach().cpu().numpy(),
            denormalize_(pred,  -1024, 3072)[0][0].detach().cpu().numpy(),
            mask_data[0][0].detach().cpu().numpy()
        )
        
        show_2Dimages(
            images = [
                denormalize_(pred * mask_data, -1024, 3072).detach().cpu().numpy()[0][0],
                denormalize_(target_data, -1024, 3072).detach().cpu().numpy()[0][0],
                # (input_data.detach().cpu().numpy() * 2000)[0][0],
                denormalize_(input_data, -1024, 3072).detach().cpu().numpy()[0][0],
                np.clip(denormalize_(pred * mask_data, -1024, 3072).detach().cpu().numpy()[0][0], -160, 240),
                np.clip(denormalize_(target_data, -1024, 3072).detach().cpu().numpy()[0][0], -160, 240)
            ],
            names = [
                "sct[-1024, 3000]", "ct[-1024, 3000]", "mr", "sct[-160, 240]", "ct[-160, 240]"
            ],
            title = f"step={step} {data['d'][0]}\n {dict2str(indicator_cache)}",
            save_path = os.path.join(config["root_dir"], "log", config["method_name"], "test"),
            save_name=f"step-{step}_{data['d'][0]}",
            shape = (2, 3)
        )
    
    return {
        "mae": indicator_cache["mae"],
        "psnr": indicator_cache["psnr"],
        "ssim": indicator_cache["ssim"],
    }

def test_3D_step(models, data, step, config):
    model = models[0]
    model.eval()
    
    with torch.no_grad():
        mr   = data['mr'][0]
        mask = data['mask'][0]
        person_id = data['person_id'][0]
        spacing   = data['spacing']
        origin    = data['origin']
        direction = data['direction']
        
        S, R, C = mr.shape
        input_V = mr * mask
        mask = mask.detach().cpu().numpy()
        result = np.zeros_like(mr, np.float32)
        
        for slice_idx in range(S):
            input_slice = input_V[slice_idx].to(config['device'])
            
            slice_pred = model(input_slice[None, None, :, :]).detach().cpu().numpy()
            slice_pred = denormalize_(np.clip(slice_pred, 0, 1) * mask[slice_idx], -1024, 3072)
            result[slice_idx] = slice_pred[0,0,:,:]
            
            show_2Dimages(
                images = [
                    # (input_slice * 2000).detach().cpu().numpy(), 
                    denormalize_(input_slice, -1024, 3072).detach().cpu().numpy()[0][0],
                    slice_pred[0,0,:,:],
                    np.clip(slice_pred[0,0,:,:], -160, 240)
                ],
                names = [
                    "mr", "sct", "sct[-160, 240]"
                ],
                title = f"{person_id}-{slice_idx}",
                save_name=f"{person_id}-{slice_idx}",
                save_path=os.path.join(config["root_dir"], "log", config["method_name"], "test")
            )
        
        
        MhaDataWrite(
            save_path = f"/data0/xcshen/research/xcshen_research/etc/sct_{person_id}.mha",
            volumn    = result,
            spacing   = spacing.flatten().cpu().numpy(),
            origin    = tuple(np.concatenate([x.flatten().cpu().numpy() for x in origin]).tolist()),
            direction = tuple(np.concatenate([x.flatten().cpu().numpy() for x in direction]).tolist()), 
        )
        return {}

test_step = test_3D_step
   

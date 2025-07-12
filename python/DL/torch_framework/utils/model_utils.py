import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim import *
from torch.optim.lr_scheduler import *
from networks.unet import *
from utils.utils import *
from indicator.metrics import *
from indicator.image_metrics import *

# from networks.Unet_SRAD.unet_model import *
from networks.nafnet import *
from networks.restormer import *
from networks.unet_denoising import *

def create_model(config):
    models = []
    for model_idx, model_name in enumerate(config["model"]):
        # if model_name == "UNet2D":
        #     model = UNet_2d(config["input_channels"], config["class_nums"])
        # if model_name == "UNet2D_SRAD":
        #     model = UNet(config["input_channels"], config["class_nums"])
        if model_name == "NAFNET":
            model = NAFNet(
                img_channel   = config["input_channels"],
                width         = config['width'], 
                middle_blk_num= config['middle_blk_num'],
                enc_blk_nums  = config['enc_blks'], 
                dec_blk_nums  = config['dec_blks']
            )
        if model_name == "Restormer":
            model = Restormer(
                inp_channels = config["inp_channels"],
                out_channels = config["out_channels"],
                dim          = config["dim"],
                num_blocks   = config["num_blocks"],
                num_refinement_blocks = config["num_refinement_blocks"],
                heads        = config["heads"],
                ffn_expansion_factor  = config["ffn_expansion_factor"],
                bias         = config["bias"],
                LayerNorm_type  = config["LayerNorm_type"],
                dual_pixel_task = config["dual_pixel_task"]
            )
        if model_name == "unet_denoising":
            model = Unet(config["input_channels"], config["class_nums"])
        model = initialize_model(model)
        model = model.to(config["device"])
        if config["DDP"] == True:
            model = DDP(model, device_ids=[config["local_rank"]], find_unused_parameters = False)
        models.append(model)
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
    if config["lr_scheduler"] == None:
        return None
    
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
    if config['DDP'] == False: models_state_dict = [model.state_dict() for model in models]
    else: models_state_dict = [model.module.state_dict() for model in models]
    
    if optimizers == None: optimizers_state_dict = None
    else: optimizers_state_dict = [optimizer.state_dict() for optimizer in optimizers]
    
    checkpoint = {
        "models": models_state_dict,
        "optimizers": optimizers_state_dict,
        "epoch": epoch,
        "indicator": indicator
    }
    if save_name == None: save_name = f"model_epoch{epoch}_{config['monitoring_indicators']}={indicator[config['monitoring_indicators']]:.4f}.pth"
    save_path = os.path.join(config["checkpoint_path"], save_name)
    
    torch.save(checkpoint, save_path)
    
    return f"The model has been saved to {save_path}"
    
def load_model(models = None, optimizers = None, config = None):
    if config["ckpt"] == "last" or config["ckpt"] == None: 
        load_path = os.path.join(config["checkpoint_path"], "last_model.pth")
    elif config["ckpt"] == "best": 
        load_path = os.path.join(config["checkpoint_path"], "best_model.pth")
    else: load_path = config["ckpt"]
    
    
    if not os.path.isfile(load_path):
        return models, optimizers, -1, -1, f"There are no pre-trained weight files! The model will be retrained!"
    
    state_dict = torch.load(load_path, weights_only=False)
    epoch = state_dict["epoch"]
    indicator = state_dict["indicator"]

    if models != None and state_dict['models'] != None:
        for model_idx in range(len(models)):
            if config['DDP'] == False: models[model_idx].load_state_dict(state_dict["models"][model_idx], strict = config['strict_loading_model_weight'])
            else: models[model_idx].module.load_state_dict(state_dict["models"][model_idx], strict = config['strict_loading_model_weight'])
    # if optimizers != None and state_dict['optimizers'] != None: 
    #     for optimizer_idx in range(len(optimizers)):
    #         optimizers[optimizer_idx].load_state_dict(state_dict["optimizers"][optimizer_idx])
    
    return models, optimizers, epoch, indicator, f"Load the model from {load_path}"

def create_indicator(config):
    if config["indicator_name"] == "IQA":
        indicator = MetricsCalculator(config["min_val"], config["max_val"], config["device"])
    if config["indicator_name"] == "MRtoCT":
        indicator = ImageMetrics(dynamic_range=[config["min_val"], config["max_val"]])
    
    return indicator

def create_loss(config):
    pass
    
    
import sys
import json
import argparse
import importlib
from torch.nn.parallel import DistributedDataParallel as DDP

from src.train import *
from src.test import *
from utils.utils import *
from utils.IO_utils import *
from utils.ddp_utils import *
from utils.data_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--main_config", type=str, default='./method/main_config.json')
parser.add_argument("--method_config", type=str, default='./method/supervised_attempt/config.json')

argspar = parser.parse_args()
config = parseJSON(argspar.main_config)
config.update(parseJSON(argspar.method_config))

sys.path.append(config["root_dir"])
method_function = importlib.import_module(config["fuction_path_of_method_import"])
config["method_function"] = method_function

set_seed(config["seed"])
create_dirs(config["root_dir"], config["method_name"])

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == 'cuda':
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in config["gpu"]])
    
    if config["DDP"] == True:
        local_rank, world_size = init_ddp(config["DDP"])
        config["device"] = torch.device(f"cuda:{config['gpu'][local_rank]}")
        config["world_size"] = world_size
        config["local_rank"] = local_rank
    else: config["device"] = torch.device(f"cuda:{config['gpu'][0]}")

recorder = Recorder(
    stage_paths = {
        "val": os.path.join(config["root_dir"], "log", config["method_name"], "val"),
        "test": os.path.join(config["root_dir"], "log", config["method_name"], "test"),
        "save_model": os.path.join(config["root_dir"], "log", config["method_name"], "checkpoint"),
    },
    method_name = config["method_name"],
    method_path = os.path.join(config["root_dir"], "log", config["method_name"]),
    DDP = config["DDP"] 
)
config["recorder"] = recorder

if config["DDP"] == True:
    recorder.message(f"[Multi-GPU Training - RANK {local_rank}] Using device: {device}:{config['gpu'][local_rank]}, name: {torch.cuda.get_device_name(local_rank)}", ignore_DDP=True, ignore_width = True, stage = "init")
else: 
    recorder.message(f"Single-GPU Training - Using device: {device}, name: {torch.cuda.get_device_name(0)}", ignore_width=True, stage = "init")
    

if config["train"] == True:
    # train(config)
    
    if (config["DDP"] == False) or (config["DDP"] == True and int(os.environ["RANK"]) == 0):
        config["ckpt"] = None
        test(config)
    
    
    
    
    
    




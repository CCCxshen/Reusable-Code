import sys
import json
import argparse
import importlib

from src.train import *
from src.test import *
from utils.utils import *
from utils.IO_utils import *
from utils.ddp_utils import *
from utils.data_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--main_config", type=str, default='./method/main_config.json')
# parser.add_argument("--method_config", type=str, default='./method/supervised_attempt/config.json')
parser.add_argument("--method_config", type=str, default='./method/SRAD25_task1/config/config_AB_lr_3e-4.json')
# parser.add_argument("--method_config", type=str, default='./method/SRAD25_task1_HN/config.json')
# parser.add_argument("--method_config", type=str, default='./method/SRAD25_task1_TH/config.json')

argspar = parser.parse_args()
config = parseJSON(argspar.method_config)
# config.update(parseJSON(argspar.etc_json_file))

config["root_dir"] = os.path.abspath(f".{os.sep}")

sys.path.append(config["root_dir"])
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in config["gpu"]])
method_function = importlib.import_module(config["fuction_path_of_method_import"])
config["method_function"] = method_function

set_seed(config["seed"])
create_dirs(config)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == 'cuda':
    if config["DDP"] == True:
        local_rank, world_size = init_ddp(config["DDP"])
        config["device"] = torch.device(f"cuda:{local_rank}")
        config["world_size"] = world_size
        config["local_rank"] = local_rank
    else: config["device"] = torch.device("cuda:0")

recorder = Recorder(
    stage_paths = {
        "val": config["val_path"],
        "test": config["test_path"],
        "save_model": config["checkpoint_path"],
    },
    method_name = config["method_name"],
    method_log_path = config["log_path"],
    DDP = config["DDP"] 
)
config["recorder"] = recorder

if config["DDP"] == True:
    recorder.message(f"[Multi-GPU Training - RANK {local_rank}] Using device: {device}:{config['gpu'][local_rank]}, name: {torch.cuda.get_device_name(local_rank)}", ignore_DDP=True, ignore_width = True, stage = "init")
else: 
    recorder.message(f"Single-GPU Training - Using device: {device}, name: {torch.cuda.get_device_name(0)}", ignore_width=True, stage = "init")
    
config = config["method_function"].prep_somethings(config)

if config["train"] == True:  
    train(config)
    config["ckpt"] = "best"
    
if config["test"] == True:
    test(config)

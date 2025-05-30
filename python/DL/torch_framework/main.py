import sys
import json
import argparse
import importlib
from src.train import *
from src.test import *
from utils.utils import *
from utils.IO_utils import *
from utils.data_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--main_config", type=str, default='./method/main_config.json')
parser.add_argument("--method_config", type=str, default='./method/demo_method_task/config.json')

argspar = parser.parse_args()
config = parseJSON(argspar.main_config)
config.update(parseJSON(argspar.method_config))

sys.path.append(config["root_dir"])
method_function = importlib.import_module(config["fuction_path_of_method_import"])
config["method_function"] = method_function

set_seed(config["seed"])
create_dirs(config["root_dir"], config["task_name"])


os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"][0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["device"] = device

recorder = Recorder(
    stage_paths = {
        "val": os.path.join(config["root_dir"], "log", config["task_name"], "val"),
        "test": os.path.join(config["root_dir"], "log", config["task_name"], "test"),
        "save_model": os.path.join(config["root_dir"], "log", config["task_name"], "checkpoint"),
    },
    task_name = config["task_name"],
    task_path = os.path.join(config["root_dir"], "log", config["task_name"]) 
)
config["recorder"] = recorder


if config["train"] == True:
    train(config)
    
    config["ckpt"] = None
    test(config)
    
    
    
    
    
    




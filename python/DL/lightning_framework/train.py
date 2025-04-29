import argparse
from utils.etc import *

parser = argparse.ArgumentParser(description="new task")
parser.add_argument("--config", type=str, default='./config/basic_settings.json')

argspar = parser.parse_args()
opt = parseJSON(argspar.config)





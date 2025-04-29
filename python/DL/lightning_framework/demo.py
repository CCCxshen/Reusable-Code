import argparse
import json
from utils.etc import *

parser = argparse.ArgumentParser(description="new task")
parser.add_argument("--config", type=str, default='./config/basic_settings.json')

argspar = parser.parse_args()
config = parseJSON(argspar.config)

print(argspar)
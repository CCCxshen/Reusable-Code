from collections import OrderedDict
import json
import os

def parseJSON(conf_path):
    # 删除以"//"开头的注释
    json_str = ''
    with open(conf_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
            
    config = json.loads(json_str, object_pairs_hook=OrderedDict)
    return config


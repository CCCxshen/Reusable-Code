from collections import OrderedDict
import torch
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

def dictionary_addition(A: dict, B: dict):
    dictA = A.copy()
    dictB = B.copy()
    for key, value in dictB.items():
        if key not in dictA: dictA[key] = value
        else: dictA[key] += value
            
    return dictA

def dictionary_division(A: dict, val: int):
    dictA = A.copy()
    for key, value in dictA.items():
        dictA[key] /= val
            
    return dictA

def dict2str(dictdata: dict, decimal_places = 4):
    text = ""
    for key, value in dictdata.items():
        text += f"{key}: {value:.{decimal_places}f}, "
    return text[:-2]

def dict_append(A: dict, B: dict):
    dictA = A.copy()
    dictB = B.copy()
    for key, value in dictB.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if key not in dictA: dictA[key] = [value] 
        else: dictA[key].append(value)
    return dictA

def check(a, b, strategy):
    if strategy == "max":
        return a > b
    if strategy == "min":
        return a < b

if __name__ == "__main__":
    A = {"A": 10}
    dictionary_division(A, 10)
    print(A)
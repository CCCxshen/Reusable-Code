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

def dictionary_addition(A: dict, B: dict):
    for key, value in B.items():
        if key not in A: A[key] = value
        else: A[key] += value
            
    return A

def dictionary_division(A: dict, val: int):
    for key, value in A.items():
        A[key] /= val
            
    return A

def dict2str(dictdata: dict, decimal_places = 4):
    text = ""
    for key, value in dictdata.items():
        text += f"{key}: {value:.{decimal_places}f}, "
    return text[:-2]

def dict_append(A: dict, B: dict):
    for key, value in B.items():
        if key not in A: A[key] = [value] 
        else: A[key].append(value)
    return A

def check(a, b, strategy):
    if strategy == "max":
        return a > b
    if strategy == "min":
        return a < b

if __name__ == "__main__":
    A = {}
    B = {"A": 1, "B": 1}
    A = dict_append(A, B)
    A = dict_append(A, B)
    print(A)
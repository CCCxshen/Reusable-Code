from collections import OrderedDict
from datetime import datetime
import numpy as np
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

def format_timedelta(delta):
    """将timedelta对象格式化为 时:分:秒.毫秒 的形式"""
    total_seconds = delta.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds = remainder
    milliseconds = delta.microseconds // 1000
    if hours > 0:
        return f"{int(hours):02d}h:{int(minutes):02d}m:{int(seconds):02d}s.{milliseconds:03d}ms"
    if hours == 0:
        return f"{int(minutes):02d}m:{int(seconds):02d}s.{milliseconds:03d}ms"

def time_utils(train_start_time, epoch_start_time, epoch_end_time, remain_epoch = 1):
    train_spent_time = epoch_end_time - train_start_time
    epoch_spent_time = epoch_end_time - epoch_start_time
    remain_spent_time = (epoch_end_time - epoch_start_time) * remain_epoch
    
    return format_timedelta(train_spent_time), format_timedelta(epoch_spent_time), format_timedelta(remain_spent_time)
    

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
        if key not in dictA: dictA[key] = np.array([value]) 
        else: dictA[key] = np.append(dictA[key], value)
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
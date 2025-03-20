# 一个多进程任务执行器
from multiprocessing import Process, Pool
from tqdm import tqdm

def execute(function = None, parameter = None, processes = 3):
    with Pool(processes = processes) as pool:
        result = list(tqdm(pool.imap(function, parameter), total = len(parameter)))
        return result

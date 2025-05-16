from torch.utils.data import Dataset
from natsort import natsorted
import numpy as np
import os, re

class patDataset(Dataset):
    def __init__(self, data_dir, is_train, regex_pattern = [r'.*'], process_fun = None):
        self.data_dir = os.path.abspath(data_dir)
        self.is_train = is_train
        self.process_fun = process_fun
        
        regex = [re.compile(x) for x in regex_pattern]
        self.data_names = [natsorted([x for x in os.listdir(self.data_dir) if y.match(x)]) for y in regex]
        self.data_paths = np.array([[os.path.join(self.data_dir, y) for y in x] for x in self.data_names])
    
    def __len__(self):
        return self.data_paths.shape[1]
    
    def __getitem__(self, index):
        data_path = np.array([x[index - 1] for x in self.data_paths])

        if self.process_fun != None:
            data = self.process_fun(data_path, self.is_train)

        return data
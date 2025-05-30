import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed: int = 42) -> None:
    
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def normalize_(img, MIN_HU=-1024., MAX_HU=3000.):
    img[img > MAX_HU] = MAX_HU
    img[img < MIN_HU] = MIN_HU
    return (img - MIN_HU) / (MAX_HU - MIN_HU)


def denormalize_(img, MIN_HU=-1024., MAX_HU=3000.):
    img = img * (MAX_HU - MIN_HU) + MIN_HU
    return img


def trans2img(img):
    img = img * 255.
    return np.uint8(img)





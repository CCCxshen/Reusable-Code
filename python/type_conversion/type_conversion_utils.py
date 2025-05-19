import pydicom as pyi
import numpy as np
import os
from natsort import natsorted

def dcmSequence2npySequence(dS_dir_path, save_path, prefix, suffix, clip_range = [-1024, 3000]):
    """
    将dcm序列转换为npy序列, 名称格式为: prefix_idx_suffix.npy
    """
    def readDCM(path):
        obj = pyi.dcmread(path)
        RescaleIntercept = obj.data_element('RescaleIntercept').value
        RescaleSlope = obj.data_element('RescaleSlope').value
        
        return obj.pixel_array.astype(np.int16) * RescaleSlope + RescaleIntercept
    
    dS_dir_path = os.path.abspath(dS_dir_path)
    save_path   = os.path.abspath(save_path)
    
    dcm_sequence = natsorted(os.listdir(dS_dir_path))
    dcm_abspaths  = [os.path.join(dS_dir_path, x) for x in dcm_sequence] 
    
    for dcm_idx, dcm_abspath in enumerate(dcm_abspaths):
        print(f"In {prefix}, slice at {dcm_idx + 1} / {len(dcm_abspaths)}", end = "\r")
        npy_name = f"{prefix}_{dcm_idx + 1}_{suffix}"
        
        dcm_array = np.clip(readDCM(dcm_abspath), clip_range[0], clip_range[1])
        np.save(os.path.join(save_path, npy_name), dcm_array)
    print(f"\n{prefix} is done!")
        
        
        
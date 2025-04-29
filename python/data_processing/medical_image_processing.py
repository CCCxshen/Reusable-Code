import numpy as np
import SimpleITK as sitk
import pydicom as pyi

def normalization(arr, min_val, max_val, bias_val, boundary = 1.):
    """将图像偏移bias_val个像素点，然后最大最小值归一化，最后缩放到[0, boundary]范围内

    Args:
        arr (_type_): _description_
        min_val (_type_): _description_
        max_val (_type_): _description_
        bias_val (_type_): _description_
        boundary (_type_, optional): _description_. Defaults to 1..

    Returns:
        _type_: _description_
    """
    arr_norm = arr.copy() + bias_val
    arr_norm = (arr_norm - min_val) / (max_val - min_val)
    arr_norm *= boundary
    return arr_norm

def restore(arr, min_val, max_val, bias_val, boundary = 1.0):
    """
        先将图像缩小boundary倍，然后还原到[min_val - bias_val, max_val - bias_val]范围内

    Args:
        arr (_type_): _description_
        min_val (_type_): _description_
        max_val (_type_): _description_
        bias_val (_type_): _description_
        boundary (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    arr_res = arr.copy() / boundary
    arr_res = arr_res * (max_val - min_val) + min_val
    arr_res -= bias_val
    return arr_res

import SimpleITK as sitk
import os
import os.path as osp
import pydicom
import numpy as np
import scipy.io

# 读取mha数据
def MhaDataRead(path, as_type=np.float32):
    """
        返回的数据有：
        1. 图像的数据数组
        2. [z, y, x]之间的物理距离
        3. 原点信息
        4. 方向余弦矩阵 
    """
    mha = sitk.ReadImage(path)
    spacing = mha.GetSpacing()            # [x,y,z] 获得xyz轴上的物理间距
    volumn = sitk.GetArrayFromImage(mha)  # [z,y,x] 获得数据
    origin = mha.GetOrigin()              # 获得图像的原点信息
    direction = mha.GetDirection()        # 获得图像的方向余弦矩阵

    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    spacing_ = np.array([spacing_z, spacing_y, spacing_x])
    return volumn.astype(as_type), spacing_.astype(np.float32), origin, direction


# 根据数据保存mha文件
def MhaDataWrite(save_path, volumn, spacing, origin, direction, as_type=np.float32):

    spacing = spacing.astype(np.float64)
    raw = sitk.GetImageFromArray(volumn[:, :, :].astype(as_type))
    spacing_ = (spacing[2], spacing[1], spacing[0])
    raw.SetSpacing(spacing_)
    raw.SetOrigin(origin)
    raw.SetDirection(direction)
    sitk.WriteImage(raw, save_path)


# N4偏置场校正
def N4BiasFieldCorrection(volumn_path, save_path):  # ,mask_path,save_path):
    img = sitk.ReadImage(volumn_path)
    # mask,_ = sitk.ReadImage(mask_path)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    inputVolumn = sitk.Cast(img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    sitk.WriteImage(corrector.Execute(inputVolumn, mask), save_path)


# dcm转mha
def dcm2mha(DCM_DIR, OUT_PATH):
    """

    :param DCM_DIR: Input folder set to be converted.
    :param OUT_PATH: Output file suffixed with .nii.gz . *(Relative path)
    :return: No retuen.
    """
    fuse_list = []
    for dicom_file in os.listdir(DCM_DIR):
        dicom = pydicom.dcmread(osp.join(DCM_DIR, dicom_file))
        fuse_list.append([dicom.pixel_array, float(dicom.SliceLocation)])
    # 按照每层位置(Z轴方向)由小到大排序
    fuse_list.sort(key=lambda x: x[1])
    volume_list = [i[0] for i in fuse_list]
    volume = np.array(volume_list).astype(np.float32) - 1024
    [spacing_x, spacing_y] = dicom.PixelSpacing
    spacing = np.array([dicom.SliceThickness, spacing_x, spacing_y])
    MhaDataWrite(OUT_PATH, volume, spacing)


# mha转mat
def mha2mat(in_path, out_dir=None):
    volume, spacing = MhaDataRead(in_path)
    if out_dir is None:  # save in same dir
        out_path = in_path.split('.')[0] + '.mat'
    else:  # save in specific dir
        out_path = osp.join(out_dir, osp.split(in_path)[-1].split('.')[0] + '.mat')
    scipy.io.savemat(out_path, {'volume': volume, 'spacing': spacing})
    print(f'Saved at {out_path}')


# mat转mha
def mat2mha(in_path, out_dir=None):
    mat_contents = scipy.io.loadmat(in_path)
    if out_dir is None:  # save in same dir
        out_path = in_path.split('.')[0] + '.nii.gz'
    else:  # save in specific dir
        out_path = osp.join(out_dir, osp.split(in_path)[-1].split('.')[0] + '.nii.gz')
    MhaDataWrite(out_path, mat_contents['output'], mat_contents['spacing'][0])
    print(f'Saved at {out_path}')


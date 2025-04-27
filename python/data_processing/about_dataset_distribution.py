import os
import random
import shutil

def data_distributor(data_dir_paths, save_dir_path, proportion = [0.8, 0.1, 0.1], test_data = True, save_path = False, path_file_name = None):
    """将数据集分类为训练集、验证集、测试集

    Args:
        data_dir_paths (_type_): _description_
        save_dir_path (_type_): _description_
        proportion (list, optional): _description_. Defaults to [0.8, 0.1, 0.1].
        test_data (bool, optional): _description_. Defaults to True.
        save_path (bool, optional): _description_. Defaults to False.
        path_file_name (_type_, optional): _description_. Defaults to None.
    """
    
    file_paths = []
    
    for i, dir_path in enumerate(data_dir_paths):
        file_names    = os.listdir(os.path.abspath(dir_path))
        file_abspath = [os.path.join(os.path.abspath(dir_path), file_name) for i, file_name in enumerate(file_names)]
        file_paths += file_abspath
    
    train_data_path = os.path.join(os.path.abspath(save_dir_path), "train")
    if not os.path.exists(train_data_path): os.makedirs(train_data_path)
    vaild_data_path = os.path.join(os.path.abspath(save_dir_path), "vaild")
    if not os.path.exists(vaild_data_path): os.makedirs(vaild_data_path)
    if test_data: 
        test_data_path = os.path.join(os.path.abspath(save_dir_path), "test")
        if not os.path.exists(test_data_path): os.makedirs(test_data_path)
    
    random.shuffle(file_paths)
    
    train_file_cnt = int(len(file_paths) *  proportion[0] + 0.5)
    if test_data == True:
        vaild_file_cnt = int(len(file_paths) *  proportion[1] + 0.5)
        test_file_cnt  = len(file_paths) - train_file_cnt - vaild_file_cnt
    else:
        vaild_file_cnt = len(file_paths) - train_file_cnt
        test_file_cnt  = 0
    
    train_file_paths = file_paths[0:0 + train_file_cnt]
    vaild_file_paths = file_paths[train_file_cnt:train_file_cnt + vaild_file_cnt]
    test_file_paths  = file_paths[train_file_cnt + vaild_file_cnt:]
    
    train_save_path = []
    vaild_save_path = []
    test_save_path = []
    
    for i, file_path in enumerate(train_file_paths):
        file_name = file_path.split("/")[-1]
        file_save_path = os.path.join(os.path.abspath(train_data_path), file_name)
        shutil.copy2(file_path, file_save_path)
        train_save_path.append(file_save_path)
        print(f"Train data: {i + 1, len(train_file_paths)}    ", end = "\r")
    print("")
    
    for i, file_path in enumerate(vaild_file_paths):
        file_name = file_path.split("/")[-1]
        file_save_path = os.path.join(os.path.abspath(vaild_data_path), file_name)
        shutil.copy2(file_path, file_save_path)
        vaild_save_path.append(file_save_path)
        print(f"Vaild data: {i + 1, len(vaild_file_paths)}    ", end = "\r")
    print("")
    
    if test_data == True:
        for i, file_path in enumerate(test_file_paths):
            file_name = file_path.split("/")[-1]
            file_save_path = os.path.join(os.path.abspath(test_data_path), file_name)
            shutil.copy2(file_path, file_save_path)
            test_save_path.append(file_save_path)
            print(f"Test data: {i + 1, len(test_file_paths)}    ", end = "\r")
        print("")
    
    if save_path == True:
        try:
            train_path_file_name = f"train_{path_file_name[0]}.txt"
        except:
            train_path_file_name = f"train.txt"
            
        try:
            vaild_path_file_name = f"vaild_{path_file_name[0]}.txt"
        except:
            vaild_path_file_name = f"vaild.txt"
            
        try:
            test_path_file_name = f"test_{path_file_name[0]}.txt"
        except:
            test_path_file_name = f"test.txt"
        
        with open(os.path.join(os.path.abspath(save_dir_path), train_path_file_name), 'w') as f:
            for x in train_save_path:
                f.write(x + '\n')
        with open(os.path.join(os.path.abspath(save_dir_path), vaild_path_file_name), 'w') as f:
            for x in vaild_save_path:
                f.write(x + '\n')
        if test_data == True:
            with open(os.path.join(os.path.abspath(save_dir_path), test_path_file_name), 'w') as f:
                for x in test_save_path:
                    f.write(x + '\n')
    print("Finish!")
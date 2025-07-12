import os, math, shutil, sys, time
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import *
from datetime import datetime
from colorama import init, Fore, Back, Style
from matplotlib.ticker import MaxNLocator
import json
import pickle


def save_slices(
        inputs,
        pred,
        gt,
        out_dir,
        name_prefix,
        epoch,
        idx,
        MIN_HU=-1024.,
        MAX_HU=3000.,
        ori_psnr=0.,
        ori_ssim=0.,
        pred_psnr=0.,
        pred_ssim=0.,
):
    os.makedirs(out_dir, exist_ok=True)
    inputs = denormalize_(inputs.cpu(), MIN_HU, MAX_HU)
    pred = denormalize_(pred.cpu(), MIN_HU, MAX_HU)
    gt = denormalize_(gt.cpu(), MIN_HU, MAX_HU)

    inputs = normalize_(inputs.numpy().squeeze(), -160, 240)
    pred = normalize_(pred.numpy().squeeze(), -160, 240)
    gt = normalize_(gt.numpy().squeeze(), -160, 240)

    plt.imsave(os.path.join(out_dir, f'{name_prefix}_{epoch}_{idx}_input_{ori_psnr}_{ori_ssim}.png'),
               (inputs * 255.).astype(np.uint8),
               cmap='gray')
    plt.imsave(os.path.join(out_dir, f'{name_prefix}_{epoch}_{idx}_pred_{pred_psnr}_{pred_ssim}.png'),
               (pred * 255.).astype(np.uint8),
               cmap='gray')
    plt.imsave(os.path.join(out_dir, f'{name_prefix}_{epoch}_{idx}_gt.png'), (gt * 255.).astype(np.uint8), cmap='gray')


def create_dirs(config):
    root_dir = config["root_dir"]
    method   = config["method_name"]
    
    config["log_path"]  = os.path.join(root_dir, "log", method)
    config["val_path"]  = os.path.join(root_dir, "log", method, "val")
    config["test_path"] = os.path.join(root_dir, "log", method, "test")
    config["checkpoint_path"] = os.path.join(root_dir, "log", method, "checkpoint")
    config["etc_path"]  = os.path.join(root_dir, "log", method, "etc")
    
    os.makedirs(config["log_path"], exist_ok=True)
    os.makedirs(config["val_path"], exist_ok=True)
    os.makedirs(config["test_path"], exist_ok=True)
    os.makedirs(config["checkpoint_path"], exist_ok=True)
    os.makedirs(config["etc_path"], exist_ok=True)
    

def save_dict_to_file(dict, filename):
    np.savez(filename, **dict)

def load_dict_from_file(filename):
    if not os.path.exists(filename):
        return {}
    
    loaded = np.load(filename, allow_pickle=True)
    loaded_dict = {key: loaded[key] for key in loaded}
    return loaded_dict



class Recorder:
    def __init__(self, stage_paths, method_name, method_log_path, DDP):
        self.stage_paths = stage_paths
        self.method_name = method_name
        self.method_path = method_log_path
        self.DDP = DDP
        init(autoreset=True)
        
        
    def message(self, info, write_into = None, end = "\n", stage = "train", ignore_DDP = False, ignore_width = False):
        if (ignore_DDP == False) and (self.DDP == True) and (int(os.environ["RANK"]) != 0): return 
        
        formatted_time = datetime.now().strftime("%Y/%m/%d-%H:%M:%S.%f")[:-3]
        terminal_width = shutil.get_terminal_size().columns

        prefix = f"\033[2K\033[0G{Fore.CYAN + formatted_time + Fore.RESET}@{Fore.GREEN + self.method_name + Fore.RESET}-{Fore.RED + stage.rjust(5) + Fore.RESET}> "
        print_info = None
        
        if ignore_width == False:
            split_info = info.split(",")
            
            for ed in range(len(split_info), 0, -1):
                if len(prefix) + len(','.join(split_info[0:ed])) - 34 <= terminal_width:
                    print_info = prefix + ','.join(split_info[0:ed])
                    if ed < len(split_info):
                        print_info = print_info + " ..."
                    break
            
            if print_info == None:
                print_info = f"\033[2K\033[0G{Fore.LIGHTRED_EX + 'Width is too narrow!' + Fore.RESET}"
                
        else: print_info = prefix + info
        
        print(print_info, end = end)
        
        if write_into != None and write_into != "never":
            with open(os.path.join(self.stage_paths[write_into], "log.txt"), "a") as file:
                file.write(f"{formatted_time}@{self.method_name}-{stage.rjust(5)}: {info}\n")
        
        if write_into != "never":
            with open(os.path.join(self.method_path, "log_summary.txt"), "a") as file:
                file.write(f"{formatted_time}@{self.method_name}-{stage.rjust(5)}: {info}\n")
        
    def draw(self, dictData: dict, x_labels: list = [], y_labels: list = [], subtitle: list = [], title: str = None, shape: tuple = None,  save_path: str = None, save_name: str = "file_name", ignore_DDP = False):
        if (ignore_DDP == False) and (self.DDP == True) and (int(os.environ["RANK"]) != 0): return 
        plt.close()
        
        datas = []

        if subtitle == []:
            for key, value in dictData.items():
                subtitle.append(key)    
                datas.append(value)
        else: 
            for key, value in dictData.items():
                datas.append(value)
        
        datas = np.array(datas)
        cnt = len(datas)
        
        if len(y_labels) < cnt:  y_labels = subtitle.copy()
            
        while len(x_labels) < cnt:
            x_labels.append("epoch")
            
        idx = 1
        while len(subtitle) < cnt:
            subtitle.append(f"line {idx}")
            idx += 1
        
        if cnt > 3:
            col = math.ceil(math.sqrt(len(datas)))
            row = 1
            while row * col < cnt:
                row += 1
            shape = (row, col)
        
        if cnt == 1:
            plt.plot(list(range(len(datas[0]))), datas[0], 'b-', linewidth = 1, marker = "o", markersize=2)
            # plt.xaxis.set_major_locator(MaxNLocator(integer=True))
            if title != None: plt.title(title)
            plt.xlabel(x_labels[0])
            plt.ylabel(y_labels[0])
            plt.grid(True)
            # plt.xticks(list(range(len(datas[0]))))
            
            
        else:
            row = 1
            col = cnt
            weight = 5 * cnt
            height = 4
            
            if shape != None:
                row = shape[0]
                col = shape[1]
                weight = 5 * col 
                height = 4 * row 
            
            fig, axes = plt.subplots(row, col, figsize = (weight, height))
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            if row == 1 or col == 1:
                for i, data in enumerate(datas):
                    axes[i].plot(list(range(len(datas[0]))), data, 'b-', linewidth = 1, marker = "o", markersize=2)
                    axes[i].set_title(subtitle[i])
                    axes[i].set_xlabel(x_labels[i])
                    axes[i].set_ylabel(y_labels[i])
                    axes[i].grid(True)
                    # axes[i].set_xticks(list(range(len(datas[0]))))

            else:
                i = 0
                for r in range(row):
                    for c in range(col):
                        if (r * col + c + 1) <= cnt: 
                            axes[r][c].plot(list(range(len(datas[0]))), datas[i], 'b-', linewidth = 1, marker = "o", markersize=2)
                            axes[r][c].set_title(subtitle[i])
                            axes[r][c].set_xlabel(x_labels[i])
                            axes[r][c].set_ylabel(y_labels[i])
                            axes[r][c].grid(True)
                            # axes[r][c].set_xticks(list(range(len(datas[0]))))
                        else: axes[r][c].axis('off')
                        i += 1
            
            if title != None: fig.suptitle(title)
        if save_path != None: 
            if not os.path.exists(save_path): os.makedirs(save_path)
            if cnt == 1: plt.savefig("{}.png".format(os.path.join(save_path, save_name)), dpi = 300)
            else: fig.savefig("{}.png".format(os.path.join(save_path, save_name)), dpi = 300)
        
        # if show: plt.show()
        
        
        
if __name__ == "__main__":
    a = Recorder("1", "taks_name", "1")
    i = 1
    while True:
        time.sleep(0.5)
        a.message(f"epoch {i}/1000, step {i}/{1000}, l1_loss: 0.0037, rmse: 153.1180, psnr: 22.4237, ssim: 0.3277, lpips: 0.1234, vif: 1.1111, nqm: 0.1234", end = "\r") 
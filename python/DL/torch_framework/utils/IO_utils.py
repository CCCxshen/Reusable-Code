
import os, math
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import *
from datetime import datetime
from colorama import init, Fore, Back, Style
from matplotlib.ticker import MaxNLocator

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


def create_dirs(root_dir, method):
	os.makedirs(os.path.join(root_dir, "log", method), exist_ok=True)
	os.makedirs(os.path.join(root_dir, "log", method, "val"), exist_ok=True)
	os.makedirs(os.path.join(root_dir, "log", method, "test"), exist_ok=True)
	os.makedirs(os.path.join(root_dir, "log", method, "checkpoint"), exist_ok=True)


 
class Recorder:
    def __init__(self, stage_paths, task_name, task_path):
        self.stage_paths = stage_paths
        self.task_name = task_name
        self.task_path = task_path
        init(autoreset=True)
        
    def message(self, info, state = None, end = "\n", stage = "train"):
        formatted_time = datetime.now().strftime("%Y/%m/%d-%H:%M:%S.%f")[:-3]
        text = f"\033[2K\033[0G{Fore.CYAN + formatted_time + Fore.RESET}@{Fore.GREEN + self.task_name + Fore.RESET}-{Fore.RED + stage.rjust(5) + Fore.RESET}: {info}"
        print(text, end = end)
        
        if state != None:
            with open(os.path.join(self.stage_paths[state], "log.txt"), "a") as file:
                file.write(f"{formatted_time}@{self.task_name}-{stage.rjust(5)}: {info}\n")
        
        with open(os.path.join(self.task_path, "log_summary.txt"), "a") as file:
            file.write(f"{formatted_time}@{self.task_name}-{stage.rjust(5)}: {info}\n")
        
    def draw(self, dictData: dict, x_labels: list = [], y_labels: list = [], subtitle: list = [], title: str = None, shape: tuple = None,  save_path: str = None, save_name: str = "file_name"):
        
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
        plt.close()
        
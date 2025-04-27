import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

def get_current_date():
    return datetime.datetime.now().strftime("%YY%mM%D%Hh%Mm%Ss")


def show_2Dimages(images: np.ndarray, names: list = None, title: str = None, shape: tuple = None, axis = 'off', save: bool = False, save_path: str = "./", show = True):
    """展示一张或多张图片

    Args:
        images (np.ndarray): _description_
        names (list, optional): _description_. Defaults to None.
        title (str, optional): _description_. Defaults to None.
        shape (tuple, optional): _description_. Defaults to None.
        axis (str, optional): _description_. Defaults to 'off'.
        save (bool, optional): _description_. Defaults to False.
        save_path (str, optional): _description_. Defaults to "./".
        show (bool, optional): _description_. Defaults to True.
    """
    cnt = len(images)
    
    if cnt == 1:
        if len(image.shape) == 2:
            plt.imshow(images[0], cmap = "gray")
        else:
            plt.imshow(images[0])
        if names != None: plt.title(names[0])
        plt.axis(axis)  
         
    else:
        row = 1
        col = cnt
        weight = 4 * cnt
        height = 4
        
        if shape != None:
            row = shape[0]
            col = shape[1]
            weight = 3 * col 
            height = 3 * row 
        
        fig, axes = plt.subplots(row, col, figsize = (weight, height))
        
        if row == 1 or col == 1:
            for i, image in enumerate(images):
                if len(image.shape) == 2:
                    axes[i].imshow(image, cmap = "gray")
                else:
                    axes[i].imshow(image)
                
                if names != None: axes[i].set_title(names[i])
                axes[i].axis(axis)
        else:
            i = 0
            for r in range(row):
                for c in range(col):
                    if len(image.shape) == 2:
                        axes[r][c].imshow(images[i], cmap = "gray")
                    else:
                        axes[r][c].imshow(image)
                    
                    if names != None: axes[r][c].set_title(names[i])
                    axes[r][c].axis(axis)
                    i += 1
        
    if title != None: fig.suptitle(title)
    if save  == True: 
        if not os.path.exists(save_path): os.makedirs(save_path)
        if cnt == 1: plt.savefig(f"{os.path.join(save_path, get_current_date())}.png", dpi = 300)
        else: fig.savefig(f"{os.path.join(save_path, get_current_date())}.png", dpi = 300)
    
    if show: plt.show()
    
    
def boxplot_Compare_Two_Pic(arr_cal, arr_real, names = None, title = None, xlabel = None, ylabel = None, show = True, save = False, save_path = "./", theoretical_value = None, idx = None):
    """箱型图，两张图对比

    Args:
        arr_cal (_type_): _description_
        arr_real (_type_): _description_
        names (_type_, optional): _description_. Defaults to None.
        title (_type_, optional): _description_. Defaults to None.
        xlabel (_type_, optional): _description_. Defaults to None.
        ylabel (_type_, optional): _description_. Defaults to None.
        show (bool, optional): _description_. Defaults to True.
        save (bool, optional): _description_. Defaults to False.
        save_path (str, optional): _description_. Defaults to "./".
        theoretical_value (_type_, optional): _description_. Defaults to None.
        idx (_type_, optional): _description_. Defaults to None.
    """
    if names == None:
        names = ['data1', 'data2']
    if title == None:
        title = 'boxplot'
    if xlabel == None:
        xlabel = "images"
    if ylabel == None:
        ylabel = "value"
    
    mid = []
        
    # 生成两个示例数组
    arr_cal = arr_cal.flatten()
    arr_real = arr_real.flatten()

    # 绘制箱线图
    fig, ax = plt.subplots()
    boxplot = ax.boxplot([arr_cal, arr_real], labels=names)
    # 设置字体大小
    small_font = {'size': 8}

    # 标注箱线图的要点
    for i, box in enumerate(boxplot['boxes']):
        # 获取箱体的位置和大小
        x1, y1 = box.get_xdata()[0], box.get_ydata()[0]
        x2, y2 = box.get_xdata()[2], box.get_ydata()[2]
        q1 = y1
        q3 = y2
        median = boxplot['medians'][i].get_ydata()[0]
        whisker_min = boxplot['whiskers'][2 * i].get_ydata()[1]
        whisker_max = boxplot['whiskers'][2 * i + 1].get_ydata()[1]
        mid.append(median)

        ax.text(i + 1.1, q1, f'Q1: {q1:.6f}', fontdict=small_font)
        ax.text(i + 1.1, q3, f'Q3: {q3:.6f}', fontdict=small_font)
        ax.text(i + 1.1, median, f'Median: {median:.6f}', fontdict=small_font)
        ax.text(i + 1.1, whisker_min, f'Min: {whisker_min:.6f}', fontdict=small_font)
        ax.text(i + 1.1, whisker_max, f'Max: {whisker_max:.6f}', fontdict=small_font)

    title = f"{title} | Median difference: {abs(mid[0]-mid[1]):.6f}"
    if theoretical_value != None: title = f"{title}--The theoretical_value is {theoretical_value[idx]:.6f}"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    if show: plt.show()
    if save:
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(f"{os.path.join(save_path, get_current_date())}.png")
    
    
   
def boxplot_Compare_Stdval(arrs, std, shape = None, names = None, ylabels = None, title = None, save = False, save_path = "./", show = True):
    """箱型图，一张或多张图与标准值对比

    Args:
        arrs (_type_): _description_
        std (_type_): _description_
        shape (_type_, optional): _description_. Defaults to None.
        names (_type_, optional): _description_. Defaults to None.
        ylabels (_type_, optional): _description_. Defaults to None.
        title (_type_, optional): _description_. Defaults to None.
        save (bool, optional): _description_. Defaults to False.
        save_path (str, optional): _description_. Defaults to "./".
        show (bool, optional): _description_. Defaults to True.
    """
    cnt = len(arrs)
    
    if title == None: title = "Boxplot"
    if names == None: names = [f"Data {x}" for x in range(1, cnt + 1)]
    if ylabels == None: ylabels = [f"Value" for x in range(1, cnt + 1)]
    
    if cnt == 1:
        data = arrs[0].flatten()
        plt.boxplot(data, vert=True)

        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)

        # 在图上显示中位数、上下四分位数
        plt.text(1.1, median, f'Median: {median:.6f}', verticalalignment='center', color='r')
        plt.text(1.1, q1, f'Q1: {q1:.6f}', verticalalignment='center', color='g')
        plt.text(1.1, q3, f'Q3: {q3:.6f}', verticalalignment='center', color='g')

        # 隐藏 x 轴刻度
        plt.xticks([])
        plt.xlabel(f'{names[0]} \n Median:{median: .6f} | distance:{abs(std - median): .6f}')

    else:
        
        row = 1
        col = cnt
        weight = 6 * cnt
        height = 6
        
        if shape != None:
            row = shape[0]
            col = shape[1]
            weight = 6 * col 
            height = 8 * row 
            
        fig, ax = plt.subplots(row, col, figsize = (weight, height))
        
        if row == 1:
            for i in range(0, cnt):
                data = arrs[i].flatten()

                ax[i].boxplot(data, vert=True)

                median = np.median(data)
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)

                ax[i].text(1.1, median, f'Median: {median:.6f}', verticalalignment='center', color='r')
                ax[i].text(1.1, q1, f'Q1: {q1:.6f}', verticalalignment='center', color='g')
                ax[i].text(1.1, q3, f'Q3: {q3:.6f}', verticalalignment='center', color='g')

                ax[i].set_title(f'{names[i]} \n Median:{median: .6f} | distance:{abs(std - median): .6f}')

                ax[i].set_xticks([])
        else:
            i = 1
            for r in range(row):
                for c in range(col):
                    data = arrs[i].flatten()

                    ax[r][c].boxplot(data, vert=True)

                    median = np.median(data)
                    q1 = np.percentile(data, 25)
                    q3 = np.percentile(data, 75)

                    ax[r][c].text(1.1, median, f'Median: {median:.6f}', verticalalignment='center', color='r')
                    ax[r][c].text(1.1, q1, f'Q1: {q1:.6f}', verticalalignment='center', color='g')
                    ax[r][c].text(1.1, q3, f'Q3: {q3:.6f}', verticalalignment='center', color='g')

                    ax[r][c].set_title(f'{names[i]} \n Median:{median: .6f} | distance:{abs(std - median): .6f}')

                    ax[r][c].set_xticks([])
                    i += 1

    proTitle = f"{title} | std: {std}"
    if cnt == 1: plt.title(proTitle)
    else: fig.suptitle(proTitle)
    
    if save == True: 
        if not os.path.exists(save_path): os.makedirs(save_path)
        if cnt == 1: plt.savefig(f"{os.path.join(save_path, f'{title}-{get_current_date()}')}.png")
        else: fig.savefig(f"{os.path.join(save_path, f'{title}-{get_current_date()}')}.png")
    
    if show == True: plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Display one or more images
def show_2Dimages(images: np.ndarray, names: list = None, title: str = None, shape: tuple = None, axis = 'off'):
    """Display one or more 2D images, and the type of picture is recommended as numpy.ndarray.

    Args:
        images (np.ndarray):     Data of images.
        names (list, optional):  List of image names. Defaults to None.
        title (str, optional):   General title. Defaults to None.
        shape (tuple, optional): Canvas size. Defaults to None.
        axis (str, optional):    Whether each image displays axes. Defaults to 'off'.
    """
    cnt = len(images)
    if cnt == 1:
        plt.imshow(images[0], cmap = "gray")
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
        for i, image in enumerate(images):
            axes[i].imshow(image, cmap = "gray")
            if names != None:
                axes[i].set_title(names[i])
            axes[i].axis(axis)
        
        if title != None: fig.suptitle(title)
    plt.show()
    

# Box diagram showing one or more pictures
def showBoxPicOf2DImages(images: np.ndarray, names: list = None, xlabel: str = None, ylabel: str = None, title: str = None):
    """Box diagram showing one or more 2D images, and the type of picture is recommended as numpy.ndarray.

    Args:
        images (np.ndarray):    Data of images.
        names (list, optional): List of image names. Defaults to None.
        xlabel (str, optional): name of x axis. Defaults to None.
        ylabel (str, optional): name of y axis. Defaults to None.
        title (str, optional):  General title.  Defaults to None.
    """

    if names == None:
        names = [f"image{x}" for x in range(1, len(images) + 1)]
    if title == None:
        title = 'BoxPicOfTwoImage'
    if xlabel == None:
        xlabel = "images"
    if ylabel == None:
        ylabel = "value"
        
    images = [image.flatten() for image in enumerate(images)]

    fig, ax = plt.subplots()
    boxplot = ax.boxplot(images, labels=names)

    small_font = {'size': 5}

    for i, box in enumerate(boxplot['boxes']):

        x1, y1 = box.get_xdata()[0], box.get_ydata()[0]
        x2, y2 = box.get_xdata()[2], box.get_ydata()[2]
        q1 = y1
        q3 = y2
        median = boxplot['medians'][i].get_ydata()[0]
        whisker_min = boxplot['whiskers'][2 * i].get_ydata()[1]
        whisker_max = boxplot['whiskers'][2 * i + 1].get_ydata()[1]
       
        # print(f"{names[i]}: Max={whisker_max:.4f}, Min={whisker_min:.4f}, Q1={q1:.4f}, Median={median:.4f}, Q3={q3:.4f}")

        ax.text(i + 1.1, q1, f'Q1: {q1:.4f}', fontdict=small_font)
        ax.text(i + 1.1, q3, f'Q3: {q3:.4f}', fontdict=small_font)
        ax.text(i + 1.1, median, f'Median: {median:.4f}', fontdict=small_font)
        ax.text(i + 1.1, whisker_min, f'Min: {whisker_min:.4f}', fontdict=small_font)
        ax.text(i + 1.1, whisker_max, f'Max: {whisker_max:.4f}', fontdict=small_font)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    plt.show()
import cv2
import matplotlib.pyplot as plt
import numpy as np

def image_to_signature(image):
    x= image.shape[0]
    y= image.shape[1]
    signature=np.empty((image.size,3),dtype=np.float32)
    index=0
    for i in range(x):
        for j in range(y):
            signature[index]=np.array([image[i,j],i,j])
            index+=1
    return signature 

def emd(image1,image2):
    sum1=np.sum(image1)
    sum2=np.sum(image2)
    if sum1 ==0 or sum2 == 0 or np.isnan(sum1) or np.isnan(sum2):
        print('one of the images is all zeros or nan \n emd will be set to -1')
        return -1
    sig1=image_to_signature(image1)
    sig2=image_to_signature(image2)
    res=cv2.EMD(sig1,sig2,cv2.DIST_L2)
    return res[0]

def plot_3d_grid(title,plots, row_names, col_names, numbers,results_dir,save_name):
    fig = plt.figure(figsize=(10, 8),dpi=200)
    fig.suptitle(title)
    grid = plt.GridSpec(3, 4, wspace=0.4, hspace=0.3)
    for i in range(2):
        #writes row name
        ax = fig.add_subplot(grid[i+1, 0])
        ax.axis('off')
        ax.text(0.5, 0.5, row_names[i], fontsize=14)
        #writes emd of row
        ax = fig.add_subplot(grid[i+1, 3])
        ax.axis('off')
        ax.text(0.5, 0.5, numbers[i], fontsize=14)
    for j in range(3):
        #writes column name
        ax = fig.add_subplot(grid[0, j+1])
        ax.axis('off')
        ax.text(0.5, 0.5, col_names[j], fontsize=14)
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(grid[i+1, j+1], projection='3d')
            ax.plot_surface(*plots[2*i+j], cmap='viridis')
    plt.savefig(f"{results_dir}/{save_name}.jpg")
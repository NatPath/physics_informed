import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

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


def plot_4_report(u,y,z=9,ckpt_name='default_ckpt.pt',results_dir='default_dir_name'):
    results_dir=results_dir+f'/{ckpt_name[:-3]}'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    N,nx,ny,nz,u_nfields = u.shape
    y_nfields = y.shape[4]
    u = u.reshape(N,nx, ny, nz,2,u_nfields//2)
    y = y.reshape(N,nx, ny, nz,2,y_nfields//2)[...,-2:]
    u = (u[...,0,:] + 1j*u[...,1,:]).detach().numpy()
    y = (y[...,0,:] + 1j*y[...,1,:]).detach().numpy()

    dict = {0:"signal out", 1:"idler out"}
    maxXY = 120e-6
    XY = u.shape[1]
    xy = np.linspace(-maxXY, maxXY, XY + 1)[:-1]
    X,Y = np.meshgrid(xy,xy)
    fig, ax = plt.subplots(2,2,dpi=200,figsize=[10,10])

    for j in range(2):
        for i,sol,src in zip(range(2),[u,y],["Prediction", "Ground truth"]):
            # 
            plt.subplot(2,2,i+1+2*j)
            if j==0:
                ax[0][i].set_title(src)
            pic = np.mean(np.abs(sol[...,z,j])**2,axis=0) 
            pic = pic / np.sum(pic)

            # loc = range(coincidence_rate.shape[0])
            # ticks = [str(t)  for t in range(-max_mode2,max_mode2+1)]
            vmax = np.mean(np.abs(y[...,z,0])**2,axis=0) 
            vmax = np.max(vmax / np.sum(vmax))

            im = plt.imshow(pic,vmin=0,vmax=vmax) 
            # plt.xlabel(r'signal mode i')
            # plt.ylabel(r'idle mode j')
            # plt.xticks(loc,ticks)
            # plt.yticks(loc,ticks)
            plt.colorbar(im)

    ax[0][0].text(-3.5,1,"(a)",fontsize=14)
    ax[1][0].text(-3.5,1,"(b)",fontsize=14)

    plt.figure(fig)
    plt.savefig(f"{results_dir}/fields.jpg")
    plt.close('all')
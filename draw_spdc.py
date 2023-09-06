from argparse import ArgumentParser
import yaml

import torch
import numpy as np
from models import FNO3d
from train_utils.datasets import SPDCLoader
from train_utils.utils import save_checkpoint
from tqdm import tqdm
import torch.nn.functional as F
import gc
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import draw_utils 
import wandb


def draw_spdc_from_train(config,save_name,model,first_pump_dl,device,id,train_or_validate):
    fake_config={'data':config['data'],'test':{'ckpt':save_name}}
    draw_SPDC(model,first_pump_dl,fake_config,{},device,test_name=f'{train_or_validate}_first_pump_id_{id}',emd=False)
    prefix=f'draw_spdc_results/{train_or_validate}_first_pump_id_{id}/{save_name[:-3]}'
    idler_pred_image_loc=prefix+'/idler out-prediction.jpg'
    signal_pred_image_loc=prefix+'/signal out-prediction.jpg'
    idler_grt_image_loc=prefix+'/idler out-grt.jpg'
    signal_grt_image_loc=prefix+'/signal out-grt.jpg'
    wandb.log({f"idler_pred_{train_or_validate}ed_on":wandb.Image(idler_pred_image_loc)},commit=False)
    wandb.log({f"signal_pred_{train_or_validate}ed_on":wandb.Image(signal_pred_image_loc)},commit=False)
    wandb.log({f"idler_grt_{train_or_validate}ed_on":wandb.Image(idler_grt_image_loc)},commit=False)
    wandb.log({f"signal_grt_{train_or_validate}ed_on":wandb.Image(signal_grt_image_loc)},commit=False)
    for z in range(config['data']['nz']):
        results_together_i_loc=prefix+f'/all_results_together_z={z}.jpg'
        wandb.log({f"results_together z={z} {train_or_validate}ed on":wandb.Image(results_together_i_loc)},commit=False)

def plot_av_sol(u,y,z=9,ckpt_name='default_ckpt.pt',results_dir='default_dir_name',emd=True):
    # y = torch.ones_like(y)
    results_dir=results_dir+f'/{ckpt_name[:-3]}'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    N,nx,ny,nz,u_nfields = u.shape
    y_nfields = y.shape[4]
    u = u.reshape(N,nx, ny, nz,2,u_nfields//2)
    y = y.reshape(N,nx, ny, nz,2,y_nfields//2)[...,-2:]
    u = (u[...,0,:] + 1j*u[...,1,:]).detach().numpy()
    y = (y[...,0,:] + 1j*y[...,1,:]).detach().numpy()
    pics=[]
    dict = {0:"signal out", 1:"idler out"}
    maxXY = 120e-6
    XY = u.shape[1]
    xy = np.linspace(-maxXY, maxXY, XY + 1)[:-1]
    X,Y = np.meshgrid(xy,xy)
    
    for sol,src in zip([u,y],["prediction", "grt"]):
        for i in range(2):
            fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
            pic=np.mean(np.abs(sol[...,z,i])**2,axis=0) 
            sum_pic=np.sum(pic)
            if sum_pic == 0:
                pics.append(pic)
            else:
                pics.append(pic/sum_pic)
            surf = ax.plot_surface(X, Y, pic, cmap=cm.coolwarm,linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title(f"{dict[i]}-{src}")
            plt.savefig(f"{results_dir}/{dict[i]}-{src}.jpg")

    #calculate emd
    # EMD ==-1 means an uncalculable emd, emd==-2 means it wasn't calculated intentionally
    if emd:
        emd_signal=draw_utils.emd(pics[0],pics[2])
        emd_idler=draw_utils.emd(pics[1],pics[3])
    else:
        emd_signal=-2
        emd_idler=-2

    plots = [(X, Y, pics[0]), (X, Y, pics[2]), (X, Y, pics[1]), (X, Y, pics[3])]
    row_names = ['signal', 'idler']
    col_names = ['prediction', 'grt','emd']
    numbers = [emd_signal, emd_idler]
    title=f'{ckpt_name[:-3]} predicts {results_dir} on z={z}'
    save_name=f'all_results_together_z={z}'
    
    draw_utils.plot_3d_grid(title,plots, row_names, col_names, numbers,results_dir,save_name)

def plot_sol_with_phase(u,y,z=9,ckpt_name='default_ckpt.pt',results_dir='default_dir_name'):
    # y = torch.ones_like(y)
    results_dir=results_dir+f'/{ckpt_name[:-3]}'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    N,nx,ny,nz,u_nfields = u.shape
    y_nfields = y.shape[4]
    u = u.reshape(N,nx, ny, nz,2,u_nfields//2)
    y = y.reshape(N,nx, ny, nz,2,y_nfields//2)[...,-2:]
    u = (u[...,0,:] + 1j*u[...,1,:]).detach().numpy()
    y = (y[...,0,:] + 1j*y[...,1,:]).detach().numpy()


    
    fig, ax = plt.subplots(2,4,dpi=200)

    dict = {0:"signal out", 1:"idler out"}
    for i in [0,1]:
        for sol,src,j in zip([u,y],["pred", "grt"],[0,1]):
            plt.suptitle(f"z={z}")
            if N==1:
                    I = np.abs(sol[0,:,:,z,i])**2
                    phase = np.angle(sol[0,:,:,:,i]) 
            else:
                    I = np.mean(np.abs(sol[...,z,i])**2,axis=0)
                    phase = np.mean(np.angle(sol[...,z,i]),axis=0)
            Imin = np.min(I)
            Imax = np.max(I)

            im1 = ax[0][2*i+j].matshow(I,cmap='coolwarm',vmin=Imin,vmax=Imax)
            ax[0][2*i+j].set_title(f"{dict[i]}-{src}",fontdict={'fontsize':7})
            ax[0][2*i+j].set_xticks([])
            ax[0][2*i+j].set_yticks([])

            Pmin = np.min(phase)
            Pmax = np.max(phase)
            im2 = ax[1][2*i+j].imshow(phase,cmap='coolwarm',vmin=Pmin,vmax=Pmax)
            # ax[1][2*i+j].set_title("phase [rad]")
            ax[1][2*i+j].set_xticks([])
            ax[1][2*i+j].set_yticks([])

    # Create the first colorbar for the upper subplots
    fig.subplots_adjust(right=0.8)
    cbar_ax1 = fig.add_axes([0.85, 0.55, 0.05, 0.35])
    fig.colorbar(im1, cax=cbar_ax1)

    # Create the second colorbar for the lower subplots
    cbar_ax2 = fig.add_axes([0.85, 0.15, 0.05, 0.35])
    fig.colorbar(im2, cax=cbar_ax2)

    plt.savefig(f"{results_dir}/new_z={z}.jpg")
    plt.close('all')

def plot_av_sol_old(u,y,ckpt_name):
    # y = torch.ones_like(y)
    N,nx,ny,nz,u_nfields = u.shape
    y_nfields = y.shape[4]
    u = u.reshape(N,nx, ny, nz,2,u_nfields//2)
    y = y.reshape(N,nx, ny, nz,2,y_nfields//2)[...,-2:]
    u = (u[...,0,:] + 1j*u[...,1,:]).detach().numpy()
    y = (y[...,0,:] + 1j*y[...,1,:]).detach().numpy()
    pics=[]
    for sol,src in zip([u,y],["prediction", "grt"]):
        dict = {0:"signal out", 1:"idler out"}
        maxXY = 120e-6
        XY = u.shape[1]
        xy = np.linspace(-maxXY, maxXY, XY + 1)[:-1]
        X,Y = np.meshgrid(xy,xy)
        for i in range(2):
            fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
            pic=np.mean(np.abs(sol[...,-1,i])**2,axis=0) 
            pics.append(pic/np.sum(pic))
            surf = ax.plot_surface(X, Y, pic, cmap=cm.coolwarm,linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title(f"{dict[i]}-{src}")
            plt.savefig(f"tmp_fig/{ckpt_name}-{dict[i]}-{src}.jpg")

    #calculate emd
    emd_signal=draw_utils.emd(pics[0],pics[2])
    emd_idler=draw_utils.emd(pics[1],pics[3])
    print(f'emd signal is {emd_signal}')
    print(f'emd idler is {emd_idler}')

def plot_singel_sol(u,y,j,ckpt_name):

    N,nx,ny,nz,nfields = u.shape
    u = u.reshape(N,nx, ny, nz,2,nfields//2)
    y = y.reshape(N,nx, ny, nz,2,nfields//2)
    u = (u[...,0,:] + 1j*u[...,1,:]).detach().numpy()
    y = (y[...,0,:] + 1j*y[...,1,:]).detach().numpy()
    for sol,src in zip([u,y],["prediction", "grt"]):
        dict = {0:"signal vac", 1:"idler vac", 2:"single out", 3:"idler out"}
        maxXY = 120e-6
        XY = u.shape(1)
        xy = np.linspace(-maxXY, maxXY, XY + 1)[:-1]
        X,Y = np.meshgrid(xy,xy)
        for i in range(4):
            fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, np.real(sol[j,...,-1,i]), cmap=cm.coolwarm,linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title(f"{dict[i]}-{src}")
            plt.savefig(f"tmp_fig/{ckpt_name}-{dict[i]}-{src}-real.jpg")
            fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, np.imag(sol[j,...,-1,i]), cmap=cm.coolwarm,linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title(f"{dict[i]}-{src}")
            plt.savefig(f"tmp_fig/{ckpt_name}-{dict[i]}-{src}-imag.jpg")

def draw_SPDC(model,
                 dataloader,
                 config,
                 equation_dict,
                 device,
                 padding = 0,
                 use_tqdm=True,
                 test_name='tmp_test_name',
                 emd=True):
    model.eval()
    nout = config['data']['nout']
    ckpt_path=config['test']['ckpt']
    #ckpt_name=os.path.basename(ckpt_path) 
    ckpt_name=ckpt_path
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    total_out = torch.tensor([],device="cpu")
    total_y = torch.tensor([],device="cpu")
    for x, y in pbar:
        gc.collect()
        torch.cuda.empty_cache()
        x, y = x.to(device), y.to(device)
        x_in = F.pad(x,(0,0,0,padding),"constant",0)
        out = model(x_in).reshape(y.size(0),y.size(1),y.size(2),y.size(3) + padding, 2*nout)
        # out, y, x = out.to("cpu"), y.to("cpu"), x.to("cpu")
            # out = out[...,:-padding,:, :] # if padding is not 0
        total_out = torch.cat((total_out,out.detach().cpu()),dim=0)
        total_y = torch.cat((total_y,y.detach().cpu()),dim=0)

    script_dir=os.path.dirname(__file__)
    results_dir_name=f'draw_spdc_results/{test_name}'
    results_dir=os.path.join(script_dir,results_dir_name)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    for z in range(config['data']['nz']):
        plot_sol_with_phase(total_out,total_y,z,ckpt_name,results_dir)
        plot_av_sol(total_out,total_y,z,ckpt_name,results_dir,emd)



def run(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_config = config['data']
    dataset = SPDCLoader(   datapath = data_config['datapath'],
                            nx=data_config['nx'], 
                            ny=data_config['ny'],
                            nz=data_config['nz'],
                            nin = data_config['nin'],
                            nout = data_config['nout'],
                            sub_xy=data_config['sub_xy'],
                            sub_z=data_config['sub_z'],
                            N=data_config['total_num'],
                            device=device)
    
    equation_dict = dataset.data_dict
    dataloader = dataset.make_loader(n_sample=data_config['n_sample'],
                                     batch_size=config['test']['batchsize'],
                                     start=data_config['offset'])
    del dataset
    gc.collect()
    torch.cuda.empty_cache()

    model = FNO3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  in_dim=config['model']['in_dim'],
                  out_dim=config['model']['out_dim'],
                  activation_func=config['model']['act']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    
    if 'test_name' in config['test']:
        test_name=config['test']['test_name']
    else:
        test_name='tmp_test_name'
    
    draw_SPDC(model=model,dataloader=dataloader, config=config, equation_dict=equation_dict, device=device,test_name=test_name,emd=not args.emd_off)


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--emd_off', action='store_true', help='Turn off the EMD calculation')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
        run(args, config)

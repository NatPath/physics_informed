from argparse import ArgumentParser
import yaml

import torch
import numpy as np
from models import FNO3d
from train_utils.datasets import SPDCLoader
from tqdm import tqdm
import torch.nn.functional as F
import gc
from time import process_time

# For n_sample = 1000 (with tqdm)
# Took 95.31417 seconds
# Took in avarge 0.09531 seconds for sample

# For n_sample = 1000 (without tqdm)
# Took 94.75268 seconds
# Took in avarge 0.09475 seconds for sample


def time_eval_SPDC(model,
                 dataloader,
                 config,
                 device,
                 padding = 0,
                 use_tqdm=True,
                 test_name='tmp_test_name'):
    model.eval()
    nout = config['data']['nout']
    ckpt_path=config['test']['ckpt']
    ckpt_name=ckpt_path
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    start_time = process_time()

    for x, y in pbar:
        gc.collect()
        torch.cuda.empty_cache()
        x, y = x.to(device), y.to(device)
        x_in = F.pad(x,(0,0,0,padding),"constant",0)
        out = model(x_in).reshape(y.size(0),y.size(1),y.size(2),y.size(3) + padding, 2*nout)

    end_time = process_time()
    duration_time = end_time - start_time
    print(f'Took {duration_time:.5f} seconds')

    n_samples = config["data"]["n_sample"]
    avarge_time = duration_time / n_samples
    print(f'Took in avarge {avarge_time:.5f} seconds for sample')





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
    
    time_eval_SPDC(model=model,dataloader=dataloader, config=config, device=device,test_name=test_name)


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
        run(args, config)
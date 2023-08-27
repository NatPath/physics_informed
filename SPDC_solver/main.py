from SPDC_solver import *
import numpy as np
from argparse import ArgumentParser
import pickle
from tqdm import tqdm

seed = 1702
np.random.seed(seed)
config = Config(pump_waist=80e-6)

max_mode1 = 3
max_mode2 = 3
total_modes = max_mode1*(2*max_mode2+1)
coeff = np.random.rand(2,total_modes)
pump_coef = {"max_mode1": max_mode1, 
"max_mode2":max_mode2, 
"real_coef":coeff[0],
"img_coef":coeff[1]}
A = SPDC_solver(N=100,config=config,pump_coef=pump_coef,data_creation=True,draw_sol=True)
A.solve()

print("Done!")
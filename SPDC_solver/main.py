from create_data import fixed_pump_single_mode
from SPDC_solver import Config
import numpy as np


config = Config(pump_waist=80e-6) # 40e-6, 60e-6, 100e-6, 120e-6
crystal_coef = {"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])}

for l in range(5):
    fixed_pump_single_mode(N_samples=500,crystal_coef=crystal_coef,is_crystal=False, config=config, p=0, l=l, loc=None, seed=123)


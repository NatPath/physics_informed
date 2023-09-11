from SPDC_solver import *
import numpy as np
from argparse import ArgumentParser
import pickle
from tqdm import tqdm

def save_data(data,file_name):
    print(f"saving data to: {file_name}")
    with open(file_name, "wb") as file:
        pickle.dump(obj=data,file=file,protocol=4)
    print("Done!")

config = Config()
p = 0
l = 3
max_mode1 = p + 1
max_mode2 = abs(l)
coeffs_arr = np.zeros(max_mode1*(2*max_mode2 + 1))
idx = p * (2*max_mode2 + 1) + l + max_mode2
coeffs_arr[idx] = 1

crystal_coef = {"max_mode1": max_mode1, "max_mode2":max_mode2, "real_coef":coeffs_arr.real,"img_coef":coeffs_arr.imag}
A = SPDC_solver(N=100,config=config,crystal_coef=crystal_coef,is_crystal=True,data_creation=True,draw_sol=False)
A.solve()

save_data(A.data,"crystal_data.bin")



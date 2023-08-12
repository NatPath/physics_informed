from SPDC_solver import *
import numpy as np
from argparse import ArgumentParser
import pickle
from tqdm import tqdm

N_samples = 10
seed = 1701
fixed_pump = True
spp = 1
config = Config(pump_waist=80e-6)

if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('-N', type=int, help='Number of samples that will be created')
    parser.add_argument('--seed', type=int, help='Seed for the random pump profile (if needed)')
    parser.add_argument('--loc', type=str, help='Location to save the file, if not specifed save at a deafult location')
    parser.add_argument('--change_pump', action='store_true', help='Creates different pump profiles')
    parser.add_argument('--spp', type=int, help='Number of samples that will be created for each pump (Only if "chang_pump" is in use)')

    args = parser.parse_args()
    N_samples = args.N
    seed = args.seed
    fixed_pump = not args.change_pump
    loc = args.loc
    spp = args.spp


defult_loc = "/home/dor-hay.sha/project/data/spdc/"
# creating data with fixed pump
if fixed_pump:
    if loc is not None:
        file_name = str(f"{loc}/fixed_pump_{N_samples}.bin")
    else:
        file_name = str(f"{defult_loc}/fixed_pump_{N_samples}.bin")

    print("creating data")
    A = SPDC_solver(N=N_samples,config=config,data_creation=True)
    A.solve()

# creating data with changin pump
else:
    if loc is not None:
        file_name = str(f"{loc}/random_pump_{N_samples}.bin")
    else:
        file_name = str(f"{defult_loc}/random_pump_{N_samples}.bin")
    
    print("creating data")
    np.random.seed(seed)
    for n in tqdm(range(N_samples//spp)):
        max_mode1 = 20
        max_mode2 = 20
        total_modes = max_mode1*(2*max_mode2+1)
        coeff = np.random.rand(2,total_modes)
        pump_coef = {"max_mode1": max_mode1, 
        "max_mode2":max_mode2, 
        "real_coef":coeff[0],
        "img_coef":coeff[1]}
        A = SPDC_solver(N=spp,config=config,pump_coef=pump_coef,data_creation=True)
        A.solve()
        if n==0:
            fields = A.data["fields"]
        else: 
            fields = np.append(fields,A.data["fields"],axis=0)
    data = A.data
    data["fields"] = fields




print("saving data")
with open(file_name, "wb") as file:
    pickle.dump(obj=A.data,file=file,protocol=4)
print("Done!")
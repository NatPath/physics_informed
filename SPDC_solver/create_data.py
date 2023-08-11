from SPDC_solver import *
import numpy as np
from argparse import ArgumentParser
import pickle

N_samples = 10
seed = 1701
fixed_pump = True
config = Config(pump_waist=80e-6)

if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('-N', type=int, help='Number of samples that will be created')
    parser.add_argument('--seed', type=int, help='Seed for the random pump profile (if needed)')
    parser.add_argument('--loc', type=str, help='Location to save the file, if not specifed save at a deafult location')
    parser.add_argument('--change_pump', action='store_true', help='Creates different pump profiles')
    args = parser.parse_args()
    N_samples = args.N
    seed = args.seed
    fixed_pump = not args.change_pump
    loc = args.loc


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
    for n in range(N_samples):

        key = random.PRNGKey(seed)
        rand_key, subkey = random.split(key)
        self.vacuum_states = random.normal(subkey,shape=(N,2,2,shape.Nx,shape.Ny))





print("saving data")
with open(file_name, "wb") as file:
    pickle.dump(obj=A.data,file=file,protocol=4)
print("Done!")
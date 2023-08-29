from SPDC_solver import *
import numpy as np
from argparse import ArgumentParser
import pickle
from tqdm import tqdm

N_samples = 10
seed = 1701
fixed_pump = True
config = Config(pump_waist=80e-6)

if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('-N', type=int, help='Number of samples that will be created')
    parser.add_argument('--seed', type=int, help='Seed for the random pump profile (if needed)')
    parser.add_argument('--loc', type=str, help='Location to save the file, if not specifed save at a deafult location')
    parser.add_argument('--random_pump', action='store_true', help='Creates random pump profiles')
    parser.add_argument('--spp', type=int, help='Number of samples that will be created for each pump (Only if "chang_pump" is in use)')

    args = parser.parse_args()
    N_samples = args.N
    seed = args.seed
    fixed_pump = not args.random_pump
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
        file_name = str(f"{loc}/random_pump_N-{N_samples}_spp-{spp}.bin")
    else:
        file_name = str(f"{defult_loc}/random_pump_N-{N_samples}_spp-{spp}.bin")
    
    print("creating data")
    max_mode = 5
    np.random.seed(seed)
    rng = np.random.default_rng()
    lam = 1.
    poisson = lambda k: lam**k * np.e**(-lam) / np.math.factorial(k)

    p = list(range(max_mode))
    l = list(range(-max_mode,max_mode+1))
    Pr = np.array([poisson(np.abs(i)+j) for i in l for j in p])

    for n in tqdm(range(N_samples//spp)):
        no_coeffs = True
        while no_coeffs:
            Pu = 1 - rng.uniform(low=0.0, high=1.0, size=len(l)*len(p)) # in (0, 1]
            coeffs_arr = Pu * np.where(Pu <= Pr, 1., 0.) 
            no_coeffs = coeffs_arr.sum() == 0
            coeffs_arr = coeffs_arr * np.exp(1j*rng.uniform(low=0.0, high=2*np.pi, size=len(coeffs_arr)))

        pump_coef = {"max_mode1": max_mode, "max_mode2":max_mode, "real_coef":coeffs_arr.real,"img_coef":coeffs_arr.imag}


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
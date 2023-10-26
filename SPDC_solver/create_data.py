from SPDC_solver import *
import numpy as np
from argparse import ArgumentParser
import pickle
from tqdm import tqdm


def single_mode(p,l,config,N_samples,seed,crystal_coef={"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])},is_crystal=False):
    '''
    creates a solution of a single mode of pump given by p,l
    Args:
        p,l - parmeters of the mode
        config - config of the pump
        N_samples - number of vac modes
        seed - seed for random vacum modes
    Return:
        A.data - The solution tensor
    '''
    max_mode1 = p + 1
    max_mode2 = abs(l)
    coeffs_arr = np.zeros(max_mode1*(2*max_mode2 + 1))
    idx = p * (2*max_mode2 + 1) + l + max_mode2
    coeffs_arr[idx] = 1

    pump_coef = {"max_mode1": max_mode1, "max_mode2":max_mode2, "real_coef":coeffs_arr.real,"img_coef":coeffs_arr.imag}
    A = SPDC_solver(N=N_samples,crystal_coef=crystal_coef,is_crystal=is_crystal,config=config,pump_coef=pump_coef,data_creation=True, seed=seed)
    A.solve()
    return A.data

def multi_modes(config,N_samples,spp,seed):
    '''
    creates a solution of a single mode with spp vac modes for N_samples//spp modes
    Args:
        config - config of the pump
        N_samples - number of samples in the data
        spp - number of vac modes per pump mode
        seed - seed for random vacum modes
    Return:
        A.data - The solution tensor
    '''
    number_of_modes = N_samples // spp
    # iterates over the modes in diaganol way and concatnaing the different modes
    idx = 0
    col,row,diag = 0,0,0

    while idx < number_of_modes:
        while (row >= 0) and (idx < number_of_modes):
            data = single_mode(p=row, l=col, config=config ,N_samples=spp, seed=(seed+idx))
            if idx==0:
                fields = data["fields"]
            else: 
                fields = np.append(fields,data["fields"],axis=0)
            row -= 1
            col += 1
            idx += 1
        diag += 1
        row = diag
        col = 0

    data["fields"] = fields
    return data

def save_data(data,file_name):
    print(f"saving data to: {file_name}")
    with open(file_name, "wb") as file:
        pickle.dump(obj=data,file=file,protocol=4)
    print("Done!")

    
def fixed_pump(N_samples, config ,spp = None, loc = None, seed = 1701,crystal_coef={"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])},is_crystal=False):

    default_loc = "/home/dor-hay.sha/project/data/spdc"
    file_name = str(f"fixed_pump_N-{N_samples}_seed-{seed}")
    if loc is not None:
        file_name = str(f"{loc}/{file_name}")
    else:
        file_name = str(f"{default_loc}/{file_name}")

    if spp is None:
        print("Creating data: only one mode, p=0, l=0")
        file_name = file_name + ".bin"
        A = SPDC_solver(N=N_samples,crystal_coef=crystal_coef,is_crystal=is_crystal,config=config,data_creation=True,seed=seed)
        A.solve()
        data = A.data

    elif spp is not None:
        file_name = str(f"{file_name}_spp-{spp}.bin")

        if not (N_samples == (N_samples // spp) * spp):
            raise Exception("Error! (N /spp) is not a round number")
        
        print(f"Creating data: The first {(N_samples // spp)} LG modes")
        data = multi_modes(config = config, N_samples = N_samples, spp = spp, seed = seed)

    save_data(data,file_name)

def fixed_pump_single_mode(N_samples, config ,p,l, loc = None, seed = 1701,crystal_coef={"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])},is_crystal=False):
    default_loc = "/home/dor-hay.sha/project/data/spdc"
    file_name = str(f"single_mode-({p},{l})_N-{N_samples}_seed-{seed}.bin")
    if loc is not None:
        file_name = str(f"{loc}/{file_name}")
    else:
        file_name = str(f"{default_loc}/{file_name}")

    print(f"Creating data: only a single mode, p={p}, l={l}")
    data = single_mode(p = p,l = l,config = config,N_samples = N_samples, seed = seed)

    save_data(data,file_name)


def random_pump(N_samples, config ,spp = 1, max_mode = 5, seed = 1701, loc = None,crystal_coef={"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])},is_crystal=False):

    if not (N_samples == (N_samples // spp) * spp):
        raise Exception("Error! (N /spp) is not a round number")

    default_loc = "/home/dor-hay.sha/project/data/spdc"
    if loc is not None:
        file_name = str(f"{loc}/random_pump_N-{N_samples}_seed-{seed}_spp-{spp}.bin")
    else:
        file_name = str(f"{default_loc}/random_pump_N-{N_samples}_seed-{seed}_spp-{spp}.bin")
    
    print(f"creating data: {(N_samples // spp)} random pump modes with max_mode = {max_mode} and seed = {seed}")

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

        A = SPDC_solver(N=spp,config=config,crystal_coef=crystal_coef,is_crystal=is_crystal,pump_coef=pump_coef,data_creation=True, seed = (seed + n))
        A.solve()
        if n==0:
            fields = A.data["fields"]
        else: 
            fields = np.append(fields,A.data["fields"],axis=0)
    data = A.data
    data["fields"] = fields

    save_data(data,file_name)


def uniform_pump(N_samples, config ,spp = 1, max_mode = 10, seed = 1701, loc = None,crystal_coef={"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])},is_crystal=False):

    if not (N_samples == (N_samples // spp) * spp):
        raise Exception("Error! (N /spp) is not a round number")

    default_loc = "/home/dor-hay.sha/project/data/spdc"
    if loc is not None:
        file_name = str(f"{loc}/uniform_pump_N-{N_samples}_seed-{seed}_spp-{spp}.bin")
    else:
        file_name = str(f"{default_loc}/uniform_pump_N-{N_samples}_seed-{seed}_spp-{spp}.bin")
    
    print(f"creating data: {(N_samples // spp)} uniform pump modes with max_mode = {max_mode} and seed = {seed}")

    np.random.seed(seed)
    rng = np.random.default_rng()

    p = list(range(max_mode))
    l = list(range(-max_mode,max_mode+1))

    for n in tqdm(range(N_samples//spp)):
        coeffs_arr = rng.uniform(low=0.0, high=1.0, size=len(l)*len(p)) # in (0, 1]
        coeffs_arr = coeffs_arr * np.exp(1j*rng.uniform(low=0.0, high=2*np.pi, size=len(coeffs_arr)))
        pump_coef = {"max_mode1": max_mode, "max_mode2":max_mode, "real_coef":coeffs_arr.real,"img_coef":coeffs_arr.imag}

        A = SPDC_solver(N=spp,config=config,crystal_coef=crystal_coef,is_crystal=is_crystal,pump_coef=pump_coef,data_creation=True, seed = (seed + n))
        A.solve()
        if n==0:
            fields = A.data["fields"]
        else: 
            fields = np.append(fields,A.data["fields"],axis=0)
    data = A.data
    data["fields"] = fields

    save_data(data,file_name)





if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('-N', type=int, default=10, help='Number of samples that will be created (default N=10)')
    parser.add_argument('--seed', type=int, default=1701, help='Seed for the random pump profile and the vacum states (default seed = 1701)')
    parser.add_argument('--loc', type=str, help='Location to save the file, if not specifed save at a deafult location')
    parser.add_argument('--mode', type=str, default="fixed", help='Pick mode of data creation from the following:\nfixed - create pure fixed LG modes for pump.\nsingle - create pure fixed single LG modes for pump according to given p,l.\nrandom - create mixed random pump modes.\nuniform - create a combination of all modes up to max mode with uniformly distribuited range [0,1] and phase')
    parser.add_argument('--spp', type=int,help='Number of samples that will be created for each pump')
    parser.add_argument('-p', type=int, default=0, help = 'p number if creation mode is \'single\' (default p = 0)')
    parser.add_argument('-l', type=int, default=0, help = 'l number if creation mode is \'single\' (default l = 0)')
    parser.add_argument('--crystal', action='store_true',default=False, help = 'Specify to create a custome crystal')

    args = parser.parse_args()
    config = Config(pump_waist=80e-6) # 40e-6, 60e-6, 100e-6, 120e-6

    if args.crystal==True:
        
        p = 0
        l = 3
        max_mode1 = p + 1
        max_mode2 = abs(l)
        coeffs_arr = np.zeros(max_mode1*(2*max_mode2 + 1))
        idx = p * (2*max_mode2 + 1) + l + max_mode2
        coeffs_arr[idx] = 1

        crystal_coef = {"max_mode1": max_mode1, "max_mode2":max_mode2, "real_coef":coeffs_arr.real,"img_coef":coeffs_arr.imag}
        is_crystal=True
    else:
        crystal_coef = {"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])}
        is_crystal=False


    if args.mode == 'fixed':
        fixed_pump(N_samples=args.N,crystal_coef=crystal_coef,is_crystal=is_crystal, config=config, spp=args.spp, loc=args.loc, seed=args.seed)
    
    elif args.mode == 'single':
        fixed_pump_single_mode(N_samples=args.N,crystal_coef=crystal_coef,is_crystal=is_crystal, config=config, p=args.p, l=args.l, loc=args.loc, seed=args.seed)

    elif args.mode == 'random':
        random_pump(N_samples=args.N ,crystal_coef=crystal_coef,is_crystal=is_crystal, config=config, spp=args.spp, seed=args.seed, loc=args.loc)

    elif args.mode == 'uniform':
        uniform_pump(N_samples=args.N ,crystal_coef=crystal_coef,is_crystal=is_crystal, config=config, spp=args.spp, seed=args.seed, loc=args.loc)
    
    else:
        raise Exception(f"Error! \'{args.mode}\' is not a valid creation mode'")
    
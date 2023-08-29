from SPDC_solver import *
import numpy as np
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm


config = Config(pump_waist=80e-6)
A = SPDC_solver(N=500,config=config,data_creation=True)
A.solve()
dict = ["pump","signal_vac", "idler_vac", "signal_out", "idler_out"]
X,Y = np.meshgrid(A.shape.x,A.shape.y, indexing='ij')
for z in range(10):
    for i in range(5):
            if True:
                        fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
                        surf = ax.plot_surface(X, Y, np.mean(np.abs(A.data["fields"][:,i,:,:,z])**2,axis=0), cmap=cm.coolwarm,linewidth=0, antialiased=False)
                        fig.colorbar(surf, shrink=0.5, aspect=5)
                        plt.title(f"{dict[i]}")
                        plt.savefig(f"{dict[i]}_z={z}.jpg")
                        plt.close()

print("Done!")
exit()

np.random.seed(1701)
max_mode = 5
rng = np.random.default_rng()
lam = 1.
poisson = lambda k: lam**k * np.e**(-lam) / np.math.factorial(k)

p = list(range(max_mode))
l = list(range(-max_mode,max_mode+1))

Pu = 1 - rng.uniform(low=0.0, high=1.0, size=len(l)*len(p)) # in (0, 1]
Pr = np.array([poisson(np.abs(i)+j) for i in l for j in p])
modes = [f'mode {i} {j}' for i in l for j in p]
no_coeffs = True
coeffs_arr = Pu * np.where(Pu <= Pr, 1., 0.) 
coeffs_arr = coeffs_arr * np.exp(1j*rng.uniform(low=0.0, high=2*np.pi, size=len(coeffs_arr)))
# no_coeffs = coeffs_arr.sum() == 0
coeffs = {mode: coeff for mode, coeff in zip(modes, coeffs_arr)}
print(coeffs)
print(coeffs_arr)
# print("Pu:",Pu)
# print("Pr:",Pr)

# seed = 1702
# np.random.seed(seed)
config = Config(pump_waist=80e-6)

# max_mode1 = 3
# max_mode2 = 3
# total_modes = max_mode1*(2*max_mode2+1)
# coeff = np.random.rand(2,total_modes)
pump_coef = {"max_mode1": max_mode, "max_mode2":max_mode, "real_coef":coeffs_arr.real,"img_coef":coeffs_arr.imag}
A = SPDC_solver(N=500,config=config,pump_coef=pump_coef,data_creation=True,draw_sol=True)
A.solve()

print("Done!")
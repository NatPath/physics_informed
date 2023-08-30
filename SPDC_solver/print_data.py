from utils import Shape
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle

datapath = "single_mode-(1,2)_N-10" + ".bin"
N = 1
spp = 1

with open(file=datapath,mode="rb") as file:
    data = pickle.load(file)

dict = ["pump","signal_vac", "idler_vac", "signal_out", "idler_out"]
shape = Shape()

X,Y = np.meshgrid(shape.x,shape.y, indexing='ij')
for z in range(9,10):
    for i in range(1):
            if True:
                        fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
                        if spp==1:
                               I = np.abs(data["fields"][N,i,:,:,z])**2
                        else:
                               I = np.mean(np.abs(data["fields"][N:N+spp,i,:,:,z])**2,axis=0)
                               
                        surf = ax.plot_surface(X, Y, I, cmap=cm.coolwarm,linewidth=0, antialiased=False)
                        fig.colorbar(surf, shrink=0.5, aspect=5)
                        plt.title(f"{dict[i]}")
                        plt.savefig(f"{dict[i]}_z={z}.jpg")
                        plt.close()

print("Done!")
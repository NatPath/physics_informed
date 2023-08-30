from SPDC_solver import *
import numpy as np
import pickle

datapath = "./"

with open(file=datapath,mode="rb") as file:
    data = pickle.load(file)

dict = ["pump","signal_vac", "idler_vac", "signal_out", "idler_out"]
X,Y = np.meshgrid(A.shape.x,A.shape.y, indexing='ij')
for z in range(9,10):
    for i in range(1):
            if True:
                        fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
                        surf = ax.plot_surface(X, Y, np.mean(np.abs(A.data["fields"][:,i,:,:,z])**2,axis=0), cmap=cm.coolwarm,linewidth=0, antialiased=False)
                        fig.colorbar(surf, shrink=0.5, aspect=5)
                        plt.title(f"{dict[i]}")
                        plt.savefig(f"{dict[i]}_z={z}.jpg")
                        plt.close()

print("Done!")
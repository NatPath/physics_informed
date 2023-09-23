from utils import Shape
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.express as px
import plotly
import pickle

datapath = "/home/dor-hay.sha/project/data/spdc/uniform_pump_N-5_seed-1701_spp-1.bin"
# datapath = "./random_pump_N-12_spp-4.bin"
N = 0
spp = 1
z0 = 9

with open(file=datapath,mode="rb") as file:
    data = pickle.load(file)

dict = ["pump","signal_vac", "idler_vac", "signal_out", "idler_out"]
shape = Shape()

X,Y = np.meshgrid(shape.x,shape.y, indexing='ij')
for z in range(z0,10):
    for i in range(1):
         if "vac" not in dict[i]:
            if True:
                     fig, (ax1,ax2) = plt.subplots(1,2)
                     plt.suptitle(f"{dict[i]}_z={z}")

                     if spp==1:
                              I = np.abs(data["fields"][N,i,:,:,z])**2
                              phase = np.angle(data["fields"][N,i,:,:,z]) 
                     else:
                              I = np.mean(np.abs(data["fields"][N:N+spp,i,:,:,z])**2,axis=0)
                              phase = np.mean(np.angle(data["fields"][N:N+spp,i,:,:,z]),axis=0)
                              
                     im1 = ax1.matshow(I,cmap='coolwarm')
                     ax1.set_title("Intensity")
                     fig.colorbar(im1, ax=ax1)
                     ax1.set_xticks([])
                     ax1.set_yticks([])

                     im2 = ax2.imshow(phase,cmap='coolwarm')
                     ax2.set_title("phase [rad]")
                     fig.colorbar(im2, ax=ax2)
                     ax2.set_xticks([])
                     ax2.set_yticks([])

                     fig.tight_layout()
                     plt.savefig(f"{dict[i]}_z={z}.jpg")
                     plt.close()

print("Done!")
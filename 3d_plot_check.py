import matplotlib.pyplot as plt
import numpy as np

def plot_3d_grid(title,plots, row_names, col_names, numbers):
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(title)
    grid = plt.GridSpec(3, 4, wspace=0.4, hspace=0.3)
    for i in range(2):
        #writes row name
        ax = fig.add_subplot(grid[i+1, 0])
        ax.axis('off')
        ax.text(0.5, 0.5, row_names[i], fontsize=14)
        #writes emd of row
        ax = fig.add_subplot(grid[i+1, 3])
        ax.axis('off')
        ax.text(0.5, 0.5, numbers[i], fontsize=14)
    for j in range(3):
        #writes column name
        ax = fig.add_subplot(grid[0, j+1])
        ax.axis('off')
        ax.text(0.5, 0.5, col_names[j], fontsize=14)
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(grid[i+1, j+1], projection='3d')
            ax.plot_surface(*plots[2*i+j], cmap='viridis')
    plt.show()

x = y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z1 = np.sin(np.sqrt(X**2 + Y**2))
Z2 = np.cos(np.sqrt(X**2 + Y**2))
Z3 = np.tan(np.sqrt(X**2 + Y**2))
Z4 = np.exp(-np.sqrt(X**2 + Y**2))

plots = [(X, Y, Z1), (X, Y, Z2), (X, Y, Z3), (X, Y, Z4)]
row_names = ['signal', 'idler']
col_names = ['grt', 'prediction','emd']
numbers = [1.23, 4.56]
title='title'

# Call the function
plot_3d_grid(title,plots, row_names, col_names, numbers)
import numpy as np
import matplotlib.pyplot as plt
import findiff
import torch

import derivative as d
from utils import knn

N = 64

########################################################################################
x, y = np.meshgrid(np.linspace(0+0.5/N, 1-0.5/N, N), np.linspace(0+0.5/N, 1-0.5/N, N))
# Define a function to evaluate
u = np.load("data/example_64/example_64_data.npy")
u = u[50]
#u = np.sin(2*np.pi*np.power(x+y,2)) #*np.cos(2*np.pi*y)
Dxy = findiff.FinDiff((0, 1/N, 1), (1, 1/N, 1))
Dx = findiff.FinDiff(0, 1/N, 1)
Dy = findiff.FinDiff(1, 1/N, 1)
Dxx = findiff.FinDiff(0, 1/N, 2)
Dyy = findiff.FinDiff(1, 1/N, 2)
# Derivatives
du_dxy = Dxy(u)
du_dx = Dx(u)
du_dy = Dy(u)
du_dxx = Dxx(u)
du_dyy = Dyy(u)
########################################################################################

# Points and Values
#x, y = np.meshgrid(np.linspace(0+0.5/N, 1-0.5/N, N), np.linspace(0+0.5/N, 1-0.5/N, N))
x, y = np.load("data/example_64/example_64_points.npy")
flat_points = np.hstack([x.reshape(-1, 1), y.reshape(-1,1)])
flatpoints = torch.from_numpy(flat_points)
values = torch.from_numpy(u).flatten()

# KNN - Algorithm
k = 20
num_d = 5
#num_k = torch.randint(low=1, high=10, size=(k, 1)).flatten()
dist, edge_index = knn.knn_torch(points_from=flatpoints, points_to=flatpoints, k_neighbors=k, spatial_dimension=2)

# select the neighbors vertices
neighbors = torch.index_select(flatpoints, 0, edge_index)
neighbors = torch.reshape(neighbors, (flatpoints.size()[0], k, 2))
neighbors_values = torch.index_select(values, 0, edge_index)
neighbors_values = torch.reshape(neighbors_values, (flatpoints.size()[0], k))

# Derivatives
derivatives = d.solveDerivatives(vertices=flatpoints,
                                 vertices_values=values,
                                 neighbors=neighbors,
                                 neighbors_values=neighbors_values,
                                 knn=k,
                                 num_derivatives=5,
                                 order=2)
derivatives = torch.transpose(derivatives, 0, 1)
derivatives = torch.reshape(derivatives, (num_d, N, N))
derivatives = derivatives.numpy()

fig, ax = plt.subplots(2,6)
ax[0, 0].imshow(u)
ax[0, 1].imshow(du_dy)
ax[0, 2].imshow(du_dyy)
ax[0, 3].imshow(du_dx)
ax[0, 4].imshow(du_dxy)
ax[0, 5].imshow(du_dxx)
ax[1, 0].imshow(u)
ax[1, 1].imshow(derivatives[0]) #dx
ax[1, 2].imshow(derivatives[1]) #dyy
ax[1, 3].imshow(derivatives[2]) #dx
ax[1, 4].imshow(derivatives[3]) #dxy
ax[1, 5].imshow(derivatives[4]) #dxx


plt.show()
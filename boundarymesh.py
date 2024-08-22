import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

# Create a simple mesh
mesh = fd.UnitSquareMesh(10, 10)

# Function space to represent the solution (P1 space is usually enough)
V = fd.FunctionSpace(mesh, "CG", 1)
all_node_coords = mesh.coordinates.dat.data
boundary_nodes = fd.DirichletBC(V, 0, "on_boundary").nodes
boundary_node_coords = all_node_coords[boundary_nodes]

distances = np.zeros(all_node_coords.shape[0])

for i, coord in enumerate(all_node_coords):
    distances[i] = np.min(np.linalg.norm(boundary_node_coords - coord, axis=1))

distance_function = fd.Function(V)
distance_function.dat.data[:] = distances

plt.plot(distances, np.zeros_like(distances), "o")
plt.show()
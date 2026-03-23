import jax.numpy as jnp
import jax
# jax.config.update('jax_enable_x64', True)
jax.config.update("jax_debug_nans", True)
import sys
sys.path.append('../../../..')  
from jax_fvm.src.mesh.mesh import Mesh # pyright: ignore[reportMissingImports]
import jax_fvm.src.mesh.plot as plot # pyright: ignore[reportMissingImports]
import jax_fvm.src.Cases.Test_Cases as Test_Cases # pyright: ignore[reportMissingImports]
import jax_fvm.src.mesh.Mesh_cases as Mesh_cases # pyright: ignore[reportMissingImports]
import time
import jax_fvm.src.solvers.helper as helper # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt
import numpy as np
size = 14
params = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'cm',  # Computer Modern font
	'legend.fontsize':size,
    'axes.labelsize' : size,
	'axes.titlesize' : size +2,
    'xtick.labelsize' : size+1,
    'ytick.labelsize' : size+1
}
plt.rcParams.update(params)
###########################################################################################################
#############################               gradient                 ######################################
###########################################################################################################

def getgradientLSQ(W_L, W_R, mesh):
	Delta_x = mesh.barycenter[mesh.neighbors] - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	
	replace = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2)
	replace = 2 * (replace - mesh.barycenter[...,None,:]) # trick in case the face is on the boundary = use face midpoint instead of neighbor cell center
	
	Delta_x = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 0)[...,None], 2, axis=-1), replace, Delta_x)

	Delta_w = W_R - W_L
	weights = 1 / jnp.linalg.norm(Delta_x, axis = -1)**2  # (N_cells, 3)

	A = jnp.einsum('ijk,ijl->ikl', weights[...,None] * Delta_x, Delta_x)  # (N_cells, 2, 2)

	b = jnp.einsum('ijk,ijl->ikl',  weights[...,None] * Delta_w, Delta_x)  # (N_cells, 2, N_vars)

	grad = jax.vmap(jax.vmap(jnp.linalg.solve))(jnp.repeat(A[:,None,...], b.shape[-2], axis=-3), b)  # (N_cells, 2, N_vars)
	return grad

def gradient_GG(W_L, W_R, mesh):
	surfaces = mesh.surface[mesh.face_connectivity]  # (N_cells, 3)
	grad = jnp.sum(0.5 * (W_R + W_L)[...,None] * mesh.normals[...,None,:] * surfaces[...,None,None], axis=-3) / mesh.area[...,None,None]  # (N_cells, N_vars, 2)
	return grad

mesh = Mesh()
mesh.mesh_generator(maxV = 2e-5, marker_boundary=1)
mesh.neighbors[1]
print("barycenter of triangle 1: ")
print(mesh.barycenter[1])
print("barycenter of the neighbor of triangle 1: ")
print(mesh.barycenter[mesh.neighbors[1]])

Primitives = (- jnp.sin(2 * jnp.pi * mesh.barycenter[...,0]) * jnp.cos(2 * jnp.pi * mesh.barycenter[...,1]) + 1.5)[...,None]

Primitives_L = jnp.repeat(Primitives[...,None,:], 3, axis=-2)
Primitives_R = Primitives[mesh.neighbors]
grad = getgradientLSQ(Primitives_L, Primitives_R, mesh)

mesh.plot_solution(grad[...,0,0], labels = r'$d\rho/dx$')
mesh.plot_solution(grad[...,0,1], labels = r'$d\rho/dy$')

# true grad 
x = mesh.barycenter[...,0]
y = mesh.barycenter[...,1]
grad_x = - 2 * jnp.pi * jnp.cos(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
grad_y = 2 * jnp.pi * jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
mesh.plot_solution(grad_x, labels = r'$d\rho/dx$ (true)')
mesh.plot_solution(grad_y, labels = r'$d\rho/dy$ (true)')

# difference
mesh.plot_solution(jnp.abs(grad[...,0,0] - grad_x), labels = r'$d\rho/dx$ (error)')
mesh.plot_solution(jnp.abs(grad[...,0,1] - grad_y), labels = r'$d\rho/dy$ (error)')


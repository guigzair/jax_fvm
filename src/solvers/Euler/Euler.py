import jax.numpy as jnp
import jax
# jax.config.update('jax_enable_x64', True)
jax.config.update("jax_debug_nans", True)
import sys
sys.path.append('../../../..')  
from FVM.src.mesh.mesh import Mesh # pyright: ignore[reportMissingImports]
import FVM.src.mesh.plot as plot # pyright: ignore[reportMissingImports]
import FVM.src.Cases.Test_Cases as Test_Cases # pyright: ignore[reportMissingImports]
import FVM.src.mesh.Mesh_cases as Mesh_cases # pyright: ignore[reportMissingImports]
import time
import FVM.src.solvers.helper as helper # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt
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

"""
Finite Volume Method for 2D Euler equations
With jax.jit compilation = ~ 100x faster for large data
"""
###########################################################################################################
##############################                Solver                 ######################################
###########################################################################################################

def venkatakrishnan(a, b, h = 0, K = 0):
	omega = (K * h)**3 
	L = (a**2 + 2 * a*b + omega) / (a**2 + 2*b**2 + a*b + omega + 1e-09)
	return L

def getlimiting(W_L, W_R, grad, mesh):
	W_m  = jnp.min(jnp.concatenate([W_L, W_R], axis = -2), axis = -2)
	W_M  = jnp.max(jnp.concatenate([W_L, W_R], axis = -2), axis = -2)

	mid_point_faces = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2) # (N_cells,3,2)
	delta_x = mid_point_faces - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	Delta = jnp.einsum('ijl,ikl->ijk', delta_x, grad)  # (N_cells, 3, N_vars)

	phi = jnp.ones_like(Delta)
	phi = jnp.where(Delta > 1e-8,
					venkatakrishnan(W_M[...,None,:] - W_L, Delta),
					phi)
	phi = jnp.where(Delta < -1e-8,
					venkatakrishnan(W_m[...,None,:] - W_L, Delta),
					phi)
	phi = jnp.min(phi, axis = -2)  # (N_cells, N_vars)
	return phi

def MUSCL(W_L, W_R, grad, mesh):
	phi = getlimiting(W_L, W_R, grad, mesh)  # (N_cells, N_vars)

	mid_point_faces = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2) # (N_cells,3,2)
	
	delta_x = mid_point_faces - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	Delta = jnp.einsum('ijl,ikl->ijk', delta_x, grad)  # (N_cells, 3, N_vars)

	delta_x_neigh = mid_point_faces - mesh.barycenter[mesh.neighbors]  # (N_cells, 3, 2)
	delta_x_neigh = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 0)[...,None], 2, axis=-1), -delta_x, delta_x_neigh) # Boundary faces: reverse the direction
	Delta_neigh = jnp.einsum('ijl,ijkl->ijk', delta_x_neigh, grad[mesh.neighbors])


	W_L_MUSCL = W_L + phi[...,None,:] * Delta  # (N_cells, 3, N_vars)
	W_R_MUSCL = W_R + phi[mesh.neighbors] * Delta_neigh  # (N_cells, 3, N_vars)

	return W_L_MUSCL, W_R_MUSCL

def getFlux(W_L, W_R, normals, surfaces, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
	# i did not put mesh as input in order to vmap this function
	# Get the cell state for each edge
	rho_L = W_L[...,0]
	mom_x_L = W_L[...,1]
	mom_y_L = W_L[...,2]
	E_L = W_L[...,3]
	u_L = mom_x_L / rho_L
	v_L = mom_y_L / rho_L
	P_L = (gamma - 1) * (E_L - 0.5 * rho_L * (u_L**2 + v_L**2))

	# Get the corresponding neighbors state
	rho_R = W_R[...,0]
	mom_x_R = W_R[...,1]
	mom_y_R = W_R[...,2]
	E_R = W_R[...,3]
	u_R = mom_x_R / rho_R
	v_R = mom_y_R / rho_R
	P_R = (gamma - 1) * (E_R - 0.5 * rho_R * (u_R**2 + v_R**2))

	# Get corresponding normals
	nx = normals[...,0]
	ny = normals[...,1]

    # Maximum wavelenghts
	C_L = jnp.sqrt(jnp.abs(gamma*P_L/rho_L))  + jnp.abs(u_L * nx + v_L * ny)
	C_R = jnp.sqrt(jnp.abs(gamma*P_R/rho_R))  + jnp.abs(u_R * nx + v_R * ny)
	C_max = jnp.maximum(C_R, C_L)

	# Energy
	en_L = P_L/(gamma-1) + 0.5 * rho_L * (u_L**2 + v_L**2)
	en_R = P_R/(gamma-1) + 0.5 * rho_R * (u_R**2 + v_R**2)

	# Flux
	flux_rho_L = rho_L * (u_L * nx + v_L * ny)
	flux_ru_L = rho_L * u_L* ( u_L * nx + v_L * ny) + P_L * nx
	flux_rv_L = rho_L * v_L * (u_L * nx + v_L * ny) + P_L * ny
	flux_E_L = (en_L + P_L) * (u_L * nx + v_L * ny)

	flux_rho_R = rho_R * (u_R * nx + v_R * ny)
	flux_ru_R = rho_R * u_R * ( u_R * nx + v_R * ny) + P_R * nx
	flux_rv_R = rho_R * v_R * (u_R * nx + v_R * ny) + P_R * ny
	flux_E_R = (en_R + P_R) * (u_R * nx + v_R * ny)

	# Total flux
	# local_mach = jnp.maximum(jnp.abs(u_L)/jnp.maximum(C_L, 1e-09), jnp.abs(u_R)/jnp.maximum(C_R, 1e-09))
	# alpha = jnp.sin(jnp.pi * local_mach /2)  # andea s damping 2024
	# alpha = jnp.where(local_mach >= 1, 1.0, alpha)
	alpha = kwargs.get('alpha', 0.1)

	flux_rho = (flux_rho_L + flux_rho_R)/2 - alpha * C_max * 0.5 * (rho_R - rho_L)
	flux_ru = (flux_ru_L + flux_ru_R)/2 - alpha * C_max * 0.5 * (rho_R * u_R - rho_L * u_L)
	flux_rv = (flux_rv_L + flux_rv_R)/2 - alpha * C_max * 0.5 * (rho_R * v_R - rho_L * v_L)
	flux_E = (flux_E_L + flux_E_R)/2 - alpha * C_max * 0.5 * (en_R - en_L)

	Flux = jnp.stack([surfaces * flux_rho, 
						surfaces * flux_ru, 
						surfaces * flux_rv, 
						surfaces * flux_E], axis = -1)

	Flux = jnp.sum(Flux, axis = -2)
	return Flux


###########################################################################################################
############################           Time integration                 ###################################
###########################################################################################################

@jax.jit(static_argnums=(1,))
def time_step(W, mesh, dt, **kwargs):
	# 1st order
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]

	# 2nd order - MUSCL with least-square gradient
	W_R = helper.BC_state(W_R, W_L, mesh)
	grad = helper.getgradientLSQ(W_L, W_R, mesh)

	W_L, W_R = MUSCL(W_L, W_R, grad, mesh)
	W_R = helper.BC_state(W_R, W_L, mesh)

	Flux = getFlux(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], 
				gamma = 1.4, alpha = kwargs.get('alpha', 1.)) 
		
	W = W - dt / mesh.area[...,None] * (Flux) 
	return W

@jax.jit(static_argnums=(1,))
def residual(W, mesh, **kwargs):
	# 1st order
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]

	# 2nd order - MUSCL with least-square gradient
	W_R = helper.BC_state(W_R, W_L, mesh, **kwargs)
	grad = helper.getgradientLSQ(W_L, W_R, mesh)

	W_L, W_R = MUSCL(W_L, W_R, grad, mesh)
	W_R = helper.BC_state(W_R, W_L, mesh, **kwargs)

	Flux = getFlux(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], **kwargs) 
	# Flux = jax.vmap(getFlux, 
	# 			in_axes=(0,0,0,0,None,None))(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], 1.4, kwargs.get('alpha', 1.))
	return Flux / mesh.area[...,None] 

@jax.jit(static_argnums=(1,))
def time_step_RK2(W, mesh, dt, **kwargs):
	F1 = residual(W, mesh, **kwargs)
	W1 = W - dt/2 * F1
	F2 = residual(W1, mesh, **kwargs)
	W = W - dt * F2
	return W

@jax.jit(static_argnums=(1,))
def time_step_RK4(W, mesh, dt, **kwargs):
	F1 = residual(W, mesh, **kwargs)
	W1 = W - dt/2 * F1
	F2 = residual(W1, mesh, **kwargs)
	W2 = W - dt/2 * F2
	F3 = residual(W2, mesh, **kwargs)
	W3 = W - dt * F3
	F4 = residual(W3, mesh, **kwargs)

	W = W - dt/6 * (F1 + 2*F2 + 2*F3 + F4)
	return W

@jax.jit(static_argnums=(1,))
def time_step_Newton(W, mesh, dt, **kwargs):
	W_old = W
	for _ in range(3):
		Fval = W - W_old + dt * residual(W, mesh, **kwargs)
		def Jv(v):
			_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
						(W,),
						(v,))
			return v + dt * jvp
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, tol=5e-3, maxiter=2, restart = 25)
		W = W + delta * 0.6  # under-relaxation to ensure convergence
	return W


@jax.jit(static_argnums=(1,))
def SDIRK2(W, mesh, dt, **kwargs):
	x = 1 - 1/jnp.sqrt(2) # singly diagonally implicit RK
	n_newton = 4
	# first step
	W1 = W
	for _ in range(n_newton):
		Fval = W1 - W + dt * residual(W1, mesh, **kwargs)
		def Jv(v):
			_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
						(W1,),
						(v,))
			return v + dt * jvp
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, tol=5e-3, maxiter=3, restart = 25)
		W1 = W1 + delta * 0.6  # under-relaxation to ensure convergence
	F1 = residual(W1, mesh, **kwargs)

	# second step
	W2 = W1
	for _ in range(n_newton):
		Fval = W2 - W + dt * (x * residual(W2, mesh, **kwargs) + (1-2 * x) * F1)
		def Jv(v):
			_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
						(W2,),
						(v,))
			return v + dt * x * jvp
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, tol=5e-3, maxiter=3, restart = 25)
		W2 = W2 + delta * 0.6  # under-relaxation to ensure convergence
	F2 = residual(W2, mesh, **kwargs)

	W = W - 0.5 * dt * (F1 + F2) 
	return W



if __name__ == "__main__":
	mesh = Mesh_cases.TestDipoleVortex().build(h = 5e-5, L = 1.)

	# Initial condition
	Primitives, mesh = Test_Cases.TestDipoleVortex2(R = 0.1, omega = 300, mach = 0.01).build(mesh)

	W = helper.getConserved(Primitives)
	mesh.plot_mesh()

	# Time loop
	t_final = 0.2 #/ jnp.mean(Primitives[...,1]) # to get real time
	CFL = 0.5
	dt = helper.get_dt(W, mesh, CFL = CFL)
	N_t = int(t_final / dt) + 1

	start_time = time.time()

	T_interval_snapshots = 100
	Snapshots = jnp.zeros((int(N_t/T_interval_snapshots), *W.shape))
	for n in range(N_t):
		W = time_step_RK2(W, mesh, dt, alpha = 0.1)
		# W = time_step_Newton(W, mesh, dt, alpha = 0.1)
		# W = SDIRK2(W, mesh, dt, alpha = 0.1)
		if n % 100 == 0:
			print(f'It : {n} / {N_t}')
		if n % T_interval_snapshots == 0:
			Snapshots = Snapshots.at[int(n/T_interval_snapshots)].set(W)

	print(f'Simulation time: {time.time() - start_time} seconds')

	# Plot solution
	Primitives = helper.getPrimitive(W)
	mesh.plot_solution(Primitives[...,0], labels = r'$\rho$')
	mesh.plot_solution(Primitives[...,1], labels = r'$u$')
	mesh.plot_solution(Primitives[...,2], labels = r'$v$')
	mesh.plot_solution(Primitives[...,3], labels = r'$P$')

	# vorticity
	vorticity = helper.get_vorticity_from_field(W, mesh)
	mesh.plot_solution(vorticity, labels = r'$\omega$')

	# entropy plot
	Total_entropy = jax.lax.map(lambda x: helper.get_total_entropy(x, mesh), Snapshots)
	fig, ax = plt.subplots()
	ax.plot(jnp.arange(Snapshots.shape[0]) * T_interval_snapshots * dt, Total_entropy)
	ax.set_xlabel(r't')
	ax.set_ylabel(r'$\mathcal{S}$')
	ax.grid()

	# enstrophy plot
	Total_enstrophy = jax.lax.map(lambda x: helper.get_total_enstrophy(x, mesh), Snapshots)
	fig, ax = plt.subplots()
	ax.plot(jnp.arange(Snapshots.shape[0]) * T_interval_snapshots * dt, Total_enstrophy)
	ax.set_xlabel(r't')
	ax.set_ylabel(r'$\xi$')
	ax.grid()

	# energy plot
	Total_energy = jax.lax.map(lambda x: helper.get_total_kinetic_energy(x, mesh), Snapshots)
	fig, ax = plt.subplots()
	ax.plot(jnp.arange(Snapshots.shape[0]) * T_interval_snapshots * dt, Total_energy)
	ax.set_xlabel(r't')
	ax.set_ylabel(r'$E$')
	ax.grid()

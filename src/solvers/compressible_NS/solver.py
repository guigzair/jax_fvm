import jax.numpy as jnp
import jax
jax.config.update("jax_debug_nans", True)
import sys

sys.path.append('../../../..')  
from jax_fvm.src.mesh.mesh import Mesh 
import jax_fvm.src.Cases.Test_Cases as Test_Cases
import jax_fvm.src.mesh.Mesh_cases as Mesh_cases 
import time
import jax_fvm.src.solvers.helper as helper 
import jax_fvm.src.solvers.Euler.Euler as Euler

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


def getFlux_diffusive(grad_prim, Prim_L, Prim_R, mesh, **kwargs):
	mu = kwargs.get('mu', 1.716e-4)
	R = kwargs.get('R', 287)
	k = kwargs.get('k', 0.0257)

	# compute stress tensor
	div_u = grad_prim[:,1,0] + grad_prim[:,2,1]

	temp_L = helper.get_temperature(Prim_L, R = R)
	temp_R = helper.get_temperature(Prim_R, R = R)
	grad_T = helper.getgradientLSQ(temp_L[...,None], temp_R[...,None], mesh)

	tau = grad_prim[...,1:3,:] + jnp.transpose(grad_prim[...,1:3,:], (0,2,1))
	tau = tau - 2/3 * jnp.einsum('ij,kl->ikl', div_u[...,None],jnp.eye(2))
	tau = mu * tau 

	tau_L = jnp.repeat(tau[...,None,:,:], 3, axis=-3)
	tau_R = tau[mesh.neighbors]
	tau_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 1)[...,None,None], 2, axis=-2), tau_L, tau_R)

	Flux_tau = 1/2 * (tau_L + tau_R)

	# Get corresponding normals
	nx = mesh.normals[...,0]
	ny = mesh.normals[...,1]

	Flux_tau_x = Flux_tau[...,0,0] * nx + Flux_tau[...,0,1] * ny
	Flux_tau_y = Flux_tau[...,1,0] * nx + Flux_tau[...,1,1] * ny

	Flux_energy = tau_R[...,0,:] * Prim_R[...,1][...,None] +  tau_R[...,1,:] * Prim_R[...,2][...,None]# q = - k grad(T)  and here we have grad(u) instead of grad(T)
	Flux_energy = Flux_energy + tau_L[...,0,:] * Prim_L[...,1][...,None] +  tau_L[...,1,:] * Prim_L[...,2][...,None]
	Flux_energy = 0.5 * (Flux_energy[...,0] * nx + Flux_energy[...,1] * ny)
	Flux_energy = Flux_energy + k * (grad_T[...,0] * nx + grad_T[...,1] * ny)

	surfaces = mesh.surface[mesh.face_connectivity]

	Flux = jnp.stack([surfaces * jnp.zeros_like(Flux_tau_x), 
						surfaces * Flux_tau_x, 
						surfaces * Flux_tau_y, 
						surfaces * Flux_energy], axis = -1)

	Flux = jnp.sum(Flux, axis = -2)

	return Flux


###########################################################################################################
############################           Time integration                 ###################################
###########################################################################################################

@jax.jit(static_argnums=(1,))
def residual(W, mesh, **kwargs):
	# 1st order
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]

	# 2nd order - MUSCL with least-square gradient
	W_R = helper.BC_state(W_R, W_L, mesh, **kwargs)

	W_L = helper.getPrimitive(W_L, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
	W_R = helper.getPrimitive(W_R, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
	grad = helper.getgradientLSQ(W_L, W_R, mesh)
	W_L, W_R = Euler.MUSCL(W_L, W_R, grad, mesh)
	W_L = helper.getConserved(W_L, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
	W_R = helper.getConserved(W_R, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
	Flux = Euler.getFlux(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], **kwargs) 
	
	# compute diffusive flux based on primitive variable gradients
	Flux_diffusive = getFlux_diffusive(grad, W_L, W_R, mesh, **kwargs)

	R = 1/ mesh.area[...,None] * (Flux - Flux_diffusive) 
	return R

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
	n_newton = 2
	maxiter = 2
	tol= 1e-2
	# first step
	W1 = W
	for _ in range(n_newton):
		Fval = W1 - W + dt * residual(W1, mesh, **kwargs)
		def Jv(v):
			_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
						(W1,),
						(v,))
			return v + dt * jvp
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, tol=tol, maxiter= maxiter, restart = 25)
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
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, tol=tol, maxiter=maxiter, restart = 25)
		W2 = W2 + delta * 0.6  # under-relaxation to ensure convergence
	F2 = residual(W2, mesh, **kwargs)

	W = W - 0.5 * dt * (F1 + F2) 
	return W


if __name__ == "__main__":
	# for air at 300K
	mu = 1.716e-4
	R = 287
	gamma = 1.4
	C_v = R / (gamma - 1)
	C_p = C_v * gamma
	k = 5/2 * mu * C_v
	kwargs = {'gamma': gamma, 'mu': mu, 'R': R, 'k': k, 'flag_NS': True, 'alpha': 0.1}

	# little test case: Forward facing step
	mesh = Mesh_cases.TestDipoleVortex().build(h = 9e-6, L = 1.)
	Primitives, mesh = Test_Cases.TestDipoleVortex2(R = 0.1, omega = 300, mach = 0.01).build(mesh)
	W = helper.getConserved(Primitives)

	# Time loop
	t_final = 0.3 
	CFL = 0.4
	dt = helper.get_dt(W, mesh, CFL = CFL)
	dt_viscous = helper.get_dt_viscous(mesh, CFL = CFL, nu = mu / jnp.mean(Primitives[...,0]))
	dt = jnp.min(jnp.array([dt, dt_viscous]))
	print(f"dt convective: {dt:.2e}, dt viscous: {dt_viscous:.2e}")
	N_t = int(t_final / dt) + 1

	start_time = time.time()

	T_interval_snapshots = 5000
	Snapshots = jnp.zeros((int(N_t/T_interval_snapshots), *W.shape))
	for n in range(N_t):
		W = time_step_RK2(W, mesh, dt, **kwargs)
		# W = time_step_Newton(W, mesh, dt, **kwargs)
		if n % 100 == 0:
			print(f'It : {n} / {N_t}')
		if n % T_interval_snapshots == 0:
			Snapshots = Snapshots.at[int(n/T_interval_snapshots)].set(W)
		
	print(f'Simulation time: {time.time() - start_time} seconds')

	# Plot solution
	Primitives = helper.getPrimitive(W)
	# mesh.plot_solution(Primitives[...,0], labels = r'$\rho$')
	# mesh.plot_solution(Primitives[...,3], labels = r'$P$')

	# vorticity
	vorticity = helper.get_vorticity_from_field(W, mesh, **kwargs)
	mesh.plot_contour_solution(vorticity, labels = r'$\omega$')
	mesh.plot_solution(vorticity, labels = r'$\omega$')

	# entropy plot
	Total_entropy = jax.lax.map(lambda x: helper.get_total_entropy(x, mesh), Snapshots)
	fig, ax = plt.subplots()
	ax.plot(jnp.arange(Snapshots.shape[0]) * T_interval_snapshots * dt, Total_entropy)
	ax.set_xlabel(r't')
	ax.set_ylabel(r'$\mathcal{S}$')
	ax.grid()

	# enstrophy plot
	Total_enstrophy = jax.lax.map(lambda x: helper.get_total_enstrophy(x, mesh, **kwargs), Snapshots)
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


	
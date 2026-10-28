import jax.numpy as jnp
import jax
jax.config.update("jax_debug_nans", True)
import sys
# jax.config.update('jax_enable_x64', True)
sys.path.append('../../../..') 
from jax_fvm.src.config import Case 
from jax_fvm.src.mesh.mesh import Mesh 
import jax_fvm.src.Cases.Test_Cases as Test_Cases
import jax_fvm.src.mesh.Mesh_cases as Mesh_cases 
import time
import jax_fvm.src.solvers.helper as helper 
import jax_fvm.src.solvers.Euler.Euler as Euler
import jax_fvm.src.solvers.Time_Integration as TimeIntegration

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


###########################################################################################################
##############################                Solver                 ######################################
###########################################################################################################


def getFlux_diffusive2(grad_face, Prim_L, Prim_R, mesh, **kwargs):
	# grad_face: face gradients of [rho, u, v, P, T], (N_cells, 3, 5, 2)
	mu = kwargs.get('mu', 1.716e-4)
	k = kwargs.get('k', 0.0257)

	# stress tensor at each face
	grad_u = grad_face[...,1:3,:]  # (N_cells, 3, 2, 2)
	div_u = grad_u[...,0,0] + grad_u[...,1,1]
	tau = grad_u + jnp.swapaxes(grad_u, -1, -2)
	tau = tau - 2/3 * div_u[...,None,None] * jnp.eye(2)
	tau = mu * tau

	grad_T = grad_face[...,4,:]  # (N_cells, 3, 2)

	# Get corresponding normals
	nx = mesh.normals[...,0]
	ny = mesh.normals[...,1]

	Flux_tau_x = tau[...,0,0] * nx + tau[...,0,1] * ny
	Flux_tau_y = tau[...,1,0] * nx + tau[...,1,1] * ny

	# (tau . u) . n + k grad(T) . n   (q = - k grad(T)), tau symmetric
	u_f = 0.5 * (Prim_L[...,1] + Prim_R[...,1])
	v_f = 0.5 * (Prim_L[...,2] + Prim_R[...,2])
	Flux_energy = u_f * Flux_tau_x + v_f * Flux_tau_y
	Flux_energy = Flux_energy + k * (grad_T[...,0] * nx + grad_T[...,1] * ny)

	# per-face flux (N_cells, 3, 4); surface scaling and face sum are done in residual
	Flux = jnp.stack([jnp.zeros_like(Flux_tau_x), 
						Flux_tau_x, 
						Flux_tau_y, 
						Flux_energy], axis = -1)

	return Flux


def getFlux_diffusive_ES(W_L, W_R, mesh, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
	mu = kwargs.get('mu', 1.716e-4)
	k = kwargs.get('k', 0.0257)
	R = kwargs.get('R', 287)

	V_L = helper.getEntropyVariables(W_L, gamma = gamma)
	V_R = helper.getEntropyVariables(W_R, gamma = gamma)
	# grad_V = helper.gradient_GG(V_L, V_R, mesh)  # (N_cells, 4, 2)
	grad_V = helper.getgradientLSQ(V_L, V_R, mesh)  # (N_cells, 4, 2)

	V = V_L[:,0]
	u = - V[:,1] / V[:,3]
	v = - V[:,2] / V[:,3]
	grad_u = - (grad_V[:,1] + u[:,None] * grad_V[:,3]) / V[:,3,None]  # (N_cells, 2)
	grad_v = - (grad_V[:,2] + v[:,None] * grad_V[:,3]) / V[:,3,None]
	grad_T = grad_V[:,3] / (R * V[:,3,None]**2)

	div_u = grad_u[:,0] + grad_v[:,1]
	tau_xx = mu * (2 * grad_u[:,0] - 2/3 * div_u)
	tau_yy = mu * (2 * grad_v[:,1] - 2/3 * div_u)
	tau_xy = mu * (grad_u[:,1] + grad_v[:,0])

	f_v = jnp.stack([jnp.zeros((V.shape[0], 2)),
					jnp.stack([tau_xx, tau_xy], axis = -1),
					jnp.stack([tau_xy, tau_yy], axis = -1),
					jnp.stack([u * tau_xx + v * tau_xy + k * grad_T[:,0],
							   u * tau_xy + v * tau_yy + k * grad_T[:,1]], axis = -1)], axis = -2)  # (N_cells, 4, 2)

	# central face flux (own cell flux on boundary faces)
	f_v_R = jnp.where((mesh.face_markers[mesh.face_connectivity] > 1)[...,None,None], f_v[:,None], f_v[mesh.neighbors])
	f_face = 0.5 * (f_v[:,None] + f_v_R)  # (N_cells, 3, 4, 2)

	# per-face flux (N_cells, 3, 4); surface scaling and face sum are done in residual
	return jnp.einsum('ijkl,ijl->ijk', f_face, mesh.normals)


def getFlux_diffusive(grad, Prim_L, Prim_R, mesh, **kwargs):
	mu = kwargs.get('mu', 1e-5)
	k  = kwargs.get('k',  1e-5)
	R  = kwargs.get('R',  287.)

	# Face-averaged primitive variables
	Prim_face = 0.5 * (Prim_L + Prim_R)  # (N_cells, 3, 4)
	rho_f = Prim_face[..., 0]
	u_f   = Prim_face[..., 1]
	v_f   = Prim_face[..., 2]
	P_f   = Prim_face[..., 3]

	# Face-averaged gradients: average of left cell and right (neighbor) cell gradients
	# grad: (N_cells, N_vars, 2) ; grad[mesh.neighbors]: (N_cells, 3, N_vars, 2)
	grad_face = 0.5 * (grad[:, None, :, :] + grad[mesh.neighbors])  # (N_cells, 3, N_vars, 2)

	du_dx   = grad_face[..., 1, 0]  # (N_cells, 3)
	du_dy   = grad_face[..., 1, 1]
	dv_dx   = grad_face[..., 2, 0]
	dv_dy   = grad_face[..., 2, 1]
	drho_dx = grad_face[..., 0, 0]
	drho_dy = grad_face[..., 0, 1]
	dP_dx   = grad_face[..., 3, 0]
	dP_dy   = grad_face[..., 3, 1]

	# Viscous stress tensor (Newtonian fluid, Stokes hypothesis: zero bulk viscosity)
	div_u  = du_dx + dv_dy
	tau_xx = mu * (2 * du_dx - 2/3 * div_u)
	tau_yy = mu * (2 * dv_dy - 2/3 * div_u)
	tau_xy = mu * (du_dy + dv_dx)

	# Temperature gradient via T = P / (rho * R)
	dT_dx = (dP_dx * rho_f - P_f * drho_dx) / (rho_f**2 * R)
	dT_dy = (dP_dy * rho_f - P_f * drho_dy) / (rho_f**2 * R)

	# Heat flux (Fourier's law): q = -k * grad(T)
	q_x = -k * dT_dx
	q_y = -k * dT_dy

	nx = mesh.normals[..., 0]  # (N_cells, 3)
	ny = mesh.normals[..., 1]

	# Diffusive flux projected onto face normal F_d · n for each conserved variable
	f_rho   = jnp.zeros_like(rho_f)
	f_mom_x = tau_xx * nx + tau_xy * ny
	f_mom_y = tau_xy * nx + tau_yy * ny
	f_E     = (u_f * tau_xx + v_f * tau_xy - q_x) * nx + (u_f * tau_xy + v_f * tau_yy - q_y) * ny

	Flux_d = jnp.stack([f_rho, f_mom_x, f_mom_y, f_E], axis=-1)  # (N_cells, 3, 4)

	# Integrate over face surfaces and sum over all faces
	# Flux_d = mesh.surface[mesh.face_connectivity][..., None] * Flux_d
	return Flux_d


###########################################################################################################
############################           Time integration                 ###################################
###########################################################################################################

@jax.jit(static_argnums=(1,), static_argnames=("flux",))
def residual(W, mesh, **kwargs):
	# 1st order
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]
	W_R = helper.BC_state(W_R, W_L, mesh, **kwargs)


	Prim_L = helper.getPrimitive(W_L, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
	Prim_R = helper.getPrimitive(W_R, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
	grad = helper.getgradientLSQ(Prim_L, Prim_R, mesh)
	# Flux_diffusive = getFlux_diffusive_ES(W_L, W_R, mesh, **kwargs)
	Prim_L, Prim_R = Euler.MUSCL(Prim_L, Prim_R, grad, mesh, **kwargs)

	PrimT_L = jnp.concatenate([Prim_L, helper.get_temperature(Prim_L, R = kwargs.get('R', 287))[...,None]], axis=-1)
	PrimT_R = jnp.concatenate([Prim_R, helper.get_temperature(Prim_R, R = kwargs.get('R', 287))[...,None]], axis=-1)
	grad_T = helper.getgradientLSQ(PrimT_L[...,4:], PrimT_R[...,4:], mesh)
	grad_face = helper.face_gradient(jnp.concatenate([grad, grad_T], axis=-2), PrimT_L, PrimT_R, mesh)
	Flux_diffusive = getFlux_diffusive2(grad_face, Prim_L, Prim_R, mesh, **kwargs)

	W_L = helper.getConserved(Prim_L, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
	W_R = helper.getConserved(Prim_R, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))

	Flux_convective = Euler.getFlux_Tadmor(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], **kwargs) 

	Flux = Flux_convective - Flux_diffusive
	Flux = mesh.surface[mesh.face_connectivity][...,None] * Flux
	Flux = jnp.sum(Flux, axis = -2)

	R = 1/ mesh.area[...,None] * Flux
	return R



if __name__ == "__main__":
	# cfg_path = "../../Cases/config/NS/corotating_merge.yaml"
	# cfg_path = "../../Cases/config/NS/KelvinHelmholtz.yaml"
	# cfg_path = "../../Cases/config/NS/KelvinHelmholtzAsym.yaml"
	cfg_path = "../../Cases/config/NS/KelvinHelmholtzMultimode.yaml"
	# cfg_path = "../../Cases/config/NS/DipoleVortex.yaml"
	case = Case.build(cfg_path, MeshClass=Mesh)

	W = helper.getConserved(case.primitives)

	# Time loop
	t_final = case.config.time.t_end
	dt = helper.get_dt(W, case.mesh, CFL = case.config.time.cfl)
	dt_viscous = helper.get_dt_viscous(case.mesh, 
									CFL = case.config.time.cfl, 
									# max of kinematic viscosity mu/rho and thermal diffusivity k/(rho cv), cv = R/(gamma-1)
									nu = max(case.config.kwargs.get('mu', 1.0),
											case.config.kwargs.get('k', 0.0) * (case.config.kwargs.get('gamma', 1.4) - 1) / case.config.kwargs.get('R', 287)) 
										/ jnp.mean(case.primitives[...,0]))
	dt = jnp.min(jnp.array([dt, dt_viscous]))
	print(f"dt convective: {dt:.2e}, dt viscous: {dt_viscous:.2e}")
	N_t = int(t_final / dt) + 1

	start_time = time.time()

	T_interval_snapshots = 1000
	Snapshots = jnp.zeros((int(N_t/T_interval_snapshots), *W.shape))
	for n in range(N_t):
		W = TimeIntegration.time_step_RK2(W, dt, case.mesh, residual, **case.config.kwargs)
		# W = time_step_Newton(W, mesh, dt, **kwargs)
		if n % 1000 == 0:
			print(f'It : {n} / {N_t}')
		if n % T_interval_snapshots == 0:
			Snapshots = Snapshots.at[int(n/T_interval_snapshots)].set(W)
		
	print(f'Simulation time: {time.time() - start_time} seconds')

	# Plot solution
	Primitives = helper.getPrimitive(W)
	fig, ax = case.mesh.plot_solution(Primitives[...,0], labels = r'$\rho$')
	# fig.savefig('density.pdf', dpi=1000)

	case.mesh.plot_solution(Primitives[...,3], labels = r'$P$')

	# vorticity
	vorticity = helper.get_vorticity_from_field(W, case.mesh, interior_only=True, **case.config.kwargs)
	# case.mesh.plot_contour_solution(vorticity, labels = r'$\omega$')
	case.mesh.plot_solution(vorticity, labels = r'$\omega$')

	# entropy plot
	Total_entropy = jax.lax.map(lambda x: helper.get_total_entropy(x, case.mesh), Snapshots)
	fig, ax = plt.subplots()
	ax.plot(jnp.arange(Snapshots.shape[0]) * T_interval_snapshots * dt, Total_entropy)
	ax.set_xlabel(r't')
	ax.set_ylabel(r'$\mathcal{S}$')
	ax.grid()

	# enstrophy plot
	Total_enstrophy = jax.lax.map(lambda x: helper.get_total_enstrophy(x, case.mesh, **case.config.kwargs), Snapshots)
	fig, ax = plt.subplots()
	ax.plot(jnp.arange(Snapshots.shape[0]) * T_interval_snapshots * dt, Total_enstrophy)
	ax.set_xlabel(r't')
	ax.set_ylabel(r'$\xi$')
	ax.grid()

	# energy plot
	Total_energy = jax.lax.map(lambda x: helper.get_total_kinetic_energy(x, case.mesh), Snapshots)
	fig, ax = plt.subplots()
	ax.plot(jnp.arange(Snapshots.shape[0]) * T_interval_snapshots * dt, Total_energy)
	ax.set_xlabel(r't')
	ax.set_ylabel(r'$E$')
	ax.grid()

	# vorticity_snap = jax.lax.map(lambda x: helper.get_vorticity_from_field(x, case.mesh), Snapshots)
	# case.mesh.animate_field(Snapshots[::1,:,0], path = "density.gif", interval = 100)



	
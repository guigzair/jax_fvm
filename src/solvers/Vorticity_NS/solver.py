import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mesh as mesh_vort
import helpers as helpers
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


# -----------------------------
# RHS
# -----------------------------

def get_velocities(omega_hat, mesh):
    psi_hat = + omega_hat / mesh.k2
    psi_hat = psi_hat.at[0,0].set(0.0)

    # velocity in Fourier
    u_hat =  1j * mesh.ky * psi_hat
    v_hat = -1j * mesh.kx * psi_hat

    # back to real
    u = jnp.real(jnp.fft.ifft2(u_hat))
    v = jnp.real(jnp.fft.ifft2(v_hat))

    return u, v

def get_viscous_term(omega_hat, mesh, nu = 1e-2):
    return nu * mesh.k2 * omega_hat

def get_convection_term(omega_hat, mesh, **kwargs):
    # stream function
    u, v = get_velocities(omega_hat * mesh.dealias, mesh)

    # vorticity gradients
    omega_x = jnp.real(jnp.fft.ifft2(1j * mesh.kx * omega_hat))
    omega_y = jnp.real(jnp.fft.ifft2(1j * mesh.ky * omega_hat))

    # nonlinear term
    adv = u * omega_x + v * omega_y
    adv_hat = jnp.fft.fft2(adv) * mesh.dealias

    return adv_hat

def get_forcing_term(mesh, key, dt, k=8, dk = 1., eps0=0.1):
    # mask
    k_mag = jnp.sqrt(mesh.k2)
    mask = (k_mag >= k - dk/2) & (k_mag <= k + dk/2)
    subkey1, subkey1 = jax.random.split(key)

    noise_real = jax.random.normal(subkey1, (mesh.N, mesh.N))
    noise_image = jax.random.normal(subkey1, (mesh.N, mesh.N))

    noise = noise_real + 1j * noise_image
    f_hat = jnp.where(mask, noise, 0.)/ jnp.sqrt(dt)

    f = jnp.real(jnp.fft.ifft2(f_hat))
    f_hat = jnp.fft.fft2(f)

    eps = helpers.get_energy(f_hat, mesh)
    f_hat = f_hat *  jnp.sqrt(eps0 / eps)  # normalize energy to E0
    return f_hat  


@jax.jit(static_argnums=(1,))
def step_IMEX(omega_hat, mesh, dt, **kwargs):
    omega_hat = omega_hat.at[0,0].set(0.0)
    adv = get_convection_term(omega_hat, mesh, **kwargs)
    nu = kwargs.get('nu', 1e-2)

    # forcing
    k = kwargs.get('k', 8)
    dk = kwargs.get('dk', 1.)
    eps0 = kwargs.get('eps0', 0.1)
    f_hat = get_forcing_term(mesh, jax.random.PRNGKey(0), dt, k=k, dk=dk, eps0=eps0)

    omega_hat = (omega_hat - dt * adv + dt * f_hat) / (1.0 + dt * nu * mesh.k2) 

    Energy = helpers.get_energy(omega_hat, mesh)
    Enstrophy = helpers.get_enstrophy(omega_hat, mesh)
    Palinstrophy = helpers.get_palinstrophy(omega_hat, mesh)
    return omega_hat, (Energy, Enstrophy, Palinstrophy)



if __name__ == "__main__":
    # -----------------------------
    # Parameters
    # -----------------------------
    N = 512
    L = 2 * jnp.pi
    nu = 1e-2

    # forcing 
    k = 5.
    dk = 1.
    eps0 = 0.2

    kwargs = {'nu': nu, 'k': k, 'dk': dk, 'eps0': eps0}

    # -----------------------------
    # Random
    # -----------------------------
    key = jax.random.PRNGKey(0)

    # -----------------------------
    # Mesh
    # -----------------------------
    mesh = mesh_vort.Mesh()
    mesh.mesh_generator(N=N, L=L)

    omega_hat = helpers.gaussian_noise(mesh, jax.random.PRNGKey(0), E0=2.0)
    # omega_hat = helpers.Taylor_green_vortex(mesh)
    # omega_hat = helpers.dipole_vortex(mesh, E0=2.0)


    u, v = get_velocities(omega_hat, mesh)
    CFL = 0.5
    dt = CFL * (L / N) / jnp.max(jnp.sqrt(u**2 + v**2))
    dt_viscous = 0.1 * (L / N)**2 / nu
    T = 10.
    N_t = int(T / dt)
    print(f"dt = {dt:.3e}, nsteps = {N_t}")
    E = []
    Eta = []
    P = []
    for n in range(N_t):
        omega_hat, (Energy, Enstrophy, Palinstrophy) = step_IMEX(omega_hat, mesh, dt, **kwargs)
        if n%50 == 0:
            omega_hat = omega_hat * mesh.dealias
        if n % 100 == 0:
            print(f"Step {n}/{N_t}")
        E.append(Energy)
        Eta.append(Enstrophy)
        P.append(Palinstrophy)
            

    omega = jnp.real(jnp.fft.ifft2(omega_hat))
    u, v = get_velocities(omega_hat, mesh)

    mesh.plot_field(omega)
    mesh.plot_field(u, clb_title=r"$u$")
    mesh.plot_field(v, clb_title=r"$v$")

    fig, ax = plt.subplots()
    ax.plot(jnp.arange(N_t) * dt, E, label=r"Energy")
    # ax.plot(jnp.arange(N_t) * dt, E[0] * jnp.exp(-4 * nu * jnp.arange(N_t) * dt), label=r"Energy")
    ax.set_xlabel("Time")
    ax.legend()

    # energy spectrum
    k_bins,E_k_bin  = helpers.get_energy_spectrum(omega_hat, mesh)
    fig, ax = plt.subplots()
    ax.loglog(k_bins, E_k_bin, label=r"$E(k)$")
    ax.set_xlabel(r"Wavenumber $k$")
    ax.set_ylabel(r"Energy Spectrum $E(k)$")
    ax.legend()
    ax.grid(True)

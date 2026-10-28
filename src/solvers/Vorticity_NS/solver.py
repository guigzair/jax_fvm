import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mesh as mesh_vort
import helpers as helpers
import os
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
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

def get_forcing_term(mesh, key, k=8, dk = 1., eps0=0.1):
    # white-in-time random forcing in the shell |k| in [k - dk/2, k + dk/2]
    # normalized so that the mean energy injection rate is eps0
    # (to be applied as sqrt(dt) * f_hat)
    k_mag = jnp.sqrt(mesh.k2)
    mask = (k_mag >= k - dk/2) & (k_mag <= k + dk/2)
    subkey1, subkey2 = jax.random.split(key)

    noise_real = jax.random.normal(subkey1, (mesh.N, mesh.N))
    noise_image = jax.random.normal(subkey2, (mesh.N, mesh.N))

    noise = noise_real + 1j * noise_image
    f_hat = jnp.where(mask, noise, 0.)

    # enforce Hermitian symmetry (real forcing)
    f = jnp.real(jnp.fft.ifft2(f_hat))
    f_hat = jnp.fft.fft2(f)

    eps = helpers.get_energy(f_hat, mesh)
    f_hat = f_hat * jnp.sqrt(eps0 / eps)  # injection rate = eps0
    return f_hat

def get_forcing_term_paper(mesh, key, eps=1.):
    #   eta = W1 sin(6x) + W2 cos(7x) + W3 sin(5(x+y)) + W4 cos(8(x+y))
    #       + W5 cos(6x) + W6 sin(7x) + W7 cos(5(x+y)) + W8 sin(8(x+y))
    # with W_i independent Wiener processes; the equation is forced by eps * d(eta)
    # (to be applied as sqrt(dt) * f_hat, since dW_i = sqrt(dt) * xi_i)
    X, Y = mesh.X, mesh.Y
    basis = jnp.stack([
        jnp.sin(6 * X),       jnp.cos(7 * X),       jnp.sin(5 * (X + Y)), jnp.cos(8 * (X + Y)),
        jnp.cos(6 * X),       jnp.sin(7 * X),       jnp.cos(5 * (X + Y)), jnp.sin(8 * (X + Y)),
    ])
    xi = jax.random.normal(key, (8,))
    f = eps * jnp.tensordot(xi, basis, axes=1)
    return jnp.fft.fft2(f)


def get_forcing_term_kolmogorov(mesh, n=4, F=1.):
    # Kolmogorov flow: steady body force F sin(n y) e_x on the velocity,
    # i.e. curl f = -dy(F sin(n y)) = -n F cos(n y) on the vorticity
    # (to be applied as dt * f_hat, deterministic)
    f = -n * F * jnp.cos(n * mesh.Y)
    return jnp.fft.fft2(f)


def get_forcing(mesh, key, forcing='shell', **kwargs):
    # returns (f_hat, stochastic): stochastic forcings are white in time and applied
    # as sqrt(dt) * f_hat, deterministic ones as dt * f_hat
    if forcing == 'none':
        return jnp.zeros_like(mesh.X), False
    if forcing == 'paper':
        return get_forcing_term_paper(mesh, key, eps=kwargs.get('eps', 1.)), True
    if forcing == 'kolmogorov':
        n = kwargs.get('kolmo_n', 4)
        F = kwargs.get('kolmo_F', 1.)
        return get_forcing_term_kolmogorov(mesh, n=n, F=F), False
    k = kwargs.get('k', 8)
    dk = kwargs.get('dk', 1.)
    eps0 = kwargs.get('eps0', 0.1)
    return get_forcing_term(mesh, key, k=k, dk=dk, eps0=eps0), True

def get_diagnostics(omega_hat, mesh):
    Energy = helpers.get_energy(omega_hat, mesh)
    Enstrophy = helpers.get_enstrophy(omega_hat, mesh)
    Palinstrophy = helpers.get_palinstrophy(omega_hat, mesh)
    return Energy, Enstrophy, Palinstrophy


@jax.jit(static_argnums=(1,), static_argnames=('forcing',))
def step_IMEX(omega_hat, mesh, dt, key, forcing='shell', **kwargs):
    # IMEX Euler: explicit Euler for advection, implicit Euler for viscosity + drag
    omega_hat = omega_hat.at[0,0].set(0.0)
    adv = get_convection_term(omega_hat, mesh, **kwargs)
    nu = kwargs.get('nu', 1e-2)
    alpha = kwargs.get('alpha', 0.)  # linear (Ekman) drag

    f_hat, stochastic = get_forcing(mesh, key, forcing=forcing, **kwargs)
    f_hat = jnp.sqrt(dt) * f_hat if stochastic else dt * f_hat

    omega_hat = (omega_hat - dt * adv + f_hat) / (1.0 + dt * (nu * mesh.k2 + alpha))
    return omega_hat, get_diagnostics(omega_hat, mesh)


# ARS(2,2,2) coefficients (Ascher, Ruuth & Spiteri 1997)
GAMMA_ARS = 1. - 1. / jnp.sqrt(2.)
DELTA_ARS = 1. - 1. / (2. * GAMMA_ARS)

@jax.jit(static_argnums=(1,), static_argnames=('forcing',))
def step_IMEX_RK2(omega_hat, mesh, dt, key, forcing='shell', **kwargs):
    # IMEX RK2, ARS(2,2,2): explicit RK2 for advection, L-stable DIRK for viscosity + drag
    #   implicit tableau          explicit tableau
    #   0 | 0  0    0             0 | 0  0    0
    #   g | 0  g    0             g | g  0    0
    #   1 | 0  1-g  g             1 | d  1-d  0
    # stiffly accurate: omega^{n+1} = third stage
    # A stochastic forcing sqrt(dt) * f is added at the start of the step (Euler-Maruyama);
    # a deterministic forcing is part of the explicit term, so the scheme stays 2nd order.
    g, d = GAMMA_ARS, DELTA_ARS
    nu = kwargs.get('nu', 1e-2)
    alpha = kwargs.get('alpha', 0.)  # linear (Ekman) drag
    Lin = -(nu * mesh.k2 + alpha)    # diagonal implicit operator
    Lin = Lin.at[0,0].set(0.0)

    f_hat, stochastic = get_forcing(mesh, key, forcing=forcing, **kwargs)
    if stochastic:
        omega0 = omega_hat.at[0,0].set(0.0) + jnp.sqrt(dt) * f_hat
        f_det = 0.
    else:
        omega0 = omega_hat.at[0,0].set(0.0)
        f_det = f_hat

    # stage 1 (explicit only)
    N1 = -get_convection_term(omega0, mesh, **kwargs) + f_det

    # stage 2
    omega1 = (omega0 + dt * g * N1) / (1.0 - dt * g * Lin)
    N2 = -get_convection_term(omega1, mesh, **kwargs) + f_det

    # stage 3
    omega_hat = (omega0 + dt * (d * N1 + (1. - d) * N2) + dt * (1. - g) * Lin * omega1) / (1.0 - dt * g * Lin)
    omega_hat = omega_hat.at[0,0].set(0.0)
    return omega_hat, get_diagnostics(omega_hat, mesh)



if __name__ == "__main__":
    # -----------------------------
    # Parameters
    # -----------------------------
    # N = 2048
    # L = 2 * jnp.pi
    # nu = 1e-3
    N = 1024
    L = 1.        # Kelvin-Helmholtz multimode, as KelvinHelmholtzMultimode.yaml
    nu = 1e-5     # = mu / rho with rho = 1

    # forcing = 'paper' # 'paper', 'shell', 'kolmogorov'
    forcing = 'none' # 'none', 'paper', 'shell', 'kolmogorov'
    eps = 1.     # 'paper': amplitude of the forcing
    k = 5.       # 'shell': forcing wavenumber
    dk = 1.      # 'shell': shell width
    eps0 = 1.    # 'shell': energy injection rate
    kolmo_n = 4  # 'kolmogorov': forcing wavenumber
    kolmo_F = 1. # 'kolmogorov': forcing amplitude
    # alpha = 0.1  # linear drag, saturation energy ~ eps0 / (2 alpha)
    alpha = 0.   # linear drag, saturation energy ~ eps0 / (2 alpha)

    kwargs = {'nu': nu, 'k': k, 'dk': dk, 'eps0': eps0, 'eps': eps,
              'kolmo_n': kolmo_n, 'kolmo_F': kolmo_F, 'alpha': alpha}

    # -----------------------------
    # Random
    # -----------------------------
    key = jax.random.PRNGKey(0)

    # -----------------------------
    # Mesh
    # -----------------------------
    mesh = mesh_vort.Mesh()
    mesh.mesh_generator(N=N, L=L)

    # omega_hat = helpers.gaussian_noise(mesh, jax.random.PRNGKey(0), E0=2.0)
    omega_hat = helpers.kelvin_helmholtz_multimode(mesh, sigma=0.01, alpha=0.02, m_max=20, seed=0)
    # omega_hat = helpers.Taylor_green_vortex(mesh)
    # omega_hat = helpers.dipole_vortex(mesh, E0=2.0)


    u, v = get_velocities(omega_hat, mesh)
    CFL = 0.1
    dt_cfl = CFL * (L / N) / jnp.max(jnp.sqrt(u**2 + v**2))
    # explicit advection damped by implicit viscosity: |G| <= 1 requires dt <= 2 nu / |u|^2
    # (von Neumann estimate for IMEX Euler, conservative for IMEX-RK2)
    dt_viscous = 2 * nu / jnp.max(u**2 + v**2)
    dt = min(float(dt_cfl), float(dt_viscous))
    # T = 4
    T = 3.
    N_t = int(T / dt)
    print(f"dt_cfl = {dt_cfl:.3e}, dt_viscous = {dt_viscous:.3e} -> dt = {dt:.3e}, nsteps = {N_t}")
    E = []
    Eta = []
    P = []
    T_interval_snapshots = 1000
    Snapshots_vort = jnp.zeros((N_t//T_interval_snapshots, N, N), dtype=jnp.float32)
    for n in range(N_t):
        key, subkey = jax.random.split(key)
        omega_hat, (Energy, Enstrophy, Palinstrophy) = step_IMEX_RK2(omega_hat, mesh, dt, subkey, forcing=forcing, **kwargs)
        if n%50 == 0:
            omega_hat = omega_hat * mesh.dealias
        if n % 100 == 0:
            print(f"Step {n}/{N_t}")
        E.append(Energy)
        Eta.append(Enstrophy)
        P.append(Palinstrophy)
        if n % T_interval_snapshots == 0:
            Snapshots_vort = Snapshots_vort.at[n//T_interval_snapshots].set(jnp.real(jnp.fft.ifft2(omega_hat)))

    # one device->host transfer instead of one per scalar
    E, Eta, P = (jnp.stack(x).tolist() for x in (E, Eta, P))

    omega = jnp.real(jnp.fft.ifft2(omega_hat))
    u, v = get_velocities(omega_hat, mesh)

    fig, ax = mesh.plot_field(omega)
    fig.savefig('omega_final.pdf', dpi=500)
    # mesh.plot_field(u, clb_title=r"$u$")
    # mesh.plot_field(v, clb_title=r"$v$")

    fig, ax = plt.subplots()
    ax.plot(jnp.arange(N_t) * dt, E, label=r"Energy")
    # ax.plot(jnp.arange(N_t) * dt, E[0] * jnp.exp(-4 * nu * jnp.arange(N_t) * dt), label=r"Energy")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True)

    # energy spectrum
    k_bins,E_k_bin  = helpers.get_energy_spectrum(omega_hat, mesh)
    fig, ax = plt.subplots()
    ax.loglog(k_bins, E_k_bin, label=r"$E(k)$")
    ax.set_xlabel(r"Wavenumber $k$")
    ax.set_ylabel(r"Energy Spectrum $E(k)$")
    ax.legend()
    ax.grid(True)

    # gif
    mesh.animate_field(Snapshots_vort, interval = 100)


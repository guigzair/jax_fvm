import sys
import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
from matplotlib.colors import ListedColormap, to_rgb

# position in [0, 1] -> colour.  0 = most negative, 0.5 = zero, 1 = most positive
_NODES = [
    # negative vorticity: dark core -> lavender rim -> deep blue -> blue -> cyan -> teal
    (0.000, "#14081f"),
    (0.025, "#2b1552"),
    (0.055, "#7c58b4"),
    (0.085, "#2e3c98"),
    (0.120, "#0a42a0"),
    (0.180, "#024fa8"),
    (0.240, "#005db4"),
    (0.300, "#006dbf"),
    (0.350, "#007dc9"),
    (0.390, "#0088cc"),
    (0.420, "#019cc4"),
    (0.450, "#018aa6"),
    (0.475, "#016d85"),
    (0.500, "#015563"),  # zero: background teal
    # positive vorticity: dark teal -> black -> sage -> cream white -> olive
    #                     -> yellow -> orange -> red -> dark red core
    (0.525, "#024551"),
    (0.550, "#03363c"),
    (0.575, "#05292c"),
    (0.600, "#0c1a1b"),
    (0.620, "#1a1a16"),
    (0.645, "#485246"),
    (0.670, "#849387"),
    (0.695, "#cbcdba"),
    (0.720, "#ecebd8"),
    (0.740, "#cdbfa5"),
    (0.760, "#7a6628"),
    (0.785, "#c9900e"),
    (0.810, "#f0ad08"),
    (0.845, "#f98e04"),
    (0.880, "#fb6603"),
    (0.915, "#f23f02"),
    (0.950, "#cf2202"),
    (0.980, "#8a0c04"),
    (1.000, "#4c0600"),
]
 
 
def make_vortex_cmap(gamma=1.0, smooth=0.006, N=1024, name="vortex"):
    """
    gamma  : >1 widens the teal band around zero so only strong vorticity gets
             coloured (reference-like look); 1 = nodes used as-is.
             Keeps zero at the centre, so still use symmetric vmin/vmax.
    smooth : width of the Gaussian blur applied along the colormap (fraction of
             its length). Removes kinks between nodes so all transitions are
             continuous; 0 = plain piecewise-linear.
    """
    pos = np.array([p for p, _ in _NODES])
    rgb = np.array([to_rgb(c) for _, c in _NODES])
 
    x = np.linspace(0, 1, N)
    ramp = np.column_stack([np.interp(x, pos, rgb[:, i]) for i in range(3)])
 
    if smooth > 0:
        s = smooth * N
        h = int(np.ceil(4 * s))
        k = np.exp(-0.5 * (np.arange(-h, h + 1) / s) ** 2)
        k /= k.sum()
        pad = np.pad(ramp, ((h, h), (0, 0)), mode="edge")
        ramp = np.column_stack([np.convolve(pad[:, i], k, mode="valid")
                                for i in range(3)])
 
    if gamma != 1.0:
        u = 2 * x - 1
        xs = 0.5 + 0.5 * np.sign(u) * np.abs(u) ** gamma
        ramp = np.column_stack([np.interp(xs, x, ramp[:, i]) for i in range(3)])
 
    return ListedColormap(np.clip(ramp, 0, 1), name=name)
 
 


class Mesh:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def mesh_generator(self, N = 128, L = 2 * jnp.pi):
        self.L = L
        self.N = N
        x = jnp.linspace(0, L, N, endpoint=False)
        y = jnp.linspace(0, L, N, endpoint=False)
        self.X, self.Y = jnp.meshgrid(x, y, indexing="ij")

        # -----------------------------
        # Fourier wavenumbers
        # -----------------------------
        k = jnp.fft.fftfreq(N, L/N) * 2 * jnp.pi
        self.kx, self.ky = jnp.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.k2 = self.k2.at[0,0].set(1.0)


        # -----------------------------
        # Dealias mask (2/3 rule)
        # -----------------------------
        kmax = jnp.max(jnp.abs(k))
        self.dealias = (jnp.abs(self.kx) < 2/3*kmax) & (jnp.abs(self.ky) < 2/3*kmax)

    def plot_field(self, field, clb_title=r"$\omega$"):
        fig, ax = plt.subplots(dpi = 500)
        # imshow is much faster than pcolormesh on a uniform grid (field is indexed [x, y])
        c = ax.imshow(np.asarray(field).T, origin='lower', extent=[0, self.L, 0, self.L], cmap='twilight')
        clb = fig.colorbar(c)
        clb.ax.set_title(clb_title)
        return fig, ax

    def animate_field(self, field_sequence, interval=100, clb_title=r"$\omega$"):
        fig, ax = plt.subplots(figsize=(6,5))

        im = ax.imshow(field_sequence[0]/jnp.max(jnp.abs(field_sequence[0])),
                    origin="lower",
                    extent=[0, self.L, 0, self.L])
        ax.set_title("2D Navier-Stokes Vorticity")
        plt.colorbar(im, ax=ax)

        def update(frame):
            im.set_data(field_sequence[frame]/jnp.max(jnp.abs(field_sequence[frame])))
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(field_sequence), interval=interval)

        ani.save('vorticity_evolution.gif', writer='imagemagick')
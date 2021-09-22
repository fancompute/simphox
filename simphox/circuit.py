import jax.numpy as jnp
import numpy as np
import xarray as xr
from dphox.device import Device
from dphox.pattern import Pattern

from .fdfd import FDFD
from .typing import Callable, Iterable, List, Optional, Size, Union


class Component:
    def __init__(self, structure: Union[Pattern, Device],
                 model: Union[xr.DataArray, Callable[[jnp.ndarray], xr.DataArray]], name: str):
        """A component in a circuit will have some structure that can be simulated
        (a pattern or device defined in DPhox), a model, and a name string.

        Args:
            structure: Structure of the device.
            model: Model of the device (in terms of wavelength).
            name: Name of the component (string representing the model name).
        """
        self.structure = structure
        self.model = model
        self.name = name

    @classmethod
    def from_fdfd(cls, pattern: Pattern, core_eps: float, clad_eps: float, spacing: float, wavelengths: Iterable[float],
                  boundary: Size, pml: float, name: str, in_ports: Optional[List[str]] = None,
                  out_ports: Optional[List[str]] = None, component_t: float = 0, component_zmin: Optional[float] = None,
                  rib_t: float = 0, sub_z: float = 0, height: float = 0, bg_eps: float = 1, profile_size_factor: int = 3,
                  pbar: Optional[Callable] = None):
        """From FDFD, this classmethod produces a component model based on a provided pattern
        and simulation attributes (currently configured for scalar photonics problems).

        Args:
            pattern: component provided by DPhox
            core_eps: core epsilon
            clad_eps: clad epsilon
            spacing: spacing required
            wavelengths: wavelengths
            boundary: boundary size around component
            pml: PML size (see :code:`FDFD` class for details)
            name: component name
            in_ports: input ports
            out_ports: output ports
            height: height for 3d simulation
            sub_z: substrate minimum height
            component_zmin: component height (defaults to substrate_z)
            component_t: component thickness
            rib_t: rib thickness for component (partial etch)
            bg_eps: background epsilon (usually 1 or air)
            profile_size_factor: profile size factor (multiply port size dimensions to get mode dimensions at each port)
            pbar: Progress bar (e.g. TQDM in a notebook which can be a valuable progress indicator).

        Returns:
            Initialize a component which contains a structure (for port specificication and visualization purposes)
            and model describing the component behavior.

        """
        sparams = []

        iterator = wavelengths if pbar is None else pbar(wavelengths)
        for wl in iterator:
            fdfd = FDFD.from_pattern(
                component=pattern,
                core_eps=core_eps,
                clad_eps=clad_eps,
                spacing=spacing,
                height=height,
                boundary=boundary,
                pml=pml,
                component_t=component_t,
                component_zmin=component_zmin,
                wavelength=wl,
                rib_t=rib_t,
                sub_z=sub_z,
                bg_eps=bg_eps,
                name=f'{name}_{wl}um'
            )
            sparams_wl = []
            for port in fdfd.port:
                s, _ = fdfd.get_sim_sparams_fn(port, profile_size_factor=profile_size_factor)(fdfd.eps)
                sparams_wl.append(s)
            sparams.append(sparams_wl)

        model = xr.DataArray(
            data=sparams,
            dims=["wavelengths", "in_ports", "out_ports"],
            coords={
                "wavelengths": wavelengths,
                "in_ports": in_ports,
                "out_ports": out_ports
            }
        )

        return cls(pattern, model=model, name=name)




def dc(epsilon):
    return jnp.array([
        [jnp.cos(np.pi / 4 + epsilon), 1j * jnp.sin(np.pi / 4 + epsilon)],
        [1j * jnp.sin(np.pi / 4 + epsilon), jnp.cos(np.pi / 4 + epsilon)]
    ])


def ps(upper, lower):
    return jnp.array([
        [jnp.exp(1j * upper), 0],
        [0, jnp.exp(1j * lower)]
    ])


def mzi(theta, phi, n=2, i=0, j=None,
        theta_upper=0, phi_upper=0, epsilon=0, dtype=np.complex128):
    j = i + 1 if j is None else j
    epsilon = epsilon if isinstance(epsilon, tuple) else (epsilon, epsilon)
    mat = np.eye(n, dtype=dtype)
    mzi_mat = dc(epsilon[1]) @ ps(theta_upper, theta) @ dc(epsilon[0]) @ ps(phi_upper, phi)
    mat[i, i], mat[i, j] = mzi_mat[0, 0], mzi_mat[0, 1]
    mat[j, i], mat[j, j] = mzi_mat[1, 0], mzi_mat[1, 1]
    return mat


def balanced_tree(n):
    # this is just defined for powers of 2 for simplicity
    assert np.floor(np.log2(n)) == np.log2(n)
    return [(2 * j * (2 ** k), 2 * j * (2 ** k) + (2 ** k))
            for k in range(int(np.log2(n)))
            for j in reversed(range(n // (2 * 2 ** k)))], n


def diagonal_tree(n, m=0):
    return [(i, i + 1) for i in reversed(range(m, n - 1))], n


def mesh(thetas, phis, network, phases=None, epsilons=None):
    ts, n = network
    u = jnp.eye(n)
    epsilons = jnp.zeros_like(thetas) if epsilons is None else epsilons
    for theta, phi, t, eps in zip(thetas, phis, ts, epsilons):
        u = mzi(theta, phi, n, *t, eps) @ u
    if phases is not None:
        u = np.diag(phases) @ u
    return u


def nullify(vector, i, j=None):
    n = len(vector)
    j = i + 1 if j is None else j
    theta = -jnp.arctan2(np.abs(vector[i]), np.abs(vector[j])) * 2
    phi = jnp.angle(vector[i]) - np.angle(vector[j])
    nullified_vector = mzi(theta, phi, n, i, j) @ vector
    return jnp.mod(theta, 2 * jnp.pi), jnp.mod(phi, 2 * jnp.pi), nullified_vector


# assumes topologically-ordered tree (e.g. above tree functions)
def analyze(v, tree):
    ts, n = tree
    thetas, phis = jnp.zeros(len(ts)), jnp.zeros(len(ts))
    for i, t in enumerate(ts):
        thetas[i], phis[i], v = nullify(v, *t)
    return thetas, phis, v[0]


def triangular(u):
    thetas, phis, mzi_lists = [], [], []
    n = u.shape[0]
    for i in range(n - 1):
        tree = diagonal_tree(n, i)
        mzi_lists_i, _ = tree
        thetas_i, phis_i, _ = analyze(u.T[i], tree)
        u = mesh(thetas_i, phis_i, tree) @ u
        thetas.extend(thetas_i)
        phis.extend(phis_i)
        mzi_lists.extend(mzi_lists_i)
    phases = jnp.angle(jnp.diag(u))
    return jnp.asarray(thetas), jnp.asarray(phis), (mzi_lists, n), phases


def generate(thetas, phis, tree, epsilons=None):
    return mesh(thetas, phis, tree, epsilons)[0]


def random_complex(n):
    return jnp.array(np.random.randn(n) + np.random.randn(n) * 1j)

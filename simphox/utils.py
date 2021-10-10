import numpy as np
import scipy.sparse as sp
from pydantic.dataclasses import dataclass
import jax.numpy as jnp

from typing import Tuple, Union, Optional
from copy import deepcopy
import xarray as xr

from .typing import List, Callable, Size2, Size3

SMALL_NUMBER = 1e-20


def fix_dataclass_init_docs(cls):
    """Fix the ``__init__`` documentation for a :class:`dataclasses.dataclass`.

    See Also:
        https://github.com/agronholm/sphinx-autodoc-typehints/issues/123

    Attributes:
        cls: The class whose docstring needs fixing

    Returns:
        The class that was passed so this function can be used as a decorator
    """
    cls.__init__.__qualname__ = f'{cls.__name__}.__init__'
    return cls


@fix_dataclass_init_docs
@dataclass
class Material:
    """Helper class for materials.

    Attributes:
        name: Name of the material.
        eps: Constant epsilon (relative permittivity) assigned for the material.
        facecolor: Facecolor in red-green-blue (RGB) for drawings (default is black or :code:`(0, 0, 0)`).
    """
    name: str
    eps: float = 1.
    facecolor: Size3 = (0, 0, 0)

    def __str__(self):
        return self.name


SILICON = Material('Silicon', 3.4784 ** 2, (0.3, 0.3, 0.3))
POLYSILICON = Material('Poly-Si', 3.4784 ** 2, (0.5, 0.5, 0.5))
AIR = Material('Air')
OXIDE = Material('Oxide', 1.4442 ** 2, (0.6, 0, 0))
NITRIDE = Material('Nitride', 1.996 ** 2, (0, 0, 0.7))
LS_NITRIDE = Material('Low-Stress Nitride', facecolor=(0, 0.4, 1))
LT_OXIDE = Material('Low-Temp Oxide', 1.4442 ** 2, (0.8, 0.2, 0.2))
ALUMINUM = Material('Aluminum', facecolor=(0, 0.5, 0))
ALUMINA = Material('Alumina', 1.75, (0.2, 0, 0.2))
ETCH = Material('Etch')

TEST_ZERO = Material('Zero', 0, (0, 0, 0))
TEST_ONE = Material('One', 1, (0, 0, 0))
TEST_INF = Material('Inf', 1e10, (0, 0, 0))


@fix_dataclass_init_docs
@dataclass
class Box:
    """Helper class for quickly generating functions for design region placements.

    Attributes:
        size: size of box
        spacing: spacing for pixelation
        material: :code:`Material` for this Box
        min: min x and min y of box
    """
    size: Union[float, Size2]
    spacing: float = 1
    material: Optional[Material] = None
    min: Size2 = (0., 0.)

    def __post_init_post_parse__(self):
        self.size = (self.size, 0) if isinstance(self.size, float) else self.size
        self.eps = self.material.eps if self.material is not None else None

    @property
    def max(self):
        return self.min[0] + self.size[0], self.min[1] + self.size[1]

    @property
    def min_i(self):
        return int(self.min[0] / self.spacing), int(self.min[1] / self.spacing)

    @property
    def max_i(self):
        return int(self.max[0] / self.spacing), int(self.max[1] / self.spacing)

    @property
    def shape(self):
        return self.max_i[0] - self.min_i[0], self.max_i[1] - self.min_i[1]

    @property
    def center(self) -> Size2:
        return self.min[0] + self.size[0] / 2, self.min[1] + self.size[1] / 2

    @property
    def slice(self) -> Tuple[slice, slice]:
        return slice(self.min_i[0], self.max_i[0]), slice(self.min_i[1], self.max_i[1])

    @property
    def copy(self) -> "Box":
        return deepcopy(self)

    def mask(self, array: Union[np.ndarray, jnp.ndarray]):
        mask = np.zeros_like(array)
        mask[self.slice[0], self.slice[1]] = 1.0
        return mask

    def translate(self, dx: float = 0, dy: float = 0) -> "Box":
        self.min = (self.min[0] + dx, self.min[1] + dy)
        return self

    def align(self, c: Union["Box", Tuple[float, float]]) -> "Box":
        center = c.center if isinstance(c, Box) else c
        self.translate(center[0] - self.center[0], center[1] - self.center[1])
        return self

    def halign(self, c: Union["Box", float], left: bool = True, opposite: bool = True):
        x = self.min[0] if left else self.max[0]
        p = c if isinstance(c, float) or isinstance(c, int) \
            else (c.min[0] if left and not opposite or opposite and not left else c.max[0])
        self.translate(dx=p - x)
        return self

    def valign(self, c: Union["Box", float], bottom: bool = True, opposite: bool = True):
        y = self.min[1] if bottom else self.max[1]
        p = c if isinstance(c, float) or isinstance(c, int) \
            else (c.min[1] if bottom and not opposite or opposite and not bottom else c.max[1])
        self.translate(dy=p - y)
        return self


def poynting_fn(axis: int = 2, use_jax: bool = False):
    ax = np.roll((1, 2, 0), -axis)
    xp = jnp if use_jax else np

    def poynting(e: np.ndarray, h: np.ndarray):
        e_cross = xp.stack([(e[ax[0]] + xp.roll(e[ax[0]], shift=1, axis=1)) / 2,
                            (e[ax[1]] + xp.roll(e[ax[1]], shift=1, axis=0)) / 2])
        h_cross = xp.stack([(h[ax[0]] + xp.roll(h[ax[0]], shift=1, axis=0)) / 2,
                            (h[ax[1]] + xp.roll(h[ax[1]], shift=1, axis=1)) / 2])
        return e_cross[ax[0]] * h_cross.conj()[ax[1]] - e_cross[ax[1]] * h_cross.conj()[ax[0]]
    return poynting


def d2curl_op(d: List[sp.spmatrix]) -> sp.spmatrix:
    o = sp.csr_matrix((d[0].shape[0], d[0].shape[0]))
    return sp.bmat([[o, -d[2], d[1]],
                    [d[2], o, -d[0]],
                    [-d[1], d[0], o]])


def curl_fn(df: Callable[[np.ndarray, int], np.ndarray], use_jax: bool = False, beta: float = None):
    xp = jnp if use_jax else np
    if beta is not None:
        def _curl(f: np.ndarray):
            return xp.stack([df(f[2], 1) + 1j * beta * f[1],
                             -1j * beta * f[0] - df(f[2], 0),
                             df(f[1], 0) - df(f[0], 1)])
    else:
        def _curl(f: np.ndarray):
            return xp.stack([df(f[2], 1) - df(f[1], 2),
                             df(f[0], 2) - df(f[2], 0),
                             df(f[1], 0) - df(f[0], 1)])
    return _curl


def yee_avg(params: np.ndarray, shift: int = 1) -> np.ndarray:
    p = params
    p_x = (p + np.roll(p, shift=shift, axis=1)) / 2
    p_y = (p + np.roll(p, shift=shift, axis=0)) / 2
    p_z = (p_y + np.roll(p_y, shift=shift, axis=1)) / 2
    return np.stack([p_x, p_y, p_z])


def yee_avg_2d_z(params: jnp.ndarray) -> jnp.ndarray:
    p = params
    p_y = (p + jnp.roll(p, shift=1, axis=0)) / 2
    p_z = (p_y + jnp.roll(p_y, shift=1, axis=1)) / 2
    return p_z


def yee_avg_jax(params: jnp.ndarray) -> jnp.ndarray:
    p = params
    p_x = (p + jnp.roll(p, shift=1, axis=1)) / 2
    p_y = (p + jnp.roll(p, shift=1, axis=0)) / 2
    p_z = (p_y + jnp.roll(p_y, shift=1, axis=1)) / 2
    return jnp.stack((p_x, p_y, p_z))


def pml_params(pos: np.ndarray, t: int, exp_scale: float, log_reflection: float, absorption_corr: float):
    d = np.vstack(((pos[:-1] + pos[1:]) / 2, pos[:-1])).T
    d_pml = np.vstack((
        (d[t] - d[:t]) / (d[t] - pos[0]),
        np.zeros_like(d[t:-t]),
        (d[-t:] - d[-t]) / (pos[-1] - d[-t])
    )).T
    sigma = (exp_scale + 1) * (d_pml ** exp_scale) * log_reflection / (2 * absorption_corr)
    alpha = (1 - d_pml) ** exp_scale
    return sigma, alpha


# Real-time splitter metrics
def splitter_metrics(sparams: xr.DataArray):
    powers = np.abs(sparams) ** 2
    return {
        'reflectivity': powers.loc["b0"] / (powers.loc["b0"] + powers.loc["b1"]),
        'transmissivity': powers.loc["b1"] / (powers.loc["b0"] + powers.loc["b1"]),
        'reflection': powers.loc["a0"],
        'insertion': powers.sum(),
        'upper': powers.loc["b0"],
        'lower': powers.loc["b1"],
    }


def random_complex(n: int, normed: bool = False) -> np.ndarray:
    """Generate a random complex normal vector.

    Args:
        n: Number of inputs.
        normed: Whether to norm the random complex vector so that the norm of the vector is 1.

    Returns:
        The random complex normal vector.

    """
    z = np.array(0.5 * np.random.randn(n) + 0.5 * np.random.randn(n) * 1j)
    return z / np.linalg.norm(z) if normed else z

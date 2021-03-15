import numpy as np
import scipy.sparse as sp
import jax.ops
import jax.numpy as jnp
from jax.scipy.signal import convolve as conv
from skimage.draw import circle
from typing import Tuple, Union, Optional

from .typing import List, Callable, Dim2


class Box:
    def __init__(self, size: Dim2, spacing: float, min: Dim2 = (0, 0), reverse: Tuple[bool, bool] = (False, False)):
        """Helper class for quickly generating functions for design region placements

        Args:
            size: size of box
            min: min x and min y of box
            spacing: spacing for pixelation
        """
        self.min = np.asarray(min)
        self.size = np.asarray(size)
        self.spacing = spacing
        self.reverse = reverse

    @property
    def max(self):
        return self.min[0] + self.size[0], self.min[1] + self.size[1]

    @property
    def min_i(self):
        return int(self.min[0] // self.spacing), int(self.min[1] // self.spacing)

    @property
    def max_i(self):
        return int(self.max[0] // self.spacing), int(self.max[1] // self.spacing)

    @property
    def center(self):
        return self.min[0] + self.size[0] / 2, self.min[1] + self.size[1] / 2

    @property
    def slice(self):
        return slice(self.min_i[0], self.max_i[0], -1 if self.reverse[0] else 1),\
               slice(self.min_i[1], self.max_i[1], -1 if self.reverse[1] else 1)

    def mask(self, array: Union[np.ndarray, jnp.ndarray]):
        mask = np.zeros_like(array)
        mask[self.slice[0], self.slice[1]] = 1.0
        return mask

    def rot90(self) -> "Box":
        self.size = (self.size[1], self.size[0])
        return self

    def flipx(self) -> "Box":
        self.reverse = (not self.reverse[0], self.reverse[1])
        return self

    def flipy(self) -> "Box":
        self.reverse = (self.reverse[0], not self.reverse[1])
        return self

    def translate(self, dx: float = 0, dy: float = 0) -> "Box":
        self.min[0] += dx
        self.min[1] += dy
        return self

    def align(self, c: Union["Box", Tuple[float, float]]) -> "Box":
        center = c.center if isinstance(c, Box) else c
        self.translate(center[0] - self.center[0], center[1] - self.center[1])
        return self

    def halign(self, c: Union["Box", float], left: bool = True, opposite = False):
        x = self.min[0] if left else self.max[0]
        p = c if isinstance(c, float) or isinstance(c, int) \
            else (c.min[0] if left and not opposite or opposite and not left else c.max[0])
        self.translate(dx=p - x)
        return self

    def valign(self, c: Union["Box", float], bottom: bool = True, opposite=False):
        y = self.min[1] if bottom else self.max[1]
        p = c if isinstance(c, float) or isinstance(c, int) \
            else (c.min[1] if bottom and not opposite or opposite and not bottom else c.max[1])
        self.translate(dy=p - y)
        return self


def poynting_z(e: np.ndarray, h: np.ndarray):
    e_cross = np.stack([(e[0] + np.roll(e[0], shift=1, axis=1)) / 2,
                        (e[1] + np.roll(e[1], shift=1, axis=0)) / 2])
    h_cross = np.stack([(h[0] + np.roll(h[0], shift=1, axis=0)) / 2,
                        (h[1] + np.roll(h[1], shift=1, axis=1)) / 2])
    return e_cross[0] * h_cross.conj()[1] - e_cross[1] * h_cross.conj()[0]


def poynting_x(e: np.ndarray, h: np.ndarray):
    e_cross = np.stack([(e[1] + np.roll(e[1], shift=1, axis=1)) / 2,
                        (e[2] + np.roll(e[2], shift=1, axis=0)) / 2])
    h_cross = np.stack([(h[1] + np.roll(h[1], shift=1, axis=0)) / 2,
                        (h[2] + np.roll(h[2], shift=1, axis=1)) / 2])
    return e_cross[1] * h_cross.conj()[2] - e_cross[2] * h_cross.conj()[1]


def overlap(e1: np.ndarray, h1: np.ndarray, e2: np.ndarray, h2: np.ndarray):
    return (np.sum(poynting_z(e1, h2)) * np.sum(poynting_z(e2, h1)) /
            np.sum(poynting_z(e1, h1))).real / np.sum(poynting_z(e2, h2)).real


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
    if len(params.shape) == 1:
        p = (params + np.roll(params, shift=shift) + np.roll(params, shift=-shift)) / 3
        return np.stack((p, p, p))
    p = params[..., np.newaxis] if len(params.shape) == 2 else params
    p_x = (p + np.roll(p, shift=shift, axis=1)) / 2
    p_y = (p + np.roll(p, shift=shift, axis=0)) / 2
    p_z = (p_y + np.roll(p_y, shift=shift, axis=1)) / 2
    return np.stack([p_x, p_y, p_z])


def yee_avg_2d(params: jnp.ndarray) -> jnp.ndarray:
    p = params[..., jnp.newaxis]
    p_y = (p + jnp.roll(p, shift=1, axis=0)) / 2
    p_z = (p_y + jnp.roll(p_y, shift=1, axis=1)) / 2
    return p_z


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


def smooth_rho_2d_fn(eta: float, beta: float, radius: float):
    def smooth_rho(rho: jnp.ndarray):
        rr, cc = circle(radius, radius, radius + 1)
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float)
        kernel[rr, cc] = 1
        kernel = kernel / kernel.sum()
        rho = conv(rho, kernel, mode='same')
        return jnp.divide(jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta)),
                          jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))
    return smooth_rho


def place_rho_2d_fn(rho_init: jnp.ndarray, box: Union[Box, List[Box]]):
    """

    Args:
        rho_init: initial rho definition
        box: Box or list of Box instances that define position and orientation of the design regions,
            where the first box is considered the upright-oriented design region

    Returns:

    """
    boxes = box if isinstance(box, list) else [box]
    slices = [b.slice for b in boxes]
    rho_design_init = rho_init[slices[0][0], slices[0][1]]

    def place_rho(rho_design):
        rho = jnp.array(rho_init)
        for s in slices:
            rho.at[s[0], s[1]].set(rho_design)
        return rho
    return place_rho, rho_design_init


def place_rho_2d_fn_temp(rho_init: jnp.ndarray, box: Box):
    """

    Args:
        rho_init: initial rho definition
        box: Box defines position and orientation of the design region

    Returns:

    """
    mask = box.mask(rho_init)

    def place_rho(rho):
        return jnp.array(rho_init) * (1 - mask) + rho * mask
    return place_rho


def match_mode_2d(s):
    def obj(ez):
        return -jnp.sum(jnp.abs(ez.conj() * s.ravel()))  # want to minimize this to maximize mode match
    return obj

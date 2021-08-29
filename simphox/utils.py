import numpy as np
import scipy.sparse as sp
import jax.ops
import jax.numpy as jnp

from typing import Tuple, Union, Optional
from copy import deepcopy
import xarray as xr

from .typing import List, Callable, Dim2

SMALL_NUMBER = 1e-20


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


def overlap(e1: np.ndarray, h1: np.ndarray, e2: np.ndarray, h2: np.ndarray):
    return (np.sum(poynting_fn(2)(e1, h2)) * np.sum(poynting_fn(2)(e2, h1)) /
            np.sum(poynting_fn(2)(e1, h1))).real / np.sum(poynting_fn(2)(e2, h2)).real


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

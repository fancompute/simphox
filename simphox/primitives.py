from jax import core
from typing import Tuple

from .mkl import spsolve_pardiso
import jax.numpy as jnp
import jax
from jax.config import config
import jax.experimental.host_callback as hcb
config.parse_flags_with_absl()

import scipy.sparse as sp


def _spsolve_hcb(ab: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
    a_entries, b, a_indices = ab
    a = sp.coo_matrix((a_entries, (a_indices[0], a_indices[1])), shape=(b.size, b.size))
    # caching is not necessary since the pardiso spsolve we are using caches the matrix factorization by default
    # replace with a better solver if available
    return spsolve_pardiso(a, b.flatten())


@jax.custom_vjp
def spsolve(a_entries: jnp.ndarray, b: jnp.ndarray, a_indices: jnp.ndarray):
    return hcb.call(_spsolve_hcb, (a_entries, b, a_indices), result_shape=b)


def spsolve_fwd(a_entries: jnp.ndarray, b: jnp.ndarray, a_indices: jnp.ndarray):
    x = spsolve(a_entries, b, a_indices)
    return x, (a_entries, x, a_indices)


def spsolve_bwd(res, g):
    a_entries, x, a_indices = res
    lambda_ = spsolve(a_entries, g, a_indices[::-1])
    i, j = a_indices
    return -lambda_[i] * x[j], lambda_, None


spsolve.defvjp(spsolve_fwd, spsolve_bwd)

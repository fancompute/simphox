from typing import Tuple

from .mkl import spsolve_pardiso

import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp
import jax
from jax.config import config
import jax.experimental.host_callback as hcb

config.parse_flags_with_absl()


def _spsolve_hcb(ab: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    a_entries, b, a_indices = ab
    a = sp.coo_matrix((a_entries, (a_indices[0], a_indices[1])), shape=(b.size, b.size))
    # caching is not necessary since the pardiso spsolve we are using caches the matrix factorization by default
    # replace with a better solver if available
    return spsolve_pardiso(a, b.flatten())


def _spdot_hcb(ab: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]):
    t, a_entries, b_entries, a_indices, b_indices = ab
    a = sp.coo_matrix((a_entries * t[a_indices[1]], (a_indices[0], a_indices[1])), shape=(t.size, t.size))
    b = sp.coo_matrix((b_entries, (b_indices[0], b_indices[1])), shape=(t.size, t.size))
    c = a.dot(b)
    c.sort_indices()
    c = c.tocoo()
    return c.data, np.stack((np.array(c.row), np.array(c.col)))


def _spdot_bwdop_hcb(abg: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    t, a_entries, b_entries, g_entries, a_indices, b_indices, g_indices = abg
    a = sp.coo_matrix((a_entries, (a_indices[0], a_indices[1])), shape=(t.size, t.size))
    b = sp.coo_matrix((b_entries, (b_indices[0], b_indices[1])), shape=(t.size, t.size))
    g = sp.coo_matrix((g_entries, (g_indices[0], g_indices[1])), shape=(t.size, t.size))
    return b.dot(g.T.dot(a)).diagonal()


@jax.custom_vjp
def spsolve(a_entries: jnp.ndarray, b: jnp.ndarray, a_indices: jnp.ndarray) -> jnp.ndarray:
    return hcb.call(_spsolve_hcb, (a_entries, b, a_indices), result_shape=b)


def spsolve_fwd(a_entries: jnp.ndarray, b: jnp.ndarray, a_indices: jnp.ndarray) -> Tuple[jnp.ndarray,
                                                                                         Tuple[jnp.ndarray, ...]]:
    x = spsolve(a_entries, b, a_indices)
    return x, (a_entries, x, a_indices)


def spsolve_bwd(res, g):
    a_entries, x, a_indices = res
    lambda_ = spsolve(a_entries, g, a_indices[::-1])
    i, j = a_indices
    return -lambda_[i] * x[j], lambda_, None


spsolve.defvjp(spsolve_fwd, spsolve_bwd)


class SparseDot:
    def __init__(self, size):
        self.size = size

    def compile_spdot(self):
        size = self.size

        @jax.custom_vjp
        def spdot(t: jnp.ndarray, a_entries: jnp.ndarray, a_indices: jnp.ndarray,
                  b_entries: jnp.ndarray, b_indices: jnp.ndarray):
            return hcb.call(_spdot_hcb, (t, a_entries, b_entries, a_indices, b_indices),
                            result_shape=(jax.ShapeDtypeStruct((size,), np.complex128),
                                          jax.ShapeDtypeStruct((2, size), np.int32)))
            # result_shape=(jnp.zeros(size, dtype=np.complex128), jnp.zeros((2, size), dtype=np.int32)))

        def spdot_fwd(t: jnp.ndarray, a_entries: jnp.ndarray, a_indices: jnp.ndarray,
                      b_entries: jnp.ndarray, b_indices: jnp.ndarray):
            c_entries, c_indices = spdot(t, a_entries, a_indices, b_entries, b_indices, size)
            return c_entries, (t, c_entries, c_indices, a_entries, a_indices, b_entries, b_indices, size)

        def spdot_bwd(res, g):
            t, c_entries, c_indices, a_entries, a_indices, b_entries, b_indices, size = res
            v = hcb.call(_spdot_bwdop_hcb, (a_entries, b_entries, g, a_indices, b_indices, c_indices), result_shape=t)
            return v, None, None, None, None

        spdot.defvjp(spdot_fwd, spdot_bwd)

        return spdot

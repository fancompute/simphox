from typing import Tuple, List

try:  # pardiso (using Intel MKL) is much faster than scipy's solver
    from .mkl import spsolve_pardiso as _spsolve
except OSError:  # if mkl isn't installed
    from scipy.sparse.linalg import spsolve as _spsolve

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
    return _spsolve(a, b.flatten())


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


def _coo_to_jnp(mat):
    mat.sort_indices()
    mat = mat.tocoo()
    return jnp.array(mat.data, dtype=np.complex128), jnp.vstack((jnp.array(mat.row), jnp.array(mat.col)))


class TMOperator:
    """This class generates some helpful TE primitives based on the input discrete derivatives provided by the FDFD
    class for a 2D problem.

    Attributes:
        df: A list of forward discrete derivative in order (:code:`df_x`, :code:`df_y`, :code:`df_z`).
        db: A list of backward discrete derivative in order (:code:`db_x`, :code:`db_y`, :code:`db_z`).
    """
    def __init__(self, df: List[sp.spmatrix], db: List[sp.spmatrix]):
        self.df, self.db = df, db
        data_x, self.x_indices = _coo_to_jnp(self.df[0] @ self.db[0])
        data_y, self.y_indices = _coo_to_jnp(self.df[1] @ self.db[1])
        self.size = (data_x.size, data_y.size)
        self.n = df[0].diagonal().size

    def compile_operator_along_axis(self, axis: int):
        """Compiles the TE mode operator along a certain axis (0 or 1)

        Args:
            axis: Axis along which to compute the operator.

        Returns:
            The contribution to the TE operator along axis 0 or 1 specified by the input.

        """
        if axis != 0 and axis != 1:
            raise ValueError("axis must be either 0 or 1.")

        n = self.n
        size = self.size[axis]
        a = self.db[axis]
        b = self.df[axis]
        c_indices = (self.x_indices, self.y_indices)[axis]

        def _te_hcb(t: jnp.ndarray):
            tm = sp.diags(t)
            c = a.dot(tm).dot(b)
            c.sort_indices()
            c = c.tocoo()
            return c.data

        def _te_backward_hcb(g: jnp.ndarray) -> jnp.ndarray:
            g = sp.coo_matrix((g, (c_indices[1], c_indices[0])), shape=(n, n))
            complex_res = b.dot(g.dot(a)).diagonal()
            return complex_res.real

        @jax.custom_vjp
        def te(t: jnp.ndarray):
            return hcb.call(_te_hcb, t, result_shape=jax.ShapeDtypeStruct((size,), np.complex128))

        def te_fwd(t: jnp.ndarray):
            return te(t), None

        def te_bwd(_, g):
            v = hcb.call(_te_backward_hcb, g, result_shape=jax.ShapeDtypeStruct((n,), np.float))
            return v,

        te.defvjp(te_fwd, te_bwd)

        return te

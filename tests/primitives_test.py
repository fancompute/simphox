import pytest
from jax import vjp
import jax.numpy as jnp
from jax.config import config
import jax.test_util as jtu
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve as spsolve_scipy

from simphox.primitives import spsolve, TMOperator

np.random.seed(0)
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')


@pytest.mark.parametrize(
    "mat, v",
    [
        (sp.spdiags(np.array([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]]), [0, 1], 5, 5), np.ones(5, dtype=np.complex128)),
        (sp.spdiags(np.array([[1, 2, 3, 8], [6, 5, 8, 300]]), [0, 1], 4, 4).transpose(), np.arange(4, dtype=np.complex128))
    ],
)
def test_spsolve_matches_scipy(mat: sp.spmatrix, v: np.ndarray):
    mat = mat.tocsr()
    expected = spsolve_scipy(mat, v)
    mat = mat.tocoo()
    mat_entries = jnp.array(mat.data, dtype=np.complex128)
    mat_indices = jnp.vstack((jnp.array(mat.row), jnp.array(mat.col)))
    x = spsolve(mat_entries, jnp.array(v), mat_indices)
    np.testing.assert_allclose(x, expected)


@pytest.mark.parametrize(
    "mat, v, g, expected",
    [
        (sp.spdiags(np.array([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]]), [0, 1], 5, 5), np.ones(5, dtype=np.complex128),
         np.ones(5, dtype=np.complex128), np.array([(1, -2, 17 / 3, -12.5, 25.2)], dtype=np.complex128)),
    ],
)
def test_spsolve_vjp_b(mat: sp.spmatrix, v: np.ndarray, g: np.ndarray, expected: np.ndarray):
    mat = mat.tocoo()
    mat_entries = jnp.array(mat.data, dtype=np.complex128)
    mat_indices = jnp.vstack((jnp.array(mat.row), jnp.array(mat.col)))
    _, vjp_fun = vjp(lambda x: spsolve(mat_entries, jnp.asarray(x), mat_indices), v)
    np.testing.assert_allclose(vjp_fun(g), expected)


@pytest.mark.parametrize(
    "mat, v, g, expected",
    [
        (sp.spdiags(np.array([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]]), [0, 1], 5, 5),
         np.ones(5, dtype=np.complex128), np.ones(5, dtype=np.complex128),
         np.array([[-726, -276, -221, -112.5, -181.44,  138,   78,   51, 90]], dtype=np.complex128) / 36),
    ],
)
def test_spsolve_vjp_mat(mat: sp.spmatrix, v: np.ndarray, g: np.ndarray, expected: np.ndarray):
    mat = mat.tocoo()
    mat_entries = jnp.array(mat.data, dtype=np.complex128)
    mat_indices = jnp.vstack((jnp.array(mat.row), jnp.array(mat.col)))
    _, vjp_fun = vjp(lambda x: spsolve(x, jnp.asarray(v), mat_indices), mat_entries)
    np.testing.assert_allclose(vjp_fun(g), expected)


# These only work when run individually at the moment...


@pytest.mark.skip(reason="This currently fails at the test tree column...")
@pytest.mark.parametrize(
    "mat1, mat2, v",
    [
        (sp.spdiags(np.array([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]]), [0, 1], 5, 5),
         sp.spdiags(np.array([[6, 5, 8, 9, 10], [1, 2, 3, 4, 5]]), [0, 1], 5, 5),
         np.ones(5, dtype=np.complex128)),
    ],
)
def test_tmoperator_numerical_grads(mat1: sp.spmatrix, mat2: sp.spmatrix, v: np.ndarray):
    operator = TMOperator([mat1, mat2], [mat2, mat1])
    op = operator.compile_operator_along_axis(axis=0)
    f = lambda x: jnp.sum(op(x)).real
    jtu.check_grads(f, (v,), order=1, modes=['rev'])


@pytest.mark.skip(reason="This currently fails at the test tree column...")
@pytest.mark.parametrize(
    "mat, v",
    [
        (sp.spdiags(np.array([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]]), [0, 1], 5, 5),
         np.ones(5, dtype=np.complex128)),
    ],
)
def test_spsolve_numerical_grads(mat, v):
    mat = mat.tocoo()
    mat_entries = jnp.array(mat.data, dtype=np.complex128)
    mat_indices = jnp.vstack((jnp.array(mat.row), jnp.array(mat.col)))
    f = lambda x: jnp.sum(spsolve(x, jnp.asarray(v), mat_indices).real)
    jtu.check_grads(f, (mat_entries,), order=1, modes=['rev'])

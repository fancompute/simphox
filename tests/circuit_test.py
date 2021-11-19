from itertools import product, zip_longest

import numpy as np
import pytest
from scipy.stats import unitary_group

from simphox.circuit import cascade, vector_unit, rectangular, balanced_tree
from simphox.circuit.vector import tree, hessian_fd, hessian_vector_unit
from simphox.utils import random_vector

import copy

np.random.seed(0)

N = [2, 4, 7, 10, 15, 16]

RAND_VECS = [random_vector(n, normed=True) for n in N]
RAND_UNITARIES = [unitary_group.rvs(n) for n in N]


@pytest.mark.parametrize(
    "n, balanced, expected_node_idxs, expected_num_columns, expected_num_top, expected_num_bottom",
    [
        (6, True, [2, 0, 1, 3, 4], 3, [3, 1, 1, 1, 1], [3, 2, 1, 2, 1]),
        (8, True, [3, 1, 0, 2, 5, 4, 6], 3, [4, 2, 1, 1, 2, 1, 1], [4, 2, 1, 1, 2, 1, 1]),
        (11, True, [4, 1, 0, 2, 3, 7, 5, 6, 8, 9], 4, [5, 2, 1, 1, 1, 3, 1, 1, 1, 1], [6, 3, 1, 2, 1, 3, 2, 1, 2, 1]),
        (6, False, [4, 3, 2, 1, 0], 5, [1, 2, 3, 4, 5], [1, 1, 1, 1, 1]),
        (8, False, [6, 5, 4, 3, 2, 1, 0], 7, [1, 2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 1, 1, 1]),
        (11, False, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0], 10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    ]
)
def test_tree_network(n: int, balanced: bool, expected_node_idxs: np.ndarray, expected_num_columns: np.ndarray,
                      expected_num_top: np.ndarray, expected_num_bottom: np.ndarray):
    circuit = tree(n, balanced=balanced)
    np.testing.assert_allclose(circuit.node_idxs, expected_node_idxs)
    np.testing.assert_allclose(circuit.num_columns, expected_num_columns)
    np.testing.assert_allclose(circuit.beta, expected_num_top)
    np.testing.assert_allclose(circuit.alpha, expected_num_bottom)


@pytest.mark.parametrize(
    "v, balanced",
    product(RAND_VECS, [True, False])
)
def test_vector_configure(v: np.ndarray, balanced: bool):
    circuit, _ = vector_unit(v, balanced=balanced)
    res = circuit.matrix_fn(use_jax=False)(circuit.params) @ v
    np.testing.assert_allclose(res, np.eye(v.size)[v.size - 1], atol=1e-10)


@pytest.mark.parametrize(
    "u, balanced",
    product(RAND_UNITARIES, [True, False])
)
def test_unitary_configure(u: np.ndarray, balanced: bool):
    circuit = cascade(u, balanced=balanced)
    np.testing.assert_allclose(circuit.matrix(), u, atol=1e-10)


@pytest.mark.parametrize(
    "u, num_columns",
    zip_longest(RAND_UNITARIES, [2 * n - 3 for n in N])
)
def test_triangular_columns(u: np.ndarray, num_columns: int):
    circuit = cascade(u, balanced=False)
    np.testing.assert_allclose(circuit.num_columns, num_columns, atol=1e-10)


@pytest.mark.parametrize(
    "u, num_columns",
    zip_longest(RAND_UNITARIES, [1, 5, 14, 25, 45, 49])
)
def test_cascade_columns(u: np.ndarray, num_columns: int):
    circuit = cascade(u, balanced=True)
    np.testing.assert_allclose(circuit.num_columns, num_columns, atol=1e-10)


@pytest.mark.parametrize(
    "u", RAND_UNITARIES
)
def test_rectangular(u: np.ndarray):
    circuit = rectangular(u)
    np.testing.assert_allclose(circuit.matrix(), u, atol=1e-10)


@pytest.mark.parametrize(
    "u", RAND_UNITARIES
)
def test_inverse(u: np.ndarray):
    circuit = rectangular(u)
    np.testing.assert_allclose(circuit.matrix(), circuit.matrix(back=True).T, atol=1e-10)


@pytest.mark.parametrize(
    "u", RAND_UNITARIES
)
def test_program_null_basis(u: np.ndarray):
    circuit = rectangular(u)
    basis = circuit.nullification_basis
    params = copy.deepcopy(circuit.params)
    circuit.program_by_null_basis(basis)
    for param, param_expected in zip(params, circuit.params):
        np.testing.assert_allclose(param, param_expected, atol=1e-10)


@pytest.mark.parametrize(
    "u, balanced", product(RAND_UNITARIES[:3], [True, False])
)
def test_hessian(u: np.ndarray, balanced: bool):
    h = hessian_vector_unit(u[0], balanced=balanced)
    h_fd = hessian_fd(u[0], balanced=balanced)
    np.testing.assert_allclose(h, h_fd, atol=1e-4)


@pytest.mark.parametrize(
    "u, balanced", product(RAND_UNITARIES[:3], [True, False])
)
def test_hessian_correlated_error(u: np.ndarray, balanced: bool):
    mesh = balanced_tree(u[0])
    vhat = mesh.matrix(params=(mesh.thetas + 0.0001, mesh.phis + 0.0001, mesh.gammas))[-1]
    np.testing.assert_allclose(
        np.sum(hessian_vector_unit(u[0], balanced=True)),
        2 * np.linalg.norm(u[0] - vhat) ** 2 / 0.0001 ** 2, atol=1e-3
    )

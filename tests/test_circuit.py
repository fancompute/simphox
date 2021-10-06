import numpy as np
import jax.numpy as jnp
import pytest
from scipy.stats import unitary_group
from itertools import product

from simphox.circuit import configure_vector, configure_unitary, CouplingNode, CouplingCircuit, tree, random_complex

np.random.seed(0)

N = np.arange(2, 16)

RAND_VECS = [random_complex(n, normed=True) for n in N]
RAND_UNITARIES = [unitary_group.rvs(n) for n in N]


@pytest.mark.parametrize(
    "n, balanced, expected_node_idxs, expected_num_levels, expected_num_top, expected_num_bottom",
    [
        (6, True, [0, 1, 2, 3, 4], 3, [3, 1, 1, 1, 1], [3, 2, 1, 2, 1]),
        (8, True, [0, 1, 2, 3, 4, 5, 6], 3, [4, 2, 1, 1, 2, 1, 1], [4, 2, 1, 1, 2, 1, 1]),
        (11, True, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4, [5, 2, 1, 1, 1, 3, 1, 1, 1, 1], [6, 3, 1, 2, 1, 3, 2, 1, 2, 1]),
        (6, False, [0, 1, 2, 3, 4], 5, [5, 4, 3, 2, 1], [1, 1, 1, 1, 1]),
        (8, False, [0, 1, 2, 3, 4, 5, 6], 7, [7, 6, 5, 4, 3, 2, 1], [1, 1, 1, 1, 1, 1, 1]),
        (11, False, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    ]
)
def test_tree_network(n: int, balanced: bool, expected_node_idxs: np.ndarray, expected_num_levels: np.ndarray,
                      expected_num_top: np.ndarray, expected_num_bottom: np.ndarray):
    circuit = tree(n, balanced=balanced)
    np.testing.assert_allclose(circuit.node_idxs, expected_node_idxs)
    np.testing.assert_allclose(circuit.num_levels, expected_num_levels)
    np.testing.assert_allclose(circuit.num_top, expected_num_top)
    np.testing.assert_allclose(circuit.num_bottom, expected_num_bottom)


@pytest.mark.parametrize(
    "v, balanced",
    product(RAND_VECS, [True, False])
)
def test_vector_configure(v: np.ndarray, balanced: bool):
    circuit, thetas, phis, gammas, _ = configure_vector(v, balanced=balanced)
    res = circuit.matrix_fn(use_jax=False)(thetas, phis, gammas) @ v
    np.testing.assert_allclose(res, np.eye(v.size)[v.size - 1], atol=1e-10)


@pytest.mark.parametrize(
    "u, balanced",
    product(RAND_UNITARIES, [True, False])
)
def test_unitary_configure(u: np.ndarray, balanced: bool):
    circuit, thetas, phis, gammas = configure_unitary(u, balanced=balanced)
    res = circuit.matrix_fn(use_jax=False)(thetas, phis, gammas)
    np.testing.assert_allclose(res, u.T.conj(), atol=1e-10)



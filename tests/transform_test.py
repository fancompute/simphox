import numpy as np
import pytest

from simphox.transform import get_symmetry_fn, get_mask_fn, get_smooth_fn
from simphox.typing import Union, List
from simphox.utils import Box

np.random.seed(0)

TEST_ARRAY = np.array([[4., 0., 4., 4., 0.],
                       [0., 4., 0., 4., 4.],
                       [4., 0., 0., 4., 0.],
                       [0., 4., 4., 0., 0.],
                       [4., 0., 0., 4., 4.],
                       [4., 4., 4., 0., 4.],
                       [0., 0., 0., 4., 0.]])

TEST_ARRAY_SQUARE = np.array([[0., 0., 0., 8., 0.],
                              [8., 8., 0., 0., 8.],
                              [8., 0., 0., 8., 0.],
                              [0., 8., 8., 0., 0.],
                              [8., 0., 0., 8., 8.]])

TEST_ARRAY_SQUARE_ONES = np.ones_like(TEST_ARRAY_SQUARE)


@pytest.mark.parametrize(
    "ortho_x, ortho_y, expected_output",
    [
        (False, False, TEST_ARRAY),
        (False, True, [[2., 2., 4., 2., 2.],
                       [2., 4., 0., 4., 2.],
                       [2., 2., 0., 2., 2.],
                       [0., 2., 4., 2., 0.],
                       [4., 2., 0., 2., 4.],
                       [4., 2., 4., 2., 4.],
                       [0., 2., 0., 2., 0.]]),
        (True, False, [[2., 0., 2., 4., 0.],
                       [2., 4., 2., 2., 4.],
                       [4., 0., 0., 4., 2.],
                       [0., 4., 4., 0., 0.],
                       [4., 0., 0., 4., 2.],
                       [2., 4., 2., 2., 4.],
                       [2., 0., 2., 4., 0.]]),
        (True, True, [[1., 2., 2., 2., 1.],
                      [3., 3., 2., 3., 3.],
                      [3., 2., 0., 2., 3.],
                      [0., 2., 4., 2., 0.],
                      [3., 2., 0., 2., 3.],
                      [3., 3., 2., 3., 3.],
                      [1., 2., 2., 2., 1.]])
    ]
)
def test_get_symmetry_fn(ortho_x: bool, ortho_y: bool, expected_output: np.ndarray):
    actual = get_symmetry_fn(ortho_x, ortho_y)(TEST_ARRAY)
    np.testing.assert_allclose(actual, expected_output)


@pytest.mark.parametrize(
    "ortho_x, ortho_y, diag_p, diag_n, expected_output",
    [
        (False, False, False, False, TEST_ARRAY_SQUARE),
        (False, False, False, True, [[4., 0., 0., 8., 0.],
                                     [8., 4., 4., 0., 8.],
                                     [4., 4., 0., 4., 0.],
                                     [0., 8., 4., 4., 0.],
                                     [8., 0., 4., 8., 4.]]),
        (False, False, True, False, [[0., 4., 4., 4., 4.],
                                     [4., 8., 0., 4., 4.],
                                     [4., 0., 0., 8., 0.],
                                     [4., 4., 8., 0., 4.],
                                     [4., 4., 0., 4., 8.]]),
        (False, False, True, True, [[4., 4., 2., 4., 4.],
                                    [4., 4., 4., 4., 4.],
                                    [2., 4., 0., 4., 2.],
                                    [4., 4., 4., 4., 4.],
                                    [4., 4., 2., 4., 4.]]),
        (True, False, True, False, [[4., 2., 4., 6., 4.],
                                    [2., 8., 2., 4., 2.],
                                    [4., 2., 0., 6., 0.],
                                    [6., 4., 6., 0., 6.],
                                    [4., 2., 0., 6., 4.]]),
        (False, True, True, False, [[0., 6., 2., 2., 4.],
                                    [6., 4., 2., 4., 6.],
                                    [2., 2., 0., 6., 2.],
                                    [2., 4., 6., 4., 2.],
                                    [4., 6., 2., 2., 8.]]),
        (True, True, True, False, [[4., 4., 2., 4., 4.],
                                   [4., 4., 4., 4., 4.],
                                   [2., 4., 0., 4., 2.],
                                   [4., 4., 4., 4., 4.],
                                   [4., 4., 2., 4., 4.]]),
        (True, True, True, True, [[4., 4., 2., 4., 4.],
                                  [4., 4., 4., 4., 4.],
                                  [2., 4., 0., 4., 2.],
                                  [4., 4., 4., 4., 4.],
                                  [4., 4., 2., 4., 4.]]),
    ]
)
def test_get_symmetry_fn_square(ortho_x: bool, ortho_y: bool, diag_p: bool, diag_n: bool, expected_output: np.ndarray):
    actual = get_symmetry_fn(ortho_x, ortho_y, diag_p, diag_n)(TEST_ARRAY_SQUARE)
    np.testing.assert_allclose(actual, expected_output)


@pytest.mark.parametrize(
    "rho_init, box, rho, expected_output",
    [
        (TEST_ARRAY, Box((2, 2), min=(2, 2)), 1, [[4., 0., 4., 4., 0.],
                                                  [0., 4., 0., 4., 4.],
                                                  [4., 0., 1., 1., 0.],
                                                  [0., 4., 1., 1., 0.],
                                                  [4., 0., 0., 4., 4.],
                                                  [4., 4., 4., 0., 4.],
                                                  [0., 0., 0., 4., 0.]]),
        (TEST_ARRAY, Box((4, 2), min=(1, 2)), 1, [[4., 0., 4., 4., 0.],
                                                  [0., 4., 1., 1., 4.],
                                                  [4., 0., 1., 1., 0.],
                                                  [0., 4., 1., 1., 0.],
                                                  [4., 0., 1., 1., 4.],
                                                  [4., 4., 4., 0., 4.],
                                                  [0., 0., 0., 4., 0.]]),
        (TEST_ARRAY, [Box((4, 2), min=(1, 2)), Box((2, 4), min=(4, 1))], 1, [[4., 0., 4., 4., 0.],
                                                                             [0., 4., 1., 1., 4.],
                                                                             [4., 0., 1., 1., 0.],
                                                                             [0., 4., 1., 1., 0.],
                                                                             [4., 1., 1., 1., 1.],
                                                                             [4., 1., 1., 1., 1.],
                                                                             [0., 0., 0., 4., 0.]]),
        (TEST_ARRAY_SQUARE,
         [Box((4, 2), min=(1, 2)), Box((2, 4), min=(4, 1))], TEST_ARRAY_SQUARE_ONES, [[0., 0., 0., 8., 0.],
                                                                                      [8., 8., 1., 1., 8.],
                                                                                      [8., 0., 1., 1., 0.],
                                                                                      [0., 8., 1., 1., 0.],
                                                                                      [8., 1., 1., 1., 1.]]),
    ]
)
def test_mask_fn(rho_init: np.ndarray, box: Union[Box, List[Box]], rho: np.ndarray, expected_output: np.ndarray):
    actual = get_mask_fn(rho_init, box)(rho)
    np.testing.assert_allclose(actual, expected_output)


@pytest.mark.parametrize(
    "rho, eta, beta, radius, expected_output",
    [
        (TEST_ARRAY, 0.5, 1, 2, [[0.65049475, 1.0963078, 1.2062135, 0.96534103, 0.81519336],
                                 [0.96534103, 1.2954934, 1.3661214, 1.2062135, 0.96534103],
                                 [1.0963078, 1.4208316, 1.4939058, 1.3661214, 1.2062135],
                                 [1.2062135, 1.4208316, 1.5173032, 1.4208316, 1.2062135],
                                 [1.0963078, 1.3661213, 1.4625252, 1.2954934, 1.0963076],
                                 [0.96534103, 1.2062135, 1.3661213, 1.2062135, 0.96534103],
                                 [0.65049475, 0.96534103, 1.2062135, 0.96534103, 0.81519336]]),
        (TEST_ARRAY, 0.5, 2, 2, [[0.6791669, 1.0550566, 1.1009896, 0.97656447, 0.8525824],
                                 [0.97656447, 1.1266409, 1.1405926, 1.1009896, 0.97656447],
                                 [1.0550566, 1.148072, 1.1541586, 1.1405926, 1.1009896],
                                 [1.1009896, 1.148072, 1.1552726, 1.148072, 1.1009896],
                                 [1.0550566, 1.1405926, 1.1520507, 1.1266409, 1.0550565],
                                 [0.97656447, 1.1009896, 1.1405926, 1.1009896, 0.97656447],
                                 [0.6791669, 0.97656447, 1.1009896, 0.97656447, 0.8525824]]),
        (TEST_ARRAY, 0.8, 1, 3, [[0.7708701, 0.87378895, 0.87378895, 0.87378895, 0.5666377],
                                 [0.87378883, 1.0733042, 1.167072, 1.0733042, 0.87378883],
                                 [1.2553324, 1.4124601, 1.4806038, 1.4124601, 1.0733042],
                                 [1.167072, 1.5417402, 1.5417402, 1.4806038, 1.2553324],
                                 [0.9751025, 1.2553325, 1.3372947, 1.3372947, 0.9751025],
                                 [0.77087015, 1.0733042, 1.0733042, 0.9751025, 0.77087015],
                                 [0.66795135, 0.87378895, 0.87378895, 0.87378895, 0.5666377]]),
    ]
)
def test_smooth_fn(rho: np.ndarray, eta: float, beta: float, radius: float, expected_output: float):
    actual = get_smooth_fn(beta, radius, eta)(rho)
    np.testing.assert_allclose(actual, expected_output, rtol=1e-6)

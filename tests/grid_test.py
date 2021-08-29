import pytest
from typing import List, Tuple, Union, Optional

from simphox.typing import Shape, Dim, Dim3

from simphox.grid import Grid, YeeGrid

import numpy as np


@pytest.mark.parametrize(
    "shape, spacing, eps, expected_cell_sizes",
    [
        ((5, 5, 2), 0.5, 1, [np.array([0.5, 0.5, 0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
                             np.array([0.5, 0.5])]),
        ((5, 5), 0.2, 1, [np.array([0.2, 0.2, 0.2, 0.2, 0.2]), np.array([0.2, 0.2, 0.2, 0.2, 0.2]), np.array([1])]),
        ((5, 4), 0.2, 1, [np.ones(5) * 0.2, np.ones(4) * 0.2, np.array([1])]),
        ((5,), 3, 1, [np.ones(5) * 3, np.array([1]), np.array([1])]),
        ((5, 3, 2), (1, 2, 3), 1, [np.ones(5) * 1, np.ones(3) * 2, np.ones(2) * 3])
    ],
)
def test_cell_size(shape: Shape, spacing: Dim, eps: Union[float, np.ndarray],
                   expected_cell_sizes: List[np.ndarray]):
    grid = Grid(shape, spacing, eps)
    for actual, expected in zip(grid.cell_sizes, expected_cell_sizes):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "shape, spacing, eps, expected_pos",
    [
        ((5, 5, 2), 0.5, 1,
         [np.array([0, 0.5, 1, 1.5, 2, 2.5]), np.array([0, 0.5, 1, 1.5, 2, 2.5]), np.array([0, 0.5, 1])]),
        ((5, 5), 0.2, 1, [np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), np.array([0])]),
        ((5, 4), 0.2, 1, [np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), np.array([0, 0.2, 0.4, 0.6, 0.8]), np.array([0])]),
        ((5,), 3, 1, [np.arange(6) * 3, np.array([0]), np.array([0])]),
        ((5, 3, 2), (1, 2, 3), 1, [np.arange(6) * 1, np.arange(4) * 2, np.arange(3) * 3])
    ],
)
def test_pos(shape: Shape, spacing: Dim, eps: Union[float, np.ndarray],
             expected_pos: List[np.ndarray]):
    grid = Grid(shape, spacing, eps)
    for actual, expected in zip(grid.pos, expected_pos):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "shape, spacing, eps, expected_spacing",
    [
        ((5, 5, 2), 0.5, 1, np.asarray((0.5, 0.5, 0.5))),
        ((5, 5), 0.2, 1, np.ones(2) * 0.2),
        ((5, 4), 0.2, 1, np.ones(2) * 0.2),
        ((5, 3, 2), (1, 2, 3), 1, np.array((1, 2, 3)))
    ],
)
def test_spacing(shape: Shape, spacing: Dim,
                 eps: Union[float, np.ndarray], expected_spacing: np.ndarray):
    grid = Grid(shape, spacing, eps)
    np.testing.assert_allclose(grid.spacing, expected_spacing)


@pytest.mark.parametrize(
    "shape, eps",
    [
        ((2, 3), np.asarray(((1, 1), (1, 1)))),
        ((2,), np.asarray(((1, 1), (1, 1))))
    ],
)
def test_error_raised_for_shape_eps_mismatch(shape: Shape, eps: Union[float, np.ndarray]):
    with pytest.raises(AttributeError, match=f'Require grid.shape == eps.shape but got'):
        Grid(shape, 1, eps)


@pytest.mark.parametrize(
    "shape, spacing",
    [
        ((2, 3), (1, 1, 1)),
        ((2, 3, 2), (1, 1))
    ],
)
def test_error_raised_for_shape_spacing_mismatch(shape: Shape, spacing: Dim):
    with pytest.raises(AttributeError, match='Require shape.size == spacing.size but got '):
        Grid(shape, spacing)


@pytest.mark.parametrize(
    "shape, spacing, size",
    [
        ((5, 5, 2), 0.5, (2.5, 2.5, 1)),
        ((5, 5), 0.2, (1, 1)),
    ],
)
def test_size(shape: Shape, spacing: Dim, size: Dim):
    grid = Grid(shape, spacing)
    np.testing.assert_allclose(grid.size, size)


@pytest.mark.parametrize(
    "shape, spacing, size",
    [
        ((5, 5, 2), 0.5, (2.5, 2.5, 1)),
        ((5, 5), 0.2, (1, 1)),
    ],
)
def test_size(shape: Shape, spacing: Dim, size: Dim):
    grid = Grid(shape, spacing)
    np.testing.assert_allclose(grid.size, size)


@pytest.mark.parametrize(
    "shape, spacing, center, size, squeezed, expected_slice",
    [
        ((5, 5, 2), 0.5, (1, 1, 1), (0.5, 1, 1), True, [slice(2, 3, None), slice(1, 3, None), slice(1, 3, None)]),
        ((5, 5, 2), 0.5, (1, 1, 1), (0.5, 0.1, 1), True, [slice(2, 3, None), 2, slice(1, 3, None)]),
        ((5, 5, 2), 0.5, (1, 1, 1), (0.5, 0.1, 1), False, [slice(2, 3, None), slice(2, 3, None), slice(1, 3, None)]),
        ((5, 5), 0.2, (1, 1, 0), (0.5, 1, 1), True, [slice(4, 6, None), slice(3, 8, None), 0]),
    ],
)
def test_slice(shape: Shape, spacing: Dim, center: Dim3, size: Dim3, squeezed: bool,
               expected_slice: Tuple[Union[slice, int]]):
    grid = Grid(shape, spacing)
    actual = grid.slice(center, size, squeezed=squeezed)
    assert tuple(actual) == tuple(expected_slice)


@pytest.mark.parametrize(
    "shape, spacing, pml, expected_df_data, expected_df_indices",
    [
        ((3, 3, 2), 0.5, None,
         [2., -2., 2., -2., 2., -2., 2., -2., 2., -2., 2., -2., 2., -2.,
          2., -2., 2., -2., 2., -2., 2., -2., 2., -2., -2., 2., -2., 2.,
          -2., 2., -2., 2., -2., 2., -2., 2.],
         [6, 0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,
          8, 15, 9, 16, 10, 17, 11, 12, 0, 13, 1, 14, 2, 15, 3, 16, 4,
          17, 5]
         ),
        ((3, 3), 1, None,
         [1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., -1., 1.,
          -1., 1., -1., 1.],
         [3, 0, 4, 1, 5, 2, 6, 3, 7, 4, 8, 5, 6, 0, 7, 1, 8, 2]
         ),
        ((3,), 2, None,
         [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
         [1, 0, 2, 1, 0, 2]
         ),
    ],
)
def test_df(shape: Shape, spacing: Dim, pml: Optional[Dim3], expected_df_data: np.ndarray,
            expected_df_indices: np.ndarray):
    grid = YeeGrid(shape, spacing, pml=pml)
    actual_df = grid.df
    np.testing.assert_allclose(actual_df[0].data, expected_df_data)
    np.testing.assert_allclose(actual_df[0].indices, expected_df_indices)


@pytest.mark.parametrize(
    "shape, spacing, pml, expected_db_data, expected_db_indices",
    [
        ((3, 3, 2), 0.5, None,
         [-2.,  2., -2.,  2., -2.,  2., -2.,  2., -2.,  2., -2.,  2.,  2.,
          -2.,  2., -2.,  2., -2.,  2., -2.,  2., -2.,  2., -2.,  2., -2.,
          2., -2.,  2., -2.,  2., -2.,  2., -2.,  2., -2.],
         [12, 0, 13, 1, 14, 2, 15, 3, 16, 4, 17, 5, 6, 0, 7, 1, 8,
          2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14, 8, 15, 9, 16, 10,
          17, 11]
         ),
        ((3, 3), 1, None,
         [-1., 1., -1., 1., -1., 1., 1., -1., 1., -1., 1., -1., 1.,
          -1., 1., -1., 1., -1.],
         [6, 0, 7, 1, 8, 2, 3, 0, 4, 1, 5, 2, 6, 3, 7, 4, 8, 5]
         ),
        ((3,), 2, None,
         [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
         [2, 0, 0, 1, 1, 2]
         ),
    ],
)
def test_db(shape: Shape, spacing: Dim, pml: Optional[Dim3], expected_db_data: np.ndarray,
            expected_db_indices: np.ndarray):
    grid = YeeGrid(shape, spacing, pml=pml)
    actual_db = grid.db
    np.testing.assert_allclose(actual_db[0].data, expected_db_data)
    np.testing.assert_allclose(actual_db[0].indices, expected_db_indices)


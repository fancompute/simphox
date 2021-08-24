import pytest
from typing import List, Tuple, Union

from simphox.typing import Shape, Dim, Dim3

from simphox.grid import Grid, FDGrid

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
        ((5, 5, 2), 0.5, 1, [np.array([0, 0.5, 1, 1.5, 2, 2.5]), np.array([0, 0.5, 1, 1.5, 2, 2.5]), np.array([0, 0.5, 1])]),
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
    "shape, spacing, center, size, expected_field",
    [
        ((3, 3, 3), 0.5, (1, 1, 1), (1.5, 1.5, 0), [[[[1], [2]], [[1], [2]]],
                                                    [[[1], [1]], [[2], [2]]],
                                                    [[[2], [2]], [[2], [2]]]]),
        ((4, 4, 4), 0.3, (1, 1, 1), (0, 1.5, 1.5), [[[[3], [3], [3]], [[3], [3], [3]], [[3], [3], [3]]],
                                                    [[[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]]],
                                                    [[[1], [1], [1]], [[2], [2], [2]], [[3], [3], [3]]]]),
        ((3, 3, 3), 0.5, (1, 1, 1), (1, 0, 1), [[[[2], [2]], [[2], [2]]],
                                                [[[1], [2]], [[1], [2]]],
                                                [[[1], [1]], [[2], [2]]]]),
        ((5, 5), 0.5, (1, 1, 0), (2, 2, 0), [[[[0, 1, 2, 3]], [[0, 1, 2, 3]], [[0, 1, 2, 3]], [[0, 1, 2, 3]]],
                                             [[[0, 0, 0, 0]], [[0, 0, 0, 0]], [[0, 0, 0, 0]], [[0, 0, 0, 0]]],
                                             [[[0, 0, 0, 0]], [[1, 1, 1, 1]], [[2, 2, 2, 2]], [[3, 3, 3, 3]]]]),
    ],
)
def test_view(shape: Shape, spacing: Dim, center: Dim3, size: Dim3, expected_field: np.ndarray):
    grid = Grid(shape, spacing)
    view_fn = grid.view_fn(center, size)
    field_to_view = np.stack(np.meshgrid(np.arange(grid.shape3[0]), np.arange(grid.shape3[1]),
                                         np.arange(grid.shape3[2])))
    actual_field = view_fn(field_to_view)
    np.testing.assert_allclose(actual_field, expected_field)

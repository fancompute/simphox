import pytest
from typing import List, Tuple, Union, Optional

from simphox.typing import Shape, Size, Size2, Size3
from simphox.utils import TEST_ZERO, TEST_ONE, Box
from simphox.grid import Grid, YeeGrid

import numpy as np


@pytest.mark.parametrize(
    "size, spacing, eps, expected_cell_sizes",
    [
        ((2.5, 2.5, 1), 0.5, 1, [np.array([0.5, 0.5, 0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
                                 np.array([0.5, 0.5])]),
        ((1, 1), 0.2, 1, [np.array([0.2, 0.2, 0.2, 0.2, 0.2]), np.array([0.2, 0.2, 0.2, 0.2, 0.2]), np.array([1])]),
        ((1, 0.8), 0.2, 1, [np.ones(5) * 0.2, np.ones(4) * 0.2, np.array([1])]),
        ((15,), 3, 1, [np.ones(5) * 3, np.array([1]), np.array([1])]),
        ((5, 6, 6), (1, 2, 3), 1, [np.ones(5) * 1, np.ones(3) * 2, np.ones(2) * 3])
    ],
)
def test_cell_size(size: Size, spacing: Size, eps: Union[float, np.ndarray],
                   expected_cell_sizes: List[np.ndarray]):
    grid = Grid(size, spacing, eps)
    for actual, expected in zip(grid.cells, expected_cell_sizes):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "size, spacing, eps, expected_pos",
    [
        ((2.5, 2.5, 1), 0.5, 1,
         [np.array([0, 0.5, 1, 1.5, 2, 2.5]), np.array([0, 0.5, 1, 1.5, 2, 2.5]), np.array([0, 0.5, 1])]),
        ((1, 1), 0.2, 1, [np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), np.array([0])]),
        ((1, 0.8), 0.2, 1, [np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), np.array([0, 0.2, 0.4, 0.6, 0.8]), np.array([0])]),
        ((15,), 3, 1, [np.arange(6) * 3, np.array([0]), np.array([0])]),
        ((5, 6, 6), (1, 2, 3), 1, [np.arange(6) * 1, np.arange(4) * 2, np.arange(3) * 3])
    ],
)
def test_pos(size: Size, spacing: Size, eps: Union[float, np.ndarray],
             expected_pos: List[np.ndarray]):
    grid = Grid(size, spacing, eps)
    for actual, expected in zip(grid.pos, expected_pos):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "size, spacing, eps, expected_spacing",
    [
        ((5, 5, 2), 0.5, 1, np.asarray((0.5, 0.5, 0.5))),
        ((5, 5), 0.2, 1, np.ones(2) * 0.2),
        ((5, 4), 0.2, 1, np.ones(2) * 0.2),
        ((5, 3, 2), (1, 2, 3), 1, np.array((1, 2, 3)))
    ],
)
def test_spacing(size: Shape, spacing: Size,
                 eps: Union[float, np.ndarray], expected_spacing: np.ndarray):
    grid = Grid(size, spacing, eps)
    np.testing.assert_allclose(grid.spacing, expected_spacing)


@pytest.mark.parametrize(
    "shape, eps",
    [
        ((2, 3), np.asarray(((1, 1), (1, 1)))),
        ((2,), np.asarray(((1, 1), (1, 1))))
    ],
)
def test_error_raised_for_shape_eps_mismatch(shape: Shape, eps: Union[float, np.ndarray]):
    with pytest.raises(AttributeError, match=f'Require grid.shape == eps.shape but got '):
        Grid(shape, 1, eps)


@pytest.mark.parametrize(
    "shape, spacing",
    [
        ((2, 3), (1, 1, 1)),
        ((2, 3, 2), (1, 1))
    ],
)
def test_error_raised_for_shape_spacing_mismatch(shape: Shape, spacing: Size):
    with pytest.raises(AttributeError, match='Require size.size == ndim == spacing.size but got '):
        Grid(shape, spacing)


@pytest.mark.parametrize(
    "shape, spacing, size",
    [
        ((5, 5, 2), 0.5, (2.5, 2.5, 1)),
        ((5, 5), 0.2, (1, 1)),
    ],
)
def test_shape(shape: Shape, spacing: Size, size: Size):
    grid = Grid(size, spacing)
    np.testing.assert_allclose(grid.shape, shape)


@pytest.mark.parametrize(
    "shape, spacing, size",
    [
        ((5, 5, 2), 0.5, (2.5, 2.5, 1)),
        ((5, 5, 1), 0.2, (1, 1)),
    ],
)
def test_shape3(shape: Shape, spacing: Size, size: Size):
    grid = Grid(size, spacing)
    np.testing.assert_allclose(grid.shape3, shape)


@pytest.mark.parametrize(
    "sim_spacing3, spacing, size",
    [
        ((0.5, 0.5, 0.5), 0.5, (2.5, 2.5, 1)),
        ((0.2, 0.2, np.inf), 0.2, (1, 1)),
    ],
)
def test_spacing3(sim_spacing3: Size, spacing: Size, size: Size):
    grid = Grid(size, spacing)
    np.testing.assert_allclose(grid.spacing3, sim_spacing3)


@pytest.mark.parametrize(
    "sim_size, spacing, center, size, squeezed, expected_slice",
    [
        ((2.5, 2.5, 1), 0.5, (1, 1, 1), (0.5, 1, 1), True, [slice(2, 3, None), slice(1, 3, None), slice(1, 3, None)]),
        ((2.5, 2.5, 1), 0.5, (1, 1, 1), (0.5, 0.1, 1), True, [slice(2, 3, None), 2, slice(1, 3, None)]),
        (
        (2.5, 2.5, 1), 0.5, (1, 1, 1), (0.5, 0.1, 1), False, [slice(2, 3, None), slice(2, 3, None), slice(1, 3, None)]),
        ((1, 1), 0.2, (1, 1, 0), (0.5, 1, 1), True, [slice(4, 6, None), slice(3, 8, None), 0]),
    ],
)
def test_slice(sim_size: Shape, spacing: Size, center: Size3, size: Size3, squeezed: bool,
               expected_slice: Tuple[Union[slice, int]]):
    grid = Grid(sim_size, spacing)
    actual = grid.slice(center, size, squeezed=squeezed)
    assert tuple(actual) == tuple(expected_slice)


@pytest.mark.parametrize(
    "size, spacing, pml, expected_df_data, expected_df_indices",
    [
        ((1.5, 1.5, 1), 0.5, None,
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
        ((6,), 2, None,
         [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
         [1, 0, 2, 1, 0, 2]
         ),
    ],
)
def test_df(size: Size, spacing: Size, pml: Optional[Size3], expected_df_data: np.ndarray,
            expected_df_indices: np.ndarray):
    grid = YeeGrid(size, spacing, pml=pml)
    actual_df = grid.deriv_forward
    np.testing.assert_allclose(actual_df[0].data, expected_df_data)
    np.testing.assert_allclose(actual_df[0].indices, expected_df_indices)


@pytest.mark.parametrize(
    "size, spacing, pml, expected_db_data, expected_db_indices",
    [
        ((1.5, 1.5, 1), 0.5, None,
         [-2., 2., -2., 2., -2., 2., -2., 2., -2., 2., -2., 2., 2.,
          -2., 2., -2., 2., -2., 2., -2., 2., -2., 2., -2., 2., -2.,
          2., -2., 2., -2., 2., -2., 2., -2., 2., -2.],
         [12, 0, 13, 1, 14, 2, 15, 3, 16, 4, 17, 5, 6, 0, 7, 1, 8,
          2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14, 8, 15, 9, 16, 10,
          17, 11]
         ),
        ((3, 3), 1, None,
         [-1., 1., -1., 1., -1., 1., 1., -1., 1., -1., 1., -1., 1.,
          -1., 1., -1., 1., -1.],
         [6, 0, 7, 1, 8, 2, 3, 0, 4, 1, 5, 2, 6, 3, 7, 4, 8, 5]
         ),
        ((6,), 2, None,
         [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
         [2, 0, 0, 1, 1, 2]
         ),
    ],
)
def test_db(size: Size, spacing: Size, pml: Optional[Size3], expected_db_data: np.ndarray,
            expected_db_indices: np.ndarray):
    grid = YeeGrid(size, spacing, pml=pml)
    actual_db = grid.deriv_backward
    np.testing.assert_allclose(actual_db[0].data, expected_db_data)
    np.testing.assert_allclose(actual_db[0].indices, expected_db_indices)


@pytest.mark.parametrize(
    "waveguide, sub, size, wg_height, spacing, rib_y, vertical, block, gap, seps, expected",
    [
        (Box((0.2, 0.4), material=TEST_ZERO), (1.4, 0.2), (1.4, 1), 0.2, 0.2, 0, False,
         None, 0.2, (0.2, 0.4), np.array(
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 0., 0., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 0., 0., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]]
        )),
        (Box((0.2, 0.4), material=TEST_ZERO), (1.4, 0.2), (1.4, 1), 0.2, 0.2, 0, False,
         Box((0.2, 0.2), material=TEST_ZERO), 0.2, (0.2, 0.2), np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1]
        ])),
        (Box((0.2, 0.4), material=TEST_ZERO), (1.4, 0.2), (1.4, 1), 0.2, 0.2, 0, True,
         Box((0.2, 0.2), material=TEST_ZERO), 0.2, (0.2, 0.2), np.array([
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 0., 0., 1., 0.],
            [1., 1., 1., 1., 1.],
            [1., 0., 0., 1., 0.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]
        ])),
        (Box((0.6, 0.6), material=TEST_ZERO), (1, 1), (1, 1), 0.2, 0.2, 0, False, None, 0, 0, np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ])),
        (Box((0.4, 0.4), material=TEST_ZERO), (1, 1), (1, 1), 0.2, 0.2, 0, False, None, 0, 0, np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ])),
        (Box((0.4, 0.4), material=TEST_ZERO), (1, 0.2), (1, 1), 0.2, 0.2, 0.2, False, None, 0, 0, np.array([
            [1, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1]
        ])),
        (Box((0.2, 0.4), material=TEST_ZERO), (1, 0.2), (1, 1), 0.2, 0.2, 0, False,
         Box((0.2, 0.4), material=TEST_ZERO), 0, 0.2, np.array([
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1]
        ])),
        (Box((0.4, 0.2), material=TEST_ZERO), (1, 0.2), (1, 1), 0.2, 0.2, 0, True,
         Box((0.4, 0.2), material=TEST_ZERO), 0, 0.2, np.array([
            [1., 1., 1., 1., 1.],
            [1., 1., 0., 1., 0.],
            [1., 1., 0., 1., 0.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]
        ])),
    ],
)
def test_block_design_eps_matches_expected(waveguide: Box, sub: Size2, size: Size2, wg_height: float, spacing: float,
                                           rib_y: float, vertical: bool, block: Box, gap: float,
                                           seps: Size2, expected: np.ndarray):
    actual = Grid(size, spacing).block_design(waveguide=waveguide,
                                              wg_height=wg_height,
                                              sub_height=wg_height,
                                              sub_eps=TEST_ONE.eps,
                                              coupling_gap=gap,
                                              rib_y=rib_y,
                                              block=block,
                                              vertical=vertical,
                                              sep=seps
                                              ).eps
    np.testing.assert_allclose(actual, expected)

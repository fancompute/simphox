from simphox.mode import ModeSolver, ModeLibrary
from simphox.utils import TEST_ONE, TEST_ZERO, SILICON, AIR, Box
from simphox.typing import Size2

import pytest
import numpy as np

np.random.seed(0)


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
    actual = ModeSolver(size, spacing).block_design(waveguide=waveguide,
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


@pytest.mark.parametrize(
    "waveguide, sub, size, wg_height, spacing, rib_y, vertical, block, gap, seps, expected_beta",
    [
        (Box((0.2, 0.4), material=SILICON), Box((1.4, 0.2), material=AIR),
         (1.4, 1), 0.2, 0.2, 0, False, Box((0.2, 0.2), material=SILICON), 0.2, (0.2, 0.4), [8.511193, 8.208999,
                                                                                            7.785885, 6.190572,
                                                                                            5.509391, 4.78587]),
        (Box((0.2, 0.4), material=SILICON), Box((1.4, 0.2), material=AIR),
         (1.4, 1), 0.2, 0.2, 0, False, None, 0.2, (0, 0), [8.497423, 8.193003,
                                                           7.732571, 5.94686,
                                                           5.247048, 3.93605]),
        (Box((0.2, 0.4), material=SILICON), Box((1.4, 0.2), material=AIR),
         (1.4, 1), 0.2, 0.2, 0, True, Box((0.2, 0.2), material=SILICON), 0.2, (0.2, 0.4), [8.539984, 8.22446,
                                                                                           7.776173, 6.022444,
                                                                                           5.283903, 4.618694]),
    ],
)
def test_mode_matches_expected_beta(waveguide: Box, sub: Box, size: Size2, wg_height: float, spacing: float,
                                    rib_y: float, vertical: bool, block: Box, gap: float, seps: Size2,
                                    expected_beta: float):
    actual_beta, _ = ModeSolver(size, spacing).block_design(waveguide=waveguide,
                                                            wg_height=wg_height,
                                                            sub_height=wg_height,
                                                            sub_eps=TEST_ONE.eps,
                                                            coupling_gap=gap,
                                                            rib_y=rib_y,
                                                            block=block,
                                                            vertical=vertical,
                                                            sep=seps
                                                            ).solve()
    np.testing.assert_allclose(actual_beta, expected_beta, atol=1e-6)

import numpy as np
import pytest

from simphox.mode import ModeSolver, ModeLibrary
from simphox.grid import Grid
from simphox.typing import Size2
from simphox.utils import TEST_ONE, TEST_INF, SILICON, AIR, Box


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


@pytest.mark.parametrize(
    "waveguide, sub, size, wg_height, spacing, rib_y, vertical, block, gap, seps, expected_mean, expected_std",
    [
        (Box((0.2, 0.4), material=SILICON), Box((1.4, 0.2), material=AIR),
         (1.4, 1), 0.2, 0.2, 0, False, Box((0.2, 0.2), material=SILICON), 0.2, (0.2, 0.4),
         [0.025206, -0.000639, 0.03445, 0.002703, 0.019679, 0.02683],
         [0.105274, 0.107789, 0.094922, 0.121647, 0.147402, 0.099317]),
        (Box((0.2, 0.4), material=SILICON), Box((1.4, 0.2), material=AIR),
         (1.4, 1), 0.2, 0.2, 0, False, None, 0.2, (0, 0),
         [4.084419e-03, 0, 3.316460e-02, 0, 2.053727e-02, 0],
         [0.105418, 0.107831, 0.096243, 0.124457, 0.167402, 0.110291]),
        (Box((0.2, 0.4), material=SILICON), Box((1.4, 0.2), material=AIR),
         (1.4, 1), 0.2, 0.2, 0, True, Box((0.2, 0.2), material=SILICON), 0.2, (0.2, 0.4),
         [-0.02573, -0.001929, 0.03454, 0.00172, 0.021061, -0.025712],
         [0.104182, 0.107348, 0.095774, 0.124315, 0.166756, 0.146658]),
    ],
)
def test_mode_matches_expected_mean_std(waveguide: Box, sub: Box, size: Size2, wg_height: float, spacing: float,
                                        rib_y: float, vertical: bool, block: Box, gap: float, seps: Size2,
                                        expected_mean: float, expected_std: float):
    _, actual_modes = ModeSolver(size, spacing).block_design(waveguide=waveguide,
                                                             wg_height=wg_height,
                                                             sub_height=wg_height,
                                                             sub_eps=TEST_ONE.eps,
                                                             coupling_gap=gap,
                                                             rib_y=rib_y,
                                                             block=block,
                                                             vertical=vertical,
                                                             sep=seps
                                                             ).solve()
    actual_mean = np.mean(actual_modes, axis=1).real
    actual_std = np.std(actual_modes, axis=1).real
    np.testing.assert_allclose(actual_mean, expected_mean, atol=1e-6)
    np.testing.assert_allclose(actual_std, expected_std, atol=1e-6)


@pytest.mark.parametrize(
    "waveguide, size, wg_height, spacing",
    [
        (Box((0.36, 0.16), material=TEST_INF), (0.48, 0.24), 0.04, 0.02),
        (Box((0.16, 0.36), material=TEST_INF), (0.24, 0.64), 0.06, 0.02),
        # (Box((0.16, 0), material=TEST_INF), (0.24,), 0.06, 0.02),
    ],
)
def test_mode_matches_expected_analytical_2d(waveguide: Box, size: Size2, wg_height: float, spacing: float):
    modes = ModeLibrary.from_block_design(
        size=size,
        spacing=spacing,
        waveguide=waveguide,
        wg_height=wg_height
    )
    wg_grid = Grid(waveguide.size, spacing)
    y, x, z = np.meshgrid(wg_grid.pos[1], wg_grid.pos[0] + spacing / 2, wg_grid.pos[2])
    analytical = (np.sin(y / waveguide.size[1] * np.pi) * np.sin(x / waveguide.size[0] * np.pi))[:-1, :-1].squeeze()
    numerical = np.abs(modes.h(0)[1][modes.eps == 1e10].reshape(analytical.shape))
    numerical = numerical / np.max(numerical)
    np.testing.assert_allclose(numerical, analytical, atol=2e-2)

from typing import Tuple

from simphox.mode import ModeDevice, ModeSolver, ModeLibrary
from simphox.material import TEST_ONE, TEST_ZERO, MaterialBlock, SILICON, AIR
import pytest
import numpy as np

np.random.seed(0)


@pytest.mark.parametrize(
    "core, sub, size, wg_height, spacing, rib_y, lat_ps, vert_ps, sep, expected",
    [
        ((0.6, 0.6), (1, 1), (1, 1), 0.2, 0.2, 0, None, None, 0, np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ])),
        ((0.4, 0.4), (1, 1), (1, 1), 0.2, 0.2, 0, None, None, 0, np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ])),
        ((0.4, 0.4), (1, 0.2), (1, 1), 0.2, 0.2, 0.2, None, None, 0, np.array([
            [1, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1]
        ])),
        ((0.2, 0.4), (1, 0.2), (1, 1), 0.2, 0.2, 0, (0.2, 0.4), None, 0.2, np.array([
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1]
        ])),
        ((0.4, 0.2), (1, 0.2), (1, 1), 0.2, 0.2, 0, None, (0.4, 0.2), 0.2, np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ])),
    ],
)
def test_single_eps_matches_expected(core: Tuple[float, float], sub: Tuple[float, float], size: Tuple[float, float],
                                     wg_height: float, spacing: float, rib_y: float, lat_ps: Tuple[float, float],
                                     vert_ps: Tuple[float, float], sep: float, expected: np.ndarray):
    device = ModeDevice(MaterialBlock(core, TEST_ZERO), MaterialBlock(sub, TEST_ONE),
                        size, wg_height, spacing, rib_y)
    lat_ps = MaterialBlock(lat_ps, TEST_ZERO) if lat_ps is not None else None
    vert_ps = MaterialBlock(vert_ps, TEST_ZERO) if vert_ps is not None else None
    np.testing.assert_allclose(device.single(lat_ps=lat_ps, sep=sep, vert_ps=vert_ps), expected)


@pytest.mark.parametrize(
    "core, sub, size, wg_height, spacing, rib_y, lat_ps, vert_ps, gap, seps, expected",
    [
        ((0.2, 0.4), (1.4, 0.2), (1.4, 1), 0.2, 0.2, 0, (0.2, 0.2), None, 0.2, (0.2, 0.4), np.array([
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1]
        ])),
    ],
)
def test_coupled_eps_matches_expected(core: Tuple[float, float], sub: Tuple[float, float], size: Tuple[float, float],
                                      wg_height: float, spacing: float, rib_y: float, lat_ps: Tuple[float, float],
                                      vert_ps: Tuple[float, float], gap: float,
                                      seps: Tuple[float, float], expected: np.ndarray):
    device = ModeDevice(MaterialBlock(core, TEST_ZERO), MaterialBlock(sub, TEST_ONE),
                        size, wg_height, spacing, rib_y)
    lat_ps = MaterialBlock(lat_ps, TEST_ZERO) if lat_ps is not None else None
    vert_ps = MaterialBlock(vert_ps, TEST_ZERO) if vert_ps is not None else None
    actual = device.coupled(gap=gap, lat_ps=lat_ps, seps=seps, vert_ps=vert_ps)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "core, sub, size, wg_height, spacing, rib_y, lat_ps, vert_ps, gap, seps, expected_max, expected_mean",
    [
        ((0.2, 0.4), (1.4, 0.2), (1.4, 1), 0.2, 0.2, 0, (0.2, 0.2), None, 0.2, (0.2, 0.4), 0.456573 + 0j, 0.044715 + 0j),
        ((0.2, 0.4), (1.4, 0.2), (1.4, 1), 0.2, 0.2, 0, None, None, 0.2, (0, 0), 0.658336 + 0j, 0.034605 + 0j),
        ((0.2, 0.4), (1.4, 0.2), (1.4, 1), 0.2, 0.2, 0, None, (0.2, 0.2), 0.2, (0.2, 0.4), 0.65787 + 0j, 0.034798 + 0j),
    ],
)
def test_mode_matches_expected_max_mean(core: Tuple[float, float], sub: Tuple[float, float], size: Tuple[float, float],
                                        wg_height: float, spacing: float, rib_y: float, lat_ps: Tuple[float, float],
                                        vert_ps: Tuple[float, float], gap: float, seps: Tuple[float, float],
                                        expected_max: complex, expected_mean: complex):
    device = ModeDevice(MaterialBlock(core, SILICON), MaterialBlock(sub, AIR),
                        size, wg_height, spacing, rib_y)
    lat_ps = MaterialBlock(lat_ps, SILICON) if lat_ps is not None else None
    vert_ps = MaterialBlock(vert_ps, SILICON) if vert_ps is not None else None
    eps = device.coupled(gap=gap, lat_ps=lat_ps, seps=seps, vert_ps=vert_ps)
    actual = device.solve(eps).modes[0]
    np.testing.assert_allclose(np.max(actual), expected_max, atol=1e-6)
    np.testing.assert_allclose(np.mean(actual), expected_mean, atol=1e-6)


if __name__ == '__main__':
    pytest.main()

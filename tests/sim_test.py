import numpy as np
import pytest

from simphox.grid import Port
from simphox.sim import SimGrid
from simphox.typing import Size, Size2, Size3, Size4, Union, List, MeasureInfo

np.random.seed(0)

EPS_20_20 = 1 + np.random.rand(20, 20) + np.random.rand(20, 20) * 1j
SOURCE_20_20 = np.random.rand(2, 3, 20, 20) + np.random.rand(2, 3, 20, 20) * 1j
EPS_20_20_15 = 1 + np.random.rand(20, 20, 15) + np.random.rand(20, 20, 15) * 1j
SOURCE_20_20_15 = np.random.rand(2, 3, 20, 20, 15) + np.random.rand(2, 3, 20, 20, 15) * 1j
EPS_40_40_30 = 1 + np.random.rand(40, 40, 30) + np.random.rand(40, 40, 30) * 1j
SOURCE_40_40_30 = np.random.rand(2, 3, 40, 40, 30) + 1j * np.random.rand(2, 3, 40, 40, 30)


@pytest.mark.parametrize(
    "size, spacing, eps, port_list, fields, measure_info, profile_size_factor, expected_params",
    [
        ((20, 20), 1, EPS_20_20, [(7, 7, 0, 2), (13, 13, 0, 2)], SOURCE_20_20, None, 2,
         [[0.494765 - 0.574412j, -1.002997 + 0.462249j], [-0.161786 + 0.069348j, 0.017847 + 0.599384j]]),
        ((20, 20, 15), 1, EPS_20_20_15, [(7, 7, 0, 2), (13, 13, 0, 2)], SOURCE_20_20_15,
         None, 2, [[4.061429 + 1.472988j, 4.727095 + 1.122546j], [-0.415584 + 0.575708j, 1.061077 - 0.429472j]]),
        ((20, 20, 15), 0.5, EPS_40_40_30, [(7, 7, 0, 2, 7, 3), (13, 13, 0, 2, 8, 3)],  # x,y,a,w,z,h
         SOURCE_40_40_30, None, 2,
         [[-2.78854 + 0.608026j, 3.406128 - 3.809541j], [-1.546855 - 0.402073j, 1.833765 - 1.124782j]])
    ]
)
def test_measure_fn(size: Size, spacing: Size, eps: Union[float, np.ndarray],
                    port_list: List[Size4], fields: np.ndarray, measure_info: MeasureInfo, profile_size_factor: float,
                    expected_params: np.ndarray):
    grid = SimGrid(size, spacing, eps=eps)
    grid.port = {i: Port(*port_tuple) for i, port_tuple in enumerate(port_list)}
    measure_fn = grid.get_measure_fn(measure_info=measure_info)
    actual = measure_fn(fields)
    np.testing.assert_allclose(actual, expected_params, rtol=1e-5)


@pytest.mark.parametrize(
    "size, spacing, eps, pml, pml_params, port_list,"
    "fields, measure_info, profile_size_factor, expected_center, expected_size,"
    "expected_mode_mean_std",
    [
        ((20, 20), 1, EPS_20_20, None, (4, -16, 1), [(7, 7, 0, 2), (13, 13, 0, 2)],
         SOURCE_20_20, None, 2, [(7, 7, 0), (13, 13, 0)],
         [(0, 4, 0), (0, 4, 0)], [[-0.085174, 0.408666], [-0.109273, 0.420226]]),
        ((20, 20, 15), 1, EPS_20_20_15, None, (4, -16, 1), [(7, 7, 0, 2, 7, 3), (13, 13, 0, 2, 8, 3)],
         SOURCE_20_20_15, None, 2, [(7, 7, 7), (13, 13, 8)], [(0, 4, 6), (0, 4, 6)],
         [[0.027211, 0.117614], [0.010568, 0.118461]]),
        ((20, 20), 1, EPS_20_20, 8, (4, -16, 1), [(7, 7, 0, 2), (13, 13, 0, 2)],
         SOURCE_20_20, None, 2, [(9, 9, 0), (11, 11, 0)],
         [(0, 4, 0), (0, 4, 0)], [[0.330118, 0.374979], [0.3065, 0.39371]]),
        ((20, 20), 1, EPS_20_20, 8, (4, -16, 1), [(7, 7, 0, 2), (13, 13, 90, 2)],
         SOURCE_20_20, None, 2, [(9, 9, 0), (11, 11, 0)],
         [(0, 4, 0), (4, 0, 0)], [[0.330118, 0.374979], [0.262414, 0.410319]]),
        ((20, 20, 15), 0.5, EPS_40_40_30, None, (4, -16, 1),
         [(7, 7, 0, 2, 7, 3), (13, 13, 90, 2, 8, 4)], SOURCE_40_40_30, None, 2, [(7, 7, 7), (13, 13, 8)],
         [(0, 4, 6), (4, 0, 8)], [[0.000196, 0.061155], [-0.012119, 0.049334]])
    ]
)
def test_port_modes(size: Size, spacing: Size, eps: Union[float, np.ndarray], pml: Size, pml_params: Size3,
                    port_list: List[Size4], fields: np.ndarray, measure_info: MeasureInfo, profile_size_factor: float,
                    expected_center: List[Size3], expected_size: List[Size3], expected_mode_mean_std: List[Size2]):
    grid = SimGrid(size, spacing, eps=eps, pml=pml, pml_params=pml_params)
    grid.port = {i: Port(*port_tuple) for i, port_tuple in enumerate(port_list)}
    port_modes = grid.port_modes(profile_size_factor=profile_size_factor)
    actual_center = [port_modes[i].center for i in port_modes]
    actual_size = [port_modes[i].size for i in port_modes]
    actual_mode_mean_std = [(np.mean(port_modes[i].io.modes[0]).real, np.std(port_modes[i].io.modes[0]).real)
                            for i in port_modes]
    np.testing.assert_allclose(actual_center, expected_center)
    np.testing.assert_allclose(actual_size, expected_size)
    np.testing.assert_allclose(actual_mode_mean_std, expected_mode_mean_std, atol=1e-6)

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
         [[0.467498 + 0.711332j, 0.850372 + 0.758913j],
          [0.110112 - 0.128359j, 0.522125 - 0.292459j]]),
        ((20, 20, 15), 1, EPS_20_20_15, [(7, 7, 0, 2), (13, 13, 0, 2)], SOURCE_20_20_15,
         None, 2, [[2.642182 + 3.919623j, 3.514617 + 3.548969j], [-0.786236 + 0.378376j, 1.098336 + 0.295231j]]),
        ((20, 20, 15), 0.5, EPS_40_40_30, [(7, 7, 0, 2, 7, 3), (13, 13, 0, 2, 8, 3)],  # x,y,a,w,z,h
         SOURCE_40_40_30, None, 2,
         [[2.702992 + 2.683246j, 3.689899 + 4.437999j], [-1.134477 - 1.556019j, -0.764333 - 0.596152j]])
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
         [(0, 4, 0), (0, 4, 0)], [[0.308511, 0.393473], [0.286622, 0.409692]]),
        ((20, 20, 15), 1, EPS_20_20_15,  None, (4, -16, 1), [(7, 7, 0, 2, 7, 3), (13, 13, 0, 2, 8, 3)],
         SOURCE_20_20_15, None, 2, [(7, 7, 7), (13, 13, 8)], [(0, 4, 6), (0, 4, 6)],
         [[0.059369, 0.105632], [0.041292, 0.113077]]),
        ((20, 20), 1, EPS_20_20, 8, (4, -16, 1), [(7, 7, 0, 2), (13, 13, 0, 2)],
         SOURCE_20_20, None, 2, [(9, 9, 0), (11, 11, 0)],
         [(0, 4, 0), (0, 4, 0)], [[0.343611, 0.363224], [0.32445, 0.380437]]),
        ((20, 20), 1, EPS_20_20, 8, (4, -16, 1), [(7, 7, 0, 2), (13, 13, 90, 2)],
         SOURCE_20_20, None, 2, [(9, 9, 0), (11, 11, 0)],
         [(0, 4, 0), (4, 0, 0)], [[0.343611, 0.363224], [0.289147, 0.407914]]),
        ((20, 20, 15), 0.5, EPS_40_40_30, None, (4, -16, 1),
         [(7, 7, 0, 2, 7, 3), (13, 13, 90, 2, 8, 4)], SOURCE_40_40_30, None, 2, [(7, 7, 7), (13, 13, 8)],
         [(0, 4, 6), (4, 0, 8)], [[0.030234, 0.053373], [0.026255, 0.044831]])
    ]
)
def test_port_modes(size: Size, spacing: Size, eps: Union[float, np.ndarray], pml: Size, pml_params: Size3,
                    port_list: List[Size4], fields: np.ndarray, measure_info: MeasureInfo, profile_size_factor: float,
                    expected_center: List[Size3], expected_size: List[Size3], expected_mode_mean_std: List[Size2]):
    grid = SimGrid(size, spacing, eps=eps, pml=pml, pml_params=pml_params)
    grid.port = {i: Port(*port_tuple) for i, port_tuple in enumerate(port_list)}
    port_modes = grid.port_modes()
    actual_center = [port_modes[i].center for i in port_modes]
    actual_size = [port_modes[i].size for i in port_modes]
    actual_mode_mean_std = [(np.mean(port_modes[i].io.modes[0]), np.std(port_modes[i].io.modes[0])) for i in port_modes]
    np.testing.assert_allclose(actual_center, expected_center)
    np.testing.assert_allclose(actual_size, expected_size)
    np.testing.assert_allclose(actual_mode_mean_std, expected_mode_mean_std, rtol=2e-5)

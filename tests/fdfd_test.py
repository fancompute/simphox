from jax.config import config
import numpy as np
import pytest

from simphox.fdfd import FDFD
from simphox.typing import Size, Size3, Optional, List, Union

np.random.seed(0)


config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')

EPS_3_3_2 = 1 + np.random.rand(3, 3, 2).astype(np.complex128)
SOURCE_3_3_2 = np.random.rand(54).astype(np.complex128)
EPS_3_3 = 1 + np.random.rand(3, 3).astype(np.complex128)
SOURCE_3_3 = np.random.rand(9).astype(np.complex128)
EPS_10 = 1 + np.random.rand(10)
EPS_6_5_10 = 1 + np.random.rand(6, 5, 10)
EPS_5_5 = 1 + np.random.rand(5, 5)
SOURCE_5_5 = np.random.rand(25).astype(np.complex128)
EPS_10_10 = 1 + np.random.rand(10, 10)
EPS_10_10_10 = 1 + np.random.rand(10, 10, 10)
SOURCE_10 = np.random.rand(10).astype(np.complex128)


@pytest.mark.parametrize(
    "size, spacing, pml, pml_params, selected_indices, expected_df_data, expected_df_indices",
    [
        ((5, 5, 5), 0.5, 2, (4, -16, 1), [0, 2, 10, 40, 100, 1000, 400, 203, 314],
         [0.05134884 + 0.31632416j, 0.05134884 + 0.31632416j,
          0.05134884 + 0.31632416j, 0.05134884 + 0.31632416j, 0.05134884 + 0.31632416j, 2.,
          1.74179713 + 0.67062435j, -0.41673505 - 0.81228197j, 0.41673505 + 0.81228197j],
         [100, 101, 105, 120, 150, 600, 300, 101, 257]),
        ((10, 10), 1, 4, (4, -16, 1), [0, 2, 10, 40, 100, 130, 190, 80],
         [0.02567442 + 0.15816208j, 0.02567442 + 0.15816208j,
          0.02567442 + 0.15816208j, 0.87089856 + 0.33531218j, 1., 1.,
          -0.03404911 - 0.18135536j, 1.],
         [10, 11, 15, 30, 60, 75, 95, 50]),
        ((20,), 2, 8, (4, -16, 1), [0, 2, 4, 6, 8],
         [0.01283721 + 0.07908104j, 0.10418376 + 0.20307049j, 0.43544928 + 0.16765609j,
          0.49971064 + 0.01202487j, 0.5], [1, 2, 3, 4, 5]),
    ],
)
def test_df_pml_selected_indices(size: Size, spacing: Size, pml: Optional[Size], pml_params: Size3,
                                 selected_indices: List[int],
                                 expected_df_data: np.ndarray, expected_df_indices: np.ndarray):
    grid = FDFD(size, spacing, pml=pml, pml_params=pml_params, pml_sep=1)
    actual_df = grid.deriv_forward
    np.testing.assert_allclose(actual_df[0].data[selected_indices], expected_df_data)
    np.testing.assert_allclose(actual_df[0].indices[selected_indices], expected_df_indices)


@pytest.mark.parametrize(
    "size, spacing, pml, pml_params, selected_indices, expected_db_data, expected_db_indices",
    [
        ((5, 5, 5), 0.5, 2, (4, -16, 1), [0, 2, 10, 40, 100, 1000, 400, 203, 314],
         [-0.02033147 - 0.20062297j, -0.02033147 - 0.20062297j, -0.02033147 - 0.20062297j, -0.02033147 - 0.20062297j,
          -0.02033147 - 0.20062297j, 2., 1.44890765 + 0.89357816j, -0.18608182 - 0.58097951j,
          0.18608182 + 0.58097951j],
         [900, 901, 905, 920, 950, 500, 200, 1, 157]),
        ((10, 10), 1, 4, (4, -16, 1), [0, 2, 10, 40, 100, 130, 190, 80],
         [-0.01016574 - 0.10031149j, -0.01016574 - 0.10031149j,
          -0.01016574 - 0.10031149j, 0.72445382 + 0.44678908j,
          1., 1., 0.09304091 + 0.29048976j, 1.],
         [90, 91, 95, 20, 50, 65, 95, 40]),
        ((20,), 2, 8, (4, -16, 1), [0, 2, 4, 6, 8],
         [-0.00508287 - 0.05015574j, -0.04652045 - 0.14524488j, -0.36222691 - 0.22339454j, -0.49925823 - 0.01924408j,
          -0.5], [9, 0, 1, 2, 3]),
    ],
)
def test_db_pml_selected_indices(size: Size, spacing: Size, pml: Optional[Size], pml_params: Size3,
                                 selected_indices: List[int],
                                 expected_db_data: np.ndarray, expected_db_indices: np.ndarray):
    grid = FDFD(size, spacing, pml=pml, pml_params=pml_params, pml_sep=1)
    actual_db = grid.deriv_backward
    np.testing.assert_allclose(actual_db[0].data[selected_indices], expected_db_data)
    np.testing.assert_allclose(actual_db[0].indices[selected_indices], expected_db_indices)


@pytest.mark.parametrize(
    "size, spacing, eps, pml, pml_params, selected_indices, expected_db_data, expected_db_indices",
    [
        ((5, 5, 5), 0.5, EPS_10_10_10, 2, (4, -16, 1), [0, 2, 10, 40, 100, 1000, 400, 203, 314],
         [-30.165956 + 0.075539j, 0.071384 - 0.021037j, -0.062418 + 0.016733j,
          -16.707972 + 1.761025j, 0.078196 + 0.635662j, 0.102698 + 0.632648j,
          -0.062418 + 0.016733j, -0.174223 + 0.088695j, -18.585439 + 4.17361j],
         [0, 9, 2009, 3, 2006, 2176, 2039, 1115, 24]),
        ((3, 2.5, 5), 0.5, EPS_6_5_10, None, (4, -16, 1), [0, 2, 10, 40, 100, 1000, 400, 203, 314],
         [-13.367367, -4., 4., -15.665815, 4., 4., 4., 4., -10.600918],
         [0, 9, 609, 3, 606, 726, 639, 365, 24]),
        ((10, 10,), 1, EPS_10_10, None, (4, -16, 1), [0, 2, 10, 40, 100, 130, 190, 80],
         [-24.113075, -1., 1., -1., -1., -1., -28.132924, 1.],
         [0, 9, 100, 114, 15, 118, 27, 110]),
        ((20,), 2, EPS_10, None, (4, -16, 1), [0, 2, 4, 6, 8],
         [-21.66702, -18.597956, -21.187809, -26.069936, -30.053552],
         [0, 2, 4, 6, 8])
    ],
)
def test_mat_selected_indices(size: Size, spacing: Size, eps: Union[float, np.ndarray],
                              pml: Optional[Size], pml_params: Size3, selected_indices: List[int],
                              expected_db_data: np.ndarray, expected_db_indices: np.ndarray):
    grid = FDFD(size, spacing, eps=eps, pml=pml, pml_params=pml_params, pml_sep=1)
    actual_mat = grid.mat
    np.testing.assert_allclose(actual_mat.data[selected_indices], expected_db_data, rtol=1e-5)
    np.testing.assert_allclose(actual_mat.indices[selected_indices], expected_db_indices)


@pytest.mark.parametrize(
    "size, spacing, eps, pml, pml_params, selected_indices, expected_mat_ez_data, expected_mat_ez_indices",
    [
        ((10, 10), 1, EPS_10_10, 4, (4, -16, 1), [0, 2, 10, 40, 100, 130, 190, 80],
         [-2.272260e+01 + 0.018885j, 1.784589e-02 - 0.005259j,
          3.050671e-02 - 0.387327j, -6.920820e-01 - 0.492298j,
          3.050671e-02 - 0.387327j, 3.050671e-02 - 0.387327j,
          -8.567010e-01 - 0.368334j, 4.355569e-02 - 0.022174j],
         [0, 9, 1, 7, 10, 16, 28, 6]),
        ((20,), 2, EPS_10, 8, (4, -16, 1), [0, 2, 4, 6, 8],
         [-1.909656e+01 + 0.002361j, 4.461473e-03 - 0.001315j,
          -2.456868e+01 + 0.030123j, 7.626676e-03 - 0.096832j,
          -1.202780e-01 - 0.158007j], [0, 9, 1, 1, 3]),
    ],
)
def test_mat_ez_selected_indices(size: Size, spacing: Size, eps: Union[float, np.ndarray],
                                 pml: Optional[Size], pml_params: Size3, selected_indices: List[int],
                                 expected_mat_ez_data: np.ndarray, expected_mat_ez_indices: np.ndarray):
    grid = FDFD(size, spacing, eps=eps, pml=pml, pml_params=pml_params, pml_sep=1)
    actual_mat_ez = grid.mat_ez
    np.testing.assert_allclose(actual_mat_ez.data[selected_indices], expected_mat_ez_data, rtol=1e-4)
    np.testing.assert_allclose(actual_mat_ez.indices[selected_indices], expected_mat_ez_indices, rtol=1e-4)


@pytest.mark.parametrize(
    "size, spacing, eps, pml, pml_params, selected_indices, expected_mat_hz_data, expected_mat_hz_indices",
    [
        ((10, 10), 1, EPS_10_10, 4, (4, -16, 1), [0, 2, 10, 40, 100, 130, 190, 80],
         [0.017608 - 0.005189j, 0.013573 - 0.003639j, 0.011075 - 0.003264j,
          0.011218 - 0.003306j, -0.346769 - 0.455543j, -0.305578 - 0.401431j,
          -0.503263 - 0.031538j, 0.062817 - 0.062641j],
         [90, 10, 92, 98, 30, 36, 48, 26]),
        ((20,), 2, EPS_10, 8, (4, -16, 1), [0, 2, 4, 6, 8],
         [3.148939e-03 - 0.000928j, 3.358308e-03 - 0.0009j,
          -1.645811e+01 + 0.021235j, 5.108325e-03 - 0.064858j,
          -8.593715e-02 - 0.112894j], [9, 1, 1, 1, 3]),
    ],
)
def test_mat_hz_selected_indices(size: Size, spacing: Size, eps: Union[float, np.ndarray],
                                 pml: Optional[Size], pml_params: Size3, selected_indices: List[int],
                                 expected_mat_hz_data: np.ndarray, expected_mat_hz_indices: np.ndarray):
    grid = FDFD(size, spacing, eps=eps, pml=pml, pml_params=pml_params, pml_sep=1)
    actual_mat_hz = grid.mat_hz
    np.testing.assert_allclose(actual_mat_hz.data[selected_indices], expected_mat_hz_data, rtol=3e-4)
    np.testing.assert_allclose(actual_mat_hz.indices[selected_indices], expected_mat_hz_indices, rtol=3e-4)


@pytest.mark.parametrize(
    "size, spacing, eps, pml, pml_params, src, expected, tm_2d",
    [
        ((5, 5), 1, EPS_5_5, None, (4, -16, 1), SOURCE_5_5,
         [-0.07167871 - 0.j, -0.14298599 - 0.j, 0.01107421 + 0.j, -0.1477444 - 0.j,
          -0.02976408 - 0.j, -0.03151538 - 0.j, -0.1121591 - 0.j, -0.03932945 - 0.j,
          -0.11261177 - 0.j, -0.08187067 - 0.j, -0.09351526 - 0.j, -0.10450915 - 0.j,
          -0.0381832 - 0.j, -0.15488618 - 0.j, -0.07711778 - 0.j, -0.06523334 - 0.j,
          -0.17933909 - 0.j, -0.22449252 - 0.j, 0.03195563 + 0.j, -0.03765066 - 0.j,
          -0.04298976 - 0.j, -0.12478716 - 0.j, -0.1828108 - 0.j, -0.22549907 - 0.j,
          -0.17781745 - 0.j], False),
        ((5,), 0.5, EPS_10, 2, (4, -16, 1), SOURCE_10,
         [-0.17089647 - 1.02674552e-05j, -0.16491481 - 2.76797540e-04j,
          -0.15285368 - 6.93158247e-03j, -0.09097409 + 4.74820947e-03j,
          0.07009409 - 3.09620835e-04j, -0.33076997 + 4.20679477e-04j,
          0.09635553 - 9.85560547e-04j, -0.08525008 + 3.21510276e-03j,
          -0.16716785 - 8.39563779e-03j, -0.16756679 - 1.39350922e-04j], False),
        ((5, 5), 1, EPS_5_5, None, (4, -16, 1), SOURCE_5_5,
         [-0.12267507, -0.26522232, 0.02130549, -0.20746095, -0.05265192, -0.04341361, -0.17348322, -0.05401931,
          -0.16864395, -0.13477245, -0.10720724, -0.13857351, -0.04744684, -0.19776741, -0.09765291, -0.08476391,
          -0.2110588, -0.25837641, 0.03507694, -0.04967016, -0.07139072, -0.20058036, -0.25911829, -0.25718025,
          -0.24592517], True),
        ((5,), 0.5, EPS_10, 2, (4, -16, 1), SOURCE_10,
         [-0.19882722 + 7.14214969e-05j, -0.24548843 - 8.43838043e-04j, -0.21431744 - 9.28589671e-03j,
          -0.12438099 + 7.56693867e-03j, 0.07728352 - 9.30301413e-04j, -0.3811222 + 9.69347403e-04j,
          0.12419099 - 2.39105654e-03j, -0.10017246 + 7.44759445e-03j, -0.24136862 - 1.68686342e-02j,
          -0.23727428 - 3.29575035e-04j], True),
    ],
)
def test_solve_2d(size: Size, spacing: Size, eps: Union[float, np.ndarray],
                  pml: Optional[Size], pml_params: Size3, src: np.ndarray, expected: np.ndarray, tm_2d: bool):
    grid = FDFD(size, spacing, eps=eps, pml=pml, pml_params=pml_params, pml_sep=1)
    actual = grid.solve(src, tm_2d=tm_2d)[int(tm_2d), 2].ravel()  # hz if tm_2d, ez if not tm_2d
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "size, spacing, eps, pml, pml_params, src, expected",
    [
        ((1.5, 1.5, 1), 0.5, EPS_3_3_2, None, (4, -16, 1), SOURCE_3_3_2,
         [-1.22372285, -0.2033493, -2.93547346, -1.57145287,
          3.60707297, 1.71612853, -0.5059581, 0.47474773,
          -1.52122203, -0.17910732, 1.99674375, -1.06473848,
          -0.27784698, 0.99227531, 2.72554383, 4.7693412,
          -3.30200237, -6.50685786, 0.86746106, -3.05483556,
          5.96681911, 4.1779766, -0.18377168, -4.25069235,
          0.12800289, 0.14021134, -1.86431569, -0.3491506,
          0.66900647, 0.88677132, -0.40220728, 1.94473361,
          -5.38257397, -1.9850864, 0.36696246, 2.42108452,
          0.16731768, 0.30936384, 0.93888465, 0.04561664,
          -1.42294774, -0.71444213, -0.25400074, -0.1811002,
          0.02535975, -0.93109824, -0.13218408, 1.52497524,
          0.13498791, -0.23291909, 0.19398719, -0.48559814,
          -0.62364214, 0.46228806])
    ],
)
def test_solve_full(size: Size, spacing: Size, eps: Union[float, np.ndarray],
                    pml: Optional[Size], pml_params: Size3, src: np.ndarray, expected: np.ndarray):
    grid = FDFD(size, spacing, eps=eps, pml=pml, pml_params=pml_params, pml_sep=1)
    actual = grid.solve(src)[0].ravel()  # just check the e field
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize(
    "size, spacing, eps, pml, pml_params, src",
    [
        ((3, 3), 1, EPS_3_3, None, (4, -16, 1), np.ones(27)),
        ((5,), 0.5, EPS_10, 2, (4, -16, 1), np.ones(30)),
    ],
)
def test_solve_src_size_error(size: Size, spacing: Size, eps: Union[float, np.ndarray],
                              pml: Optional[Size], pml_params: Size3, src: np.ndarray):
    with pytest.raises(ValueError, match='Expected src.size == '):
        FDFD(size, spacing, eps=eps, pml=pml, pml_params=pml_params, pml_sep=1).solve(src)


@pytest.mark.parametrize(
    "size, spacing, eps, pml, pml_params, src, tm_2d",
    [
        # ((3, 3, 2), 1, EPS_3_3_2, None, (4, -16, 1), SOURCE_3_3_2, False),  # takes a while, bicgstab test
        ((3, 3), 1, EPS_3_3, None, (4, -16, 1), SOURCE_3_3, True),
        ((5,), 0.5, EPS_10, 2, (4, -16, 1), SOURCE_10, True),
        ((3, 3), 1, EPS_3_3, None, (4, -16, 1), SOURCE_3_3, False),
        ((5,), 0.5, EPS_10, 2, (4, -16, 1), SOURCE_10, False)
    ]
)
def test_solve_fn_jax(size: Size, spacing: Size, eps: Union[float, np.ndarray],
                      pml: Optional[Size], pml_params: Size3, src: np.ndarray, tm_2d: bool):
    grid = FDFD(size, spacing, eps=eps, pml=pml, pml_params=pml_params, pml_sep=1)
    solve_jax = grid.get_fields_fn(src, tm_2d=tm_2d)
    jax_result = solve_jax(grid.eps).ravel()
    numpy_result = grid.solve(src, tm_2d=tm_2d).ravel()
    np.testing.assert_allclose(jax_result, numpy_result, rtol=2e-3)

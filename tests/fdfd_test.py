import numpy as np
import pytest

from simphox.fdfd import FDFD
from simphox.typing import Shape, Dim, Dim3, Optional, List, Union

np.random.seed(0)

from jax.config import config

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')


@pytest.mark.parametrize(
    "shape, spacing, pml, pml_params, selected_indices, expected_df_data, expected_df_indices",
    [
        ((10, 10, 10), 0.5, 4, (4, -16, 1), [0, 2, 10, 40, 100, 1000, 400, 203, 314],
         [0.05134884 + 0.31632416j, 0.05134884 + 0.31632416j,
          0.05134884 + 0.31632416j, 0.05134884 + 0.31632416j, 0.05134884 + 0.31632416j, 2.,
          1.74179713 + 0.67062435j, -0.41673505 - 0.81228197j, 0.41673505 + 0.81228197j],
         [100, 101, 105, 120, 150, 600, 300, 101, 257]),
        ((10, 10), 1, 4, (4, -16, 1), [0, 2, 10, 40, 100, 130, 190, 80],
         [0.02567442 + 0.15816208j, 0.02567442 + 0.15816208j,
          0.02567442 + 0.15816208j, 0.87089856 + 0.33531218j, 1., 1.,
          -0.03404911 - 0.18135536j, 1.],
         [10, 11, 15, 30, 60, 75, 95, 50]),
        ((10,), 2, 4, (4, -16, 1), [0, 2, 4, 6, 8],
         [0.01283721 + 0.07908104j, 0.10418376 + 0.20307049j, 0.43544928 + 0.16765609j,
          0.49971064 + 0.01202487j, 0.5], [1, 2, 3, 4, 5]),
    ],
)
def test_df_pml_selected_indices(shape: Shape, spacing: Dim, pml: Optional[Dim], pml_params: Dim3,
                                 selected_indices: List[int],
                                 expected_df_data: np.ndarray, expected_df_indices: np.ndarray):
    grid = FDFD(shape, spacing, pml=pml, pml_params=pml_params)
    actual_df = grid.df
    np.testing.assert_allclose(actual_df[0].data[selected_indices], expected_df_data)
    np.testing.assert_allclose(actual_df[0].indices[selected_indices], expected_df_indices)


@pytest.mark.parametrize(
    "shape, spacing, pml, pml_params, selected_indices, expected_db_data, expected_db_indices",
    [
        ((10, 10, 10), 0.5, 4, (4, -16, 1), [0, 2, 10, 40, 100, 1000, 400, 203, 314],
         [-0.02033147 - 0.20062297j, -0.02033147 - 0.20062297j, -0.02033147 - 0.20062297j, -0.02033147 - 0.20062297j,
          -0.02033147 - 0.20062297j, 2., 1.44890765 + 0.89357816j, -0.18608182 - 0.58097951j,
          0.18608182 + 0.58097951j],
         [900, 901, 905, 920, 950, 500, 200, 1, 157]),
        ((10, 10), 1, 4, (4, -16, 1), [0, 2, 10, 40, 100, 130, 190, 80],
         [-0.01016574 - 0.10031149j, -0.01016574 - 0.10031149j,
          -0.01016574 - 0.10031149j, 0.72445382 + 0.44678908j,
          1., 1., 0.09304091 + 0.29048976j, 1.],
         [90, 91, 95, 20, 50, 65, 95, 40]),
        ((10,), 2, 4, (4, -16, 1), [0, 2, 4, 6, 8],
         [-0.00508287 - 0.05015574j, -0.04652045 - 0.14524488j, -0.36222691 - 0.22339454j, -0.49925823 - 0.01924408j,
          -0.5], [9, 0, 1, 2, 3]),
    ],
)
def test_db_pml_selected_indices(shape: Shape, spacing: Dim, pml: Optional[Dim], pml_params: Dim3,
                                 selected_indices: List[int],
                                 expected_db_data: np.ndarray, expected_db_indices: np.ndarray):
    grid = FDFD(shape, spacing, pml=pml, pml_params=pml_params)
    actual_db = grid.db
    np.testing.assert_allclose(actual_db[0].data[selected_indices], expected_db_data)
    np.testing.assert_allclose(actual_db[0].indices[selected_indices], expected_db_indices)


@pytest.mark.parametrize(
    "shape, spacing, eps, pml, pml_params, selected_indices, expected_db_data, expected_db_indices",
    [
        ((10, 10, 10), 0.5, 1 + np.random.rand(10, 10, 10), 4, (4, -16, 1), [0, 2, 10, 40, 100, 1000, 400, 203, 314],
         [-23.826338 + 0.0755392j, 0.07138356 - 0.02103651j, -0.0624179 + 0.01673309j, -19.513415 + 1.76102538j,
          0.07819586 + 0.63566241j, 0.10269768 + 0.63264833j, -0.0624179 + 0.01673309j, -0.17422277 + 0.0886948j,
          -8.1875137 + 4.17361047j],
         [0, 9, 2009, 3, 2006, 2176, 2039, 1115, 24]),
        ((6, 5, 10), 0.5, 1 + np.random.rand(6, 5, 10), None, (4, -16, 1), [0, 2, 10, 40, 100, 1000, 400, 203, 314],
         [-6.46858101, -4., 4., -12.2331403, 4., 4., 4., 4., -6.99875595],
         [0, 9, 609, 3, 606, 726, 639, 365, 24]),
        ((10, 10,), 1, 1 + np.random.rand(10, 10), None, (4, -16, 1), [0, 2, 10, 40, 100, 130, 190, 80],
         [-27.87600171, -1., 1., -1., -1., -1., -22.86141228, 1.],
         [0, 9, 100, 114, 15, 118, 27, 110]),
        ((6,), 2, 1 + np.random.rand(6), None, (4, -16, 1), [0, 2, 4, 6, 8],
         [-21.585215 + 0.j, -19.739484 + 0.j, -32.706181 + 0.j, -26.003054 + 0.j, -0.25 + 0.j],
         [0, 2, 4, 6, 11])
    ],
)
def test_mat_selected_indices(shape: Shape, spacing: Dim, eps: Union[float, np.ndarray],
                              pml: Optional[Dim], pml_params: Dim3, selected_indices: List[int],
                              expected_db_data: np.ndarray, expected_db_indices: np.ndarray):
    grid = FDFD(shape, spacing, eps=eps, pml=pml, pml_params=pml_params)
    actual_mat = grid.mat
    np.testing.assert_allclose(actual_mat.data[selected_indices], expected_db_data)
    np.testing.assert_allclose(actual_mat.indices[selected_indices], expected_db_indices)


@pytest.mark.parametrize(
    "shape, spacing, eps, pml, pml_params, selected_indices, expected_mat_ez_data, expected_mat_ez_indices",
    [
        ((10, 10), 1, 1 + np.random.rand(10, 10), 4, (4, -16, 1), [0, 2, 10, 40, 100, 130, 190, 80],
         [-2.07684849e+01 + 0.0188848j, 1.78458909e-02 - 0.00525913j,
          3.05067059e-02 - 0.38732672j, -6.92081956e-01 - 0.49229807j,
          3.05067059e-02 - 0.38732672j, 3.05067059e-02 - 0.38732672j,
          -8.56701006e-01 - 0.36833401j, 4.35556934e-02 - 0.0221737j],
         [0, 9, 1, 7, 10, 16, 28, 6]),
        ((10,), 2, 1 + np.random.rand(10), 4, (4, -16, 1), [0, 2, 4, 6, 8],
         [-2.639165e+01 + 0.002361j, 4.461473e-03 - 0.001315j, -2.755190e+01 + 0.030123j, 7.626676e-03 - 0.096832j,
          -1.202780e-01 - 0.158007j], [0, 9, 1, 1, 3]),
    ],
)
def test_mat_ez_selected_indices(shape: Shape, spacing: Dim, eps: Union[float, np.ndarray],
                                 pml: Optional[Dim], pml_params: Dim3, selected_indices: List[int],
                                 expected_mat_ez_data: np.ndarray, expected_mat_ez_indices: np.ndarray):
    grid = FDFD(shape, spacing, eps=eps, pml=pml, pml_params=pml_params)
    actual_mat_ez = grid.mat_ez
    np.testing.assert_allclose(actual_mat_ez.data[selected_indices], expected_mat_ez_data, rtol=1e-4)
    np.testing.assert_allclose(actual_mat_ez.indices[selected_indices], expected_mat_ez_indices, rtol=1e-4)


@pytest.mark.parametrize(
    "shape, spacing, eps, pml, pml_params, selected_indices, expected_mat_hz_data, expected_mat_hz_indices",
    [
        ((10, 10), 1, 1 + np.random.rand(10, 10), 4, (4, -16, 1), [0, 2, 10, 40, 100, 130, 190, 80],
         [0.011478 - 0.003383j, 0.009269 - 0.002485j, 0.010008 - 0.002949j,
          0.009969 - 0.002938j, -0.333122 - 0.437615j, -0.319263 - 0.419409j,
          -0.764422 - 0.047904j, 0.057465 - 0.057304j],
         [90, 10, 92, 98, 30, 36, 48, 26]),
        ((10,), 2, 1 + np.random.rand(10), 4, (4, -16, 1), [0, 2, 4, 6, 8],
         [2.463368e-03 - 0.000726j, 2.042910e-03 - 0.000548j,
          -1.645445e+01 + 0.019385j, 5.114120e-03 - 0.064931j,
          -1.076240e-01 - 0.141383j], [9, 1, 1, 1, 3]),
    ],
)
def test_mat_hz_selected_indices(shape: Shape, spacing: Dim, eps: Union[float, np.ndarray],
                                 pml: Optional[Dim], pml_params: Dim3, selected_indices: List[int],
                                 expected_mat_hz_data: np.ndarray, expected_mat_hz_indices: np.ndarray):
    grid = FDFD(shape, spacing, eps=eps, pml=pml, pml_params=pml_params)
    actual_mat_hz = grid.mat_hz
    np.testing.assert_allclose(actual_mat_hz.data[selected_indices], expected_mat_hz_data, rtol=3e-4)
    np.testing.assert_allclose(actual_mat_hz.indices[selected_indices], expected_mat_hz_indices, rtol=3e-4)


@pytest.mark.parametrize(
    "shape, spacing, eps, pml, pml_params, src, expected, tm_2d",
    [
        ((5, 5), 1, 1 + np.random.rand(5, 5), None, (4, -16, 1), np.random.rand(25),
         [-0.0098678 - 0.j, -0.02835023 - 0.j, -0.08116987 - 0.j, -0.09871585 - 0.j,
          -0.08672967 - 0.j, -0.15901065 - 0.j, -0.02310777 - 0.j, -0.1461786 - 0.j,
          -0.15291198 - 0.j, -0.09105158 - 0.j, -0.06854252 - 0.j, -0.10924188 - 0.j,
          -0.08740056 - 0.j, -0.09849117 - 0.j, -0.06651749 - 0.j, -0.05385054 - 0.j,
          -0.13523573 - 0.j, -0.00408954 - 0.j, -0.01164929 - 0.j, -0.13425466 - 0.j,
          -0.12931978 - 0.j, -0.03954953 - 0.j, -0.06191786 - 0.j, -0.16296683 - 0.j,
          -0.13406839 - 0.j], False),
        ((10,), 0.5, 1 + np.random.rand(10), 4, (4, -16, 1), np.random.rand(10),
         [-0.03209711 + 0.00014801j, -0.13980539 - 0.00027873j,
          -0.17055568 - 0.02337514j, -0.05307879 + 0.02511928j,
          -0.22670035 - 0.00703862j, 0.04975806 + 0.00152104j,
          -0.17760366 - 0.00085765j, 0.02621358 + 0.00324852j,
          -0.04440355 + 0.00326276j, -0.13017986 - 0.00331717j], False),
        ((5, 5), 1, 1 + np.random.rand(5, 5), None, (4, -16, 1), np.random.rand(25),
         [-0.25829168 - 0.j, -0.01587057 - 0.j, -0.042272 - 0.j, -0.19711486 - 0.j,
          -0.02698197 - 0.j, -0.06255242 - 0.j, -0.03830037 - 0.j, -0.04244731 - 0.j,
          -0.1260402 - 0.j, -0.0198188 - 0.j, -0.08023085 - 0.j, -0.1108811 - 0.j,
          0.02258808 + 0.j, -0.10976114 - 0.j, -0.00987089 - 0.j, -0.11014809 - 0.j,
          -0.22365001 - 0.j, -0.22621251 - 0.j, 0.011518 + 0.j, -0.16253209 - 0.j,
          -0.21603078 - 0.j, -0.03588253 - 0.j, -0.19797152 - 0.j, -0.23330676 - 0.j,
          -0.20691924 - 0.j], True),
        ((10,), 0.5, 1 + np.random.rand(10), 4, (4, -16, 1), np.random.rand(10),
         [-2.18255337e-01 - 6.93905290e-06j, -2.30688897e-01 - 2.54425861e-03j,
          -5.03957093e-02 + 6.65292518e-02j, -3.09433408e-01 - 7.18964433e-02j,
          8.79757981e-02 + 2.77606488e-02j, -3.70644312e-01 - 8.86546006e-03j,
          8.81885112e-02 + 7.70259865e-03j, -2.57952836e-01 - 2.40989836e-02j,
          -3.25580867e-04 + 5.65293064e-02j, -2.33226392e-01 - 7.09875575e-03j], True),
    ],
)
def test_solve_2d(shape: Shape, spacing: Dim, eps: Union[float, np.ndarray],
                  pml: Optional[Dim], pml_params: Dim3, src: np.ndarray, expected: np.ndarray, tm_2d: bool):
    grid = FDFD(shape, spacing, eps=eps, pml=pml, pml_params=pml_params)
    actual = grid.solve(src, tm_2d=tm_2d)[int(tm_2d), 2].ravel()  # hz if tm_2d, ez if not tm_2d
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "shape, spacing, eps, pml, pml_params, src, expected",
    [
        ((3, 3, 2), 0.5, 1 + np.random.rand(3, 3, 2), None, (4, -16, 1), np.random.rand(54),
         [0.24237113 + 0.j, -0.07359536 + 0.j, 0.28886727 + 0.j, 0.05057506 + 0.j,
          -0.33484689 - 0.j, -0.61932041 - 0.j, -0.69873692 - 0.j, -0.05731527 - 0.j,
          -0.88403197 - 0.j, -0.02088915 - 0.j, 0.12789415 + 0.j, 0.94119096 + 0.j,
          -0.05135766 - 0.j, -0.16643756 + 0.j, -0.12434779 - 0.j, 0.07834147 + 0.j,
          -0.06479193 + 0.j, -0.24350049 - 0.j, -0.04799194 + 0.j, -0.09193079 - 0.j,
          -0.19140449 - 0.j, -0.11670119 - 0.j, 0.48765872 + 0.j, -0.3220854 + 0.j,
          -0.55414875 - 0.j, 0.18994315 + 0.j, 0.1759667 + 0.j, 0.7065071 + 0.j,
          -0.69168417 - 0.j, -0.60620362 - 0.j, -0.10326846 + 0.j, -0.05431738 - 0.j,
          -0.65365688 + 0.j, -0.18599579 - 0.j, 0.35364707 + 0.j, -0.11638765 - 0.j,
          0.08164954 + 0.j, -0.06130199 - 0.j, 0.18063669 + 0.j, 0.09881442 + 0.j,
          -0.72792423 - 0.j, -0.25104903 - 0.j, -0.86529585 - 0.j, -1.60343367 - 0.j,
          -0.87728383 - 0.j, -1.25095393 + 0.j, 2.65436477 + 0.j, 2.345954 + 0.j,
          0.94036481 + 0.j, 0.99622357 + 0.j, 0.63561156 + 0.j, 0.80567092 + 0.j,
          -2.38460573 - 0.j, -1.92793007 - 0.j])
    ],
)
def test_solve_full(shape: Shape, spacing: Dim, eps: Union[float, np.ndarray],
                    pml: Optional[Dim], pml_params: Dim3, src: np.ndarray, expected: np.ndarray):
    grid = FDFD(shape, spacing, eps=eps, pml=pml, pml_params=pml_params)
    actual = grid.solve(src)[0].ravel()  # just check the e field
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize(
    "shape, spacing, eps, pml, pml_params, src",
    [
        ((3, 3), 1, 1 + np.random.rand(3, 3), None, (4, -16, 1), np.random.rand(27)),
        ((10,), 0.5, 1 + np.random.rand(10), 4, (4, -16, 1), np.random.rand(30)),
    ],
)
def test_solve_src_size_error(shape: Shape, spacing: Dim, eps: Union[float, np.ndarray],
                          pml: Optional[Dim], pml_params: Dim3, src: np.ndarray):
    with pytest.raises(ValueError, match='Expected src.size == '):
        FDFD(shape, spacing, eps=eps, pml=pml, pml_params=pml_params).solve(src)


@pytest.mark.parametrize(
    "shape, spacing, eps, pml, pml_params, src, tm_2d",
    [
        # ((3, 3, 2), 1, 1 + np.random.rand(3, 3, 2).astype(np.complex128), None, (4, -16, 1),
        #  np.random.rand(54).astype(np.complex128), False),
        ((3, 3), 1, 1 + np.random.rand(3, 3).astype(np.complex128), None, (4, -16, 1),
         np.random.rand(9).astype(np.complex128), True),
        ((10,), 0.5, 1 + np.random.rand(10), 4, (4, -16, 1), np.random.rand(10).astype(np.complex128), True),
        ((3, 3), 1, 1 + np.random.rand(3, 3).astype(np.complex128), None, (4, -16, 1),
         np.random.rand(9).astype(np.complex128), False),
        ((10,), 0.5, 1 + np.random.rand(10), 4, (4, -16, 1), np.random.rand(10).astype(np.complex128), False)
    ]
)
def test_solve_fn_jax(shape: Shape, spacing: Dim, eps: Union[float, np.ndarray],
                      pml: Optional[Dim], pml_params: Dim3, src: np.ndarray, tm_2d: bool):
    grid = FDFD(shape, spacing, eps=eps, pml=pml, pml_params=pml_params)
    solve_jax = grid.get_fields_fn(src, tm_2d=tm_2d)
    jax_result = solve_jax(grid.eps).ravel()
    numpy_result = grid.solve(src, tm_2d=tm_2d).ravel()
    np.testing.assert_allclose(jax_result, numpy_result, rtol=2e-3)

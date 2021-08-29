from typing import Union, Tuple, List, Optional, Dict, Callable, Iterable
import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp

Shape2 = Tuple[int, int]
Shape3 = Tuple[int, int, int]
Dim2 = Tuple[float, float]
Dim3 = Tuple[float, float, float]
Dim4 = Tuple[float, float, float, float]
Shape = Union[Shape2, Shape3]
Dim = Union[Dim2, Dim3]
GridSpacing = Union[float, Tuple[float, float, float]]
Op = Callable[[np.ndarray], np.ndarray]
SpSolve = Callable[[sp.spmatrix, np.ndarray], np.ndarray]
Source = Union[Callable[[float], Tuple[np.ndarray, np.ndarray]], np.ndarray]
State = Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]], Optional[List[np.ndarray]]]
Array = Union[jnp.ndarray, np.ndarray]
MeasureInfo = Dict[str, List[int]]
PortLabel = Union[str, int]
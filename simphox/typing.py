from typing import Union, Tuple, List, Optional, Dict, Callable, Iterable

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

Shape2 = Tuple[int, int]
Shape3 = Tuple[int, int, int]
Size2 = Tuple[float, float]
Size3 = Tuple[float, float, float]
Size4 = Tuple[float, float, float, float]
Shape = Union[Shape2, Shape3]
Size = Union[Size2, Size3]
Spacing = Union[float, Tuple[float, float, float]]
Op = Callable[[np.ndarray], np.ndarray]
SpSolve = Callable[[sp.spmatrix, np.ndarray], np.ndarray]
Array = Union[jnp.ndarray, np.ndarray]
State = Tuple[Array, Array, Optional[List[Array]], Optional[List[Array]]]
MeasureInfo = Dict[str, List[int]]
Excitation = Union[str, Tuple[str, int], Dict[str, List[int]], Iterable[Union[str, Tuple[str, int]]]]
SourceLabel = Union[str, Dict[Tuple[str, int], float], Dict[str, float]]
Source = Tuple[Optional[np.ndarray], Optional[np.ndarray], Tuple[slice, ...]]
PortLabel = Union[str, int]
PhaseParams = Tuple[np.ndarray, np.ndarray, np.ndarray]
IndexSelect = Tuple[Union[slice, np.ndarray], ...]

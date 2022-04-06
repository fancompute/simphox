from typing import Tuple, Union, List

import numpy as np
from enum import Enum
from pydantic.dataclasses import dataclass
from ..utils import fix_dataclass_init_docs


def phase_matrix(top: float, bottom: float = 0):
    return np.array([
        [np.exp(1j * top), 0],
        [0, np.exp(1j * bottom)]
    ])


def coupling_matrix_s(s: float):
    return np.array([
        [np.sqrt(1 - s), np.sqrt(s)],
        [np.sqrt(s), -np.sqrt(1 - s)]
    ])


def loss2insertion(loss_db: float):
    return np.sqrt(np.exp(np.log(10) * loss_db / 10))


def coupling_matrix_phase(theta: float, split_error: float = 0, loss_error: float = 0):
    insertion = loss2insertion(loss_error)
    return np.array([
        [insertion * np.cos(theta / 2 + split_error / 2), insertion * 1j * np.sin(theta / 2 + split_error / 2)],
        [1j * np.sin(theta / 2 + split_error / 2), np.sin(theta / 2 + split_error / 2)]
    ])


def _embed_2x2(mat: np.ndarray, n: int, i: int, j: int):
    if mat.shape != (2, 2):
        raise AttributeError(f"Expected shape (2, 2), but got {mat.shape}.")
    out = np.eye(n, dtype=np.complex128)
    out[i, i] = mat[0, 0]
    out[i, j] = mat[0, 1]
    out[j, i] = mat[1, 0]
    out[j, j] = mat[1, 1]
    return out


class PhaseStyle(str, Enum):
    """Enumeration for the different phase styles (differential, common, top, bottom).

    A phase style is defined as

    Attributes:
        TOP: Top phase shift
        BOTTOM: Bottom phase shift
        DIFFERENTIAL: Differential phase shift
        SYMMETRIC: Symmetric phase shift

    """
    TOP = 'top'
    BOTTOM = 'bottom'
    DIFFERENTIAL = 'differential'
    SYMMETRIC = 'symmetric'


@fix_dataclass_init_docs
@dataclass
class CouplingNode:
    """A simple programmable 2x2 coupling node model.

    Attributes:
        node_id: The index of the coupling node (useful in networks).
        loss: The differential loss error of the overall coupling node in dB (one at each of the theta and phi phase shifters).
        bs_error: The splitting error of the coupling node (MZI coupling errors).
        n: Total number of inputs/outputs.
        top: Top input/output index.
        bottom: Bottom input/output index.
        num_top: Total number of inputs connected to top port (for tree architectures initialization).
        num_bottom: Total number of inputs connected to bottom port (for tree architecture initialization).
        column: The column label assigned to the node

    """
    node_id: int = 0
    loss: Tuple[float, float] = (0., 0.)
    bs_error: Union[float, Tuple[float, float]] = 0.
    n: int = 2
    top: int = 0
    bottom: int = 1
    alpha: int = 1
    beta: int = 1
    column: int = 0

    def __post_init_post_parse__(self):
        self.stride = self.bottom - self.top
        self.top_descendants = np.array([])
        self.bot_descendants = np.array([])
        self.bs_error = (self.bs_error, self.bs_error) if np.isscalar(self.bs_error) else self.bs_error

    @property
    def mzi_terms(self):
        return [
            np.cos(np.pi / 4 + self.bs_error[1]) * np.cos(np.pi / 4 + self.bs_error[0]),
            np.cos(np.pi / 4 + self.bs_error[1]) * np.sin(np.pi / 4 + self.bs_error[0]),
            np.sin(np.pi / 4 + self.bs_error[1]) * np.cos(np.pi / 4 + self.bs_error[0]),
            np.sin(np.pi / 4 + self.bs_error[1]) * np.sin(np.pi / 4 + self.bs_error[0]),
        ]

    def ideal_node(self, s: float = 0, phi: float = 0):
        """Ideal node with parameters s and phi that can be embedded in a circuit.

        Args:
            s: Cross split ratio :math:`s \\in [0, 1]` (:math:`s=1` means cross state).
            phi: Differential phase :math:`\\phi \\in [0, 2\\pi)` (set phase between inputs to the node).

        Returns:
            The embedded ideal node.

        """
        mat = phase_matrix(phi) @ coupling_matrix_s(s)
        return _embed_2x2(mat, self.n, self.top, self.bottom)

    def mzi_node_matrix(self, theta: float = 0, phi: float = 0, embed: bool = True):
        """Tunable Mach-Zehnder interferometer node matrix.

        Args:
            theta: MMI phase between odd/even modes :math:`\\theta \\in [0, \\pi]` (:math:`\\theta=0` means cross state).
            phi: Differential phase :math:`\\phi \\in [0, 2\\pi)` (set phase between inputs to the node).
            embed: Whether to return the embedded matrix in the n-waveguide system (specified in node).

        Returns:
            Tunable MMI node matrix embedded in an :math:`N`-waveguide system.

        """
        mat = self.dc(right=True) @ phase_matrix(theta) @ self.dc(right=False) @ phase_matrix(phi)
        return _embed_2x2(mat, self.n, self.top, self.bottom) if embed else mat

    def phase_matrix(self, top: float = 0, bottom: float = 0):
        """Embedded phase matrix.

        Args:
            top: Top phase of the phase matrix
            bottom: Bottom phase of the phase matrix

        Returns:
            Embedded phase matrix.

        """
        return _embed_2x2(phase_matrix(top, bottom), self.n, self.top, self.bottom)

    def dc(self, right: bool = False) -> np.ndarray:
        """Directional coupler matrix with error.

        Args:
            right: Whether to use the left or right error (:code:`error` and :code:`error_right` respectively).

        Returns:
            A directional coupler matrix with error.

        """
        error = self.bs_error[right]
        insertion = loss2insertion(self.loss[right])
        return np.array([
            [np.cos(np.pi / 4 + error) * insertion, 1j * np.sin(np.pi / 4 + error) * insertion],
            [1j * np.sin(np.pi / 4 + error), np.cos(np.pi / 4 + error)]
        ])

    def mmi_node_matrix(self, theta: float = 0, phi: float = 0, embed: bool = True):
        """Tunable multimode interferometer node matrix.

        Args:
            theta: MZI arm phase :math:`\\theta \\in [0, \\pi]` (:math:`\\theta=0` means cross state).
            phi: Differential phase :math:`\\phi \\in [0, 2\\pi)` (set phase between inputs to the node).
            embed: Whether to return the embedded matrix in the n-waveguide system (specified in node).

        Returns:
            Tunable MMI node matrix embedded in an :math:`N`-waveguide system.

        """
        mat = coupling_matrix_phase(theta, self.bs_error, self.loss) @ phase_matrix(phi)
        return _embed_2x2(mat, self.n, self.top, self.bottom) if embed else mat

    def nullify(self, vector: np.ndarray, idx: int, lower_theta: bool = False, lower_phi: bool = False):
        theta = np.arctan2(np.abs(vector[idx]), np.abs(vector[idx + 1])) * 2
        theta = -theta if lower_theta else theta
        phi = np.angle(vector[idx + 1]) - np.angle(vector[idx])
        phi = -phi if lower_phi else phi
        mat = self.mzi_node_matrix(theta, phi)
        nullified_vector = mat @ vector
        return nullified_vector, mat, np.mod(theta, 2 * np.pi), np.mod(phi, 2 * np.pi)

    def set_descendants(self, top_descendants: np.ndarray, bot_descendants: np.ndarray):
        self.top_descendants = top_descendants
        self.bot_descendants = bot_descendants
        return self


def direct_transmissivity(top: np.ndarray, bottom: np.ndarray):
    """Get the direct transmissivity between top and bottom

    Args:
        top: Top vector elements
        bottom: Bottom vector elements

    Returns:
        The transmissivities

    """
    return np.abs(top) ** 2 / (np.abs(top) ** 2 + np.abs(bottom) ** 2 + np.spacing(1))


def transmissivity_to_phase(s: Union[float, np.ndarray], mzi_terms: np.ndarray = None):
    """Convert transmissivity :math:`\\boldsymbol{s}` to phase :math:`\\boldsymbol{\\theta}`.

    Args:
        s: The transmissivity float or array.
        mzi_terms: The splitting terms :code:`(cs, sc)` for an MZI node. If :code:`None`, assumes 0.5 power for each.

    Returns:
        The phase :math:`\\boldsymbol{\\theta}` corresponding to the transmissivity :math:`\\boldsymbol{s}`.

    """

    if mzi_terms is not None:
        _, cs, sc, _ = mzi_terms
    else:
        cs = sc = 0.5
    return np.arccos(np.minimum(np.maximum((s - cs ** 2 - sc ** 2) / (2 * cs * sc), -1), 1))

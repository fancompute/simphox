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


def coupling_matrix_phase(theta: float, split_error: float = 0, loss_error: float = 0):
    return (1 - loss_error) * np.array([
        [np.cos(theta / 2 + split_error / 2), 1j * np.sin(theta / 2 + split_error / 2)],
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
        SYMMETRIC: Symemtric phase shift

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
        loss: The loss of the overall coupling node.
        error: The splitting error of the coupling node (MZI coupling errors).
        n: Total number of inputs/outputs.
        top: Top input/output index.
        bottom: Bottom input/output index.
        num_top: Total number of inputs connected to top port (for tree architectures initialization).
        num_bottom: Total number of inputs connected to bottom port (for tree architecture initialization).
        level: The level label assigned to the node

    """
    node_id: int = 0
    loss: float = 0
    error: float = 0
    error_right: float = 0
    n: int = 2
    top: int = 0
    bottom: int = 1
    alpha: int = 1
    beta: int = 1
    level: int = 0

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

    def mzi_node_matrix(self, theta: float = 0, phi: float = 0):
        """Tunable Mach-Zehnder interferometer node matrix.

        Args:
            theta: MMI phase between odd/even modes :math:`\\theta \\in [0, \\pi]` (:math:`\\theta=0` means cross state).
            phi: Differential phase :math:`\\phi \\in [0, 2\\pi)` (set phase between inputs to the node).

        Returns:
            Tunable MMI node matrix embedded in an :math:`N`-waveguide system.

        """
        mat = (1 - self.loss) * self.dc(right=True) @ phase_matrix(theta) @ self.dc(right=False) @ phase_matrix(phi)
        return _embed_2x2(mat, self.n, self.top, self.bottom)

    def dc(self, right: bool = False) -> np.ndarray:
        """Directional coupler matrix with error.

        Args:
            right: Whether toi use the left or right error (:code:`error` and :code:`error_right` respectively).

        Returns:
            A directional coupler matrix with error.

        """
        error = (self.error, self.error_right)[right]
        return np.array([
            [np.cos(np.pi / 4 + error), 1j * np.sin(np.pi / 4 + error)],
            [1j * np.sin(np.pi / 4 + error), np.cos(np.pi / 4 + error)]
        ])

    def mmi_node_matrix(self, theta: float = 0, phi: float = 0):
        """Tunable multimode interferometer node matrix.

        Args:
            theta: MZI arm phase :math:`\\theta \\in [0, \\pi]` (:math:`\\theta=0` means cross state).
            phi: Differential phase :math:`\\phi \\in [0, 2\\pi)` (set phase between inputs to the node).

        Returns:
            Tunable MMI node matrix embedded in an :math:`N`-waveguide system.

        """
        mat = (1 - self.loss) * coupling_matrix_phase(theta, self.error, self.loss) @ phase_matrix(phi)
        return _embed_2x2(mat, self.n, self.top, self.bottom)
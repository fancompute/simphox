from typing import Tuple

import numpy as np
import scipy as sp

try:
    DPHOX_IMPORTED = True
    from dphox.device import Device
    from dphox.pattern import Pattern
except ImportError:
    DPHOX_IMPORTED = False

from ..typing import List, Optional
from .coupling import CouplingNode, PhaseStyle
from .forward import ForwardMesh


def _tree(indices: np.ndarray, n_rails: int, start: int = 0, column: int = 0,
          balanced: bool = True) -> List[CouplingNode]:
    """Recursive helper function to generate balanced binary tree architecture.

    Our network data structure is similar to how arrays are generally treated as tree data structures
    in computer science, where the waveguide rail index refers to the index of the vector propagating through
    the coupling network. This is the basis for our recursive definitions of binary trees,
    which ultimately form universal photonic networks when cascaded.

    Args:
        indices: Ordered indices into the tree nodes (splitting ratio, losses, errors, etc. all use this).
        n_rails: Number of rails in the system.
        start: Starting index for the tree.
        column: column index for the tree (leaves have largest column index :math:`\\log_2 n`).
        balanced: If balanced, does balanced tree (:code:`m = n // 2`) otherwise linear chain (:code:`m = n - 1`).

    Returns:
        A list of :code:`CouplingNode`'s in order that modes visit them.

    """
    nodes = []
    n = indices.size + 1
    m = n // 2 if balanced else n - 1
    if n == 1:
        return nodes
    top = indices[1:m]
    bottom = indices[m:]
    nodes.append(CouplingNode(indices[0], top=start + m - 1, bottom=start + n - 1, n=n_rails,
                              alpha=bottom.size + 1, beta=top.size + 1, column=column))
    nodes.extend(_tree(top, n_rails, start, column + 1, balanced))
    nodes.extend(_tree(bottom, n_rails, start + m, column + 1, balanced))
    return nodes


def _butterfly(n: int, n_rails: int, start: int = 0, column: int = 0) -> List[CouplingNode]:
    """Recursive helper function to generate a balanced butterfly architecture (works best with powers of 2).

    Our network data structure is similar to how arrays are generally treated as tree data structures
    in computer science, where the waveguide rail index refers to the index of the vector propagating through
    the coupling network.

    Args:
        n: The number of modes in the binary tree (uses the top :math:`n` modes of the system).
        n_rails: Number of rails in the system.
        start: Starting index for the tree.
        column: column index for the tree (leaves have largest column index :math:`\\log_2 n`).

    Returns:
        A list of :code:`CouplingNode`'s in order that modes visit them.

    """
    nodes = []
    m = n // 2
    if n == 1:
        return nodes
    nodes.extend([CouplingNode(top=start + i, bottom=start + m + i, n=n_rails,
                               alpha=m + 1, beta=m + 1, column=column) for i in range(m)])
    nodes.extend(_butterfly(m, n_rails, start, column + 1))
    nodes.extend(_butterfly(n - m, n_rails, start + m, column + 1))
    return nodes


def tree(n: int, n_rails: Optional[int] = None, balanced: bool = True, phase_style: str = PhaseStyle.TOP) -> ForwardMesh:
    """Return a balanced or linear chain tree of MZIs.

    Args:
        n: Number of inputs into the tree.
        n_rails: Embed the first :code:`n` rails in an :code:`n_rails`-rail system (default :code:`n_rails == n`).
        balanced: If balanced, does balanced tree (:code:`m = n // 2`) otherwise linear chain (:code:`m = n - 1`).

    Returns:
        A :code:`CouplingCircuit` consisting of :code:`CouplingNode`'s arranged in a tree network.

    """
    n_rails = n if n_rails is None else n_rails
    return ForwardMesh(_tree(np.arange(n - 1), n_rails, balanced=balanced), phase_style=phase_style).invert_columns()


def butterfly(n: int, n_rails: Optional[int] = None) -> ForwardMesh:
    """Return a butterfly architecture

    Args:
        n: Number of inputs into the tree.
        n_rails: Embed the first :code:`n` rails in an :code:`n_rails`-rail system (default :code:`n_rails == n`).

    Returns:
        A :code:`CouplingCircuit` consisting of :code:`CouplingNode`'s arranged in a tree network.

    """
    n_rails = n if n_rails is None else n_rails
    return ForwardMesh(_butterfly(n - 1, n_rails)).invert_columns()


def vector_unit(v: np.ndarray, n_rails: int = None, balanced: bool = True, phase_style: str = PhaseStyle.TOP,
                error_mean_std: Tuple[float, float] = (0., 0.), loss_mean_std: Tuple[float, float] = (0., 0.)):
    """Generate an architecture based on our recursive definitions programmed to implement normalized vector :code:`v`.

    Args:
        v: The vector to be configured, if a matrix is provided
        n_rails: Embed the first :code:`n` rails in an :code:`n_rails`-rail system (default :code:`n_rails == n`).
        balanced: If balanced, does balanced tree (:code:`m = n // 2`) otherwise linear chain (:code:`m = n - 1`).
        phase_style: Phase style for the nodes (see the :code:`PhaseStyle` enum).
        error_mean_std: Mean and standard deviation for errors (in radians).
        loss_mean_std: Mean and standard deviation for losses (in dB).

    Returns:
        Coupling network, thetas and phis that initialize the coupling network to implement normalized v.
    """
    network = tree(v.shape[0], n_rails=n_rails, balanced=balanced, phase_style=phase_style)
    error_mean, error_std = error_mean_std
    loss_mean, loss_std = loss_mean_std
    network = network.add_error_mean(error_mean, loss_mean).add_error_variance(error_std, loss_std)
    thetas = np.zeros(v.shape[0] - 1)
    phis = np.zeros(v.shape[0] - 1)
    w = v.copy()
    w = w[:, np.newaxis] if w.ndim == 1 else w
    for nc in network.columns:
        # grab the elements for the top and bottom arms of the mzi.
        top = w[(nc.top,)]
        bottom = w[(nc.bottom,)]

        theta, phi = nc.parallel_nullify(w, network.mzi_terms)

        # Vectorized (efficient!) parallel mzi elements
        t11, t12, t21, t22 = nc.parallel_mzi_fn()(theta, phi)
        t11, t12, t21, t22 = t11[:, np.newaxis], t12[:, np.newaxis], t21[:, np.newaxis], t22[:, np.newaxis]

        # The final vector after the vectorized multiply
        w[(nc.top + nc.bottom,)] = np.vstack([t11 * top + t21 * bottom,
                                              t12 * top + t22 * bottom])

        # The resulting thetas and phis, indexed according to the coupling network specifications
        thetas[(nc.node_idxs,)] = theta
        phis[(nc.node_idxs,)] = phi

    final_basis_vec = np.zeros(v.shape[0])
    final_basis_vec[-1] = 1
    gammas = -np.angle(final_basis_vec * w[-1, -1])

    network.params = thetas, phis, gammas
    return network, w.squeeze()


def unitary_unit(u: np.ndarray, balanced: bool = True, phase_style: str = PhaseStyle.TOP,
                 error_mean_std: Tuple[float, float] = (0., 0.), loss_mean_std: Tuple[float, float] = (0., 0.)):
    """Generate an architecture based on our recursive definitions programmed to implement unitary :code:`u`.

    Args:
        u: The (:math:`k \\times n`) mutually orthogonal basis vectors (unitary if :math:`k=n`) to be configured.
        balanced: If balanced, does balanced tree (:code:`m = n // 2`) otherwise linear chain (:code:`m = n - 1`).
        phase_style: Phase style for the nodes (see the :code:`PhaseStyle` enum).
        error_mean_std: Mean and standard deviation for errors (in radians).
        loss_mean_std: Mean and standard deviation for losses (in dB).

    Returns:
        Node list, thetas and phis.

    """
    subunits = []
    thetas = np.array([])
    phis = np.array([])
    gammas = np.array([])
    n_rails = u.shape[0]
    num_columns = 0
    num_nodes = 0

    w = u.conj().T.copy()
    for i in reversed(range(n_rails + 1 - u.shape[1], n_rails)):
        # Generate the architecture as well as the theta and phi for each row of u.
        network, w = vector_unit(w[:i + 1, :i + 1], n_rails, balanced, phase_style,
                                 error_mean_std, loss_mean_std)

        # Update the phases.
        thetas = np.hstack((thetas, network.thetas))
        phis = np.hstack((phis, network.phis))
        gammas = np.hstack((network.gammas[-1], gammas))

        # We need to index the thetas and phis correctly based on the number of programmed nodes in previous subunits
        # For unbalanced architectures (linear chains), we can actually pack them more efficiently into a triangular
        # architecture.
        network.offset(num_nodes).offset_column(num_columns if balanced else 2 * (n_rails - 1 - i))

        # Add the nodes list to the subunits
        subunits.append(network)

        # The number of columns and nodes in the architecture are incremented by the subunit size (log_2(i))
        num_columns += subunits[-1].num_columns
        num_nodes += subunits[-1].num_nodes
    gammas = np.hstack((-np.angle(w[0, 0]), gammas))
    unit = ForwardMesh.aggregate(subunits)
    unit.params = thetas, phis, gammas
    return unit


def triangular(u: np.ndarray, phase_style: str = PhaseStyle.TOP, error_mean_std: Tuple[float, float] = (0., 0.),
               loss_mean_std: Tuple[float, float] = (0., 0.)):
    """Triangular mesh.

    Args:
        u: Unitary matrix
        phase_style: Phase style for the nodes of the mesh.
        error_mean_std: Split error mean and standard deviation
        loss_mean_std: Loss error mean and standard deviation (dB)

    Returns:
        A triangular mesh object.

    """
    return unitary_unit(u, balanced=False, phase_style=phase_style,
                        error_mean_std=error_mean_std, loss_mean_std=loss_mean_std)


def tree_cascade(u: np.ndarray, phase_style: str = PhaseStyle.TOP, error_mean_std: Tuple[float, float] = (0., 0.),
                 loss_mean_std: Tuple[float, float] = (0., 0.)):
    """Balanced cascade mesh.

    Args:
        u: Unitary matrix
        phase_style: Phase style for the nodes of the mesh.
        error_mean_std: Split error mean and standard deviation
        loss_mean_std: Loss error mean and standard deviation (dB)

    Returns:
        A tree cascade mesh object.

    """
    return unitary_unit(u, balanced=True, phase_style=phase_style,
                        error_mean_std=error_mean_std, loss_mean_std=loss_mean_std)


def dirichlet_matrix(v, embed_dim=None):
    phases = np.exp(-1j * np.angle(v))
    y = np.abs(v) ** 2
    yop = np.sqrt(np.outer(y, y))
    ysum = np.cumsum(y)
    yden = 1 / np.sqrt(ysum[:-1] * ysum[1:])
    u = np.zeros_like(yop, dtype=np.complex128)
    u[1:, :] = yden[:, np.newaxis] * yop[1:, :]
    u[np.triu_indices(v.size)] = 0
    u[1:, 1:][np.diag_indices(v.size - 1)] = -ysum[:-1] * yden
    u[0] = np.sqrt(y / ysum[-1])
    u *= phases
    u = np.roll(u, -1, axis=0) if embed_dim is None else sp.linalg.block_diag(np.roll(u, -1, axis=0),
                                                                              np.eye(embed_dim - v.size))
    return u.T

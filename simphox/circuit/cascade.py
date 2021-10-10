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
from .forward import ForwardCouplingCircuit


def _tree(indices: np.ndarray, n_rails: int, start: int = 0, level: int = 0,
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
        level: Level index for the tree (leaves have largest level index :math:`\\log_2 n`).
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
                              alpha=top.size + 1, beta=bottom.size + 1, level=level))
    nodes.extend(_tree(top, n_rails, start, level + 1, balanced))
    nodes.extend(_tree(bottom, n_rails, start + m, level + 1, balanced))
    return nodes


def tree(n: int, n_rails: Optional[int] = None, balanced: bool = True) -> ForwardCouplingCircuit:
    """Return a balanced or linear chain tree of MZIs.

    Args:
        n: Number of inputs into the tree.
        n_rails: Embed the first :code:`n` rails in an :code:`n_rails`-rail system (default :code:`n_rails == n`).
        balanced: If balanced, does balanced tree (:code:`m = n // 2`) otherwise linear chain (:code:`m = n - 1`).

    Returns:
        A :code:`CouplingCircuit` consisting of :code:`CouplingNode`'s arranged in a tree network.

    """
    n_rails = n if n_rails is None else n_rails
    return ForwardCouplingCircuit(_tree(np.arange(n - 1), n_rails, balanced=balanced)).invert_levels()


def vector_unit(v: np.ndarray, n_rails: int = None, balanced: bool = True, phase_style: str = PhaseStyle.TOP):
    """Generate an architecture based on our recursive definitions programmed to implement normalized vector :code:`v`.

    Args:
        v: The vector to be configured, if a matrix is provided
        n_rails: Embed the first :code:`n` rails in an :code:`n_rails`-rail system (default :code:`n_rails == n`).
        balanced: If balanced, does balanced tree (:code:`m = n // 2`) otherwise linear chain (:code:`m = n - 1`).
        phase_style: Phase style for the nodes (see the :code:`PhaseStyle` enum).

    Returns:
        Coupling network, thetas and phis that initialize the coupling network to implement normalized v.
    """
    network = tree(v.shape[0], n_rails=n_rails, balanced=balanced)
    thetas = np.zeros(v.shape[0] - 1)
    phis = np.zeros(v.shape[0] - 1)
    w = v.copy()
    w = w[:, np.newaxis] if w.ndim == 1 else w
    for nc in network.levels:
        # grab the elements for the top and bottom arms of the mzi.
        top = w[(nc.top,)]
        bottom = w[(nc.bottom,)]

        # Vectorized (efficient!) nullification
        if phase_style == PhaseStyle.SYMMETRIC:
            raise NotImplementedError('Require phase_style not be of the SYMMETRIC variety.')
        elif phase_style == PhaseStyle.BOTTOM:
            theta = np.arctan2(np.abs(bottom[:, -1]), np.abs(top[:, -1])) * 2
            phi = np.angle(top[:, -1]) - np.angle(bottom[:, -1])
        else:
            theta = -np.arctan2(np.abs(bottom[:, -1]), np.abs(top[:, -1])) * 2
            phi = np.angle(bottom[:, -1]) - np.angle(top[:, -1])

        # Vectorized (efficient!) parallel mzi elements used to compute the
        t11, t12, t21, t22 = nc.parallel_mzi_fn(phase_style=phase_style)(theta, phi)
        t11, t12, t21, t22 = t11[:, np.newaxis], t12[:, np.newaxis], t21[:, np.newaxis], t22[:, np.newaxis]

        # The final vector after the vectorized multiply
        w[(nc.top + nc.bottom,)] = np.vstack([t11 * top + t21 * bottom,
                                              t12 * top + t22 * bottom])

        # the resulting thetas and phis, indexed according to the coupling network specifications
        thetas[(nc.node_idxs,)] = theta
        phis[(nc.node_idxs,)] = phi

    gammas = -np.angle(np.eye(v.size)[v.size - 1] * w[-1, -1])

    return network, thetas, phis, gammas, w.squeeze()


def unitary_unit(u: np.ndarray, balanced: bool = True, phase_style: str = PhaseStyle.TOP):
    """Generate an architecture based on our recursive definitions programmed to implement unitary :code:`u`.

    Args:
        u: The (:math:`k \\times n`) mutually orthogonal basis vectors (unitary if :math:`k=n`) to be configured.
        balanced: If balanced, does balanced tree (:code:`m = n // 2`) otherwise linear chain (:code:`m = n - 1`).
        phase_style: Phase style for the nodes (see the :code:`PhaseStyle` enum).

    Returns:
        Node list, thetas and phis.

    """
    subunits = []
    thetas = np.array([])
    phis = np.array([])
    gammas = np.array([])
    n_rails = u.shape[0]
    num_levels = 0
    num_nodes = 0

    w = u.copy()
    for i in reversed(range(n_rails + 1 - u.shape[1], n_rails)):
        # Generate the architecture as well as the theta and phi for each row of u.
        nodes, theta, phi, gamma, w = vector_unit(w[:i + 1, :i + 1], n_rails, balanced, phase_style)

        # Update the phases.
        thetas = np.hstack((thetas, theta))
        phis = np.hstack((phis, phi))
        gammas = np.hstack((gamma[-1], gammas))

        # We need to index the thetas and phis correctly based on the number of programmed nodes in previous subunits
        # For unbalanced architectures (linear chains), we can actually pack them more efficiently into a triangular
        # architecture.
        nodes.offset(num_nodes).offset_level(num_levels if balanced else 2 * (n_rails - 1 - i))

        # Add the nodes list to the subunits
        subunits.append(nodes)

        # The number of levels and nodes in the architecture are incremented by the subunit size (log_2(i))
        num_levels += subunits[-1].num_levels
        num_nodes += subunits[-1].num_nodes
    gammas = np.hstack((-np.angle(w[0, 0]), gammas))
    return ForwardCouplingCircuit.aggregate(subunits), thetas, phis, gammas


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

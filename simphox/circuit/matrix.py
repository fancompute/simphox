from typing import Tuple

import scipy as sp

try:
    DPHOX_IMPORTED = True
    from dphox.device import Device
    from dphox.pattern import Pattern
except ImportError:
    DPHOX_IMPORTED = False

from .coupling import PhaseStyle
from .rectangular import rectangular
from .vector import vector_unit
from scipy.stats import unitary_group
import numpy as np

from scipy.linalg import svd, qr, block_diag
from .coupling import CouplingNode
from .forward import ForwardMesh


def cascade(u: np.ndarray, balanced: bool = True, phase_style: str = PhaseStyle.TOP,
            error_mean_std: Tuple[float, float] = (0., 0.), loss_mean_std: Tuple[float, float] = (0., 0.)):
    """Generate an architecture based on our recursive definitions programmed to implement unitary :math:`U`,
    or a set of :math:`K` mutually orthogonal basis vectors.

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
    """Triangular mesh that analyzes a unitary matrix :code:`u`.

    Args:
        u: Unitary matrix or integer representing the number of inputs and outputs
        phase_style: Phase style for the nodes of the mesh.
        error_mean_std: Split error mean and standard deviation
        loss_mean_std: Loss error mean and standard deviation (dB)

    Returns:
        A triangular mesh object.

    """
    u = unitary_group.rvs(u) if np.isscalar(u) else u
    return cascade(u, balanced=False, phase_style=phase_style,
                   error_mean_std=error_mean_std, loss_mean_std=loss_mean_std)


def tree_cascade(u: np.ndarray, phase_style: str = PhaseStyle.TOP, error_mean_std: Tuple[float, float] = (0., 0.),
                 loss_mean_std: Tuple[float, float] = (0., 0.)):
    """Balanced cascade mesh that analyzes a unitary matrix :code:`u`.

    Args:
        u: Unitary matrix
        phase_style: Phase style for the nodes of the mesh.
        error_mean_std: Split error mean and standard deviation
        loss_mean_std: Loss error mean and standard deviation (dB)

    Returns:
        A tree cascade mesh object.

    """
    u = unitary_group.rvs(u) if np.isscalar(u) else u
    return cascade(u, balanced=True, phase_style=phase_style,
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

def cs(mat: np.ndarray):
    """Cosine-sine decomposition of arbitrary matrix :math:`U`(:code:`u`)

    Args:
        mat: The unitary matrix

    Even-partition cosine decomposition:
             [ q00 | q01 ]   [ l0 | 0  ]  [  s |  c ]  [ r0 | 0  ]
         u = [-----------] = [---------]  [---------]  [---------]   .
             [ q10 | q11 ]   [  0 | l1 ]  [  c | -s ]  [  0 | r1 ]

    c = diag(cos(theta))
    s = diag(sin(theta))
    where theta is in the range [0, pi / 2]

    Returns:
        The tuple of the four matrices :code:`l0`, :code:`l1`, :code:`r0`, :code:`r1`, and
        cosine-sine phases :code:`theta` in order from top to bottom.

    """
    n = mat.shape[0]
    m = n // 2
    q00 = mat[:m, :m]
    q10 = mat[m:, :m]
    q01 = mat[:m, m:]
    q11 = mat[m:, m:]
    l0, d00, r0 = svd(q00)
    r1hp, d01 = qr(q01.conj().T @ l0)
    theta = np.arcsin(d00)
    d01 = np.append(np.diag(d01), 1) if n % 2 else np.diag(d01)
    r1 = (r1hp * np.sign(d01)).conj().T
    l1p, d10 = qr(q10 @ r0.conj().T)
    d10 = np.append(np.diag(d10), 1) if n % 2 else np.diag(d10)
    l1 = l1p * np.sign(d10)
    phasor = (l1.conj().T @ q11 @ r1.conj().T)[-1, -1] if n % 2 else None
    if n % 2:
        r1[-1] *= phasor
    return l0, l1, r0, r1, theta


def csinv(l0: np.ndarray, l1: np.ndarray, r0: np.ndarray, r1: np.ndarray, theta: np.ndarray):
    """Runs the inverse of the :code:`cs` function

    Args:
        l0: top left
        l1: bottom left
        r0: top right
        r1: bottom right
        theta: cosine-sine phases

    Returns:
        The final unitary matrix :code:`u`.

    """
    l = block_diag(l0, l1)
    r = block_diag(r0, r1)
    c = np.cos(theta)
    s = np.sin(theta)
    d = np.block([[np.diag(s), np.diag(c)],
                  [np.diag(c), -np.diag(s)]])
    if r0.shape[0] != r1.shape[1]:
        d = block_diag(d, 1).astype(np.complex128)
    return l @ d @ r


def _bowtie(u: np.ndarray, n_rails: int, thetas: np.ndarray, phis: np.ndarray, start: int, layer: int = None):
    """Recursive step for the cosine-sine bowtie architecture

    Args:
        u: Unitary matrix u
        n_rails: Number of total rails in the architecture
        thetas: The internal phase shifts or coupling phase terms :math:`\\theta`.
        phis: The external phase shifts or differential input phase terms :math:`\\phi`.
        start: Start index for interfering modes.
        layer: Layer of the bowtie recursion

    Returns:
        The list of :code:`CouplingNode`.

    """
    nodes = []
    n = u.shape[0]
    m = n // 2
    if n == 1:
        phis[layer][start] += np.angle(u[0][0])
        return nodes
    l0, l1, r0, r1, theta = cs(u)
    thetas[layer][start:start + m * 2][::2] = theta
    nodes.extend([CouplingNode(n=n_rails, top=start + shift, bottom=start + shift + m, column=layer)
                  for shift in range(m)])
    nodes.extend(_bowtie(l0, n_rails, thetas, phis, start, layer - m))
    nodes.extend(_bowtie(r0, n_rails, thetas, phis, start, layer + m))
    nodes.extend(_bowtie(l1, n_rails, thetas, phis, start + m, layer - m))
    nodes.extend(_bowtie(r1, n_rails, thetas, phis, start + m, layer + m))
    return nodes


def bowtie(u: np.ndarray):
    """Cosine-sine bowtie architecture.

    Args:
        u: The unitary matrix :code:`u` to parametrize the system.

    Returns:
        The bowtie fractal architecture.

    """
    n = u.shape[0]
    thetas = np.zeros((2 * n - 3, n))
    phis = np.zeros((2 * n - 1, n))
    circuit = ForwardMesh(_bowtie(u, n, thetas, phis, 0, n - 2))
    phis = phis[1:]
    theta = np.zeros(int(n * (n - 1) / 2))
    phi = np.zeros(int(n * (n - 1) / 2))
    columns = circuit.columns
    for col_idx, col in enumerate(columns):
        theta[(col.node_idxs,)] = thetas[col_idx][np.nonzero(thetas[col_idx])]
        phi[(col.node_idxs,)] = phis[col_idx][(col.top,)] - phis[col_idx][(col.bottom,)]
        phis[col_idx][(col.top,)] = phis[col_idx][(col.bottom,)]
        if col_idx < len(columns):
            phis[col_idx + 1] += phis[col_idx]
            phis[col_idx + 1] = np.mod(phis[col_idx + 1], 2 * np.pi)
    circuit.params = theta * 2, phi, phis[-1]
    return circuit


def psvd(a: np.ndarray):
    """Photonic SVD architecture

    Args:
        a: The matrix for which to perform the svd

    Returns:
        A tuple of singular values and the two corresponding SVD architectures :math:`U` and :math:`V^\\dagger`.

    """
    l, d, r = svd(a)
    return rectangular(l), d, rectangular(r)

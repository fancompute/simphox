import numpy as np

from scipy.linalg import svd, qr, block_diag, dft
from .coupling import CouplingNode
from .forward import ForwardCouplingCircuit


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
    circuit = ForwardCouplingCircuit(_bowtie(u, n, thetas, phis, 0, n - 2))
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

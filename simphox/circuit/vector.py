import numpy as np

from .coupling import CouplingNode, PhaseStyle
from .forward import ForwardMesh
from ..typing import Callable, List, Optional, Tuple
from ..utils import random_vector


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
    if balanced == False:
        return [CouplingNode(i, top=i, bottom=i + 1, n=n_rails,
                             alpha=1, beta=n - i - 1, column=n - 2 - i).set_descendants(indices[:i],
                                                                                        np.array([], dtype=np.int32))
                for i in reversed(range(n - 1))]
    m = n // 2 if balanced else n - 1
    if n == 1:
        return nodes
    top = indices[:m - 1]
    bottom = indices[m:]
    nodes.append(CouplingNode(indices[m - 1], top=start + m - 1, bottom=start + n - 1, n=n_rails,
                              alpha=bottom.size + 1, beta=top.size + 1, column=column).set_descendants(top, bottom))
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
                               alpha=m + 1, beta=m + 1, column=column) for i in reversed(range(m))])
    nodes.extend(_butterfly(m, n_rails, start, column + 1))
    nodes.extend(_butterfly(n - m, n_rails, start + m, column + 1))
    return nodes


def tree(n: int, n_rails: Optional[int] = None, balanced: bool = True,
         phase_style: str = PhaseStyle.TOP) -> ForwardMesh:
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
    return ForwardMesh(_butterfly(n, n_rails)).invert_columns()


def _program_vector_unit(v: np.ndarray, network: ForwardMesh):
    """Code for programming a vector unit that already exists.

    Note:
        This is a private method since we cannot assume that the input network is the appropriate
        vector unit (in size/structure) without defining a separate dataclass for it.

    Args:
        v: The vector / matrix (use final row) to program into the network
        network: The network to program

    Returns:
        The programmed vector unit

    """
    v = v + 0j  # cast to complex
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
        t11, t12, t21, t22 = nc.parallel_mzi_fn(mzi_terms=network.mzi_terms)(theta, phi)
        t11, t12, t21, t22 = t11[:, np.newaxis], t12[:, np.newaxis], t21[:, np.newaxis], t22[:, np.newaxis]

        # these are the top port powers before nulling
        network.pnsn[(nc.node_idxs,)] = np.abs(top[..., -1]
                                               if network.phase_style == PhaseStyle.TOP else bottom[..., -1]) ** 2

        # The final vector after the vectorized multiply
        w[(nc.top + nc.bottom,)] = np.vstack([t11 * top + t21 * bottom,
                                              t12 * top + t22 * bottom])

        # these are the relative powers after nulling
        network.pn[(nc.node_idxs,)] = np.abs(w[(nc.bottom,)][..., -1]) ** 2

        # The resulting thetas and phis, indexed according to the coupling network specifications
        thetas[(nc.node_idxs,)] = theta
        phis[(nc.node_idxs,)] = np.mod(phi, 2 * np.pi)

    final_basis_vec = np.zeros(v.shape[0])
    final_basis_vec[-1] = 1
    gammas = -np.angle(final_basis_vec * w[-1, -1])

    network.params = thetas, phis, gammas

    return network, w.squeeze()


def vector_unit(v: np.ndarray, n_rails: int = None, balanced: bool = True, phase_style: str = PhaseStyle.TOP,
                error_mean_std: Tuple[float, float] = (0., 0.), loss_mean_std: Tuple[float, float] = (0., 0.)):
    """Generate an architecture based on our recursive definitions programmed to implement normalized vector :code:`v`.

    Args:
        v: The number of inputs or vector to be configured. If a matrix is provided, use the final row vector of matrix.
        n_rails: Embed the first :code:`n` rails in an :code:`n_rails`-rail system (default :code:`n_rails == n`).
        balanced: If balanced, does balanced tree (:code:`m = n // 2`) otherwise linear chain (:code:`m = n - 1`).
        phase_style: Phase style for the nodes (see the :code:`PhaseStyle` enum).
        error_mean_std: Mean and standard deviation for errors (in radians).
        loss_mean_std: Mean and standard deviation for losses (in dB).

    Returns:
        A tuple of the programmed coupling network, the matrix after being fed through the network.
    """
    network = tree(v.shape[0] if not np.isscalar(v) else v, n_rails=n_rails, balanced=balanced, phase_style=phase_style)
    error_mean, error_std = error_mean_std
    loss_mean, loss_std = loss_mean_std
    network = network.add_error_mean(error_mean, loss_mean).add_error_variance(error_std, loss_std)
    if np.isscalar(v):
        return network, None
    return _program_vector_unit(v, network)


def balanced_tree(v: np.ndarray, phase_style: str = PhaseStyle.TOP,
                  error_mean_std: Tuple[float, float] = (0., 0.),
                  loss_mean_std: Tuple[float, float] = (0., 0.)):
    """Balanced tree mesh that analyzes a vector :code:`v`.

    Args:
        v: Vector unit
        phase_style: Phase style for the nodes of the mesh.
        error_mean_std: Split error mean and standard deviation
        loss_mean_std: Loss error mean and standard deviation (dB)

    Returns:
        A tree mesh object analyzing a vector.

    """
    return vector_unit(v.conj().T if not np.isscalar(v) else v,
                       phase_style=phase_style, error_mean_std=error_mean_std, loss_mean_std=loss_mean_std)[0]


def unbalanced_tree(v: np.ndarray, phase_style: str = PhaseStyle.TOP, error_mean_std: Tuple[float, float] = (0., 0.),
                    loss_mean_std: Tuple[float, float] = (0., 0.)):
    """Linear chain that analyzes a vector :code:`v`.

    Args:
        v: Vector unit
        phase_style: Phase style for the nodes of the mesh.
        error_mean_std: Split error mean and standard deviation
        loss_mean_std: Loss error mean and standard deviation (dB)

    Returns:
        A linear chain mesh object analyzing a vector.

    """
    return vector_unit(v.conj().T if not np.isscalar(v) else v,
                       phase_style=phase_style, error_mean_std=error_mean_std, loss_mean_std=loss_mean_std,
                       balanced=False)[0]


def hessian_vector_unit(v: np.ndarray, balanced: bool = True):
    """Compute the Hessian for a vector unit if size code:`n` using finite differences assuming TOP phase style.

    We use the self-configuring dynamic programming approach to generate the necessary power quantities
    at each node required to generate the Hessian directly from the vector unit.

    Args:
        v: Vector to be programmed on the vector unit.
        balanced: Whether to use the balanced or the unbalanced tree vector unit.

    Returns:
        The Hessian matrix with block matrices
         :math:`\\mathcal{H}_{\\theta \\to \\theta}`, :math:`\\mathcal{H}_{\\phi \\to \\phi}`,
        :math:`\\mathcal{H}_{\\theta \\to \\phi}`, which give the Hessian magnitudes for the matrix.

    """
    v = v / np.linalg.norm(v)
    mesh = balanced_tree(v) if balanced else unbalanced_tree(v)
    pn, pnsn = mesh.pn, mesh.pnsn

    theta_theta = np.diag(pn)
    phi_phi = 2 * np.diag(pnsn)
    theta_phi = np.zeros_like(phi_phi)
    phi_theta = np.diag(pnsn)
    for i in range(v.size - 1):
        nid = mesh.nodes[i].node_id
        theta_theta[nid][(mesh.nodes[i].top_descendants,)] = mesh.pn[(mesh.nodes[i].top_descendants,)] / 2
        theta_theta[nid][(mesh.nodes[i].bot_descendants,)] = mesh.pn[(mesh.nodes[i].bot_descendants,)] / 2
        theta_theta[..., nid] = theta_theta[nid]
        phi_phi[nid][(mesh.nodes[i].top_descendants,)] = 2 * mesh.pnsn[(mesh.nodes[i].top_descendants,)]
        phi_phi[..., nid] = phi_phi[nid]
        theta_phi[..., nid][(mesh.nodes[i].top_descendants,)] = mesh.pn[(mesh.nodes[i].top_descendants,)]
        phi_theta[nid][(mesh.nodes[i].top_descendants,)] = mesh.pnsn[(mesh.nodes[i].top_descendants,)]
        phi_theta[nid][(mesh.nodes[i].bot_descendants,)] = mesh.pnsn[(mesh.nodes[i].bot_descendants,)]
    h = np.block([[theta_theta, (theta_phi + phi_theta).T], [(theta_phi + phi_theta), phi_phi]])
    return h


def hessian_fd(v: np.ndarray, error=0.0001, balanced=False):
    """Compute the Hessian for a vector unit if size code:`n` using finite differences.
    This is mostly useful for testing, but it takes way too long in practice.

    The finite difference evaluation is given by the central differencing scheme:
    .. math::
        \\mathcal{H}_{ij} = \\frac{\\partial^2 \\epsilon^2}{\\partial \\delta_{i} \\partial \\delta_{j}} \\approx
        \\frac{\\epsilon^2(\\delta_i \\boldsymbol{e}_i + \\delta_j \\boldsymbol{e}_j) - \\epsilon^2(
        \\delta_i \\boldsymbol{e}_i - \\delta_j \\boldsymbol{e}_j)}{2\\delta_{i} \\delta_{j}}
    where we allow :math:`\\delta_{i} = \\delta_{j}` be the phase error applied to phases :math:`i, j` in the network.

    Args:
        v: Vector to be programmed on the vector unit.
        error: The tiny error to use to compute the second-order Hessian matrix.
        balanced: Whether to use the balanced or the unbalanced tree vector unit.

    Returns:
        The Hessian matrix with block matrices
         :math:`\\mathcal{H}_{\\theta \\to \\theta}`, :math:`\\mathcal{H}_{\\phi \\to \\phi}`,
        :math:`\\mathcal{H}_{\\theta \\to \\phi}`, which give the Hessian magnitudes for the matrix.

    """
    n = v.size
    v = v / np.linalg.norm(v)
    h = np.zeros((2 * n - 2, 2 * n - 2), dtype=np.complex128)
    e = np.eye(n - 1) * error
    mesh = balanced_tree(v) if balanced else unbalanced_tree(v)

    def err(params):
        return 2 - 2 * mesh.matrix_fn(inputs=v.conj())(params)[-1]

    for i in range(n - 1):
        for j in range(n - 1):
            theta_pp = (mesh.thetas + e[i] + e[j], mesh.phis, mesh.gammas)
            theta_nn = (mesh.thetas - e[i] - e[j], mesh.phis, mesh.gammas)
            theta_pn = (mesh.thetas + e[i] - e[j], mesh.phis, mesh.gammas)
            theta_np = (mesh.thetas - e[i] + e[j], mesh.phis, mesh.gammas)
            phi_pp = (mesh.thetas, mesh.phis + e[i] + e[j], mesh.gammas)
            phi_nn = (mesh.thetas, mesh.phis - e[i] - e[j], mesh.gammas)
            phi_pn = (mesh.thetas, mesh.phis + e[i] - e[j], mesh.gammas)
            phi_np = (mesh.thetas, mesh.phis - e[i] + e[j], mesh.gammas)
            theta_phi_pp = (mesh.thetas + e[i], mesh.phis + e[j], mesh.gammas)
            theta_phi_pn = (mesh.thetas + e[i], mesh.phis - e[j], mesh.gammas)
            theta_phi_np = (mesh.thetas - e[i], mesh.phis + e[j], mesh.gammas)
            theta_phi_nn = (mesh.thetas - e[i], mesh.phis - e[j], mesh.gammas)
            h[i, j] = np.real(err(theta_pp) + err(theta_nn) - err(theta_pn) - err(theta_np)) / (4 * error ** 2)
            h[i + n - 1, j + n - 1] = np.real(err(phi_pp) + err(phi_nn) - err(phi_pn) - err(phi_np)) / (4 * error ** 2)
            h[i + n - 1, j] = np.real(err(theta_phi_pp) + err(theta_phi_nn) - err(theta_phi_pn) - err(theta_phi_np)) / (
                        4 * error ** 2)
            h[j, i + n - 1] = h[i + n - 1, j]
    return h


def hessian_distribution(n: int, balanced: bool = False, n_samples=1, pbar: Callable = None):
    """Compute the Hessian distribution for a vector unit of size code:`n`.

    Args:
        n: Number of inputs
        balanced: Whether to use the balanced or the unbalanced tree vector unit.
        n_samples: Number of samples (if 1, return the mesh along with the errors).
        pbar: Progress bar.

    Returns:
        A tuple of :math:`\\mathcal{H}_{\\theta \\to \\theta}`, :math:`\\mathcal{H}_{\\phi \\to \\phi}`,
        :math:`\\mathcal{H}_{\\theta \\to \\phi}`, which give the Hessian magnitudes for the matrix.

    """
    hessians = []
    iterator = range(n_samples) if pbar is None else pbar(range(n_samples))
    for _ in iterator:
        hessians.append(hessian_vector_unit(random_vector(n, normed=True), balanced=balanced)[0])
    return np.array(hessians)

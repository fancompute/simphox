from collections import defaultdict
from enum import Enum

import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

try:
    DPHOX_IMPORTED = True
    from dphox.device import Device
    from dphox.pattern import Pattern
except ImportError:
    DPHOX_IMPORTED = False
from pydantic.dataclasses import dataclass
from scipy.stats import beta

from .fdfd import FDFD
from .typing import Callable, Iterable, List, Optional, Size, Union
from .utils import fix_dataclass_init_docs


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
        COMMON: Common phase shift

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
    num_top: int = 1
    num_bottom: int = 1
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


@fix_dataclass_init_docs
@dataclass
class ForwardCouplingCircuit:
    """A coupling circuit class is a feedforward coupling nodes that interact arbitrary waveguide pairs.

    The :code:`CouplingCircuit` has the convenient property that it is acyclic, so this means that we can simply
    use a list of nodes defined in time-order traversal to define the entire circuit. This greatly simplifies
    the notation and code necessary to define such a circuit. General graphs / circuits do not have this property.

    Attributes:
        nodes: A list of :code:`CouplingNode`'s in the order that light visits the nodes.

    """
    nodes: List[CouplingNode]

    def __post_init_post_parse__(self):
        self.top = tuple([int(node.top) for node in self.nodes])
        self.bottom = tuple([int(node.bottom) for node in self.nodes])
        self.errors_left = np.array([node.error for node in self.nodes])
        self.errors_right = np.array([node.error_right for node in self.nodes])
        self.mzi_terms = np.array(
            [np.cos(np.pi / 4 + self.errors_right) * np.cos(np.pi / 4 + self.errors_left),
             np.cos(np.pi / 4 + self.errors_right) * np.sin(np.pi / 4 + self.errors_left),
             np.sin(np.pi / 4 + self.errors_right) * np.cos(np.pi / 4 + self.errors_left),
             np.sin(np.pi / 4 + self.errors_right) * np.sin(np.pi / 4 + self.errors_left)]
        )
        self.losses = np.array([node.loss for node in self.nodes])
        self.node_idxs = tuple([node.node_id for node in self.nodes])
        self.dataframe = pd.DataFrame([node.__dict__ for node in self.nodes])
        self.n = self.nodes[0].n if len(self.nodes) > 0 else 1
        self.num_top = np.array([node.num_top for node in self.nodes])
        self.num_bottom = np.array([node.num_bottom for node in self.nodes])
        self.num_nodes = len(self.nodes)
        self.level_by_node = np.array([node.level for node in self.nodes])
        self.num_levels = np.max(self.level_by_node) + 1 if len(self.nodes) > 0 else 0

    def offset(self, offset: int):
        """Offset the :code:`node_id`'s, useful for architectures that contain many tree subunits.

        Generally, assign :math:`j := j + \\Delta j` for each node index :math:`j`,
        where the offset is :math:`\\Delta j`.

        Args:
            offset: Offset index.

        """
        for node in self.nodes:
            node.node_id += offset
        return self

    def offset_level(self, offset: int):
        """Offset the :code:`level`'s, useful for architectures that contain many tree subunits.

        Generally, assign :math:`\\ell := \\ell + \\Delta \\ell` for each layer :math:`\\ell`,
        where the offset is :math:`\\Delta \\ell`.

        Args:
            offset: Offset index.

        """
        for node in self.nodes:
            node.level += offset
        return self

    def invert_levels(self):
        """Invert the level labels for all the nodes.

        Assume :math:`L` total levels. If the leaf level is :math:`L`, then set it to 1 or vice versa.
        The opposite for the root level. Generally, assign :math:`L - \\ell` for each layer :math:`\\ell`.

        Returns:
            This circuit with inverted levels.

        """
        for node in self.nodes:
            node.level = self.num_levels - 1 - node.level
        self.level_by_node = np.array([node.level for node in self.nodes])
        return self

    def rand_s(self):
        """Randomly initialized split ratios :math:`s`.

        Returns:
            A randomly initialized :code:`s`.

        """
        return beta.pdf(np.random.rand(self.num_nodes), self.num_top, self.num_bottom)

    def rand_theta(self):
        """Randomly initialized coupling phase :math:`\\theta`.

        Returns:
            A randomly initialized :code:`theta`.

        """
        return 2 * np.arccos(np.sqrt(self.rand_s()))

    @classmethod
    def aggregate(cls, nodes_or_node_lists: List[Union[CouplingNode, "ForwardCouplingCircuit"]]):
        """Aggregate nodes and/or node lists into a single :code:`NodeList`.

        Args:
            nodes_or_node_lists: Nodes and/or node lists to aggregate into a single :code:`NodeList`.

        Returns:
            A :code:`NodeList` that contains all the nodes.

        """
        all_nodes = []
        for obj in nodes_or_node_lists:
            if isinstance(obj, CouplingNode):
                all_nodes.append(obj)
            else:
                all_nodes.extend(obj.nodes)
        return cls(all_nodes)

    @property
    def levels(self):
        """Return a list of :code:`CouplingCircuit` by level.

        An interesting property of the resulting :code:`CouplingCircuit`'s is that no two coupling nodes
        in such circuits are connected. This allows use to apply the node operators simultaneously, which
        allows for calibration and simulation to increase in efficiency.

        Returns:
            A list of :code:`NodeList` grouped by level (needed for efficient implementations, autodiff, calibrations).

        """
        nodes_by_level = defaultdict(list)
        for node in self.nodes:
            nodes_by_level[node.level].append(node)
        return [ForwardCouplingCircuit(nodes_by_level[level]) for level in range(self.num_levels)]

    def parallel_mzi_fn(self, phase_style: str = PhaseStyle.TOP, use_jax: bool = False):
        """This is a helper function for finding the matrix elements for MZIs in parallel,
        leading to significant speedup.

        Args:
            phase_style: The phase style to use for the parallel MZIs.
            use_jax: Whether to use jax.

        Returns:
            A function that accepts the thetas and phis as inputs and outputs the vectorized
            MZI matrix elements

        """

        xp = jnp if use_jax else np

        # stands for cos-cos, cos-sin, sin-cos, sin-sin terms.
        cc, cs, sc, ss = self.mzi_terms

        # insertion (assume a fixed loss that is global for each node)
        insertion = 1 - self.losses

        if phase_style == PhaseStyle.TOP:
            def parallel_mzi(theta, phi):
                t11 = (-cc + ss * xp.exp(1j * theta)) * xp.exp(1j * phi)
                t12 = 1j * (cs + sc * xp.exp(1j * theta)) * xp.exp(1j * phi)
                t21 = 1j * (sc + cs * xp.exp(1j * theta))
                t22 = (ss - cc * xp.exp(1j * theta))
                return insertion * xp.array([t11, t12, t21, t22])
        elif phase_style == PhaseStyle.BOTTOM:
            def parallel_mzi(theta, phi):
                t11 = (-cc * xp.exp(1j * theta) + ss)
                t12 = 1j * (cs * xp.exp(1j * theta) + sc)
                t21 = 1j * (sc * xp.exp(1j * theta) + cs) * xp.exp(1j * phi)
                t22 = (ss * xp.exp(1j * theta) - cc) * xp.exp(1j * phi)
                return insertion * xp.array([t11, t12, t21, t22])
        elif phase_style == PhaseStyle.SYMMETRIC:
            def parallel_mzi(theta, phi):
                t11 = (-cc + ss * xp.exp(1j * theta)) * xp.exp(1j * (phi - theta / 2))
                t12 = 1j * (cs + sc * xp.exp(1j * theta)) * xp.exp(1j * (phi - theta / 2))
                t21 = 1j * (sc + cs * xp.exp(1j * theta)) * xp.exp(1j * (phi - theta / 2))
                t22 = (ss - cc * xp.exp(1j * theta)) * xp.exp(1j * (phi - theta / 2))
                return insertion * xp.array([t11, t12, t21, t22])
        elif phase_style == PhaseStyle.DIFFERENTIAL:
            def parallel_mzi(theta, phi):
                t11 = (-cc + ss * xp.exp(1j * theta)) * xp.exp(1j * (-phi - theta) / 2)
                t12 = 1j * (cs + sc * xp.exp(1j * theta)) * xp.exp(1j * (-phi - theta) / 2)
                t21 = 1j * (sc + cs * xp.exp(1j * theta)) * xp.exp(1j * (phi - theta) / 2)
                t22 = (ss - cc * xp.exp(1j * theta)) * xp.exp(1j * (phi - theta) / 2)
                return insertion * xp.array([t11, t12, t21, t22])
        else:
            raise ValueError(f"Phase style {phase_style} is not valid.")
        return parallel_mzi

    def matrix_fn(self, phase_style: str = PhaseStyle.TOP, use_jax: bool = False):
        """Return a function that returns the matrix representation of this circuit.

        The coupling circuit is a photonic network aligned with :math:`N` waveguide rails.
        We use rail index to refer to mode index in the waveguide basis. This enables us to define
        a matrix assigning the :math:`N` inputs to :math:`N` outputs of the network. We assume :math:`N`
        is the same across all nodes in the circuit so we access it from any one of the individual nodes.

        Here, we define a matrix function that performs the equivalent matrix multiplications
        without needing to explicitly define :math:`N \\times N` matrix for each subunit or circuit level.
        We also go level-by-level to significantly improve the efficiency of the matrix multiplications.

        Args:
            phase_style: The phase style for each node in the network.
            use_jax: Use JAX to accelerate the matrix function.

        Returns:
            Return the MZI level function that transforms the inputs (no explicit matrix defined here).
            This function accepts network :code:`inputs`, and the node parameters :code:`thetas, phis`.
            Using the identity matrix as input gives the unitary representation of the rail network.

        """

        node_levels = self.levels
        level_fns = [level.parallel_mzi_fn(phase_style, use_jax) for level in node_levels]
        xp = jnp if use_jax else np
        identity = xp.eye(node_levels[0].n, dtype=xp.complex128)

        # Define a function that represents an mzi level given inputs and all available thetas and phis
        def matrix(thetas: xp.ndarray, phis: xp.ndarray, gammas: xp.ndarray):
            outputs = identity
            for nc, level_fn in zip(node_levels, level_fns):
                # collect the inputs to be interfered
                top = outputs[(nc.top,)]
                bottom = outputs[(nc.bottom,)]

                # collect the thetas and phis from all MZIs in a given column
                theta = thetas[(nc.node_idxs,)]
                phi = phis[(nc.node_idxs,)]

                # collect the matrix elements to be applied in parallel to the incoming modes.
                t11, t12, t21, t22 = level_fn(theta, phi)
                t11, t12, t21, t22 = t11[:, xp.newaxis], t12[:, xp.newaxis], t21[:, xp.newaxis], t22[:, xp.newaxis]

                # use jax or use numpy affects the syntax for assigning the outputs to the new outputs after the layer
                if use_jax:
                    outputs = outputs.at[(nc.top + nc.bottom,)].set(
                        xp.vstack([t11 * top + t21 * bottom,
                                   t12 * top + t22 * bottom])
                    )
                else:
                    outputs[(nc.top + nc.bottom,)] = xp.vstack([t11 * top + t21 * bottom,
                                                                t12 * top + t22 * bottom])

            # multiply the gamma phases present at the end of the network (sets the reference phase front).
            return (xp.exp(1j * gammas) * outputs.T).T

        return matrix


def random_complex(n: int, normed: bool = False) -> np.ndarray:
    """Generate a random complex normal vector.

    Args:
        n: Number of inputs.
        normed: Whether to norm the random complex vector so that the norm of the vector is 1.

    Returns:
        The random complex normal vector.

    """
    z = np.array(0.5 * np.random.randn(n) + 0.5 * np.random.randn(n) * 1j)
    return z / np.linalg.norm(z) if normed else z


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
                              num_top=top.size + 1, num_bottom=bottom.size + 1, level=level))
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


def configure_vector(v: np.ndarray, n_rails: int = None, balanced: bool = True, phase_style: str = PhaseStyle.TOP):
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

        # vectorized nullification
        if phase_style == PhaseStyle.SYMMETRIC:
            raise NotImplementedError('Require phase_style not be of the SYMMETRIC variety.')
        elif phase_style == PhaseStyle.BOTTOM:
            theta = np.arctan2(np.abs(bottom[:, -1]), np.abs(top[:, -1])) * 2
            phi = np.angle(top[:, -1]) - np.angle(bottom[:, -1])
        else:
            theta = -np.arctan2(np.abs(bottom[:, -1]), np.abs(top[:, -1])) * 2
            phi = np.angle(bottom[:, -1]) - np.angle(top[:, -1])

        # vectorized parallel mzi elements used to compute the
        t11, t12, t21, t22 = nc.parallel_mzi_fn(phase_style=phase_style)(theta, phi)
        t11, t12, t21, t22 = t11[:, np.newaxis], t12[:, np.newaxis], t21[:, np.newaxis], t22[:, np.newaxis]

        # the final vector after the vectorized multiply
        w[(nc.top + nc.bottom,)] = np.vstack([t11 * top + t21 * bottom,
                                              t12 * top + t22 * bottom])

        # the resulting thetas and phis, indexed according to the coupling network specifications
        thetas[(nc.node_idxs,)] = theta
        phis[(nc.node_idxs,)] = phi

    gammas = -np.angle(np.eye(v.size)[v.size - 1] * w[-1, -1])

    return network, thetas, phis, gammas, w.squeeze()


def configure_unitary(u: np.ndarray, balanced: bool = True, phase_style: str = PhaseStyle.TOP):
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
        nodes, theta, phi, gamma, w = configure_vector(w[:i + 1, :i + 1], n_rails, balanced, phase_style)

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


def checkerboard_to_param(checkerboard: np.ndarray, units: int):
    param = np.zeros((units, units // 2))
    if units % 2:
        param[::2, :] = checkerboard.T[::2, :-1:2]
    else:
        param[::2, :] = checkerboard.T[::2, ::2]
    param[1::2, :] = checkerboard.T[1::2, 1::2]
    return param


def grid_common_mode_flow(external_phases: np.ndarray, gamma: np.ndarray):
    """In a grid mesh (e.g., triangular, rectangular meshes), phases may need to be re-arranged.
     This is achieved using a procedure called "common mode flow" where common modes are shifted
     throughout the mesh until phases are correctly set.

    Args:
        external_phases: external phases in the grid mesh
        gamma: input phase shifts

    Returns:
        new external phases shifts and new gamma resulting

    """
    units, num_layers = external_phases.shape
    phase_shifts = np.hstack((external_phases, gamma[:, np.newaxis])).T
    new_phase_shifts = np.zeros_like(external_phases.T)

    for i in range(num_layers):
        current_layer = i
        start_idx = current_layer % 2
        end_idx = units - (current_layer + units) % 2

        # calculate upper and lower phases
        upper_phase = phase_shifts[current_layer][start_idx:end_idx][::2]
        lower_phase = phase_shifts[current_layer][start_idx:end_idx][1::2]
        upper_phase = np.mod(upper_phase, 2 * np.pi)
        lower_phase = np.mod(lower_phase, 2 * np.pi)

        # upper - lower
        new_phase_shifts[i][start_idx:end_idx][::2] = upper_phase - lower_phase

        # lower_phase is now the common mode for all phase shifts in this layer
        phase_shifts[current_layer] -= new_phase_shifts[i]

        # shift the phases to the next layer in parallel
        phase_shifts[current_layer + 1] += np.mod(phase_shifts[current_layer], 2 * np.pi)

        # set the current layer's phase to 0 (changing to new phase shifts)
        phase_shifts[current_layer] = 0
    new_gamma = np.mod(phase_shifts[-1], 2 * np.pi)
    return np.mod(new_phase_shifts.T, 2 * np.pi), new_gamma


def rectangular(u: np.ndarray, pbar: Callable = None):
    """Get a rectangular architecture for the unitary matrix :code:`u` using the Clements decomposition.

    Args:
        u: The unitary matrix
        pbar: The progress bar for the clements decomposition (useful for larger unitaries)

    Returns:

    """
    u_hat = u.copy()
    n = u.shape[0]
    # odd and even layer dimensions
    theta_checkerboard = np.zeros_like(u, dtype=np.float64)
    phi_checkerboard = np.zeros_like(u, dtype=np.float64)
    phi_checkerboard = np.hstack((np.zeros((n, 1)), phi_checkerboard))
    iterator = pbar(range(n - 1)) if pbar else range(n - 1)
    for i in iterator:
        if i % 2:
            for j in range(i + 1):
                pairwise_index = n + j - i - 2
                target_row, target_col = n + j - i - 1, j
                theta = np.arctan2(np.abs(u_hat[target_row - 1, target_col]), np.abs(u_hat[target_row, target_col])) * 2
                phi = np.angle(u_hat[target_row, target_col]) - np.angle(u_hat[target_row - 1, target_col])
                mzi = CouplingNode(n=n, top=pairwise_index, bottom=pairwise_index + 1)
                left_multiplier = mzi.mzi_node_matrix(theta, phi)
                u_hat = left_multiplier @ u_hat
                theta_checkerboard[pairwise_index, j] = theta
                phi_checkerboard[pairwise_index, j] = phi
        else:
            for j in range(i + 1):
                pairwise_index = i - j
                target_row, target_col = n - j - 1, i - j
                theta = np.arctan2(np.abs(u_hat[target_row, target_col + 1]), np.abs(u_hat[target_row, target_col])) * 2
                phi = np.angle(-u_hat[target_row, target_col]) - np.angle(u_hat[target_row, target_col + 1])
                mzi = CouplingNode(n=n, top=pairwise_index, bottom=pairwise_index + 1)
                right_multiplier = mzi.mzi_node_matrix(theta, phi)
                u_hat = u_hat @ right_multiplier.conj().T
                theta_checkerboard[pairwise_index, -j - 1] = theta
                phi_checkerboard[pairwise_index, -j - 1] = phi

    diag_phases = np.angle(np.diag(u_hat))
    theta = checkerboard_to_param(np.fliplr(theta_checkerboard), n)
    phi_checkerboard = np.fliplr(phi_checkerboard)
    if n % 2:
        phi_checkerboard[:, :-1] += np.fliplr(np.diag(diag_phases))
    else:
        phi_checkerboard[:, 1:] += np.fliplr(np.diag(diag_phases))

    phi, gamma = grid_common_mode_flow(external_phases=phi_checkerboard[:, :-1],
                                       gamma=phi_checkerboard[:, -1])
    phi = checkerboard_to_param(phi, n)

    nodes = []
    thetas = np.array([])
    phis = np.array([])
    node_id = 0
    for i in range(n):
        num_to_interfere = theta.shape[1] - (i % 2) * (1 - n % 2)
        nodes += [CouplingNode(node_id=node_id + j, n=n, top=2 * j + i % 2, bottom=2 * j + 1 + i % 2, level=i)
                  for j in range(num_to_interfere)]
        thetas = np.hstack([thetas, theta[i, :num_to_interfere]])
        phis = np.hstack([phis, phi[i, :num_to_interfere]])
        node_id += num_to_interfere

    return ForwardCouplingCircuit(nodes), thetas, phis, gamma


def dirichlet_matrix(v, embed_dim=None):
    dim = v.size
    phases = np.exp(-1j * np.angle(v))
    y = np.abs(v) ** 2
    yop = np.sqrt(np.outer(y, y))
    ysum = np.cumsum(y)
    yden = 1 / np.sqrt(ysum[:-1] * ysum[1:])
    u = np.zeros_like(yop, dtype=np.complex128)
    u[1:, :] = yden[:, np.newaxis] * yop[1:, :]
    u[np.triu_indices(dim)] = 0
    u[1:, 1:][np.diag_indices(dim - 1)] = -ysum[:-1] * yden
    u[0] = np.sqrt(y / ysum[-1])
    u *= phases
    u = np.roll(u, -1, axis=0) if embed_dim is None else sp.linalg.block_diag(np.roll(u, -1, axis=0),
                                                                              np.eye(embed_dim - dim))
    return u.T


def binary_svd_depth_size(n, k, interport_distance=25, device_length=200, loss_db: float = 0.3):
    num_input_layers = np.ceil(np.log2(n))
    num_unitary_layers = k * np.ceil(np.log2(n))
    attenuate_and_unitary_layer = k + 1
    num_waveguides_height = np.ceil(n / k) * n
    layers = num_input_layers + num_unitary_layers + attenuate_and_unitary_layer
    height = num_waveguides_height * interport_distance
    length = device_length * layers
    return {
        'layers': layers,
        'length (cm)': length / 1e4,
        'height (cm)': height / 1e4,
        'footprint (cm^2)': length * height / 1e8,
        'loss (dB)': -layers * loss_db + 10 * np.log10(1 / np.ceil(n / k)) + 10 * np.log10(1 / n) - 3
    }


def rectangular_depth_size(n, interport_distance=25, device_length=200, loss_db: float = 0.3, svd: bool = True):
    """


    Args:
        n: The number of outputs of the binary tree
        interport_distance: Distance between each port in the network (include the phase shifters)
        device_length: Length of the device
        loss_db: Loss of the device
        svd: Whether to use an SVD architecture (doubles the number of layers in the architecture)

    Returns:

    """
    num_input_layers = np.ceil(np.log2(n))
    num_unitary_layers = n * (1 + svd) + svd
    num_waveguides_height = n
    layers = num_input_layers + num_unitary_layers
    height = num_waveguides_height * interport_distance
    length = device_length * layers
    return {
        'layers': layers,
        'length (cm)': length / 1e4,
        'height (cm)': height / 1e4,
        'footprint (cm^2)': length * height / 1e8,
        'loss (dB)': -layers * loss_db - 10 * np.log10(n) - 3 * svd
    }


def binary_equiv_cascade_size(n, n_equiv, interport_distance=25, device_length=200, loss_db: float = 0.3):
    """

    Args:
        n: The number of outputs of the binary tree
        n_equiv: Find the k (number of inputs) required to match the flops of n_equiv x n_equiv matrix
        interport_distance: Distance between each port in the network (include the phase shifters)
        device_length: Length of the device
        loss_db: Loss of the device

    Returns:

    """
    k = np.ceil(n_equiv ** 2 / n)
    num_input_layers = np.ceil(np.log2(n))
    num_unitary_layers = k * np.ceil(np.log2(n))
    attenuate_and_unitary_layer = k + 1
    num_waveguides_height = n
    layers = num_input_layers + num_unitary_layers + attenuate_and_unitary_layer
    height = num_waveguides_height * interport_distance
    length = device_length * layers
    return {
        'layers': layers,
        'length (cm)': length / 1e4,
        'height (cm)': height / 1e4,
        'footprint (cm^2)': length * height / 1e8,
        'loss (dB)': -layers * loss_db + 10 * np.log10(1 / n)
    }


class Component:
    def __init__(self, structure: Union["Pattern", "Device"],
                 model: Union[xr.DataArray, Callable[[jnp.ndarray], xr.DataArray]], name: str):
        """A component in a circuit will have some structure that can be simulated
        (a pattern or device defined in DPhox), a model, and a name string.

        Args:
            structure: Structure of the device.
            model: Model of the device (in terms of wavelength).
            name: Name of the component (string representing the model name).
        """
        self.structure = structure
        self.model = model
        self.name = name

    @classmethod
    def from_fdfd(cls, pattern: "Pattern", core_eps: float, clad_eps: float, spacing: float,
                  wavelengths: Iterable[float],
                  boundary: Size, pml: float, name: str, in_ports: Optional[List[str]] = None,
                  out_ports: Optional[List[str]] = None, component_t: float = 0, component_zmin: Optional[float] = None,
                  rib_t: float = 0, sub_z: float = 0, height: float = 0, bg_eps: float = 1,
                  profile_size_factor: int = 3,
                  pbar: Optional[Callable] = None):
        """From FDFD, this classmethod produces a component model based on a provided pattern
        and simulation attributes (currently configured for scalar photonics problems).

        Args:
            pattern: component provided by DPhox
            core_eps: core epsilon
            clad_eps: clad epsilon
            spacing: spacing required
            wavelengths: wavelengths
            boundary: boundary size around component
            pml: PML size (see :code:`FDFD` class for details)
            name: component name
            in_ports: input ports
            out_ports: output ports
            height: height for 3d simulation
            sub_z: substrate minimum height
            component_zmin: component height (defaults to substrate_z)
            component_t: component thickness
            rib_t: rib thickness for component (partial etch)
            bg_eps: background epsilon (usually 1 or air)
            profile_size_factor: profile size factor (multiply port size dimensions to get mode dimensions at each port)
            pbar: Progress bar (e.g. TQDM in a notebook which can be a valuable progress indicator).

        Returns:
            Initialize a component which contains a structure (for port specificication and visualization purposes)
            and model describing the component behavior.

        """
        sparams = []

        iterator = wavelengths if pbar is None else pbar(wavelengths)
        for wl in iterator:
            fdfd = FDFD.from_pattern(
                component=pattern,
                core_eps=core_eps,
                clad_eps=clad_eps,
                spacing=spacing,
                height=height,
                boundary=boundary,
                pml=pml,
                component_t=component_t,
                component_zmin=component_zmin,
                wavelength=wl,
                rib_t=rib_t,
                sub_z=sub_z,
                bg_eps=bg_eps,
                name=f'{name}_{wl}um'
            )
            sparams_wl = []
            for port in fdfd.port:
                s, _ = fdfd.get_sim_sparams_fn(port, profile_size_factor=profile_size_factor)(fdfd.eps)
                sparams_wl.append(s)
            sparams.append(sparams_wl)

        model = xr.DataArray(
            data=sparams,
            dims=["wavelengths", "in_ports", "out_ports"],
            coords={
                "wavelengths": wavelengths,
                "in_ports": in_ports,
                "out_ports": out_ports
            }
        )

        return cls(pattern, model=model, name=name)

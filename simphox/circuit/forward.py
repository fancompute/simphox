import copy
from collections import defaultdict
from enum import Enum
from typing import Tuple, Optional

import jax.numpy as jnp
from jax import value_and_grad, jit
from jax.experimental.optimizers import adam
import numpy as np
import pandas as pd

try:
    DPHOX_IMPORTED = True
    from dphox.device import Device
    from dphox.pattern import Pattern
except ImportError:
    DPHOX_IMPORTED = False
from pydantic.dataclasses import dataclass
from scipy.stats import beta

from ..typing import List, Union
from ..utils import fix_dataclass_init_docs, normalized_fidelity_fn
from .coupling import CouplingNode, PhaseStyle


@fix_dataclass_init_docs
@dataclass
class ForwardCouplingCircuit:
    """A forward couping circuit is a feedforward "mesh" of coupling nodes that interact arbitrary waveguide pairs.

    The :code:`CouplingCircuit` has the convenient property that it is acyclic, so this means that we can simply
    use a list of nodes defined in time-order traversal to define the entire circuit. This greatly simplifies
    the notation and code necessary to define such a circuit.

    General graphs / circuits do not have this property. Therefore, this class is particularly suited to applications
    where light (or computation more generally) only goes in one direction.

    Attributes:
        nodes: A list of :code:`CouplingNode`s in the order that light visits the nodes (sorted by column/level).

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
        self.alpha = np.array([node.alpha for node in self.nodes])
        self.beta = np.array([node.beta for node in self.nodes])
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
        return beta.pdf(np.random.rand(self.num_nodes), self.alpha, self.beta)

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
            use_jax: Use JAX to accelerate the matrix function for autodifferentiation purposes.

        Returns:
            Return the MZI level function that transforms the inputs (no explicit matrix defined here).
            This function accepts network :code:`inputs`, and the node parameters :code:`thetas, phis`.
            Using the identity matrix as input gives the unitary representation of the rail network.

        """

        node_levels = self.levels
        level_ordered = self.level_ordered
        node_fn = level_ordered.parallel_mzi_fn(phase_style, use_jax=True)
        xp = jnp if use_jax else np
        identity = xp.eye(node_levels[0].n, dtype=xp.complex128)

        # Define a function that represents an mzi level given inputs and all available thetas and phis
        def matrix(thetas: xp.ndarray, phis: xp.ndarray, gammas: xp.ndarray):
            outputs = identity
            # get the matrix elements for all nodes in parallel
            t11, t12, t21, t22 = node_fn(thetas, phis)
            for nc in node_levels:
                # collect the inputs to be interfered
                top = outputs[(nc.top,)]
                bottom = outputs[(nc.bottom,)]

                # collect the matrix elements to be applied in parallel to the incoming modes.
                # the new axis allows us to broadcast (apply same op) over the second output dimension
                s11 = t11[(nc.node_idxs,)][:, xp.newaxis]
                s12 = t12[(nc.node_idxs,)][:, xp.newaxis]
                s21 = t21[(nc.node_idxs,)][:, xp.newaxis]
                s22 = t22[(nc.node_idxs,)][:, xp.newaxis]

                # use jax or use numpy affects the syntax for assigning the outputs to the new outputs after the layer
                if use_jax:
                    outputs = outputs.at[(nc.top + nc.bottom,)].set(
                        xp.vstack([s11 * top + s21 * bottom,
                                   s12 * top + s22 * bottom])
                    )
                else:
                    outputs[(nc.top + nc.bottom,)] = xp.vstack([s11 * top + s21 * bottom,
                                                                s12 * top + s22 * bottom])

            # multiply the gamma phases present at the end of the network (sets the reference phase front).
            return (xp.exp(1j * gammas) * outputs.T).T

        return matrix

    @property
    def level_ordered(self):
        """Level-ordered nodes for this circuit

        Returns:
            The level-ordered nodes for this circuit (useful in cases nodes are out of order).

        """
        return ForwardCouplingCircuit.aggregate(self.levels)

    def add_error_mean(self, error: Union[float, np.ndarray] = 0,
                       error_right: Optional[Union[float, np.ndarray]] = None,
                       loss_db: Union[float, np.ndarray] = 0):
        new_nodes = copy.deepcopy(self.nodes)
        if error_right is None:
            error_right = error
        error = error * np.ones_like(self.errors_left) if isinstance(error, float) else error
        error_right = error_right * np.ones_like(self.error_right) if isinstance(error_right, float) else error_right
        loss_db = loss_db * np.ones_like(self.loss) if isinstance(loss_db, float) else loss_db
        for node, e, er, loss in zip(new_nodes, error, error_right, loss_db):
            node.error = e
            node.error_right = er
            node.loss = 1 - 10 ** (loss / 10)
        return ForwardCouplingCircuit(new_nodes)

    def add_error_variance(self, error_std: float, loss_db_std: float = 0, correlated_error: bool = True):
        """

        Args:
            error_std:
            loss_db_std:
            correlated_error:

        Returns:

        """
        new_nodes = copy.deepcopy(self.nodes)
        error_std_left = error_std * np.random.randn(self.errors_left.size)
        error_std_right = error_std * np.random.randn(self.errors_right.size) if not correlated_error else error_std_left
        loss_std = loss_db_std * np.random.randn(self.losses.size)
        loss_db = np.maximum(self.loss_db + loss_std, 0)
        error = self.errors_left + error_std_left
        error_right = self.errors_right + error_std_right
        for node, e, er, loss in zip(new_nodes, error, error_right, loss_db):
            node.error = e
            node.error_right = er
            node.loss = 1 - 10 ** (loss / 10)
        return ForwardCouplingCircuit(new_nodes)

    @property
    def loss_db(self):
        return 10 * np.log10(1 - self.losses)

    def matrix_optimizer(self, uopt: np.ndarray, thetas: np.ndarray, phis: np.ndarray, gammas: np.ndarray,
                         step_size: float = 0.1, use_jit: bool = False):
        """Matrix optimizer.

        Args:
            uopt: Unitary matrix to optimize.
            thetas: Initial thetas (internal phases).
            phis: Initial phis (external phases).
            gammas: Initial gammas (output phase front).
            step_size: Step size for the optimizer.
            use_jit: Whether to use JIT to compile the JAX function (faster to optimize, slower to compile!).

        Returns:
            A tuple of the initial state :code:`init` and the :code:`update_fn`.

        """
        fidelity = normalized_fidelity_fn(uopt, use_jax=True)
        matrix_fn = self.matrix_fn(use_jax=True)
        matrix_fn = jit(matrix_fn) if use_jit else matrix_fn

        def cost_fn(params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
            return -fidelity(matrix_fn(*params))

        opt_init, opt_update, get_params = adam(step_size=step_size)
        init = opt_init((jnp.array(thetas), jnp.array(phis), jnp.array(gammas)))

        def update_fn(i, state):
            v, g = value_and_grad(cost_fn)(get_params(state))
            return v, opt_update(i, g, state)

        return init, update_fn

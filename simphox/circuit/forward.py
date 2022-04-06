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

from ..typing import List, Union, PhaseParams
from ..utils import fix_dataclass_init_docs, normalized_error
from .coupling import CouplingNode, loss2insertion, PhaseStyle, transmissivity_to_phase, direct_transmissivity
from scipy.special import betaincinv


@fix_dataclass_init_docs
@dataclass
class ForwardMesh:
    """A :code:`ForwardMesh` is a feedforward forward couping circuit of coupling nodes that interact arbitrary waveguide pairs.

    The :code:`ForwardMesh` has the convenient property that it is acyclic, so this means that we can simply
    use a list of nodes defined in time-order traversal to define the entire circuit. This greatly simplifies
    the notation and code necessary to define such a circuit.

    General graphs / circuits do not have this property. Therefore, this class is particularly suited to applications
    where light (or computation more generally) only goes in one direction.

    Attributes:
        nodes: A list of :code:`CouplingNode`s in the order that light visits the nodes (sorted by column/column).

    """
    nodes: List[CouplingNode]
    phase_style: str = PhaseStyle.TOP

    def __post_init_post_parse__(self):
        self.num_nodes = len(self.nodes)
        self.top = tuple([int(node.top) for node in self.nodes])
        self.bottom = tuple([int(node.bottom) for node in self.nodes])
        self.bs_errors = np.array([node.bs_error for node in self.nodes])
        self.losses = np.array([node.loss for node in self.nodes])
        bs_error_l, bs_error_r = self.all_errors.T
        self.mzi_terms = np.array(
            [np.cos(np.pi / 4 + bs_error_r) * np.cos(np.pi / 4 + bs_error_l),
             np.cos(np.pi / 4 + bs_error_r) * np.sin(np.pi / 4 + bs_error_l),
             np.sin(np.pi / 4 + bs_error_r) * np.cos(np.pi / 4 + bs_error_l),
             np.sin(np.pi / 4 + bs_error_r) * np.sin(np.pi / 4 + bs_error_l)]
        )
        self.node_idxs = tuple([node.node_id for node in self.nodes])
        if np.sum(self.node_idxs) == 0:
            self.node_idxs = tuple(np.arange(len(self.nodes)).tolist())
            for node, idx in zip(self.nodes, self.node_idxs):
                node.node_id = idx
        self.n = self.nodes[0].n if len(self.nodes) > 0 else 1
        self.alpha = np.array([node.alpha for node in self.nodes])
        self.beta = np.array([node.beta for node in self.nodes])
        self.column_by_node = np.array([node.column for node in self.nodes])
        self.thetas = self.rand_theta()
        self.phis = 2 * np.pi * np.random.rand(self.thetas.size)
        self.gammas = 2 * np.pi * np.random.rand(self.n)
        self.pn = np.zeros_like(self.thetas)  # only useful for vector units
        self.pnsn = np.zeros_like(self.thetas)  # only useful for vector units
        self.num_columns = np.max(self.column_by_node) + 1 if len(self.nodes) > 0 else 0

    @property
    def dataframe(self):
        return pd.DataFrame([node.__dict__ for node in self.nodes])

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

    def offset_column(self, offset: int):
        """Offset the :code:`column`'s, useful for architectures that contain many tree subunits.

        Generally, assign :math:`\\ell := \\ell + \\Delta \\ell` for each layer :math:`\\ell`,
        where the offset is :math:`\\Delta \\ell`.

        Args:
            offset: Offset index.

        """
        for node in self.nodes:
            node.column += offset
        return self

    def invert_columns(self):
        """Invert the column labels for all the nodes.

        Assume :math:`L` total columns. If the leaf column is :math:`L`, then set it to 1 or vice versa.
        The opposite for the root column. Generally, assign :math:`L - \\ell` for each layer :math:`\\ell`.

        Returns:
            This circuit with inverted columns.

        """
        for node in self.nodes:
            node.column = self.num_columns - 1 - node.column
        self.column_by_node = np.array([node.column for node in self.nodes])
        return self

    def rand_s(self):
        """Randomly initialized split ratios :math:`s`.

        Returns:
            A randomly initialized :code:`s`.

        """
        return betaincinv(self.beta, self.alpha, np.random.rand(self.num_nodes))

    def rand_theta(self):
        """Randomly initialized coupling phase :math:`\\theta`.

        Returns:
            A randomly initialized :code:`theta`.

        """
        return transmissivity_to_phase(self.rand_s(), self.mzi_terms)

    @classmethod
    def aggregate(cls, nodes_or_node_lists: List[Union[CouplingNode, "ForwardMesh"]]):
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
    def columns(self):
        """Return a list of :code:`CouplingCircuit` by column.

        An interesting property of the resulting :code:`CouplingCircuit`'s is that no two coupling nodes
        in such circuits are connected. This allows use to apply the node operators simultaneously, which
        allows for calibration and simulation to increase in efficiency.

        Returns:
            A list of :code:`NodeList` grouped by column (needed for efficient implementations, autodiff, calibrations).

        """
        nodes_by_column = defaultdict(list)
        for node in self.nodes:
            nodes_by_column[node.column].append(node)
        return [ForwardMesh(nodes_by_column[column], phase_style=self.phase_style)
                for column in range(self.num_columns)]

    def matrix(self, params: Optional[np.ndarray] = None, back: bool = False):
        return self.matrix_fn(back=back)(self.params if params is None else params)

    def parallel_dc(self, right: bool = False):
        """This is a helper function for parallel directional couplers
        across all directional couplers in the mesh.

        Args:
            right: Whether to go to the right.

        Returns:
            A function that accepts the thetas and phis as inputs and outputs the vectorized
            MZI matrix elements

        """
        right = int(right)
        insertion = loss2insertion(self.all_losses.T[right])
        errors = self.all_errors.T[right]
        t11 = insertion * np.sin(np.pi / 4 + errors)
        t12 = insertion * 1j * np.cos(np.pi / 4 + errors)
        t21 = 1j * np.cos(np.pi / 4 + errors)
        t22 = np.sin(np.pi / 4 + errors)
        return np.array([t11, t12, t21, t22])

    def parallel_mzi_fn(self, use_jax: bool = False, back: bool = False,
                        mzi_terms: np.ndarray = None):
        """This is a helper function for finding the matrix elements for MZIs in parallel,
        leading to significant speedup.

        Args:
            use_jax: Whether to use jax.
            back: Go backward through the MZIs
            mzi_terms:

        Returns:
            A function that accepts the thetas and phis as inputs and outputs the vectorized
            MZI matrix elements

        """

        xp = jnp if use_jax else np

        # stands for cos-cos, cos-sin, sin-cos, sin-sin terms.
        cc, cs, sc, ss = self.mzi_terms if mzi_terms is None else mzi_terms.T[(self.node_idxs,)].T

        # insertion (assume a fixed loss that is global for each node) expressed in terms of dB
        il, ir = np.log(10) * self.all_losses.T / 10

        if self.phase_style == PhaseStyle.TOP:
            def parallel_mzi(theta, phi):
                t11 = (-ss + cc * xp.exp(1j * theta + il)) * xp.exp(1j * phi + ir)
                t12 = 1j * (cs + sc * xp.exp(1j * theta + il)) * xp.exp((1j * phi + ir) * (1 - back))
                t21 = 1j * (sc + cs * xp.exp(1j * theta + il)) * xp.exp((1j * phi + ir) * back)
                t22 = (cc - ss * xp.exp(1j * theta + il))
                return xp.array([t11, t12, t21, t22])
        elif self.phase_style == PhaseStyle.BOTTOM:
            def parallel_mzi(theta, phi):
                t11 = (-ss * xp.exp(1j * theta + il) + cc)
                t12 = 1j * (cs * xp.exp(1j * theta + il) + sc) * xp.exp((1j * phi + ir) * back)
                t21 = 1j * (sc * xp.exp(1j * theta + il) + cs) * xp.exp((1j * phi + ir) * (1 - back))
                t22 = (cc * xp.exp(1j * theta + il) - ss) * xp.exp(1j * phi + ir)
                return xp.array([t11, t12, t21, t22])
        elif self.phase_style == PhaseStyle.SYMMETRIC:
            def parallel_mzi(theta, phi):
                t11 = (-ss + cc * xp.exp(1j * theta + il)) * xp.exp(1j * (phi - theta / 2) + ir)
                t12 = 1j * (cs + sc * xp.exp(1j * theta + il)) * xp.exp(1j * (phi - theta / 2) + ir)
                t21 = 1j * (sc + cs * xp.exp(1j * theta + il)) * xp.exp(1j * (phi - theta / 2) + ir)
                t22 = (cc - ss * xp.exp(1j * theta + il)) * xp.exp(1j * (phi - theta / 2) + ir)
                return xp.array([t11, t12, t21, t22])
        elif self.phase_style == PhaseStyle.DIFFERENTIAL:
            def parallel_mzi(theta, phi):
                t11 = (-ss + cc * xp.exp(1j * theta + il)) * xp.exp(1j * ((-phi - theta) / 2 + phi * back) + ir)
                t12 = 1j * (cs + sc * xp.exp(1j * theta + il)) * xp.exp(1j * ((-phi - theta) / 2 + phi * back) + ir)
                t21 = 1j * (sc + cs * xp.exp(1j * theta + il)) * xp.exp(1j * ((phi - theta) / 2 - phi * back) + ir)
                t22 = (cc - ss * xp.exp(1j * theta + il)) * xp.exp(1j * ((phi - theta) / 2 - phi * back) + ir)
                return xp.array([t11, t12, t21, t22])
        else:
            raise ValueError(f"Phase style {self.phase_style} is not valid.")
        return jit(parallel_mzi) if use_jax else parallel_mzi

    def matrix_fn(self, inputs: Optional[np.ndarray] = None, use_jax: bool = False, back: bool = False):
        """Return a function that returns the matrix representation of this circuit.

        The coupling circuit is a photonic network aligned with :math:`N` waveguide rails.
        We use rail index to refer to mode index in the waveguide basis. This enables us to define
        a matrix assigning the :math:`N` inputs to :math:`N` outputs of the network. We assume :math:`N`
        is the same across all nodes in the circuit so we accsess it from any one of the individual nodes.

        Here, we define a matrix function that performs the equivalent matrix multiplications
        without needing to explicitly define :math:`N \\times N` matrix for each subunit or circuit column.
        We also go column-by-column to significantly improve the efficiency of the matrix multiplications.

        Args:
            use_jax: Use JAX to accelerate the matrix function for autodifferentiation purposes.
            inputs: The inputs, of shape :code:`(N, K)`, to propagate through the network. If :code:`None`,
                use the identity matrix. This may also be a 1d vector.
            back: Whether to propagate the inputs backwards in the device

        Returns:
            Return the MZI column function that transforms the inputs (no explicit matrix defined here).
            This function accepts network :code:`inputs`, and the node parameters :code:`thetas, phis`.
            Using the identity matrix as input gives the unitary representation of the rail network.

        """

        node_columns = self.columns[::-1] if back else self.columns
        node_fn = self.parallel_mzi_fn(use_jax=use_jax, back=back)
        xp = jnp if use_jax else np
        inputs = np.eye(node_columns[0].n, dtype=np.complex128) if inputs is None else inputs
        inputs = inputs[..., np.newaxis] if inputs.ndim == 1 else inputs

        # Define a function that represents an mzi column given inputs and all available thetas and phis
        def matrix(params: Optional[Tuple[xp.ndarray, xp.ndarray, np.ndarray]]):
            thetas, phis, gammas = params
            outputs = xp.array(inputs.copy())
            # get the matrix elements for all nodes in parallel
            t11, t12, t21, t22 = node_fn(thetas, phis)

            if back:
                outputs = outputs * (xp.exp(1j * gammas) * outputs.T).T

            for nc in node_columns:
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
            if not back:
                outputs = (xp.exp(1j * gammas) * outputs.T).T
            return outputs

        return matrix

    @property
    def column_ordered(self):
        """column-ordered nodes for this circuit

        Returns:
            The column-ordered nodes for this circuit (useful in cases nodes are out of order).

        """
        return ForwardMesh.aggregate(self.columns)

    def add_error_mean(self, error: Union[float, np.ndarray] = 0, loss_db: Union[float, np.ndarray] = 0):
        """Add split error (in phase) and loss error (in dB) mean values to the circuit.

        Args:
            error: Phase-parametrized error for the left (and right if not specified) splitters
            loss_db: The loss in dB.

        Returns:
            A new :code:`ForwardCouplingCircuit` with the modified error terms.

        """
        new_nodes = copy.deepcopy(self.nodes)
        error = error * np.ones_like(self.bs_errors) if not isinstance(error, np.ndarray) else error
        loss_db = loss_db * np.ones_like(self.losses) if not isinstance(loss_db, np.ndarray) else loss_db
        for node, e, loss in zip(new_nodes, error, loss_db):
            node.bs_error = tuple(e)
            node.loss = tuple(loss)
        mesh = ForwardMesh(new_nodes, self.phase_style)
        mesh.params = self.params
        return mesh

    def add_error_variance(self, bs_error_std: float, loss_db_std: float = 0, equal_splitter_error: bool = True):
        """Add split error (in phase) and loss error (in dB) variance values to the circuit.

        Args:
            bs_error_std: Standard deviation in the beamsplitter error
            loss_db_std: Standard deviation in the loss error in dB
            equal_splitter_error: Equalize the beamsplitter error across the left and right splitters

        Returns:
            A new :code:`ForwardCouplingCircuit` with the modified error terms.

        """
        new_nodes = copy.deepcopy(self.nodes)
        bs_error_std = bs_error_std * np.random.randn(*self.bs_errors.shape)
        if equal_splitter_error:
            bs_error_std[:, 1] = bs_error_std[:, 0]
        loss_std = loss_db_std * np.random.randn(*self.losses.shape)
        loss_db = self.losses + loss_std
        error = self.bs_errors + bs_error_std
        for node, e, loss in zip(new_nodes, error, loss_db):
            node.bs_error = tuple(e)
            node.loss = tuple(loss)
        mesh = ForwardMesh(new_nodes, self.phase_style)
        mesh.params = self.params
        return mesh

    @property
    def all_errors(self):
        if (self.bs_errors.ndim == 1 or self.bs_errors.shape[-1] == 1) and self.bs_errors.size == self.num_nodes:
            return np.hstack((self.bs_errors, self.bs_errors))
        elif self.bs_errors.shape[-1] == 2 and self.bs_errors.size == self.num_nodes * 2:
            return self.bs_errors
        else:
            raise AttributeError(f"Losses should have size "
                                 f"({self.bs_errors.shape[-1]}, [1 or 2]).")

    @property
    def all_losses(self):
        if (self.losses.ndim == 1 or self.losses.shape[-1] == 1) and self.losses.size == self.num_nodes:
            return np.hstack((self.losses, self.losses))
        elif self.losses.shape[-1] == 2 and self.losses.size == self.num_nodes * 2:
            return self.losses
        else:
            raise AttributeError(f"Losses should have size "
                                 f"({self.losses.shape[-1]}, [1 or 2]).")

    def matrix_opt(self, uopt: np.ndarray, params: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                   step_size: float = 0.1, use_jit: bool = False):
        """Matrix optimizer.

        Args:
            uopt: Unitary matrix to optimize.
            params: Initial params (uses params of the class if :code:`None`).
            step_size: Step size for the optimizer.
            use_jit: Whether to use JIT to compile the JAX function (faster to optimize, slower to compile!).

        Returns:
            A tuple of the initial state :code:`init` and the :code:`update_fn`.

        """
        error = normalized_error(uopt, use_jax=True)
        matrix_fn = self.matrix_fn(use_jax=True)
        matrix_fn = jit(matrix_fn) if use_jit else matrix_fn

        def cost_fn(params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
            return error(matrix_fn(params))

        opt_init, opt_update, get_params = adam(step_size=step_size)
        thetas, phis, gammas = self.params if params is None else params
        init = opt_init((jnp.array(thetas), jnp.array(phis), jnp.array(gammas)))

        def update_fn(i, state):
            v, g = value_and_grad(cost_fn)(get_params(state))
            return v, opt_update(i, g, state)

        return init, update_fn, get_params

    def _column_transform(self, v: np.ndarray, matrix_elements: np.ndarray):
        t11, t12, t21, t22 = matrix_elements
        top = v[(self.top,)]
        bottom = v[(self.bottom,)]
        s11, s12 = t11[(self.node_idxs,)][:, np.newaxis], t12[(self.node_idxs,)][:, np.newaxis]
        s21, s22 = t21[(self.node_idxs,)][:, np.newaxis], t22[(self.node_idxs,)][:, np.newaxis]
        v[(self.top + self.bottom,)] = np.vstack([s11 * top + s21 * bottom,
                                                  s12 * top + s22 * bottom])

    def propagate(self, inputs: Optional[np.ndarray] = None, back: bool = False,
                  column_cutoff: Optional[int] = None, params: Optional[PhaseParams] = None,
                  explicit: bool = True):
        """Propagate :code:`inputs` through the mesh

        Args:
            inputs: Inputs for propagation of the modes through the mesh.
            back: send the light backward (flip the mesh)
            column_cutoff: The cutoff column where to start propagating (useful for nullification basis)
            params: parameters to use for the propagation (if :code:`None`, use the default params attribute).
            explicit: Explicitly consider the directional couplers in the propagation

        Returns:
            Propagated fields

        """

        params = self.params if params is None else params
        thetas, phis, gammas = params

        node_columns = self.columns[::-1] if back else self.columns
        inputs = np.eye(self.n, dtype=np.complex128) if inputs is None else inputs + 0j
        inputs = inputs[:, np.newaxis] if inputs.ndim == 1 else inputs
        outputs = inputs.copy()
        if explicit:
            left = self.parallel_dc(right=False)
            right = self.parallel_dc(right=True)
        else:
            mzis = self.parallel_mzi_fn(use_jax=False, back=back)(thetas, phis)

        propagated = [outputs.copy()]
        if column_cutoff is None:
            column_cutoff = -1 if back else self.num_columns

        if back and column_cutoff == -1:
            outputs = (np.exp(1j * gammas) * outputs.T).T
            propagated.append(outputs.copy())

        # get the matrix elements for all nodes in parallel
        for nc in node_columns:
            if back and (nc.column_by_node.size == 0 or self.num_columns - column_cutoff < nc.column_by_node[0]):
                continue
            if not back and (nc.column_by_node.size == 0 or column_cutoff <= nc.column_by_node[0]):
                continue

            if explicit:
                if back:
                    nc._column_transform(outputs, right)
                    propagated.append(outputs.copy())
                    outputs[(nc.top,)] *= np.exp(1j * thetas[(nc.node_idxs,)][:, np.newaxis])
                    propagated.append(outputs.copy())
                    nc._column_transform(outputs, left)
                    propagated.append(outputs.copy())
                    outputs[(nc.top,)] *= np.exp(1j * phis[(nc.node_idxs,)][:, np.newaxis])
                    propagated.append(outputs.copy())
                else:
                    outputs[(nc.top,)] *= np.exp(1j * phis[(nc.node_idxs,)][:, np.newaxis])
                    propagated.append(outputs.copy())
                    nc._column_transform(outputs, left)
                    propagated.append(outputs.copy())
                    outputs[(nc.top,)] *= np.exp(1j * thetas[(nc.node_idxs,)][:, np.newaxis])
                    propagated.append(outputs.copy())
                    nc._column_transform(outputs, right)
                    propagated.append(outputs.copy())
            else:
                nc._column_transform(outputs, mzis)
                propagated.append(outputs.copy())

        if not back and column_cutoff == self.num_columns:
            outputs = (np.exp(1j * gammas) * outputs.T).T
            propagated.append(outputs.copy())
        return np.array(propagated).squeeze() if explicit else np.array(propagated)

    @property
    def nullification_basis(self):
        """The nullificuation basis for parallel nullification and error correction of non-self-configurable
        architectures.

        Returns:
            The nullification basis for the architecture based on the internal thetas, phis, and gammas.

        """
        node_columns = self.columns
        null_vecs = []
        for nc in node_columns:
            vector = np.zeros(self.n, dtype=np.complex128)
            vector[(nc.bottom,)] = 1
            null_vecs.append(
                self.propagate(
                    vector[:, np.newaxis],
                    column_cutoff=self.num_columns - nc.column_by_node[0],
                    back=True, explicit=False)[-1]
            )
        return np.array(null_vecs)[..., 0].conj()

    def program_by_null_basis(self, nullification_basis: np.ndarray):
        """Parallel program the mesh using the null basis.

        Args:
            nullification_basis: The nullification basis for the photonic mesh network.

        Returns:
            The parameters to be programmed

        """

        node_columns = self.columns
        for nc, w in zip(node_columns, nullification_basis):
            vector = self.propagate(w.copy(), column_cutoff=nc.column_by_node[0], explicit=False)[-1]
            theta, phi = nc.parallel_nullify(vector, self.mzi_terms)
            self.thetas[(nc.node_idxs,)] = theta
            self.phis[(nc.node_idxs,)] = np.mod(phi, 2 * np.pi)

    def parallel_nullify(self, vector: np.ndarray, mzi_terms: np.ndarray):
        """Assuming the mesh is a column, this method runs a parallel nullify algorithm to set up
        the elements of the column in parallel.

        Args:
            vector: The vector entering the column.
            mzi_terms: The MZI terms account for errors in the couplers of the photonic circuit.

        Returns:
            The programmed phases

        """
        top = vector[(self.top,)]
        bottom = vector[(self.bottom,)]
        mzi_terms = mzi_terms.T[(self.node_idxs,)].T
        cc, cs, sc, ss = mzi_terms
        if self.phase_style == PhaseStyle.SYMMETRIC:
            raise NotImplementedError('Require phase_style not be of the SYMMETRIC variety.')
        elif self.phase_style == PhaseStyle.BOTTOM:
            theta = transmissivity_to_phase(direct_transmissivity(top[:, -1], bottom[:, -1]), mzi_terms)
            phi = np.angle(top[:, -1]) - np.angle(bottom[:, -1]) + np.pi
            phi += np.angle(-ss + cc * np.exp(-1j * theta)) - np.angle(1j * (cs + np.exp(-1j * theta) * sc))
        else:
            theta = transmissivity_to_phase(direct_transmissivity(top[:, -1], bottom[:, -1]), mzi_terms)
            phi = np.angle(bottom[:, -1]) - np.angle(top[:, -1])
            phi += np.angle(-ss + cc * np.exp(-1j * theta)) - np.angle(1j * (cs + np.exp(-1j * theta) * sc))
        return theta, phi

    @property
    def params(self):
        return copy.deepcopy((self.thetas, self.phis, self.gammas))

    @params.setter
    def params(self, params: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        self.thetas, self.phis, self.gammas = params

    def phases(self, error: np.ndarray = 0, constant: bool = False, gamma_error: bool = False):
        """

        Args:
            error: Error in the phases.
            constant: Whether the phase error should be constant.
            gamma_error: The error in gamma (output phases).

        Returns:
            The phases

        """
        errors = error if constant else error * np.random.randn(self.thetas.size)
        g_errors = error if constant else error * np.random.randn(self.gammas.size)
        return self.thetas + errors, self.phis + errors, self.gammas + g_errors * gamma_error
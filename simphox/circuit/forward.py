import copy
from collections import defaultdict
from typing import Tuple, Optional
from jax import custom_vjp


import jax.numpy as jnp
from jax import value_and_grad, jit, lax
from jax.example_libraries.optimizers import adam
import numpy as np
import pandas as pd
import haiku as hk

# from pydantic.dataclasses import dataclass

from ..typing import List, Union
from ..utils import fix_dataclass_init_docs, normalized_error
from .coupling import CouplingNode, PhaseStyle, transmissivity_to_phase, direct_transmissivity
from scipy.special import betaincinv

from dataclasses import dataclass


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

    # def __post_init_post_parse__(self):
    def __post_init__(self):
        self.num_nodes = len(self.nodes)
        self.top = [int(node.top) for node in self.nodes]
        self.bottom = [int(node.bottom) for node in self.nodes]
        self.bs_errors = np.array([node.bs_error for node in self.nodes])
        self.losses = np.array([node.loss for node in self.nodes])
        self.mzi_terms = _mzi_terms(self.all_errors)
        self.node_idxs = [node.node_id for node in self.nodes]
        if np.sum(self.node_idxs) == 0:
            self.node_idxs = np.arange(len(self.nodes)).tolist()
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
    def columns(self) -> List["ForwardMesh"]:
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

    def matrix_fn(self, use_jax: bool = False, back: bool = False):
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

        ncs = self.columns[::-1] if back else self.columns
        xp = jnp if use_jax else np
        all_idx = self.idx(back)

        # Define a function that represents an mzi column given inputs and all available thetas and phis
        def matrix(params: Tuple[xp.ndarray, xp.ndarray, xp.ndarray] = self.params, inputs=xp.eye(self.n, dtype=np.complex64),
                   bs_errors=xp.array(self.all_errors, dtype=np.complex64), loss_errors=xp.array(self.all_losses, dtype=np.complex64)):

            inputs += 0j  # cast to complex if not already
            thetas, phis, gammas = params
            outputs = xp.array(inputs[:, None].copy() if inputs.ndim == 1 else inputs.copy())
            mzis = _parallel_mzi(
                thetas, phis, bs_errors, loss_errors,
                phase_style=self.phase_style, use_jax=use_jax, back=back
            )

            if back:
                outputs = (xp.exp(1j * gammas) * outputs.T).T

            if use_jax:
                def body_fun(outputs, idx):
                    return _parallel_transform(outputs, mzis, idx, use_jax=True), None
                outputs, _ = lax.scan(body_fun, outputs, all_idx)
            else:
                for nc in ncs:
                    outputs = nc.parallel_transform(outputs, mzis, use_jax=False)

            if not back:
                outputs = (xp.exp(1j * gammas) * outputs.T).T
            return outputs

        return matrix

    def idx(self, back: bool = False):
        """Returns a JAX-friendly index representation ofthe forward mesh.

        Args:
            Whether to run the mesh backwards

        """
        n = self.n
        cols = self.columns
        cols = cols[::-1] if back else cols
        _top = [nc.top for nc in cols]
        num_nodes = [nc.num_nodes for nc in cols]
        m = np.max(num_nodes)
        _bottom = [nc.bottom for nc in cols]
        _node_idxs = [nc.node_idxs for nc in cols]
        top = np.zeros((len(num_nodes), m), dtype=np.int32)
        bottom = np.zeros((len(num_nodes), m), dtype=np.int32)
        node_idxs = np.zeros((len(num_nodes), m), dtype=np.int32)
        mask = np.zeros((len(num_nodes), m), dtype=np.int32)

        # indices to fill to avoid bad overwriting when mask = 0.
        dummy_fill = []
        for t, b in zip(_top, _bottom):
            dummy_fill.append(np.delete(np.arange(n), np.hstack((t, b))))

        for i, t in enumerate(_top):
            top[i, :len(t)] = t
            if dummy_fill[i].size:
                top[i, len(t):] = dummy_fill[i][0]
        for i, t in enumerate(_bottom):
            bottom[i, :len(t)] = t
            if dummy_fill[i].size:
                bottom[i, len(t):] = dummy_fill[i][0]
        for i, t in enumerate(_node_idxs):
            node_idxs[i, :len(t)] = t
        for i, t in enumerate(_node_idxs):
            mask[i, :len(t)] = 1

        return top, bottom, node_idxs, mask

    def propagate_matrix_fn(self, back: bool = False, column_cutoff: Optional[int] = None,
                            explicit: bool = True, use_jax: bool = False):
        """Return a function that propagates any given set of inputs through this circuit, returning the transfer matrix at each step.

        Args:
            inputs: Inputs for propagation of the modes through the mesh.
            back: send the light backward (flip the mesh)
            column_cutoff: The cutoff column where to start propagating (useful for nullification basis), default to all columns if None or use_jax True
            explicit: Explicitly consider the directional couplers in the propagation.
            use_jax: Whether to use jax to implement the propagation (accelerated performance!).

        Returns:
            Propagated fields

        """
        node_columns = self.columns[::-1] if back else self.columns
        if column_cutoff is None:
            column_cutoff = -1 if back else self.num_columns

        xp = jnp if use_jax else np
        all_idx = self.idx(back)

        def propagate_matrix(params=self.params, inputs=xp.eye(self.n, dtype=np.complex64),
                             bs_errors=xp.array(self.all_errors, dtype=np.complex64), loss_errors=xp.array(self.all_losses, dtype=np.complex64)):
            thetas, phis, gammas = params
            thetas, phis, gammas = xp.array(thetas), xp.array(phis), xp.array(gammas)
            inputs = inputs + 0j  # cast to complex if not already
            outputs = xp.array(inputs[:, None].copy() if inputs.ndim == 1 else inputs.copy())
            propagated = [outputs.copy()[None]]

            if back and column_cutoff == -1:
                outputs = (xp.exp(1j * gammas) * outputs.T).T
                propagated.append(outputs.copy()[None])

            if explicit:
                left = _parallel_dc(bs_errors, loss_errors, right=False, use_jax=use_jax)
                right = _parallel_dc(bs_errors, loss_errors, right=True, use_jax=use_jax)
            else:
                mzis = _parallel_mzi(thetas, phis, bs_errors, loss_errors, use_jax=use_jax, back=back, phase_style=self.phase_style)

            # get the matrix elements for all nodes in parallel
            if use_jax:
                if explicit:
                    def parallel_propagate(carry, idx):
                        return _parallel_propagate(carry, thetas, phis, left, right, idx,
                                                   phase_style=self.phase_style, back=back, use_jax=True)
                else:
                    def parallel_propagate(carry, idx):
                        carry = _parallel_transform(carry, mzis, idx, phase_style=self.phase_style, back=back, use_jax=True)
                        return carry, carry[None]
                outputs, _propagated = lax.scan(parallel_propagate, outputs, all_idx)
                propagated.extend(_propagated)
            else:
                for nc in node_columns:
                    if back and self.num_columns - column_cutoff < nc.column_by_node[0]:
                        continue
                    if not back and column_cutoff <= nc.column_by_node[0]:
                        continue
                    if explicit:
                        outputs, _propagated = nc.parallel_propagate(outputs, thetas, phis, left, right, back=back, use_jax=False)
                        propagated.extend(_propagated)
                    else:
                        outputs = nc.parallel_transform(outputs, mzis, use_jax=False)
                        propagated.append(outputs.copy()[None])

            if not back and column_cutoff == self.num_columns:
                outputs = (xp.exp(1j * gammas) * outputs.T).T
                propagated.append(outputs.copy()[None])

            return xp.vstack(propagated).squeeze() if explicit else xp.vstack(propagated)
        return propagate_matrix

    def parallel_propagate(self, outputs: np.ndarray, thetas: np.ndarray, phis: np.ndarray, left: np.ndarray, right: np.ndarray,
                           back: bool = False, use_jax: bool = True):
        return _parallel_propagate(outputs, thetas, phis, left, right,
                                   (self.top, self.bottom, self.node_idxs, None), self.phase_style, back, use_jax)

    def in_situ_matrix_fn(self, tap_pd_shot_noise: float = 0, io_amp_error_std: float = 0, io_phase_error_std: float = 0,
                          all_analog: bool = True):
        """A version of matrix function with in situ backpropagation registered as the "optical VJP."

        Args:
            tap_pd_shot_noise: Tap photodetector shot noise (proportional to power).
            io_amp_error_std: Input/output amplitude error/noise standard deviation.
            io_phase_error_std: Input/output phase error/noise standard deviation.
            all_analog: All-analog implementation of backprop involves running the backward step by sending the adjoint field forward.

        """
        forward_matrix_fn = self.matrix_fn(use_jax=True, back=False)
        backward_matrix_fn = self.matrix_fn(use_jax=True, back=True)
        forward_prop_fn = self.propagate_matrix_fn(use_jax=True, back=False)
        backward_prop_fn = self.propagate_matrix_fn(use_jax=True, back=True)

        @custom_vjp
        def matrix(params=self.params, inputs=jnp.eye(self.n, dtype=np.complex64),
                   bs_errors=jnp.array(self.all_errors), loss_errors=jnp.array(self.all_losses)):
            forward = forward_matrix_fn(params, inputs, bs_errors, loss_errors)
            forward_inputs_abs = jnp.abs(jnp.abs(forward) * (1 + io_amp_error_std * np.random.randn(*forward.shape)))
            forward_inputs_phase = jnp.angle(forward) + io_phase_error_std * np.random.randn(*forward.shape) * np.pi
            forward = forward_inputs_abs * jnp.exp(1j * forward_inputs_phase)
            return forward

        def matrix_fwd(params, inputs, bs_errors, loss_errors):
            # Returns primal output and residuals to be used in backward pass by f_bwd.
            return matrix(params, inputs, bs_errors, loss_errors), (params, inputs, bs_errors, loss_errors)

        def matrix_bwd(res, g):
            params, inputs, bs_errors, loss_errors = res

            inorm = jnp.max(jnp.linalg.norm(inputs, axis=-1))
            gnorm = jnp.max(jnp.linalg.norm(g, axis=-1))

            inputs /= inorm
            g /= gnorm

            if all_analog:
                # instead of using digital subtraction of backpropagating signals, use only forward prop'd signal
                adjoint_inputs = backward_matrix_fn(params, g, bs_errors, loss_errors).squeeze()
                adjoint_inputs_abs = jnp.abs(jnp.abs(adjoint_inputs) * (1 + io_amp_error_std * np.random.randn(*adjoint_inputs.shape)))
                adjoint_inputs_phase = jnp.angle(adjoint_inputs) + io_phase_error_std * np.random.randn(*adjoint_inputs.shape) * np.pi
                adjoint_inputs = adjoint_inputs_abs * jnp.exp(1j * adjoint_inputs_phase)
                adjoint = forward_prop_fn(params, jnp.conj(adjoint_inputs), bs_errors, loss_errors)
                sum = forward_prop_fn(params, inputs - 1j * jnp.conj(adjoint_inputs), bs_errors, loss_errors)
                diff = forward_prop_fn(params, inputs + 1j * jnp.conj(adjoint_inputs), bs_errors, loss_errors)
                sum_abs, diff_abs = jnp.abs(sum), jnp.abs(diff)
                sum_meas = sum_abs ** 2 + tap_pd_shot_noise * sum_abs * np.random.randn(*sum.shape)
                diff_meas = diff_abs ** 2 + tap_pd_shot_noise * diff_abs * np.random.randn(*diff.shape)
                grad_powers = (sum_meas - diff_meas) / 2
            else:
                forward = forward_prop_fn(params, inputs, bs_errors, loss_errors)
                adjoint = backward_prop_fn(params, g, bs_errors, loss_errors)[::-1]
                adjoint_inputs = adjoint[0]
                sum = forward_prop_fn(params, inputs - 1j * jnp.conj(adjoint_inputs), bs_errors, loss_errors)
                sum_abs, forward_abs, adjoint_abs = jnp.abs(sum), jnp.abs(forward), jnp.abs(adjoint)
                sum_meas = sum_abs ** 2 + tap_pd_shot_noise * sum_abs * np.random.randn(*forward.shape)
                forward_meas = forward_abs ** 2 + tap_pd_shot_noise * forward_abs * np.random.randn(*forward.shape)
                adjoint_meas = adjoint_abs ** 2 + tap_pd_shot_noise * adjoint_abs * np.random.randn(*forward.shape)
                grad_powers = sum_meas - forward_meas - adjoint_meas

            # returns the grad powers at the various gradient positions and adjoint inputs (needed to propagate errors backward to prev layer)
            # assumes bs errors and loss errors are not modifiable (zero backpropagated gradients for those!)
            return (self.phase_shift_localize(grad_powers / 2 * (inorm * gnorm)), adjoint_inputs, jnp.zeros_like(bs_errors), jnp.zeros_like(loss_errors))

        matrix.defvjp(matrix_fwd, matrix_bwd)
        return matrix

    def propagate(self, inputs: Optional[np.ndarray] = None, back: bool = False,
                  column_cutoff: Optional[int] = None, explicit: bool = True):
        """Propagate :code:`inputs` through the mesh

        Args:
            inputs: Inputs for propagation of the modes through the mesh.
            back: send the light backward (flip the mesh)
            column_cutoff: The cutoff column where to start propagating (useful for nullification basis)
            params: parameters to use for the propagation (if :code:`None`, use the default params attribute).
            explicit: Explicitly consider the directional couplers in the propagation.
            use_jax: Whether to use jax to implement the propagation.

        Returns:
            Propagated fields

        """
        return self.propagate_matrix_fn(back=back, column_cutoff=column_cutoff, explicit=explicit, use_jax=False)(inputs=inputs)

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
            vector = np.zeros(self.n, dtype=np.complex64)
            vector[nc.bottom] = 1
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
            self.thetas[nc.node_idxs] = theta
            self.phis[nc.node_idxs] = np.mod(phi, 2 * np.pi)

    def parallel_nullify(self, vector: np.ndarray, mzi_terms: np.ndarray):
        """Assuming the mesh is a column, this method runs a parallel nullify algorithm to set up
        the elements of the column in parallel.

        Args:
            vector: The vector entering the column.
            mzi_terms: The MZI terms account for errors in the couplers of the photonic circuit.

        Returns:
            The programmed phases

        """
        top = vector[self.top]
        bottom = vector[self.bottom]
        mzi_terms = mzi_terms.T[self.node_idxs].T
        cc, cs, sc, ss = mzi_terms
        theta = transmissivity_to_phase(direct_transmissivity(top[:, -1], bottom[:, -1]), mzi_terms)
        if self.phase_style == PhaseStyle.SYMMETRIC:
            raise NotImplementedError('Require phase_style not be of the SYMMETRIC variety.')
        elif self.phase_style == PhaseStyle.BOTTOM:
            phi = np.angle(top[:, -1]) - np.angle(bottom[:, -1]) + np.pi
        else:
            phi = np.angle(bottom[:, -1]) - np.angle(top[:, -1])
        phi += np.angle(-ss + cc * np.exp(-1j * theta)) - np.angle(1j * (cs + np.exp(-1j * theta) * sc))
        return theta, phi

    @property
    def params(self):
        return (self.thetas.copy(), self.phis.copy(), self.gammas.copy())

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

    def phase_shift_localize(self, prop_powers: np.ndarray, use_jax: bool = True):
        xp = jnp if use_jax else np
        if prop_powers.ndim == 3:
            return (
                xp.vstack([prop_powers[i * 4 + 2, nc.top] for i, nc in enumerate(self.columns)]).sum(axis=-1),
                xp.vstack([prop_powers[i * 4, nc.top] for i, nc in enumerate(self.columns)]).sum(axis=-1),
                prop_powers[-1].sum(axis=-1)
            )
        elif prop_powers.ndim == 2:
            return (
                xp.hstack([prop_powers[i * 4 + 2, nc.top] for i, nc in enumerate(self.columns)]),
                xp.hstack([prop_powers[i * 4, nc.top] for i, nc in enumerate(self.columns)]),
                prop_powers[-1]
            )

    def parallel_mzi(self, thetas: np.ndarray = None, phis: np.ndarray = None, phase_style: PhaseStyle = None, use_jax: bool = False, back: bool = False):
        phase_style = self.phase_style if phase_style is None else phase_style
        thetas = self.params[0] if thetas is None else thetas
        phis = self.params[0] if phis is None else phis
        return _parallel_mzi(thetas, phis, self.all_errors, self.all_losses, phase_style, use_jax, back)

    @property
    def column_ordered(self):
        """column-ordered nodes for this circuit

        Returns:
            The column-ordered nodes for this circuit (useful in cases nodes are out of order).

        """
        return ForwardMesh.aggregate(self.columns)

    def add_error_mean(self, error: Union[float, np.ndarray] = 0, loss_db: Union[float, np.ndarray] = 0,
                       error_right: Optional[Union[float, np.ndarray]] = None):
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

    def parallel_transform(self, v: np.ndarray, matrix_elements: np.ndarray, use_jax: bool = False):
        return _parallel_transform(v, matrix_elements,
                                   (self.top, self.bottom, self.node_idxs, None), use_jax=use_jax)


def _parallel_mzi(theta: np.ndarray, phi: np.ndarray,
                  bs_errors: np.ndarray, loss_errors: np.ndarray, phase_style: PhaseStyle,
                  use_jax: bool = False, back: bool = False):
    """This is a helper function for finding the matrix elements for MZIs in parallel,
    leading to significant speedup.

    Args:
        theta: theta phases to use.
        phi: phi phases to use.
        use_jax: Whether to use jax.
        back: Go backward through the MZIs
        mzi_terms: MZI terms (evaluation of the errors for the various mzi matrix elements)

    Returns:
        A function that accepts the thetas and phis as inputs and outputs the vectorized
        MZI matrix elements

    """

    xp = jnp if use_jax else np

    # stands for cos-cos, cos-sin, sin-cos, sin-sin terms.
    cc, cs, sc, ss = _mzi_terms(bs_errors, use_jax)
    # insertion (assume a fixed loss that is global for each node) expressed in terms of dB
    il, ir = np.log(10) * loss_errors.T / 20

    if phase_style == PhaseStyle.TOP:
        t11 = (-ss + cc * xp.exp(1j * theta + il)) * xp.exp(1j * phi + ir)
        t12 = 1j * (cs + sc * xp.exp(1j * theta + il)) * xp.exp((1j * phi + ir) * (1 - back))
        t21 = 1j * (sc + cs * xp.exp(1j * theta + il)) * xp.exp((1j * phi + ir) * back)
        t22 = (cc - ss * xp.exp(1j * theta + il))
    elif phase_style == PhaseStyle.BOTTOM:
        t11 = (-ss * xp.exp(1j * theta + il) + cc)
        t12 = 1j * (cs * xp.exp(1j * theta + il) + sc) * xp.exp((1j * phi + ir) * back)
        t21 = 1j * (sc * xp.exp(1j * theta + il) + cs) * xp.exp((1j * phi + ir) * (1 - back))
        t22 = (cc * xp.exp(1j * theta + il) - ss) * xp.exp(1j * phi + ir)
    elif phase_style == PhaseStyle.SYMMETRIC:
        t11 = (-ss + cc * xp.exp(1j * theta + il)) * xp.exp(1j * (phi - theta / 2) + ir)
        t12 = 1j * (cs + sc * xp.exp(1j * theta + il)) * xp.exp(1j * (phi - theta / 2) + ir)
        t21 = 1j * (sc + cs * xp.exp(1j * theta + il)) * xp.exp(1j * (phi - theta / 2) + ir)
        t22 = (cc - ss * xp.exp(1j * theta + il)) * xp.exp(1j * (phi - theta / 2) + ir)
    elif phase_style == PhaseStyle.DIFFERENTIAL:
        t11 = (-ss + cc * xp.exp(1j * theta + il)) * xp.exp(1j * ((-phi - theta) / 2 + phi * back) + ir)
        t12 = 1j * (cs + sc * xp.exp(1j * theta + il)) * xp.exp(1j * ((-phi - theta) / 2 + phi * back) + ir)
        t21 = 1j * (sc + cs * xp.exp(1j * theta + il)) * xp.exp(1j * ((phi - theta) / 2 - phi * back) + ir)
        t22 = (cc - ss * xp.exp(1j * theta + il)) * xp.exp(1j * ((phi - theta) / 2 - phi * back) + ir)
    else:
        raise ValueError(f"Phase style {phase_style} is not valid.")
    return xp.array([t11, t12, t21, t22])


def _parallel_dc(bs_errors: np.ndarray, loss_errors: np.ndarray, right: bool = False, use_jax: bool = False):
    """This is a helper function for parallel directional couplers
    across all directional couplers in the mesh.

    Args:
        right: Whether to go to the right.

    Returns:
        A function that accepts the thetas and phis as inputs and outputs the vectorized
        MZI matrix elements

    """
    xp = jnp if use_jax else np
    right = int(right)
    insertion = xp.exp(np.log(10) * loss_errors.T[right] / 20)
    errors = bs_errors.T[right]
    t11 = insertion * xp.sin(np.pi / 4 + errors)
    t12 = insertion * 1j * xp.cos(np.pi / 4 + errors)
    t21 = 1j * xp.cos(np.pi / 4 + errors)
    t22 = xp.sin(np.pi / 4 + errors)
    return xp.array([t11, t12, t21, t22])


def _mzi_terms(bs_errors: np.ndarray, use_jax: bool = False):
    xp = jnp if use_jax else np
    bs_error_l, bs_error_r = bs_errors.T
    return xp.array((xp.cos(np.pi / 4 + bs_error_r) * xp.cos(np.pi / 4 + bs_error_l),
                     xp.cos(np.pi / 4 + bs_error_r) * xp.sin(np.pi / 4 + bs_error_l),
                     xp.sin(np.pi / 4 + bs_error_r) * xp.cos(np.pi / 4 + bs_error_l),
                     xp.sin(np.pi / 4 + bs_error_r) * xp.sin(np.pi / 4 + bs_error_l)))


def _parallel_transform(v: np.ndarray, matrix_elements: np.ndarray, idx: np.ndarray, use_jax: bool = False):
    t11, t12, t21, t22 = matrix_elements
    top, bottom, node_idxs, mask = idx

    _top = v[top]
    _bottom = v[bottom]
    s11, s12 = t11[node_idxs][:, np.newaxis], t12[node_idxs][:, np.newaxis]
    s21, s22 = t21[node_idxs][:, np.newaxis], t22[node_idxs][:, np.newaxis]
    if use_jax:
        mask = mask[:, np.newaxis]
        s11 = s11 * mask + (1 + 0j - mask)
        s22 = s22 * mask + (1 + 0j - mask)
        s12 = s12 * mask
        s21 = s21 * mask
        return v.at[jnp.hstack((top, bottom))].set(
            jnp.vstack((s11 * _top + s21 * _bottom,
                        s12 * _top + s22 * _bottom))
        )
    else:
        v[top + bottom] = np.vstack([
            s11 * _top + s21 * _bottom,
            s12 * _top + s22 * _bottom
        ])
        return v


def _parallel_multiply(v: np.ndarray, phases: np.ndarray, idx: np.ndarray, phase_style: PhaseStyle, use_jax: bool = False):
    top, bottom, node_idxs, mask = idx
    if phase_style != PhaseStyle.TOP and phase_style != PhaseStyle.BOTTOM:
        raise ValueError(f"Phase style must be TOP or BOTTOM, phase style {phase_style} is not yet supported.")
    idx = top if phase_style == PhaseStyle.TOP else bottom
    if use_jax:
        phasors = jnp.exp(1j * phases[node_idxs] * mask)
        return v.at[idx].set(v[idx] * phasors[:, np.newaxis])
    else:
        v[idx] *= np.exp(1j * phases[node_idxs])[:, np.newaxis]
        return v

def _parallel_propagate(inputs: np.ndarray, thetas: np.ndarray, phis: np.ndarray, left: np.ndarray, right: np.ndarray,
                        idx: np.ndarray, phase_style: PhaseStyle, back: bool = False, use_jax: bool = True):
    """Propagate through a layer of MZIs
    """
    propagated = []
    if back:
        v1 = _parallel_transform(inputs, right, idx, use_jax=use_jax)
        propagated.append(v1[None] if use_jax else v1.copy()[None])
        v2 = _parallel_multiply(v1, thetas, idx, phase_style=phase_style, use_jax=use_jax)
        propagated.append(v2[None] if use_jax else v2.copy()[None])
        v3 = _parallel_transform(v2, left, idx, use_jax=use_jax)
        propagated.append(v3[None] if use_jax else v3.copy()[None])
        outputs = _parallel_multiply(v3, phis, idx, phase_style=phase_style, use_jax=use_jax)
        propagated.append(outputs[None] if use_jax else outputs.copy()[None])
    else:
        v1 = _parallel_multiply(inputs, phis, idx, phase_style=phase_style, use_jax=use_jax)
        propagated.append(v1[None] if use_jax else v1.copy()[None])
        v2 = _parallel_transform(v1, left, idx, use_jax=use_jax)
        propagated.append(v2[None] if use_jax else v2.copy()[None])
        v3 = _parallel_multiply(v2, thetas, idx, phase_style=phase_style, use_jax=use_jax)
        propagated.append(v3[None] if use_jax else v3.copy()[None])
        outputs = _parallel_transform(v3, right, idx, use_jax=use_jax)
        propagated.append(outputs[None] if use_jax else outputs.copy()[None])
    return outputs, jnp.vstack(propagated) if use_jax else propagated

class InSituBackpropLayer(hk.Module):
    def __init__(self, mesh: ForwardMesh, in_situ_matrix: callable = None,
                 activation: callable = None, name: str = None,
                 tap_pd_shot_noise: float = 0, io_amp_error_std: float = 0,
                 io_phase_error_std: float = 0, loss_db_std: float = 0, all_analog: bool = True,
                 set_gammas_zero: bool = True, use_jit: bool = True):
        """Mesh layer for backpropagation.

        Args:
            mesh: Feedforward mesh
            in_situ_matrix: Callable (use to avoid recompiling layers that do not need to be recompiled)
            activation: Activation
            name: Name of the mesh layer
            tap_pd_shot_noise: Tap photodetector shot noise (should be proportional to power)
            io_amp_error_std: Amplitude error (random, based on measurement)
            io_phase_error_std: Phase error (random, based on measurement)
            loss_db_std: Loss error variation among the components in the mesh
            all_analog: All-analog measurement
            use_jit: JIT compile (should be slow on the first step and fast after that.)

        """
        super().__init__(name=name)
        self.mesh: ForwardMesh = mesh
        self.output_size = self.n = mesh.n
        if set_gammas_zero:
            self.mesh.gammas = np.zeros_like(self.mesh.gammas)  # useful hack for this demo
        self.mesh = self.mesh.add_error_variance(0, loss_db_std)

        if in_situ_matrix is None:
            self.matrix = self.mesh.in_situ_matrix_fn(tap_pd_shot_noise, io_amp_error_std, io_phase_error_std, all_analog)
            self.matrix = jit(self.matrix) if use_jit else self.matrix
        else:
            self.matrix = in_situ_matrix

        self.activation = activation if activation is not None else (lambda x: x)

    def __call__(self, x):
        theta = hk.get_parameter("theta",
                                 shape=self.mesh.thetas.shape,
                                 init=hk.initializers.Constant(self.mesh.thetas))
        phi = hk.get_parameter("phi",
                               shape=self.mesh.phis.shape,
                               init=hk.initializers.Constant(self.mesh.phis))
        gamma = hk.get_parameter("gamma",
                                 shape=self.mesh.gammas.shape,
                                 init=hk.initializers.Constant(self.mesh.gammas))
        return self.activation((self.matrix((theta, phi, gamma)) @ x.T).T)

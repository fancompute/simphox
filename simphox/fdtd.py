from functools import lru_cache
from typing import Tuple, List, Callable

import jax.numpy as jnp
import numpy as np

from .parse import parse_excitation, parse_source_port
from .sim import SimGrid
from .typing import Array, Op, Shape, Spacing, Optional, Union, State, Size3, Size, Source
from .utils import pml_sigma, curl_pml_fn, yee_avg, yee_avg_jax, shift_slice

try:
    from dphox.pattern import Pattern
    DPHOX_INSTALLED = True
except ImportError:
    DPHOX_INSTALLED = False


class FDTD(SimGrid):
    """Stateless Finite Difference Time Domain (FDTD) implementation.

    The FDTD update consists of updating the fields and auxiliary vectors that comprise the system "state." This class
    ideally makes use of the jit capability of JAX.

    Attributes:
        size: size of the simulation
        spacing: spacing among the different dimensions
        eps: epsilon permittivity
        pml: perfectly matched layers (PML)
        pml_params: The PML parameters of the form :code:`(exp_scale, log_reflectivity, pml_eps)`.
        use_jax: Whether to use jax
        name: Name of the simulator
    """

    def __init__(self, size: Size, spacing: Spacing, eps: Union[float, np.ndarray] = 1,
                 pml: Optional[Union[Shape, Size, float]] = None, pml_params: Size3 = (3, -25, 1),
                 pml_sep: int = 5, use_jax: bool = True, name: str = 'fdtd'):
        super(FDTD, self).__init__(size, spacing, eps, pml=pml, pml_params=pml_params, pml_sep=pml_sep, name=name)
        self.dt = 1 / np.sqrt(np.sum(1 / self.spacing ** 2))  # includes courant condition!
        self.use_jax = use_jax
        self.xp = jnp if use_jax else np
        self.pml_regions = []
        self.sigma = None
        self.cpml_b, self.cpml_c = [], []
        self._curl_e_pml, self._curl_h_pml = [], []
        # pml (internal to the grid / does not affect params, so specified here!)
        if self.pml_shape is not None:
            self._set_pml(pml_params)
        self._curl_e = self.curl_fn(use_jax=self.use_jax)
        self._curl_h = self.curl_fn(of_h=True, use_jax=self.use_jax)

        # raise NotImplementedError("This class is still WIP")

    @classmethod
    def from_pattern(cls, component: "Pattern", core_eps: float, clad_eps: float, spacing: float, boundary: Size,
                     pml: float, component_t: float = 0, component_zmin: Optional[float] = None,
                     rib_t: float = 0, sub_z: float = 0, height: float = 0, bg_eps: float = 1, name: str = 'fdfd'):
        """Initialize an FDFD from a Pattern defined in DPhox.

        Args:
            component: pattern provided by DPhox
            core_eps: core epsilon (in the pattern mask region)
            clad_eps: clad epsilon
            spacing: spacing required
            boundary: boundary size around component
            pml: PML boundary size
            height: height for 3d simulation
            sub_z: substrate minimum height
            component_zmin: component height (defaults to substrate_z)
            component_t: component thickness
            rib_t: rib thickness for component (partial etch)
            bg_eps: background epsilon (usually 1 or air/vacuum)
            name: Name of the component

        Returns:
            A Grid object for the component

        """
        if not DPHOX_INSTALLED:
            raise ImportError('DPhox not installed, but it is required to run this function.')
        b = component.size
        x = b[0] + 2 * boundary[0]
        y = b[1] + 2 * boundary[1]
        component_zmin = sub_z if component_zmin is None else component_zmin
        spacing = spacing * np.ones(2 + (component_t > 0)) if isinstance(spacing, float) else np.asarray(spacing)
        size = (x, y, height) if height > 0 else (x, y)
        grid = cls(size, spacing, eps=bg_eps, pml=pml, name=name)
        grid.fill(sub_z + rib_t, core_eps)
        grid.fill(sub_z, clad_eps)
        grid.add(component, core_eps, component_zmin, component_t)
        return grid

    @property
    def zero_state(self) -> State:
        """Zero state, the default initial state for the FDTD

        Returns:
            Hidden state of the form:
                e: current :math:`\\mathbf{E}`

                h: current :math:`\\mathbf{H}`

                psi_e: current :math:`\\boldsymbol{\\Psi}_E` for CPML updates (otherwise :code:`None`)

                psi_h: current :math:`\\boldsymbol{\\Psi}_H` for CPML updates (otherwise :code:`None`)

        """
        # stored fields for fdtd
        e = self.xp.zeros(self.field_shape, dtype=np.complex128)
        h = self.xp.zeros_like(e)
        # for pml updates
        psi_e = None if self.pml_shape is None else [self.xp.zeros_like(self.xp.vstack([e[s], e[s]]))
                                                     for s in self.pml_regions]
        psi_h = None if self.pml_shape is None else [self.xp.zeros_like(p) for p in psi_e]
        return e, h, psi_e, psi_h

    def _step_e(self, state: State, sources: List[Tuple[np.ndarray, np.ndarray, Tuple[slice, ...]]]):
        e, h, psi_e, psi_h = state
        # update pml in pml regions if specified
        phi_e = []
        for pml_idx, pml_region in enumerate(self.pml_regions):
            psi_e[pml_idx], p = self._curl_h_pml[pml_idx](h, psi_e[pml_idx], self.cpml_b[pml_idx][1])
            phi_e.append(p)

        # update e field
        e += self._curl_h(h) / self.eps_t * self.dt

        for pml_idx, pml_region in enumerate(self.pml_regions):
            if self.use_jax:
                e = e.at[pml_region].add(phi_e[pml_idx] / self.eps_t[pml_region] * self.dt)
            else:
                e[pml_region] += phi_e[pml_idx] / self.eps_t[pml_region] * self.dt

        # add source
        for source, _, source_region in sources:
            if source is not None:
                if self.use_jax:
                    e = e.at[source_region].set(-source.squeeze() / self.eps_t[source_region] * self.dt)
                else:
                    e[source_region] -= source.squeeze() / self.eps_t[source_region] * self.dt

        return e, h, psi_e, psi_h

    def _step_h(self, state: State, sources: List[Source]):
        e, h, psi_e, psi_h = state
        # update h field in pml regions if specified
        phi_h = []
        for pml_idx, pml_region in enumerate(self.pml_regions):
            psi_h[pml_idx], p = self._curl_e_pml[pml_idx](e, psi_h[pml_idx], self.cpml_b[pml_idx][0])
            phi_h.append(p)

        # update h field
        h -= self._curl_e(e) * self.dt

        for pml_idx, pml_region in enumerate(self.pml_regions):
            if self.use_jax:
                h = h.at[pml_region].add(-phi_h[pml_idx] * self.dt)
            else:
                h[pml_region] -= phi_h[pml_idx] * self.dt

        # add source
        for _, source, source_region in sources:
            if source is not None:
                if self.use_jax:
                    h = h.at[source_region].set(-source.squeeze() * self.dt)
                else:
                    h[source_region] -= source.squeeze() * self.dt

        return e, h, psi_e, psi_h

    def step(self, state: State, sources: List[Source]) -> State:
        """FDTD step (in the form of an RNNCell)

        Notes:
            The updates are of the form:

            .. math::
                \\mathbf{E}(t + \\mathrm{d}t) &= \\mathbf{E}(t) + \\mathrm{d}t
                \\frac{\\mathrm{d}\\mathbf{E}}{\\mathrm{d}t}

                \\mathbf{H}(t + \\mathrm{d}t) &= \\mathbf{H}(t) +
                \\mathrm{d}t \\frac{\\mathrm{d}\\mathbf{H}}{\\mathrm{d}t}

            From Maxwell's equations, we have (for current source :math:`\\mathbf{J}(t)`):

            .. math::
                \\frac{\\mathrm{d}\\mathbf{E}}{\\mathrm{d}t} &= \\frac{1}{\\epsilon} \\nabla
                \\times \\mathbf{H}(t) + \\mathbf{J}(t)

                \\frac{\\mathrm{d}\\mathbf{H}}{\\mathrm{d}t} &= -\\frac{1}{\\mu} \\nabla \\times
                \\mathbf{E}(t) + \\mathbf{M}(t)

            The recurrent update assumes that :math:`\\mu = c = 1, \\mathbf{M}(t) = \\mathbf{0}` and factors in
            perfectly-matched layers (PML), which requires storing two additional PML arrays in the system's state
            vector, namely :math:`\\boldsymbol{\\Psi}_E(t)` and :math:`\\boldsymbol{\\Psi}_H(t)`.

            .. math::
                \\mathbf{\\Psi_E}^{(t+1/2)} &= \\mathbf{b} \\mathbf{\\Psi_E}^{(t-1/2)} +
                \\nabla_{\\mathbf{c}} \\times \\mathbf{H}^{(t)}

                \\mathbf{\\Psi_H}^{(t + 1)} &= \\mathbf{b} \\mathbf{\\Psi_H}^{(t)} +
                \\nabla_{\\mathbf{c}} \\times \\mathbf{E}^{(t)}

                \\mathbf{E}^{(t+1/2)} &= \\mathbf{E}^{(t-1/2)} + \\frac{\\Delta t}{\\epsilon} \\left(\\nabla \\times
                \\mathbf{H}^{(t)} + \\mathbf{J}^{(t)} + \\mathbf{\\Psi_E}^{(t+1/2)}\\right)

                \\mathbf{H}^{(t + 1)} &= \\mathbf{H}^{(t)} - \\Delta t \\left(\\nabla \\times \\mathbf{E}^{(t+1/2)} +
                \\mathbf{\\Psi_H}^{(t + 1)}\\right)


            Note, in Einstein notation, the weighted curl operator is given by:
            :math:`\\nabla_{\\mathbf{c}} \\times \\mathbf{v} := \\epsilon_{ijk} c_i \\partial_j v_k`.

        Args:
            state: current state of the form :code:`(e, h, psi_e, psi_h)` = :math:`(\\mathbf{E}(t),
            \\mathbf{H}(t), \\boldsymbol{\\Psi}_E(t), \\boldsymbol{\\Psi}_H(t))`.
            sources: The sources :math:`\\mathbf{J}_i(t)`, i.e. the input excitations to the system,
                and the corresponding slice or mask of the added source to be added to E in the update,
                which must be the same shape as :math:`\\mathbf{J}_i(t)`.

        Returns:
            a new :code:`State` of the form :code:`(e, h, psi_e, psi_h)` = :math:`(\\mathbf{E}(t),
            \\mathbf{H}(t), \\boldsymbol{\\Psi}_E(t), \\boldsymbol{\\Psi}_H(t))`.

        """

        state = self._step_e(state, sources)
        state = self._step_h(state, sources)
        return state

    def run_cw_port(self, num_time_steps: int, wavelength: float = 1.55,
                    source_port: Union[str, List[Tuple[str, int]]] = 'a0',
                    measure_port: Union[str, List[Tuple[str, int]]] = None,
                    tm_2d: bool = True, profile_size_factor: float = 3, pbar: Callable = None,
                    initial_state: Optional[State] = None, viz_pipes: Optional[dict] = None,
                    viz_interval: int = 1, viz_h: bool = True, viz_axis: int = 2):
        """Run the FDTD using an harmonic source with an eigenmode at a port at a specified wavelength.

        Args:
            num_time_steps: total time to run the simulation.
            wavelength: Wavelength of the CW harmonic source.
            source_port: The source port(s), default a0, generally considered to be specified default input port.
            measure_port: The measure port(s), measure at all ports if None.
            tm_2d: If 2D, use the TM mode, else use the TE mode. Ignore if 3D.
            profile_size_factor: profile size factor (multiply the port size to get profile sim region size)
            pbar: Progress bar handle (e.g. tqdm)
            initial_state: Initial state fot the FDTD (default is the zero state called by :code:`fdtd.initial_state()`)
            viz_pipes: Visualization streaming handle structure mapping port names to visualization pipes.
                This is useful for streaming live simulation results in a notebook! We assume multilayer device for now,
                which means the z-dimension is where we perform a cross section. However, you can look along another
                dimension (e.g. y) if you supply a tuple of the port and the dimension 0 or 1.
            viz_interval: Interval to send streaming data

        Returns:
            flux: The flux measurements (not averaged) as an array, which can be averaged during post-processing.
            state: final state of the form :code:`(e, h, psi_e, psi_h)`
                -:code:`e` refers to electric field :math:`\\mathbf{E}(t)`
                -:code:`h` refers to magnetic field :math:`\\mathbf{H}(t)`
                -:code:`psi_e` refers to :math:`\\boldsymbol{\\Psi}_E(t)` (for debugging PML)
                -:code:`psi_h` refers to :math:`\\boldsymbol{\\Psi}_H(t)` (for debugging PML)

        """
        source_excitation = list(parse_source_port(source_port).keys())
        source_modes = self.port_modes(excitation=source_excitation,
                                       profile_size_factor=profile_size_factor, wavelength=wavelength)
        measure_fn = self.get_measure_fn(measure_port, use_jax=self.use_jax,
                                         profile_size_factor=profile_size_factor, tm_2d=tm_2d)
        state = self.zero_state if initial_state is None else initial_state
        iterator = range(num_time_steps) if pbar is None else pbar(np.arange(num_time_steps))
        flux = [measure_fn(state[:2])]
        k0 = 2 * np.pi / wavelength
        for step in iterator:
            sources = []
            for p, midx in source_excitation:
                mode = source_modes[p]
                time_shift = np.exp(1j * step * self.dt * k0)
                src_e, src_h, src_slice = mode.profile(midx), mode.profile(midx, use_h=True), mode.slice(self)
                sources.append((src_e * time_shift, src_h * time_shift, src_slice))
            _, h_before, _, _ = state
            state = self.step(state, sources)
            e, h, _, _ = state
            synchronized_fields = np.stack((e, (h_before + h) / 2))
            flux.append(measure_fn(synchronized_fields))
            if viz_pipes and step % viz_interval == 0:
                self._viz(viz_pipes, e, h, viz_h, viz_axis)
        return flux, state

    def _viz(self, viz_pipes: dict, e: Array, h: Array, viz_h: bool, viz_axis: int):
        """Visualize the fields."""
        for port_name in viz_pipes:
            eps_pipe, field_pipe, power_pipe = viz_pipes[port_name]
            if self.ndim == 3:
                port_name = (port_name, 2) if not isinstance(port_name, tuple) else port_name
                idx = int(np.around(self.port[port_name[0]].xyz[port_name[1]] / self.spacing[port_name[1]]))
                eps_slice = tuple([idx if ax == port_name[1] else slice(None) for ax in range(3)])
                eps = self.eps[eps_slice].T
                f = np.array(h[(slice(None), *eps_slice)] if viz_h else e[(slice(None), *eps_slice)])
            else:
                eps = self.eps.T
                f = np.array(h if viz_h else e)
            eps_pipe.send((eps - np.min(eps)) / (np.max(eps) - np.min(eps)))
            field_pipe.send(f[viz_axis].T.real / np.max(f[viz_axis].T.real + np.spacing(1)))
            power = np.abs(f[viz_axis].T) ** 2
            power_pipe.send(power / np.max(power + np.spacing(1)))

    def _set_pml(self, pml_params: Size3):
        exp_scale, log_reflection, absorption_corr = pml_params
        kappa, alpha = 1, 1e-8  # TODO: make these params
        self.sigma = [-pml_sigma(self.pos[ax], thickness=self.pml_shape[ax], exp_scale=exp_scale,
                                 log_reflection=log_reflection, absorption_corr=absorption_corr) for ax in range(3)]
        # for memory and time purposes, we only update the pml slices, NOT the full field
        # therefore, we need to specify the pml regions for the fields.
        self.pml_regions = ((slice(None), slice(None, self.pml_shape[0]), slice(None), slice(None)),
                            (slice(None), slice(-self.pml_shape[0], None), slice(None), slice(None)),
                            (slice(None), slice(None), slice(None, self.pml_shape[1]), slice(None)),
                            (slice(None), slice(None), slice(-self.pml_shape[1], None), slice(None)),
                            (slice(None), slice(None), slice(None), slice(None, self.pml_shape[2])),
                            (slice(None), slice(None), slice(None), slice(-self.pml_shape[2], None)))

        for i, region in enumerate(self.pml_regions):
            ax = i // 2
            if self.pml_shape[ax]:
                pml_slice = tuple([None if idx == slice(None) else idx for idx in region[1:]])
                pml_shape = np.array((2,) + self.field_shape)
                pml_shape[ax + 2] = self.pml_shape[ax]
                sigma_ax = np.zeros(pml_shape, dtype=np.complex128)
                sigma_ax[0, ax] = self.sigma[ax][0][pml_slice]
                sigma_ax[1, ax] = self.sigma[ax][1][pml_slice]
                self.cpml_b.append(np.exp(-(alpha + sigma_ax / kappa) * self.dt))
                self.cpml_c.append((self.cpml_b[-1] - 1) * sigma_ax / (sigma_ax * kappa + alpha * kappa ** 2))
        self._curl_h_pml = [self.curl_h_pml(pml_idx) for pml_idx in range(len(self.pml_regions))]
        self._curl_e_pml = [self.curl_e_pml(pml_idx) for pml_idx in range(len(self.pml_regions))]

    def curl_e_pml(self, pml_idx: int) -> Callable[[Array, Array, Array], Array]:
        dx, _ = self._dxes
        c, s = self.cpml_c[pml_idx][0], self.pml_regions[pml_idx][1:]
        de = lambda e, ax: c[ax] * (self.xp.roll(e, -1, axis=ax)[s] - e[s]) / dx[ax][s]
        return curl_pml_fn(de, use_jax=self.use_jax)

    def curl_h_pml(self, pml_idx: int) -> Callable[[Array, Array, Array], Array]:
        _, dx = self._dxes
        c, s = self.cpml_c[pml_idx][1], self.pml_regions[pml_idx][1:]
        dh = lambda h, ax: c[ax] * (h[s] - self.xp.roll(h, 1, axis=ax)[s]) / dx[ax][s]
        return curl_pml_fn(dh, use_jax=self.use_jax)

    @property
    @lru_cache()
    def eps_t(self):
        """The epsilon tensor (assumed to be diagonal, i.e. no off-diagonal components for now).

        Returns:

        """
        yee = yee_avg_jax if self.use_jax else yee_avg
        eps_t = yee(self.xp.array(self.eps.reshape(self.shape3)))
        return eps_t

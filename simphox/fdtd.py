from functools import lru_cache
from typing import Tuple, List, Callable

import jax.numpy as jnp
import numpy as np

from .grid import YeeGrid
from .typing import Shape, Spacing, Optional, Union, Source, State, Size3, Size
from .utils import pml_params, curl_fn, yee_avg


class FDTD(YeeGrid):
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
                 pml: Optional[Union[Shape, Size]] = None, pml_params: Size3 = (3, -35, 1),
                 pml_sep: int = 5, use_jax: bool = True, name: str = 'fdtd'):
        super(FDTD, self).__init__(size, spacing, eps, pml=pml, pml_params=pml_params, pml_sep=pml_sep, name=name)
        self.dt = 1 / np.sqrt(np.sum(1 / self.spacing ** 2))  # includes courant condition!

        # pml (internal to the grid / does not affect params, so specified here!)
        if self.pml_shape is not None:
            b, c = zip(*[self._cpml(ax) for ax in range(3)])
            b_e, c_e = [b[ax][0] for ax in range(3)], [c[ax][0] for ax in range(3)]
            b_h, c_h = [b[ax][1] for ax in range(3)], [c[ax][1] for ax in range(3)]
            b_e, c_e = np.asarray(np.meshgrid(*b_e, indexing='ij')), np.asarray(np.meshgrid(*c_e, indexing='ij'))
            b_h, c_h = np.asarray(np.meshgrid(*b_h, indexing='ij')), np.asarray(np.meshgrid(*c_h, indexing='ij'))
            # for memory and time purposes, we only update the pml slices, NOT the full field
            self.pml_regions = ((slice(None), slice(None, self.pml_shape[0]), slice(None), slice(None)),
                                (slice(None), slice(-self.pml_shape[0], None), slice(None), slice(None)),
                                (slice(None), slice(None), slice(None, self.pml_shape[1]), slice(None)),
                                (slice(None), slice(None), slice(-self.pml_shape[1], None), slice(None)),
                                (slice(None), slice(None), slice(None), slice(None, self.pml_shape[2])),
                                (slice(None), slice(None), slice(None), slice(-self.pml_shape[2], None)))
            self.cpml_be, self.cpml_bh = [b_e[s] for s in self.pml_regions], [b_h[s] for s in self.pml_regions]
            self.cpml_ce, self.cpml_ch = [c_e[s] for s in self.pml_regions], [c_h[s] for s in self.pml_regions]
            if use_jax:
                self.cpml_be, self.cpml_bh = [jnp.asarray(v) for v in self.cpml_be], \
                                             [jnp.asarray(v) for v in self.cpml_bh]
                self.cpml_ce, self.cpml_ch = [jnp.asarray(v) for v in self.cpml_ce], \
                                             [jnp.asarray(v) for v in self.cpml_ch]
            self.xp = jnp if use_jax else np
            self.use_jax = use_jax
            self._curl_h_pml = [self.curl_h_pml(pml_idx) for pml_idx in range(len(self.pml_regions))]
            self._curl_e_pml = [self.curl_e_pml(pml_idx) for pml_idx in range(len(self.pml_regions))]
        self._curl_e = self.curl_fn(use_jax=self.use_jax)
        self._curl_h = self.curl_fn(of_h=True, use_jax=self.use_jax)

        raise NotImplementedError("This class is still WIP")

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
        psi_e = None if self.pml_shape is None else [self.xp.zeros_like(e[s]) for s in self.pml_regions]
        psi_h = None if self.pml_shape is None else [self.xp.zeros_like(e[s]) for s in self.pml_regions]
        return e, h, psi_e, psi_h

    def step(self, state: State, src: np.ndarray, src_region: np.ndarray) -> State:
        """FDTD step (in the form of an RNNCell)

        Notes:
            The updates are of the form:

            .. math::
                \\mathbf{E}(t + \\mathrm{d}t) &= \\mathbf{E}(t) + \\mathrm{d}t
                \\frac{\\mathrm{d}\\mathbf{E}}{\\mathrm{d}t}

                \\mathbf{H}(t + \\mathrm{d}t) &= \\mathbf{H}(t) +
                \\mathrm{d}t \\frac{\\mathrm{d}\\mathbf{H}}{\\mathrm{d}t}

            From Maxwell's equations, we have (for current source :math:`\mathbf{J}(t)`):

            .. math::
                \\frac{\\mathrm{d}\\mathbf{E}}{\\mathrm{d}t} = \\frac{1}{\\epsilon} \\nabla
                \\times \\mathbf{H}(t) + \\mathbf{J}(t)

                \\frac{\\mathrm{d}\\mathbf{H}}{\\mathrm{d}t} = -\\frac{1}{\\mu} \\nabla \\times
                \\mathbf{E}(t) + \\mathbf{M}(t)

            The recurrent update assumes that :math:`\\mu = c = 1, \\mathbf{M}(t) = \\mathbf{0}` and factors in
            perfectly-matched layers (PML), which requires storing two additional PML arrays in the system's state
            vector, namely :math:`\\boldsymbol{\\Psi}_E(t)` and :math:`\\boldsymbol{\\Psi}_H(t)`.

            .. math::
                \\mathbf{\\Psi_E}^{(t+1/2)} = \\mathbf{b} \\mathbf{\\Psi_E}^{(t-1/2)} +
                \\nabla_{\\mathbf{c}} \\times \\mathbf{H}^{(t)}

                \\mathbf{\\Psi_H}^{(t + 1)} = \\mathbf{b} \mathbf{\\Psi_H}^{(t)} +
                \\nabla_{\\mathbf{c}} \\times \\mathbf{E}^{(t)}

                \\mathbf{E}^{(t+1/2)} = \\mathbf{E}^{(t-1/2)} + \\frac{\\Delta t}{\\epsilon} \\left(\\nabla \\times
                \\mathbf{H}^{(t)} + \\mathbf{J}^{(t)} + \mathbf{\Psi_E}^{(t+1/2)}\\right)

                \\mathbf{H}^{(t + 1)} = \\mathbf{H}^{(t)} - \\Delta t \\left(\\nabla \\times \\mathbf{E}^{(t+1/2)} +
                \\mathbf{\\Psi_H}^{(t + 1)}\\right)


            Note, in Einstein notation, the weighted curl operator is given by:
            :math:`\\nabla_{\\mathbf{c}} \\times \\mathbf{v} := \\epsilon_{ijk} c_i \\partial_j v_k`.

        Args:
            state: current state of the form :code:`(e, h, psi_e, psi_h)` = :math:`(\\mathbf{E}(t),
            \\mathbf{H}(t), \\boldsymbol{\\Psi}_E(t), \\boldsymbol{\\Psi}_H(t))`.
            src: The source :math:`\\mathbf{J}(t)`, i.e. the input excitation to the system.
            src_region: slice or mask of the added source to be added to E in the update (assume same shape as :code:`e`
                if :code:`None`)

        Returns:
            a new :code:`State` of the form :code:`(e, h, psi_e, psi_h)` = :math:`(\\mathbf{E}(t),
            \\mathbf{H}(t), \\boldsymbol{\\Psi}_E(t), \\boldsymbol{\\Psi}_H(t))`.

        """
        e, h, psi_e, psi_h = state
        src_region = tuple([slice(None)] * 4) if src_region is None else src_region

        # update pml in pml regions if specified
        for pml_idx, region in enumerate(self.pml_regions):
            e_pml, h_pml = e[self.pml_regions[pml_idx]], h[self.pml_regions[pml_idx]]
            psi_e[pml_idx] = self.cpml_be[pml_idx] * psi_e[pml_idx] + self._curl_h_pml[pml_idx](h_pml)
            psi_h[pml_idx] = self.cpml_bh[pml_idx] * psi_h[pml_idx] + self._curl_e_pml[pml_idx](e_pml)

        # add source
        src = src.flatten() if isinstance(src_region, np.ndarray) else src.squeeze()
        e[src_region] += src * self.dt / self.eps_t[src_region]

        # update e field
        e += self._curl_h(h) / self.eps_t * self.dt
        for pml_idx, region in enumerate(self.pml_regions):
            e[region] += psi_e[pml_idx] / self.eps_t[region] * self.dt

        # update h field
        h -= self._curl_e(e) * self.dt
        for pml_idx, region in enumerate(self.pml_regions):
            h[region] -= psi_h[pml_idx] * self.dt

        return e, h, psi_e, psi_h

    def run(self, src: Source, src_idx: Union[np.ndarray, List[np.ndarray]], num_time_steps: int,
            pbar: Callable = None, initial_state: Optional[State] = None):
        """Run the FDTD

        Args:
            src: a function that provides the input source, or an :code:`ndarray` where :code:`src[time_step]`
                gives the source at that time step
            src_idx: source location in the grid, provided by :code:`src_idx`
            num_time_steps: total time to run the simulation
            pbar: Progress bar handle (e.g. tqdm)
            initial_state: Initial state fot the FDTD (default is the zero state called by :code:`fdtd.initial_state()`)

        Returns:
            state: final state of the form :code:`(e, h, psi_e, psi_h)`
                -:code:`e` refers to electric field :math:`\\mathbf{E}(t)`
                -:code:`h` refers to magnetic field :math:`\\mathbf{H}(t)`
                -:code:`psi_e` refers to :math:`\\boldsymbol{\\Psi}_E(t)` (for debugging PML)
                -:code:`psi_h` refers to :math:`\\boldsymbol{\\Psi}_H(t)` (for debugging PML)

        """
        state = self.zero_state if initial_state is None else initial_state
        iterator = range(num_time_steps) if pbar is None else pbar(np.arange(num_time_steps))
        for step in iterator:
            source = src[step] if isinstance(src, np.ndarray) else src(step * self.dt)
            source_idx = src_idx[step] if isinstance(src_idx, list) else src_idx
            state = self.step(state, source, source_idx)
        return state

    def _cpml(self, ax: int, alpha_max: float = 0, exp_scale: float = 3,
              kappa: float = 1, log_reflection: float = 35) -> Tuple[np.ndarray, np.ndarray]:
        if self.cells[ax].size == 1:
            return np.ones(2), np.ones(2)
        sigma, alpha = pml_params(self.pos[ax], t=self.pml_shape[ax], exp_scale=exp_scale,
                                  log_reflection=-log_reflection, absorption_corr=1)
        alpha *= alpha_max  # alpha_max recommended to be np.pi * central_wavelength / 5 if nonzero
        b = np.exp(-(alpha + sigma / kappa) * self.dt)
        factor = sigma / (sigma + alpha * kappa) if alpha_max > 0 else 1
        return b, (b - 1) * factor / kappa

    def curl_e_pml(self, pml_idx: int) -> Callable[[np.ndarray], np.ndarray]:
        dx, _ = self._dxes
        c, s = self.cpml_ce[pml_idx], self.pml_regions[pml_idx][1:]
        de = lambda e_, ax: (self.xp.roll(e_, -1, axis=ax) - e_) / dx[ax][s] * c[ax]
        return curl_fn(de, use_jax=self.use_jax)

    def curl_h_pml(self, pml_idx: int) -> Callable[[np.ndarray], np.ndarray]:
        _, dx = self._dxes
        c, s = self.cpml_ch[pml_idx], self.pml_regions[pml_idx][1:]
        dh = lambda h_, ax: (h_ - self.xp.roll(h_, 1, axis=ax)) / dx[ax][s] * c[ax]
        return curl_fn(dh, use_jax=self.use_jax)

    @property
    @lru_cache()
    def eps_t(self):
        eps_t = yee_avg(self.eps)
        return self.xp.asarray(eps_t)

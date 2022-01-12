import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.config import config
from jax.scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import eigs

from .primitives import spsolve, TMOperator
from .sim import SimGrid
from .typing import Callable, Dict, List, Optional, Shape2, Size, Size2, Size3, Spacing, SpSolve, Tuple, Union
from .utils import curl_fn, d2curl_op, yee_avg_2d_z, yee_avg_jax

try:  # pardiso (using Intel MKL) is much faster than scipy's solver
    from .mkl import spsolve_pardiso, feast_eigs
except OSError:  # if mkl isn't installed
    from scipy.sparse.linalg import spsolve

try:
    from dphox.pattern import Pattern
    DPHOX_INSTALLED = True
except ImportError:
    DPHOX_INSTALLED = False

from logging import getLogger

logger = getLogger()
config.parse_flags_with_absl()


class FDFD(SimGrid):
    """Finite Difference Frequency Domain (FDFD) simulator

    Notes:
        Finite difference frequency domain works by performing a linear solve of discretized Maxwell's equations
        at a `single` frequency (wavelength).

        The discretized version of Maxwell's equations in frequency domain is:

        .. math::
            \\nabla \\times \\mu^{-1} \\nabla \\times \\mathbf{e} - k_0^2 \\epsilon \\mathbf{e} = k_0 \\mathbf{j},

        which can be written in the form :math:`A \\mathbf{e} = \\mathbf{b}`, where:

        .. math::
            A &= \\nabla \\times \\mu^{-1} \\nabla \\times - k_0^2 \\epsilon

            \\mathbf{b} &= k_0 \\mathbf{j}

        is an operator representing the discretized EM wave operator at frequency
        :math:`\\omega = k_0 = \\frac{2\\pi}{\\lambda}`.

        Therefore, :math:`\\mathbf{e} = A^{-1}\\mathbf{b}`.

        For 2D simulations, it can be more efficient to solve for just the :math:`z`-component of the fields since
        only :math:`\\mathbf{e}_z` is non-zero. In this case, we can solve a smaller problem to improve the efficiency.
        The form of this problem is :math:`A_z \\mathbf{e}_z = \\mathbf{b}_z`, where:

        .. math::
            A &= (\\nabla \\times \\mu^{-1} \\nabla \\times)_z + k_0^2 \\epsilon_z

            \\mathbf{b}_z &= k_0 \\mathbf{j}_z

    Attributes:
        size: Tuple of size 1, 2, or 3 representing the size in the grid in arbitrary units
        spacing: Spacing (microns) between each pixel along each axis (must be same dim as :code:`size`)
        eps: Relative permittivity :math:`\\epsilon_r`
        bloch_phase: Bloch phase (generally useful for angled scattering sims, not yet implemented!)
        pml: Perfectly matched layer (PML) of thickness on both sides of the form :code:`(x_pml, y_pml, z_pml)`
        pml_sep: The PML separation distance in number of pixels for sources
        pml_params: The PML parameters of the form :code:`(exp_scale, log_reflectivity, pml_eps)`.
    """
    def __init__(self, size: Size, spacing: Spacing,
                 wavelength: float = 1.55, eps: Union[float, np.ndarray] = 1,
                 bloch_phase: Union[Size, float] = 0.0, pml: Optional[Union[Size, float]] = None,
                 pml_sep: int = 5, pml_params: Size3 = (4, -16, 1.0), name: str = 'fdfd'):

        super(FDFD, self).__init__(
            size=size,
            spacing=spacing,
            eps=eps,
            bloch_phase=bloch_phase,
            pml=pml,
            pml_params=pml_params,
            pml_sep=pml_sep,
            name=name
        )

        self.wavelength = wavelength
        self.k0 = 2 * np.pi / self.wavelength

        # overwrite dxes with PML-scaled ones if specified
        if self.pml_shape is not None:
            dxes_pml_e, dxes_pml_h = [], []
            for ax, p in enumerate(self.pos):
                scpml_e, scpml_h = self.scpml(ax)
                dxes_pml_e.append(self.cells[ax] * scpml_e)
                dxes_pml_h.append(self.cells[ax] * scpml_h)
            self._dxes = np.meshgrid(*dxes_pml_e, indexing='ij'), np.meshgrid(*dxes_pml_h, indexing='ij')

    @classmethod
    def from_simgrid(cls, simgrid: SimGrid, wavelength: float):
        """Prepare an :code:`FDFD` instance from a generic :code:`SimGrid` and wavelength :math:`\\lambda`.

        Args:
            simgrid: :code:`SimGrid` instance.
            wavelength: Wavelength (:math:`\\lambda`).

        Returns:
            The :code:`FDFD` instance

        """
        fdfd = cls(
            size=simgrid.size,
            spacing=simgrid.spacing,
            wavelength=wavelength,
            eps=simgrid.eps,
            pml=simgrid.pml_shape,
            name=simgrid.name
        )
        fdfd.port = simgrid.port
        return fdfd

    @property
    def mat(self) -> sp.csr_matrix:
        """Build the discrete Maxwell operator :math:`A(k_0)` acting on :math:`\\mathbf{e}`.

        Returns:
            Electric field operator :math:`A` for solving Maxwell's equations at frequency :math:`omega`.
        """
        curl_curl: sp.spmatrix = d2curl_op(self.deriv_backward) @ d2curl_op(self.deriv_forward)
        curl_curl.sort_indices()  # for the solver
        mat = curl_curl - self.k0 ** 2 * sp.diags(self.eps_t.flatten())
        return mat

    A = mat  # alias A (common symbol for FDFD matrix) to mat

    @property
    def mat_ez(self) -> sp.csr_matrix:
        """Build the discrete Maxwell operator :math:`A_z(k_0)` acting on :math:`\\mathbf{e}_z` (for 1D/2D problems).

        Returns:
            Electric field operator :math:`A_z` for a source with ez-polarized field.
        """
        df, db = self.deriv_forward, self.deriv_backward
        ddz = -db[0] @ df[0] - db[1] @ df[1]
        ddz.sort_indices()  # for the solver
        mat = ddz - self.k0 ** 2 * sp.diags(self.eps_t[2].flatten())
        return mat

    @property
    def mat_hz(self) -> sp.csr_matrix:
        """Build the discrete Maxwell operator :math:`A_z(k_0)` acting on :math:`\\mathbf{h}_z` (for 2D problems).

        Returns:
            Magnetic field operator :math:`A_z` for a source with hz-polarized field.
        """
        df, db = self.deriv_forward, self.deriv_backward
        t0, t1 = sp.diags(1 / self.eps_t[0].flatten()), sp.diags(1 / self.eps_t[1].flatten())
        mat = -db[0] @ t1 @ df[0] - db[1] @ t0 @ df[1] - self.k0 ** 2 * sp.identity(self.n)
        return mat

    def e2h(self, e: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """Convert magnetic field :math:`\\mathbf{e}` to electric field :math:`\\mathbf{h}`.

        Usage is: :code:`h = fdfd.e2h(e)`, where :code:`e` is grid-shaped (not flattened). If :code:`e` is flattened,
        this function attempts to reshape it.

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            i \\omega \\mu \\mathbf{h} &= \\nabla \times \\mathbf{e}.

        Returns:
            The h-field converted from the e-field.
        """
        return self.curl_fn(of_h=False, beta=beta)(self.reshape(e)) / (1j * self.k0)

    def h2e(self, h: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """Convert magnetic field :math:`\\mathbf{h}` to electric field :math:`\\mathbf{e}`.

        Usage is: :code:`e = fdfd.h2e(h)`, where :code:`h` is grid-shaped (not flattened). If :code:`h` is flattened,
        this function attempts to reshape it.

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            -i \\omega \\epsilon \\mathbf{e} &= \\nabla \\times \\mathbf{h}.

        Returns:
            The e-field converted from the h-field.
        """
        return self.curl_fn(of_h=True, beta=beta)(self.reshape(h)) / (1j * self.k0 * self.eps_t)

    def solve(self, src: np.ndarray, solver_fn: Optional[SpSolve] = None,
              iterative: int = -1, tm_2d: bool = True, callback: Optional[Callable] = None) -> np.ndarray:
        """Solves the FDFD problem (see class description for math).

        Args:
            src: normalized source (can be wgm or tfsf)
            solver_fn: any function that performs a sparse linalg solve
            iterative: default = -1, direct = 0, gmres = 1, bicgstab
            tm_2d: use the TM polarization (only relevant for 2D, ignored for 3D)
            callback: a function to run during the solve (only applies in 3d iterative solver case, not yet implemented)

        Returns:
            Electric fields that solve the problem :math:`A\\mathbf{e} = \\mathbf{b} = i \\omega \\mathbf{j}`.

        """
        b = self.k0 * src.flatten()
        if self.ndim == 3:
            if not src.size == 3 * self.n:
                raise ValueError(f'Expected src.size == {3 * self.n}, but got {b.size}.')
            if iterative > 0 and solver_fn is None and self.ndim == 3:
                # use iterative solver for 3d sims by default
                M = sp.linalg.LinearOperator(self.mat.shape, sp.linalg.spilu(self.mat).solve)
                e, _ = sp.linalg.gmres(self.mat, b, M=M) if iterative == 1 else sp.linalg.bicgstab(self.mat, b, M=M)
            else:
                e = solver_fn(self.mat, b) if solver_fn else spsolve_pardiso(self.mat, b)
            e = self.reshape(e)
            curl_e = curl_fn(self.diff_fn(of_h=False))
            h = curl_e(e) / (1j * self.k0)
            return np.array((e, h))
        else:  # assume only the z component
            if not src.size == self.n:
                raise ValueError(f'Expected src.size == {self.n}, but got {b.size}.')
            mat = self.mat_hz if tm_2d else self.mat_ez
            fz = solver_fn(mat, b) if solver_fn else spsolve_pardiso(mat, b)
            o = np.zeros_like(fz)
            field = np.vstack((o, o, fz)).reshape((3, *self.shape3))
            df = self.diff_fn(of_h=tm_2d, use_jax=False)
            eps_t = self.eps_t
            if tm_2d:
                h = field
                o = np.zeros_like(h[2])
                return np.stack([np.stack((df(h[2], 1), -df(h[2], 0), o)) / (1j * self.k0 * eps_t), h])
            else:
                e = field
                o = np.zeros_like(e[2])
                return np.stack([e, np.stack((df(e[2], 1), -df(e[2], 0), o)) / (1j * self.k0)])

    def scpml(self, ax: int) -> Tuple[np.ndarray, np.ndarray]:
        exp_scale, log_reflection, pml_eps = self.pml_params
        if self.cells[ax].size == 1:
            return np.ones(1), np.ones(1)
        p = self.pos[ax]
        pe, ph = (p[:-1] + p[1:]) / 2, p[:-1]
        absorption_corr = self.k0 * pml_eps
        t = self.pml_shape[ax]

        def _scpml(d: np.ndarray):
            d_pml = np.hstack((
                (d[t] - d[:t]) / (d[t] - p[0]),
                np.zeros_like(d[t:-t]),
                (d[-t:] - d[-t]) / (p[-1] - d[-t])
            ))
            return 1 + 1j * (exp_scale + 1) * (d_pml ** exp_scale) * log_reflection / (2 * absorption_corr)

        return _scpml(pe), _scpml(ph)

    @classmethod
    def from_pattern(cls, component: "Pattern", core_eps: float, clad_eps: float, spacing: float, boundary: Size,
                     pml: float, wavelength: float, component_t: float = 0, component_zmin: Optional[float] = None,
                     rib_t: float = 0, sub_z: float = 0, height: float = 0, bg_eps: float = 1, name: str = 'fdfd'):
        """Initialize an FDFD from a Pattern defined in DPhox.

        Args:
            component: component provided by DPhox
            core_eps: core epsilon (in the pattern region)
            clad_eps: clad epsilon
            spacing: spacing required
            boundary: boundary size around component
            pml: PML boundary size
            wavelength: Wavelength for the simulation (specific to FDFD).
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
        grid = cls(size, spacing, wavelength=wavelength, eps=bg_eps, pml=pml, name=name)
        grid.fill(sub_z + rib_t, core_eps)
        grid.fill(sub_z, clad_eps)
        grid.add(component, core_eps, component_zmin, component_t)
        return grid

    def sparams(self, port_name: str, mode_idx: int = 0, measure_info: Optional[Dict[str, List[int]]] = None):
        """Measure sparams using a port profile provided for a given port and mode index.

        Args:
            port_name: The source port name for the sparams to measure.
            mode_idx: Mode index to access for the source.
            measure_info: A list of :code:`port_name`, :code:`mode_idx` to get mode measurements at each port.

        Returns:
            Sparams measured at the ports specified in the class.

        """
        measure_fn = self.get_measure_fn(measure_info)
        measure_info = [(name, 0) for name in self.port] if measure_info is None else measure_info
        h = self.solve(self.port_source({(port_name, mode_idx): 1}))
        s_out, s_in = measure_fn(h)
        src_sparam_reference = measure_info.index((port_name, mode_idx))
        return s_out / s_in[src_sparam_reference]

    def get_fields_fn(self, src: np.ndarray, transform_fn: Optional[Callable] = None, tm_2d: bool = True) -> Callable:
        """Build a fields function of a set of parameters (e.g., density, epsilon, etc.)
        given the source and transform function.

        1. A numpy array source :code:`src`.
        2. The JAX-transformable transform function :code:`transform_fn` (e.g. transform) that yields
           the epsilon distribution used by jax.

        Args:
            src: Source for the solver.
            transform_fn: Transforms parameters to yield the epsilon parameters used by jax (if None, use identity).
            tm_2d: Whether to solve the TM polarization for this FDFD (only relevant for 2D, ignored for 3D).

        Returns:
            A solve function (2d or 3d based on defined :code:`ndim` specified for the instance of :code:`FDFD`).

        """
        src = jnp.ravel(jnp.array(src))
        k0 = self.k0
        transform_fn = transform_fn if transform_fn is not None else lambda x: x
        shape3 = self.shape3
        field_shape = self.field_shape

        def coo_to_jnp(mat: sp.coo_matrix):
            mat.sort_indices()
            mat = mat.tocoo()
            return jnp.array(mat.data, dtype=np.complex128), jnp.vstack((jnp.array(mat.row), jnp.array(mat.col)))

        if self.ndim < 3:
            shape = self.shape
            o = jnp.zeros(self.shape3, jnp.complex128)
            if tm_2d:
                # exact 2d FDFD for TM polarization
                constant_term = -jnp.ones_like(self.eps.flatten()) * k0 ** 2
                constant_term_indices = jnp.stack((jnp.arange(self.n), jnp.arange(self.n)))

                # this is temporary while we wait for sparse-sparse support in JAX.
                operator = TMOperator(self.deriv_forward, self.deriv_backward)
                x_op = operator.compile_operator_along_axis(0)
                y_op = operator.compile_operator_along_axis(1)
                x_ind, y_ind = operator.x_indices, operator.y_indices
                dh = self.diff_fn(of_h=True, use_jax=True)

                # @jax.jit
                def solve(rho: jnp.ndarray):
                    eps_t = yee_avg_jax(transform_fn(rho).reshape(self.shape3))
                    eps_x, eps_y = jnp.ravel(eps_t[0]), jnp.ravel(eps_t[1])
                    ddx_entries = x_op(-1 / eps_y)
                    ddy_entries = y_op(-1 / eps_x)
                    mat_entries = jnp.hstack((constant_term, ddx_entries, ddy_entries))
                    hz = spsolve(mat_entries, k0 * src, jnp.hstack((constant_term_indices, x_ind, y_ind)))
                    hz = hz.reshape(shape3)
                    h = jnp.stack((o, o, hz))
                    e = jnp.stack((dh(h[2], 1), -dh(h[2], 0), o)) / (1j * k0 * eps_t)
                    return jnp.stack((e, h))

            else:
                # exact 2d FDFD for TE polarization
                df, db = self.deriv_forward, self.deriv_backward
                ddz = -db[0] @ df[0] - db[1] @ df[1]
                ddz_entries, ddz_indices = coo_to_jnp(ddz)
                mat_indices = jnp.hstack((jnp.vstack((jnp.arange(self.n), jnp.arange(self.n))), ddz_indices))
                de = self.diff_fn(of_h=False, use_jax=True)

                # @jax.jit
                def solve(rho: jnp.ndarray):
                    eps = yee_avg_2d_z(transform_fn(rho).reshape(shape3)).ravel()
                    mat_entries = jnp.hstack((-eps * k0 ** 2, ddz_entries))
                    ez = spsolve(mat_entries, k0 * src, mat_indices)
                    ez = ez.reshape(shape3)
                    e = jnp.stack((o, o, ez))
                    h = jnp.stack((de(e[2], 1), -de(e[2], 0), o)) / (1j * k0)
                    return jnp.stack((e, h))
        else:
            # iterative 3d FDFD (simpler than 2D code-wise, but takes way more memory and time)
            curl_e = curl_fn(self.diff_fn(of_h=False, use_jax=True), use_jax=True)
            curl_h = curl_fn(self.diff_fn(of_h=True, use_jax=True), use_jax=True)

            def op(eps: jnp.ndarray):
                return lambda b: curl_h(curl_e(b.reshape(field_shape))) - k0 ** 2 * eps * b.reshape(field_shape)

            # @jax.jit
            def solve(rho: jnp.ndarray):
                eps = yee_avg_jax(transform_fn(rho))
                e, _ = bicgstab(op(eps), k0 * src.reshape(field_shape))
                e = jnp.stack([split_v.reshape(shape3) for split_v in jnp.split(e, 3)])
                h = curl_e(e) / (1j * k0)
                return jnp.stack((e, h))

        return solve

    def to_2d(self, wavelength: float = None, slab_x: Union[Shape2, Size2] = None,
              slab_y: Union[Shape2, Size2] = None, tm: bool = False):
        """Project a 3D FDFD into a 2D FDFD using the variational 2.5D method laid out in the
        [paper](https://ris.utwente.nl/ws/files/5413011/ishpiers09.pdf).

        Args:
            wavelength: The wavelength to use for calculating the effective 2.5 FDFD
                (useful to stabilize multi-wavelength optimizations)
            slab_x: Port location x (if None, the port is provided by reading the port location specified by the component)
            slab_y: Port location y (if None, the port is provided by reading the port location specified by the component)
            tm: Whether the mode in the 2D simulation is a TM mode (dominated by Hz component).

        Returns:
            A 2D FDFD to approximate the 3D FDFD

        """
        wavelength = self.wavelength if wavelength is None else wavelength
        return FDFD.from_simgrid(super(FDFD, self).to_2d(wavelength, tm=tm), wavelength)

    def tfsf_profile(self, q_mask: np.ndarray, k: Size, wavelength: float = None):
        mask = q_mask
        q = sp.diags(mask.flatten())
        wavelength = self.wavelength if wavelength is None else wavelength
        k0 = 2 * np.pi / wavelength
        k = np.asarray(k) / np.sum(k) * k0
        fsrc = np.einsum('i,j,k->ijk', np.exp(1j * self.pos[0][:-1] * k[0]),
                         np.exp(1j * self.pos[1][:-1] * k[1]),
                         np.exp(1j * self.pos[2][:-1] * k[2])).flatten()
        a = self.mat
        src = self.reshape((q @ a - a @ q) @ fsrc)  # qaaq = quack :)
        raise NotImplementedError('TFSF profile not yet implemented.')
        return src

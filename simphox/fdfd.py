from .sim import SimGrid
from .utils import d2curl_op, yee_avg_jax
from .typing import Shape, Dim, GridSpacing, Optional, Tuple, Union, SpSolve, Shape2, Dim2, List, Callable, Dict

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

import jax
import jax.numpy as jnp
from jax.config import config

from jax.scipy.sparse.linalg import bicgstab
from .utils import yee_avg_2d_z, curl_fn
from .primitives import spsolve, TMOperator

try:  # pardiso (using Intel MKL) is much faster than scipy's solver
    from .mkl import spsolve_pardiso, feast_eigs
except OSError:  # if mkl isn't installed
    from scipy.sparse.linalg import spsolve

try:
    from dphox.component import Pattern
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
            \nabla \\times \mu^{-1} \nabla \\times \mathbf{e} - k_0^2 \\epsilon \mathbf{e} = k_0 \mathbf{j},
        which can be written in the form :math:`A \mathbf{e} = \mathbf{b}`, where:
        .. math::
            A = \nabla \\times \mu^{-1} \nabla \\times - k_0^2 \\epsilon \\
            b = k_0 \mathbf{j}
        is an operator representing the discretized EM wave operator at frequency :math:`omega`.

        Therefore, :math:`\mathbf{e} = A^{-1}\mathbf{b}`.

        For 2D simulations, it can be more efficient to solve for just the :math:`z`-component of the fields since
        only :math:`\mathbf{e}_z` is non-zero. In this case, we can solve a smaller problem to improve the efficiency.
        The form of this problem is :math:`A_z \mathbf{e}_z = \mathbf{b}_z`, where:
        .. math::
            A = (\nabla \\times \mu^{-1} \nabla \times)_z + k_0^2 \epsilon_z \\
            \mathbf{b}_z = k_0 \mathbf{j}_z \\

    Attributes:
        shape: Tuple of size 1, 2, or 3 representing the number of pixels in the grid
        spacing: Spacing (microns) between each pixel along each axis (must be same dim as `grid_shape`)
        eps: Relative permittivity :math:`\\epsilon_r`
        bloch_phase: Bloch phase (generally useful for angled scattering sims)
        pml: Perfectly matched layer (PML) of thickness on both sides of the form :code:`(x_pml, y_pml, z_pml)`
        pml_eps: The permittivity used to scale the PML (should probably assign to 1 for now)
        yee_avg: whether to do a yee average (highly recommended)
    """

    def __init__(self, shape: Shape, spacing: GridSpacing,
                 wavelength: float = 1.55, eps: Union[float, np.ndarray] = 1,
                 bloch_phase: Union[Dim, float] = 0.0, pml: Optional[Union[int, Shape, Dim]] = None,
                 pml_eps: float = 1.0, yee_avg: bool = True, name: str = 'fdfd'):

        super(FDFD, self).__init__(
            shape=shape,
            spacing=spacing,
            eps=eps,
            bloch_phase=bloch_phase,
            pml=pml,
            pml_eps=pml_eps,
            yee_avg=yee_avg,
            name=name
        )

        self.wavelength = wavelength

        # overwrite dxes with PML-scaled ones if specified
        if self.pml_shape is not None:
            dxes_pml_e, dxes_pml_h = [], []
            for ax, p in enumerate(self.pos):
                scpml_e, scpml_h = self.scpml(ax)
                dxes_pml_e.append(self.cell_sizes[ax] * scpml_e)
                dxes_pml_h.append(self.cell_sizes[ax] * scpml_h)
            self._dxes = np.meshgrid(*dxes_pml_e, indexing='ij'), np.meshgrid(*dxes_pml_h, indexing='ij')

    @classmethod
    def from_simgrid(cls, simgrid: SimGrid, wavelength: float):
        fdfd = cls(
            shape=simgrid.shape,
            spacing=simgrid.spacing,
            wavelength=wavelength,
            eps=simgrid.eps,
            pml=simgrid.pml_shape,
            name=simgrid.name
        )
        fdfd.port = simgrid.port
        fdfd.port_thickness = simgrid.port_thickness
        fdfd.port_height = simgrid.port_height
        return fdfd

    @property
    def k0(self):
        return 2 * np.pi / self.wavelength

    @property
    def mat(self) -> Union[sp.spmatrix, Tuple[np.ndarray, np.ndarray]]:
        """Build the discrete Maxwell operator :math:`A(k_0)` acting on :math:`\mathbf{e}`.

        Returns:
            Electric field operator :math:`A` for solving Maxwell's equations at frequency :math:`omega`.
        """
        mat = self.curl_curl - self.k0 ** 2 * sp.diags(self.eps_t.flatten())
        return mat

    A = mat  # alias A (common symbol for FDFD matrix) to mat

    @property
    def mat_ez(self) -> sp.spmatrix:
        """Build the discrete Maxwell operator :math:`A_z(k_0)` acting on :math:`\mathbf{e}_z` (for 2D problems).

        Returns:
            Electric field operator :math:`A_z` for a source with z-polarized e-field.
        """
        df, db = self.df, self.db
        ddz = -db[0] @ df[0] - db[1] @ df[1]
        ddz.sort_indices()  # for the solver
        mat = ddz - self.k0 ** 2 * sp.diags(self.eps_t[2].flatten())
        return mat

    @property
    def mat_hz(self) -> sp.spmatrix:
        """Build the discrete Maxwell operator :math:`A_z(k_0)` acting on :math:`\mathbf{h}_z` (for 2D problems).

        Returns:
            Electric field operator :math:`A_z` for a source with z-polarized e-field.
        """
        df, db = self.df, self.db
        t0, t1 = sp.diags(1 / self.eps_t[0].flatten()), sp.diags(1 / self.eps_t[1].flatten())
        mat = -db[0] @ t0 @ df[0] - db[1] @ t1 @ df[1] - self.k0 ** 2 * sp.identity(self.n)
        return mat

    def e2h(self, e: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """
        Convert magnetic field :math:`\mathbf{e}` to electric field :math:`\mathbf{h}`.

        Usage is: `h = fdfd.e2h(e)`, where `e` is grid-shaped (not flattened)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            i \omega \mu \mathbf{h} = \nabla \times \mathbf{e}

        Returns:
            The h-field converted from the e-field

        """
        e = self.reshape(e) if e.ndim == 2 else e
        return self.curl_e(beta)(e) / (1j * self.k0)

    def h2e(self, h: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """
        Convert magnetic field :math:`\mathbf{h}` to electric field :math:`\mathbf{e}`.

        Usage is: `e = fdfd.h2e(h)`, where `h` is grid-shaped (not flattened)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            -i \omega \epsilon \mathbf{e} = \nabla \times \mathbf{h}

        Returns:
            Function to convert h-field to e-field

        """
        h = self.reshape(h) if h.ndim == 2 else h
        return self.curl_h(beta)(h) / (1j * self.k0 * self.eps_t)

    def solve(self, src: np.ndarray, solver_fn: Optional[SpSolve] = None, reshaped: bool = True,
              iterative: int = -1, tm_2d: bool = True, callback: Optional[Callable] = None) -> np.ndarray:
        """FDFD Solver

        Args:
            src: normalized source (can be wgm or tfsf)
            solver_fn: any function that performs a sparse linalg solve
            reshaped: reshape into the grid shape (instead of vectorized/flattened form)
            iterative: default = -1, direct = 0, gmres = 1, bicgstab
            tm_2d: use the TM polarization (only relevant for 2D, ignored for 3D)
            callback: a function to run during the solve (only applies in 3d iterative solver case, not yet implemented)

        Returns:
            Electric fields that solve the problem :math:`A\mathbf{e} = \mathbf{b} = i \omega \mathbf{j}`

        """
        b = self.k0 * src.flatten()
        if b.size == self.n * 3:
            if iterative == -1 and solver_fn is None and self.ndim == 3:
                # use iterative solver for 3d sims by default
                M = sp.linalg.LinearOperator(self.mat.shape, sp.linalg.spilu(self.mat).solve)
                field = sp.linalg.gmres(self.mat, b, M=M) if iterative == 1 else sp.linalg.bicgstab(self.mat, b, M=M)
            else:
                field = solver_fn(self.mat, b) if solver_fn else spsolve_pardiso(self.mat, b)
        elif b.size == self.n:  # assume only the z component
            mat = self.mat_hz if tm_2d else self.mat_ez
            fz = solver_fn(mat, b) if solver_fn else spsolve_pardiso(mat, b)
            o = np.zeros_like(fz)
            field = np.vstack((o, o, fz))
        else:
            raise ValueError(f'Expected src.size == {self.n * 3} or {self.n}, but got {b.size}.')
        return self.reshape(field) if reshaped else field

    def scpml(self, ax: int, exp_scale: float = 4, log_reflection: float = -16) -> Tuple[np.ndarray, np.ndarray]:
        if self.cell_sizes[ax].size == 1:
            return np.ones(1), np.ones(1)
        p = self.pos[ax]
        pe, ph = (p[:-1] + p[1:]) / 2, p[:-1]
        absorption_corr = self.k0 * self.pml_eps
        t = self.pml_shape[ax]

        def _scpml(d: np.ndarray):
            d_pml = np.hstack((
                (d[t] - d[:t]) / (d[t] - p[0]),
                np.zeros_like(d[t:-t]),
                (d[-t:] - d[-t]) / (p[-1] - d[-t])
            ))
            return 1 + 1j * (exp_scale + 1) * (d_pml ** exp_scale) * log_reflection / (2 * absorption_corr)

        return _scpml(pe), _scpml(ph)

    @property
    @lru_cache()
    def curl_curl(self) -> sp.spmatrix:
        curl_curl: sp.spmatrix = d2curl_op(self.db) @ d2curl_op(self.df)
        curl_curl.sort_indices()  # for the solver
        return curl_curl

    @classmethod
    def from_pattern(cls, component: "Pattern", core_eps: float, clad_eps: float, spacing: float, boundary: Dim,
                     pml: float, wavelength: float, component_t: float = 0, component_zmin: Optional[float] = None,
                     rib_t: float = 0, sub_z: float = 0, height: float = 0, bg_eps: float = 1, name: str = 'fdfd'):
        """Initialize an FDFD from a Pattern defined in DPhox.

        Args:
            component: component provided by DPhox
            core_eps: core epsilon (in the pattern region_
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
        npml = int(pml / spacing)
        component_zmin = sub_z if component_zmin is None else component_zmin
        spacing = spacing * np.ones(2 + (component_t > 0)) if isinstance(spacing, float) else np.asarray(spacing)
        if height > 0:
            shape = (np.asarray((x, y, height)) / spacing).astype(np.int)
        else:
            shape = (np.asarray((x, y)) / spacing).astype(np.int)
        grid = cls(shape, spacing, wavelength=wavelength, eps=bg_eps, pml=npml, name=name)
        grid.fill(sub_z + rib_t, core_eps)
        grid.fill(sub_z, clad_eps)
        grid.add(component, core_eps, component_zmin, component_t)
        return grid

    def sparams(self, port_name: str, mode_idx: int = 0, measure_info: Optional[Dict[str, List[int]]] = None):
        """Measure sparams using port profiles.

        Args:
            port_name: The port name for the sparams to measure.
            mode_idx: Mode index to access for the source.
            measure_info: A list of :code:`port_name`, :code:`mode_idx` to get mode measurements at each port.

        Returns:

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

        1. A numpy array source :code:`src`
        2. The JAX-transformable transform function :code:`transform_fn` (e.g. transform) that yields
           the epsilon distribution used by jax.

        Args:
            src: Source for the solver
            transform_fn: Transforms parameters to yield the epsilon parameters used by jax (if None, use identity)
            tm_2d: Whether to solve the TM polarization for this FDFD (only relevant for 2D, ignored for 3D)

        Returns:
            A solve function (2d or 3d based on defined :code:`ndim` specified for the instance of :code:`FDFD`)

        Returns:

        """
        src = jnp.ravel(jnp.array(src))
        k0 = self.k0
        transform_fn = transform_fn if transform_fn is not None else lambda x: x

        def coo_to_jnp(mat: sp.coo_matrix):
            mat.sort_indices()
            mat = mat.tocoo()
            return jnp.array(mat.data, dtype=np.complex), jnp.vstack((jnp.array(mat.row), jnp.array(mat.col)))

        if self.ndim == 2:
            shape = self.shape
            o = jnp.zeros(self.shape, jnp.complex128)[..., jnp.newaxis]
            if tm_2d:
                # exact 2d FDFD for TE polarization
                constant_term = -jnp.ones_like(self.eps.flatten()) * k0 ** 2
                constant_term_indices = jnp.stack((jnp.arange(self.n), jnp.arange(self.n)))

                # this is temporary while we wait for sparse-sparse support in JAX.
                operator = TMOperator(self.df, self.db)
                x_op = operator.compile_operator_along_axis(0)
                y_op = operator.compile_operator_along_axis(1)
                x_ind, y_ind = operator.x_indices, operator.y_indices
                dh = self.diff_fn(use_h=True, use_jax=True)

                @jax.jit
                def solve(rho: jnp.ndarray):
                    eps_t = yee_avg_jax(transform_fn(rho))
                    eps_x, eps_y = jnp.ravel(eps_t[0]), jnp.ravel(eps_t[1])
                    ddx_entries = x_op(-1 / eps_x)
                    ddy_entries = y_op(-1 / eps_y)
                    mat_entries = jnp.hstack((constant_term, ddx_entries, ddy_entries))
                    hz = spsolve(mat_entries, k0 * src, jnp.hstack((constant_term_indices, x_ind, y_ind)))
                    hz = jnp.reshape(hz, shape)[..., jnp.newaxis]
                    h = jnp.stack((o, o, hz))
                    e = jnp.stack((dh(h[2], 1), -dh(h[2], 0), o)) / (1j * k0 * eps_t)
                    return jnp.stack((e, h))

            else:
                # exact 2d FDFD for TM polarization
                df, db = self.df, self.db
                ddz = -db[0] @ df[0] - db[1] @ df[1]
                ddz_entries, ddz_indices = coo_to_jnp(ddz)
                mat_indices = jnp.hstack((jnp.vstack((jnp.arange(self.n), jnp.arange(self.n))), ddz_indices))

                de = self.diff_fn(use_h=False, use_jax=True)

                @jax.jit
                def solve(rho: jnp.ndarray):
                    eps = jnp.ravel(yee_avg_2d_z(transform_fn(rho)))
                    mat_entries = jnp.hstack((-eps * k0 ** 2, ddz_entries))
                    ez = spsolve(mat_entries, k0 * src, mat_indices)
                    ez = jnp.reshape(ez, shape)[..., jnp.newaxis]
                    e = jnp.stack((o, o, ez))
                    h = jnp.stack((de(e[2], 1), -de(e[2], 0), o)) / (1j * k0)
                    return jnp.stack((e, h))
        else:
            # iterative 3d FDFD (simpler than 2D code-wise, but takes way more memory and time, untested atm)
            curl_e = curl_fn(self.diff_fn(use_h=False, use_jax=True), use_jax=True)
            curl_h = curl_fn(self.diff_fn(use_h=True, use_jax=True), use_jax=True)

            shape3 = self.shape3

            def op(eps: jnp.ndarray):
                return lambda b: curl_h(curl_e(b)) - k0 ** 2 * eps * b

            @jax.jit
            def solve(rho: jnp.ndarray):
                eps = transform_fn(rho)
                e = bicgstab(op(eps), k0 * src)
                e = jnp.stack([split_v.reshape(shape3) for split_v in jnp.split(e, 3)])
                h = curl_e(e) / (1j * k0)
                return jnp.stack((e, h))

        return solve

    def to_2d(self, wavelength: float = None, slab_x: Union[Shape2, Dim2] = None, slab_y: Union[Shape2, Dim2] = None):
        """Project a 3D FDFD into a 2D FDFD using the variational 2.5D method laid out in the paper
        https://ris.utwente.nl/ws/files/5413011/ishpiers09.pdf.

        Args:
            wavelength: The wavelength to use for calculating the effective 2.5 FDFD
                (useful to stabilize multi-wavelength optimizations)
            slab_x: Port location x (if None, the port is provided by reading the port location specified by the component)
            slab_y: Port location y (if None, the port is provided by reading the port location specified by the component)

        Returns:
            A 2D FDFD to approximate the 3D FDFD

        """
        wavelength = self.wavelength if wavelength is None else wavelength
        return FDFD.from_simgrid(super(FDFD, self).to_2d(wavelength), wavelength)

    def tfsf_profile(self, q_mask: np.ndarray, wavelength: float, k: Dim):
        mask = q_mask
        q = sp.diags(mask.flatten())
        period = wavelength  # equivalent to period since c = 1!
        k0 = 2 * np.pi / period
        k = np.asarray(k) / (np.sum(k)) * k0
        fsrc = np.einsum('i,j,k->ijk', np.exp(1j * self.pos[0][:-1] * k[0]),
                         np.exp(1j * self.pos[1][:-1] * k[1]),
                         np.exp(1j * self.pos[2][:-1] * k[2])).flatten()
        a = self.mat
        src = self.reshape((q @ a - a @ q) @ fsrc)  # qaaq = quack :)
        return src

from .grid import YeeGrid
from .material import MaterialBlock
from .typing import Shape, Dim, GridSpacing, Optional, Tuple, Union, List, Callable
from .utils import poynting_fn, overlap, SMALL_NUMBER
from .viz import plot_eps_2d, plot_power_2d, plot_field_2d

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import copy
from functools import lru_cache

import jax.numpy as jnp

try:  # pardiso (using Intel MKL) is much faster than scipy's solver
    from .mkl import feast_eigs
except OSError:  # if mkl isn't installed
    from scipy.sparse.linalg import spsolve

try:
    from dphox.component import Pattern

    DPHOX_INSTALLED = True
except ImportError:
    DPHOX_INSTALLED = False

from logging import getLogger

logger = getLogger()


class ModeSolver(YeeGrid):
    """FDFD-based waveguide mode solver

    Notes:
        This class is capable of finding 1d or 2d cross-sectional modes, which are
        useful for defining sources and calculating the number of propagating modes that exist in multi-mode
        interferometers (MMIs).

        We can solve either the 1d or 2d case using 1 or 2 components of the field respectively.
        Note that in `simphox` units, we assume $k_0 = \\frac{2\\pi}{\\lambda} = \\omega$,
        letting $c = \\epsilon_0 = \\mu_0 = 1$ for simplicity. We define the *wavenumber* $\\beta_m$ for mode $m$ to be
        the square root of the eigenvalue (hence the $\beta^2$ terms in the later equations) of $C_{\\mathrm{1d}}$ and
        $C_{\\mathrm{2d}}$ for the respective problems.

        For the 1d case, we consider the case where the simulation is a line cross-section of a 2d grid in the
        $xy$-plane. In that case, we solve for the $z$-component of an $h_z$-polarized mode of the form
        $\\mathbf{h}_m = (0, 0, h_z(y)) e^{-i\\beta_m x}$. The solutions for $h_z(y)$ correspond to the simple equation:

        .. math::
            \\beta^2 h_z = \\partial_y^2 h_z + k_0^2 \\epsilon_z h_z

            \\beta_m^2 \\mathbf{h}_{m} = C_{\\mathrm{1d}} \\mathbf{h}_{m}

        For the 2d case, we cannot make this type of assumption.
        Instead we solve the frequency-domain Maxwell's equations for the case of $z$-translation symmetry
        (here, we consider propagation along $z$ instead of $x$ to match convention).
        This time, we solve for an $\\mathbf{h}$-field of the form $\\mathbf{h}_m = \\mathbf{h}(x, y) e^{-i\\beta_m z}$.
        This is made possible by the following set of coupled differential equations:

        .. math::
            \beta^2 h_x &= \\partial_x(\\partial_x h_x + \\partial_y h_y) - \\epsilon_y (\\epsilon_z^{-1}
            (\\partial_y h_x - \\partial_x h_y) + k_0^2) = C_{xy} \\begin{bmatrix} h_x \\ h_y\\end{bmatrix}

            \beta^2 h_y &= \\partial_y(\\partial_x h_x + \\partial_y h_y) - \\epsilon_x (\\epsilon_z^{-1}
            (\\partial_x h_y - \\partial_y h_x) + k_0^2) = C_{yx} \\begin{bmatrix} h_x \\ h_y\\end{bmatrix}

            \beta^2 \\begin{bmatrix} h_x \\ h_y\\end{bmatrix} &= \\begin{bmatrix} C_{xy} \\ C_{yx}\\end{bmatrix}
            \\begin{bmatrix} h_x \\ h_y\\end{bmatrix}

            \beta_m^2 \\mathbf{h}_{m} &= C_{\\mathrm{2d}} \\mathbf{h}_{m}

    Attributes:
        shape: Tuple of size 1, 2, or 3 representing the number of pixels in the grid
        spacing: Spacing (microns) between each pixel along each axis (must be same dim as `grid_shape`)
        eps: Relative permittivity :math:`\\epsilon_r`
        bloch_phase: Bloch phase (generally useful for angled scattering sims)
        yee_avg: whether to do a yee average (highly recommended)
    """

    def __init__(self, shape: Shape, spacing: GridSpacing,
                 wavelength: float = 1.55, eps: Union[float, np.ndarray] = 1,
                 bloch_phase: Union[Dim, float] = 0.0, yee_avg: bool = True, name: str = 'mode'):

        super(ModeSolver, self).__init__(
            shape=shape,
            spacing=spacing,
            eps=eps,
            bloch_phase=bloch_phase,
            pml=None,
            yee_avg=yee_avg,
            name=name
        )

        self.wavelength = wavelength
        self.k0 = 2 * np.pi / self.wavelength

    @property
    def wgm(self) -> sp.spmatrix:
        """Build the WaveGuide Mode (WGM) operator (for 1D or 2D grid only)

        The WGM operator :math:`C(\omega)` acts on the magnetic field
        :math:`\mathbf{h}` of the form `(hx, hy)`, which assumes cross-section translational x-symmetry:
        .. math::
            C \mathbf{h}_m = \lambda_m \mathbf{h}_m,
        where :math:`0 \leq m < M` for the :math:`M` modes with the largest wavenumbers
        (:math:`\beta_m = \sqrt{\lambda_m}`).

        Returns:
            Magnetic field operator :math:`C`.
        """

        if not self.ndim <= 2:
            raise AttributeError("Grid dimension must be 1 or 2")

        df, db = self.df, self.db

        if self.ndim == 2:
            eps = [e.flatten() for e in self.eps_t]
            eps_10 = sp.diags(np.hstack((eps[1], eps[0])))
            m1 = eps_10 * self.k0 ** 2
            m2 = eps_10 @ sp.vstack([-df[1], df[0]]) @ sp.diags(1 / eps[2]) @ sp.hstack([-db[1], db[0]])
            m3 = sp.vstack(db[:2]) @ sp.hstack(df[:2])
            return m1 + m2 + m3
        else:
            return sp.diags(self.eps.flatten()) * self.k0 ** 2 + df[0].dot(db[0])

    C = wgm  # C is the matrix for the guided mode eigensolver

    def e2h(self, e: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """
        Convert magnetic field :math:`\mathbf{e}` to electric field :math:`\mathbf{h}`.

        Usage is: `h = mode.e2h(e)`, where `e` is grid-shaped (not flattened)

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

        Usage is: `e = mode.h2e(h)`, where `h` is grid-shaped (not flattened)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            -i \omega \epsilon \mathbf{e} = \nabla \times \mathbf{h}

        Returns:
            Function to convert h-field to e-field

        """
        h = self.reshape(h) if h.ndim == 2 else h
        return self.curl_h(beta)(h) / (1j * self.k0 * self.eps_t)

    def solve(self, num_modes: int = 6, beta_guess: Optional[Union[float, Tuple[float, float]]] = None,
              tol: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
        """FDFD waveguide mode solver

        Solve for waveguide modes (x-translational symmetry) by finding the eigenvalues of :math:`C`.

        .. math::
            C \\mathbf{h}_m = \\lambda_m \\mathbf{h}_m,

        where :math:`0 \\leq m < M` for the :math:`M` (`num_modes`) modes with the largest wavenumbers
        (:math:`\beta_m = \\sqrt{\\lambda_m}`).

        Args:
            num_modes: Number of modes
            beta_guess: Guess for propagation constant :math:`\beta`
            tol: Tolerance of the mode solver

        Returns:
            `num_modes` (:math:`M`) largest propagation constants (:math:`\\sqrt{\\lambda_m(C)}`)
            and corresponding modes (:math:`\\mathbf{h}_m`) of shape :code:`(num_modes, n)`.
        """

        df = self.df
        if isinstance(beta_guess, float) or beta_guess is None:
            sigma = beta_guess ** 2 if beta_guess else (self.k0 * np.sqrt(np.max(self.eps))) ** 2
            eigvals, eigvecs = eigs(self.wgm, k=num_modes, sigma=sigma, tol=tol)
        elif isinstance(beta_guess, tuple):
            erange = beta_guess[0] ** 2, beta_guess[1] ** 2
            eigvals, eigvecs, _, _, _, _ = feast_eigs(self.wgm, erange=erange, k=num_modes)
        else:
            raise TypeError(f'Expected beta_guess to be None, float, or Tuple[float, float] but got {type(beta_guess)}')
        inds_sorted = np.asarray(np.argsort(np.sqrt(eigvals.real))[::-1])
        if self.ndim > 1:
            hz = sp.hstack(df[:2]) @ eigvecs / (1j * np.sqrt(eigvals))
            h = np.vstack((eigvecs, hz))
        else:
            h = eigvecs

        h = h[:, inds_sorted].T
        h *= np.exp(-1j * np.angle(h[:1, :]))  # divide by global phase or set polarity (set reference plane)
        return np.sqrt(eigvals[inds_sorted]), h

    def profile(self, mode_idx: int = 0, power: float = 1,
                beta_guess: Optional[float] = None, tol: float = 1e-5,
                return_beta: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Define waveguide mode source using waveguide mode solver (incl. pml if part of the mode solver!)

        Args:
            mode_idx: Mode index to use (default is 0, the fundamental mode)
            power: Power to scale the source (default is 1, a normalized mode in arb units),
            and if negative, the source moves in opposite direction (polarity is encoded in sign of power).
            beta_guess: Guess for propagation constant :math:`\beta`
            tol: Tolerance of the mode solver
            return_beta: Also return beta

        Returns:
            Grid-shaped waveguide mode (wgm) source (normalized h-mode for 1d, spins-b source for 2d)
        """

        polarity = np.sign(power)
        p = np.abs(power)

        beta, h = self.solve(min(mode_idx + 1, 6), beta_guess, tol)
        if self.ndim == 1:
            src = h[mode_idx] * polarity * np.sqrt(p)
        else:
            src = self.reshape(h[mode_idx]) * polarity * np.sqrt(p)
        return beta, src if return_beta else src


class ModeLibrary:
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray],
                 wavelength: float = 1.55, num_modes: int = 1):
        """A data structure to contain the information about :math:`num_modes` cross-sectional modes

        Args:
            shape: The shape of the mode solver.
            spacing: The spacing of the mode solver.
            eps: The permittivity distribution for the mode solver.
            wavelength: The wavelength for the modes.
            num_modes: Number of modes that should be solved.
        """
        self.solver = ModeSolver(
            shape=shape,
            spacing=spacing,
            eps=eps,
            wavelength=wavelength
        )
        self.ndim = self.solver.ndim
        self.betas, self.modes = self.solver.solve(num_modes)
        self.modes = self.modes * np.exp(-1j * np.angle(self.modes[:1, :]))
        self.eps = eps
        self.num_modes = self.m = len(self.betas)
        self.o = np.zeros_like(self.modes[0])

    @lru_cache()
    def h(self, mode_idx: int = 0, tm_2d: bool = True) -> np.ndarray:
        """Magnetic field :math:`\\mathbf{H}` for the mode of specified index

        Args:
            mode_idx: The mode index :math:`m \\leq M`
            tm_2d: If the mode is using a 1d distribution, this specifies if the mode is TM (otherwise TE)

        Returns:
            :math:`\\mathbf{H}_m`, an :code:`ndarray` of the form :code:`(3, X, Y)` for mode :math:`m \\leq M`

        """
        mode = self.modes[mode_idx]
        if self.ndim == 1:
            if tm_2d:
                mode = np.hstack((self.o, mode, self.o))
            else:
                mode = np.hstack((1j * self.betas[mode_idx] * mode, self.o,
                                  -(mode - np.roll(mode, 1, axis=0)) / self.solver.cell_sizes[0])) / (1j * self.solver.k0)
        return self.solver.reshape(mode)

    @lru_cache()
    def e(self, mode_idx: int = 0, tm_2d: bool = True) -> np.ndarray:
        """Electric field :math:`\\mathbf{E}` for the mode of specified index

        Args:
            mode_idx: The mode index :math:`m \\leq M`
            tm_2d: If the mode is using a 1d distribution, this specifies if the mode is TM (otherwise TE)

        Returns:
            :math:`\\mathbf{E}_m`, an :code:`ndarray` of shape :code:`(3, X, Y, Z)` for mode :math:`m \\leq M`

        """
        if self.ndim == 2:
            return self.solver.h2e(self.h(mode_idx), self.betas[mode_idx])
        else:
            mode = self.modes[mode_idx]
            if tm_2d:
                mode = np.hstack((1j * self.betas[mode_idx] * mode, self.o,
                                  -(np.roll(mode, -1, axis=0) - mode) / self.solver.cell_sizes[0])) / (
                        1j * self.solver.k0 * self.solver.eps_t.flatten())
            else:
                mode = np.hstack((self.o, mode, self.o))
            return self.solver.reshape(mode)

    @lru_cache()
    def sz(self, mode_idx: int = 0) -> np.ndarray:
        """Poynting vector :math:`\\mathbf{S}_z` for the mode of specified index

        Args:
            mode_idx: The mode index :math:`m \\leq M`

        Returns:
            :math:`\\mathbf{S}_{m, z}`, the z-component of Poynting vector (correspoding to power),
            of shape :code:`(X, Y)`

        """
        return poynting_fn(2)(self.e(mode_idx), self.h(mode_idx)).squeeze()

    def beta(self, mode_idx: int = 0) -> float:
        """Fundamental mode propagation constant :math:`\\beta` for the mode of specified index

        Args:
            mode_idx: The mode index :math:`m \\leq M`

        Returns:
            :math:`\\beta_m` for mode :math:`m \\leq M`
        """
        return self.betas[mode_idx]

    def n(self, mode_idx: int = 0):
        """Index :math:`n`

        Returns:
            :math:`n`
        """
        return self.betas[mode_idx] / self.solver.k0

    @property
    @lru_cache()
    def hs(self):
        """An array for the magnetic fields `\\mathbf{H}` corresponding to all :math:`M` modes

        Returns:
           :math:`\\mathbf{H}`, an :code:`ndarray` of shape :code:`(M, 3, X, Y)`
        """
        hs = []
        for mode in self.modes:
            hs.append(self.solver.reshape(mode))
        return np.stack(hs).squeeze()

    @property
    @lru_cache()
    def es(self):
        """An array for the magnetic fields `\mathbf{E}` corresponding to all :math:`M` modes

        Returns:
           :math:`\mathbf{E}`, an :code:`ndarray` of shape :code:`(M, 3, X, Y)`
        """
        es = []
        for beta, h in zip(self.betas, self.hs):
            es.append(self.solver.h2e(h[..., np.newaxis], beta))
        return np.stack(es).squeeze()

    @property
    @lru_cache()
    def szs(self):
        """An array for the magnetic fields `\\mathbf{S}_z` corresponding to all :math:`M` modes

        Returns:
           :math:`\\mathbf{S}_z`, an :code:`ndarray` of shape :code:`(M, X, Y)`
        """
        szs = []
        for e, h in zip(self.es, self.hs):
            szs.append(poynting_fn(2)(e[..., np.newaxis], h[..., np.newaxis]))
        return np.stack(szs).squeeze()

    @property
    @lru_cache()
    def ns(self):
        return self.betas / self.solver.k0

    @property
    def dbeta(self):
        return self.beta(0) - self.beta(1)

    @property
    def dn(self):
        return (self.beta(0) - self.beta(1)) / self.solver.k0

    @property
    def te_ratios(self):
        te_ratios = []
        for h in self.hs:
            habs = np.abs(h.squeeze())
            norms = np.asarray((np.linalg.norm(habs[0].flatten()), np.linalg.norm(habs[1].flatten())))
            te_ratios.append(norms[0] ** 2 / np.sum(norms ** 2))
        return np.asarray(te_ratios)

    def fundamental_coeff(self, other_modes: "ModeLibrary"):
        e_i, h_i = self.e(), self.h()
        e_o, h_o = other_modes.e(), other_modes.h()
        return np.sum(poynting_fn(2)(e_o, h_i) + poynting_fn(2)(e_i, h_o)).real

    def plot_sz(self, ax, idx: int = 0, title: str = "Poynting", include_n: bool = False,
                title_size: float = 16, label_size=16):
        if idx > self.m - 1:
            raise ValueError("Out of range of number of solutions")
        plot_power_2d(ax, np.abs(self.sz(idx).real), self.eps, spacing=self.solver.spacing[0])
        if include_n:
            ax.set_title(rf'{title}, $n_{idx + 1} = {self.n(idx):.4f}$', fontsize=title_size)
        else:
            ax.set_title(rf'{title}', fontsize=title_size)
        ax.text(x=0.9, y=0.9, s=rf'$s_z$', color='white', transform=ax.transAxes, fontsize=label_size)
        ratio = np.max((self.te_ratios[idx], 1 - self.te_ratios[idx]))
        polarization = "TE" if np.argmax((self.te_ratios[idx], 1 - self.te_ratios[idx])) > 0 else "TM"
        ax.text(x=0.05, y=0.9, s=rf'{polarization}[{ratio:.2f}]', color='white', transform=ax.transAxes)

    def plot_field(self, ax, idx: int = 0, axis: int = 1, use_e: bool = False, title: str = "Field",
                   title_size: float = 16, label_size=16, include_n: bool = False):
        """

        Args:
            ax: Matplotlib axis handle
            idx: Mode index
            axis: Field axis to plot
            use_e: Use electric field :math:`\mathbf{E}`, else use magnetic field :math:`\mathbf{H}` by default
            title: Title for the plot (recommended to change for application!)

        Returns:

        """
        field = self.es if use_e else self.hs
        if idx > self.m - 1:
            ValueError("Out of range of number of solutions")
        if not (axis in (0, 1, 2)):
            ValueError(f"Axis expected to be (0, 1, 2) but got {axis}.")
        plot_field_2d(ax, field[idx][axis].real, self.eps, spacing=self.solver.spacing[0])
        if include_n:
            ax.set_title(rf'{title}, $n_{idx + 1} = {self.n(idx):.4f}$', fontsize=title_size)
        else:
            ax.set_title(rf'{title}', fontsize=title_size)
        ax.text(x=0.9, y=0.9, s=rf'$e_y$' if use_e else rf'$h_y$', color='black', transform=ax.transAxes,
                fontsize=label_size)
        ratio = np.max((self.te_ratios[idx], 1 - self.te_ratios[idx]))
        polarization = "TE" if np.argmax((self.te_ratios[idx], 1 - self.te_ratios[idx])) > 0 else "TM"
        ax.text(x=0.05, y=0.9, s=rf'{polarization}[{ratio:.2f}]', color='black', transform=ax.transAxes)

    def phase(self, length: float = 1):
        return self.solver.k0 * length * self.n()

    def overlap_fundamental(self, other_sol: "ModeLibrary"):
        return overlap(self.e(), self.h(), other_sol.e(), other_sol.h()) ** 2

    def place(self, mode_idx: int, grid: YeeGrid, center: Dim, size: Dim) -> np.ndarray:
        """Place at mode_idx in device with :math:`shape` and :math:`region`.

        Args:
            mode_idx: mode index to use.
            grid: finite-difference grid to use.
            center: Specified center
            size: Specified size

        Returns:
            Places the mode into the provided grid at the requested center and size, with orientation of the mode
            automatically determined from the center and size provided.

        """
        region = grid.slice(center, size)
        if self.ndim == 2:
            x = np.zeros((3, *grid.shape), dtype=np.complex)
            x[:, region[0], region[1], region[2]] = self.h(mode_idx)
        else:
            x = np.zeros(grid.shape, dtype=np.complex)
            x[region[0], region[1]] = self.modes[mode_idx]
        return x

    def measure_fn(self, mode_idx: int = 0, use_jax: bool = False, tm_2d: bool = True):
        """Measure flux

        Args:
            mode_idx: Use the poynting, mode index
            use_jax: Use jax.
            tm_2d: Use TM polarization (only relevant in the case of 2D simulations (i.e., 1D modes))

        Returns:
            A function that takes e, h fields and outputs the a and b terms

        """
        poynting = poynting_fn(use_jax=use_jax)
        em, hm = self.e(mode_idx, tm_2d=tm_2d), self.h(mode_idx, tm_2d=tm_2d)
        sm = np.sum(poynting(em, hm))
        xp = jnp if use_jax else np

        def _measure(e, h):
            a, b = xp.sum(poynting(e, hm)) / sm / 2, (xp.sum(poynting(em, h)) / sm).conj() / 2
            return xp.array([a + b, a - b])

        return _measure


class ModeDevice:
    def __init__(self, wg: MaterialBlock, sub: MaterialBlock, size: Tuple[float, float], wg_height: float,
                 spacing: float = 0.01, rib_y: float = 0):
        """A :code:`ModeDevice` can be used to efficiently simulate various scenarios for coupled waveguides
        and phase shifters.

        Args:
            wg: Waveguide :code:`MaterialBlock`
            sub: Substrate :code:`MaterialBlock`
            size: Size of the overall simulation (in arb. units)
            wg_height: Size of the overall simulation (in arb. units)
            spacing: Spacing for the simulation (recommended at least 10 pixels per wavelength in high-index material)
            rib_y: Rib height (from substrate to partial etch cutoff)
        """
        self.size = size
        self.spacing = spacing
        self.nx = np.round(self.size[0] / spacing).astype(int)
        self.ny = np.round(self.size[1] / spacing).astype(int)
        self.wg_height = wg_height
        self.wg = wg
        self.sub = sub
        self.rib_y = rib_y

    def solve(self, eps: np.ndarray, num_modes: int = 6, wavelength: float = 1.55) -> ModeLibrary:
        """Solve the modes for the provided epsilon distribution.

        Args:
            eps: Permittivity distribution for the solver.
            num_modes: Number of modes to be input into the solver.
            wavelength: Wavelength for the mode solver.

        Returns:
            :code:`ModeLibrary` object that solves the.

        """
        return ModeLibrary(shape=(self.nx, self.ny),
                           spacing=self.spacing,
                           wavelength=wavelength,
                           eps=eps,
                           num_modes=num_modes)

    def single(self, vert_ps: Optional[MaterialBlock] = None, lat_ps: Optional[MaterialBlock] = None,
               sep: float = 0) -> np.ndarray:
        """Single-waveguide permittivity with an optional phase shifter block placed above

        Args:
            vert_ps: vertical phase shifter block
            lat_ps: lateral phase shifter block
            sep: separation betwen phase shifter and waveguide

        Returns:
            permittivity distribution for the system

        """
        nx, ny = self.nx, self.ny
        center = nx // 2
        wg, sub, dx = self.wg, self.sub, self.spacing
        wg_y = (self.wg_height, self.wg_height + wg.y)

        xr_wg = (center - int(wg.x / 2 / dx), center - int(wg.x / 2 / dx) + np.round(wg.x / dx).astype(int))
        yr_wg = slice(int(wg_y[0] / dx), int(wg_y[1] / dx))
        eps = np.ones((nx, ny))
        eps[:, :int(self.sub.y / dx)] = sub.material.eps
        eps[slice(*xr_wg), yr_wg] = wg.material.eps
        eps[:, int(self.sub.y / dx):int(self.sub.y / dx) + np.round(self.rib_y / dx).astype(int)] = wg.material.eps

        if vert_ps is not None:
            ps_y = (wg.y + self.wg_height + sep, wg.y + self.wg_height + sep + vert_ps.y)
            xr_ps = slice(center - int(vert_ps.x / 2 / dx),
                          center - int(vert_ps.x / 2 / dx) + np.round(vert_ps.x / dx).astype(int))
            yr_ps = slice(int(ps_y[0] / dx), int(ps_y[1] / dx))
            eps[xr_ps, yr_ps] = vert_ps.material.eps

        if lat_ps is not None:
            xrps_l = slice(xr_wg[0] - np.round(lat_ps.x / dx).astype(int) - int(sep / dx),
                           xr_wg[0] - int(sep / dx))
            xrps_r = slice(xr_wg[1] + int(sep / dx),
                           xr_wg[1] + int(sep / dx) + np.round(lat_ps.x / dx).astype(int))
            eps[xrps_l, yr_wg] = lat_ps.material.eps
            eps[xrps_r, yr_wg] = lat_ps.material.eps

        return eps

    def coupled(self, gap: float, vert_ps: Optional[MaterialBlock] = None,
                lat_ps: Optional[MaterialBlock] = None,
                seps: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """Coupled-waveguide permittivity with an optional pair of phase shifter blocks placed above

        Args:
            gap: coupling gap for the interaction region
            vert_ps: vertical phase shifter :code:`MaterialBlock`
            lat_ps: lateral phase shifter :code:`MaterialBlock`
            seps: separation between left and right waveguide in the coupler respectively

        Returns:
            permittivity distribution for the system

        """
        nx, ny = self.nx, self.ny
        center = nx // 2
        wg, sub, dx = self.wg, self.sub, self.spacing
        wg_y = (self.wg_height, self.wg_height + wg.y)

        xr_l = (center - int((gap / 2 + wg.x) / dx), center - int(gap / 2 / dx))
        xr_r = (center + int((gap / 2) / dx), center + int((gap / 2 + wg.x) / dx))
        yr = slice(int(wg_y[0] / dx), int(wg_y[1] / dx))

        eps = np.ones((nx, ny))
        eps[:, :int(sub.y / dx)] = sub.eps
        eps[xr_l[0]:xr_l[1], yr] = wg.eps
        eps[xr_r[0]:xr_r[1], yr] = wg.eps
        eps[:, int(self.sub.y / dx):int(self.sub.y / dx) + int(self.rib_y / dx)] = wg.material.eps

        if vert_ps is not None:
            ps_y = (self.wg.y + self.wg_height + seps[0], self.wg.y + self.wg_height + seps[0] + vert_ps.y)
            ps_y_2 = (self.wg.y + self.wg_height + seps[1], self.wg.y + self.wg_height + seps[1] + vert_ps.y)
            wg_l, wg_r = (xr_l[0] + xr_l[1]) / 2, (xr_r[0] + xr_r[1]) / 2
            xrps_l = slice(int(wg_l - vert_ps.x / dx / 2), int(wg_l + vert_ps.x / dx / 2))
            xrps_r = slice(int(wg_r - vert_ps.x / dx / 2), int(wg_r + vert_ps.x / dx / 2))
            yr_ps = slice(int(ps_y[0] / dx), int(ps_y[1] / dx))
            yr_ps2 = slice(int(ps_y_2[0] / dx), int(ps_y_2[1] / dx))
            eps[xrps_l, yr_ps] = vert_ps.eps
            eps[xrps_r, yr_ps2] = vert_ps.eps

        if lat_ps is not None:
            xrps_l = slice(xr_l[0] - int(lat_ps.x / dx) - int(seps[0] / dx), xr_l[0] - int(seps[0] / dx))
            xrps_r = slice(xr_r[1] + int(seps[1] / dx), xr_r[1] + int(seps[1] / dx) + int(lat_ps.x / dx))
            eps[xrps_l, yr] = lat_ps.eps
            eps[xrps_r, yr] = lat_ps.eps

        return eps

    def dc_grid(self, seps: np.ndarray, gap: float, ps: Optional[MaterialBlock] = None, m: int = 6,
                pbar: Callable = None) -> List[ModeLibrary]:
        """Tunable directional coupler grid sweep

        Args:
            seps: separations to sweep, for :math:`S` separations,
            the resulting solution grid will be :math:`S \\times S`
            gap: coupling gap for the interaction region
            ps: phase shifter :code:`MaterialBlock`
            m: Number of modes to find
            pbar: progress bar handle (to show progress using e.g. tqdm)

        Returns:
            A list of :math:`S^2` :code:`Modes` solution objects

        """
        solutions = []
        pbar = range if pbar is None else pbar
        for sep_1 in pbar(seps):
            for sep_2 in pbar(seps):
                eps = self.coupled(gap, ps, seps=(sep_1, sep_2))
                solutions.append(copy.deepcopy(self.solve(eps, m)))
        return solutions

    def ps_sweep(self, seps: np.ndarray, ps: Optional[MaterialBlock] = None, m: int = 6,
                 pbar: Callable = None) -> List[ModeLibrary]:
        """Phase shifter sweep

        Args:
            seps: Separations to sweep, for :math:`S` separations, the resulting solution will be of length :math:`S`
            ps: Phase shifter :code:`MaterialBlock`
            m: Number of modes to find
            pbar: Progress bar handle (to show progress using e.g. tqdm)

        Returns:
            A list of :math:`S` :code:`Modes` solution objects

        """
        solutions = []
        pbar = range if pbar is None else pbar
        for sep in pbar(seps):
            eps = self.single(ps, sep=sep)
            solutions.append(copy.deepcopy(self.solve(eps, m)))
        return solutions

    def dispersion_sweep(self, eps: np.ndarray, wavelengths: np.ndarray, m: int = 6, pbar: Callable = None):
        """Dispersion sweep for cross sectional modes

        Args:
            eps: Epsilon distribution for the sweep
            wavelengths: Wavelengths :math:`\\lambda` of length :math:`L`
            m: Number of modes to solve
            pbar: Progress bar handle (to show progress using e.g. tqdm)

        Returns:
            A list of :math:`L` :code:`Modes` solution objects

        """
        solutions = []
        pbar = range if pbar is None else pbar
        for wavelength in pbar(wavelengths):
            solutions.append(self.solve(eps, m, wavelength))
        return solutions

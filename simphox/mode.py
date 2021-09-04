from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from .grid import YeeGrid
from .typing import Size, GridSpacing, Optional, Tuple, Union, Callable, Size2
from .utils import poynting_fn, overlap, Box
from .viz import plot_power_2d, plot_field_2d

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
            \\beta^2 h_x &= \\partial_x(\\partial_x h_x + \\partial_y h_y) - \\epsilon_y (\\epsilon_z^{-1}
            (\\partial_y h_x - \\partial_x h_y) + k_0^2) = C_{xy} \\begin{bmatrix} h_x \\ h_y\\end{bmatrix}

            \\beta^2 h_y &= \\partial_y(\\partial_x h_x + \\partial_y h_y) - \\epsilon_x (\\epsilon_z^{-1}
            (\\partial_x h_y - \\partial_y h_x) + k_0^2) = C_{yx} \\begin{bmatrix} h_x \\ h_y\\end{bmatrix}

            \\beta^2 \\begin{bmatrix} h_x \\ h_y\\end{bmatrix} &= \\begin{bmatrix} C_{xy} \\ C_{yx}\\end{bmatrix}
            \\begin{bmatrix} h_x \\ h_y\\end{bmatrix}

            \\beta_m^2 \\mathbf{h}_{m} &= C_{\\mathrm{2d}} \\mathbf{h}_{m}

    Attributes:
        size: Tuple of size 1, 2, or 3 representing the size in arbitrary units
        spacing: Spacing (microns) between each pixel along each axis (MUST be in same units as `size`)
        eps: Relative permittivity :math:`\\epsilon_r`
        bloch_phase: Bloch phase (generally useful for angled scattering sims)
        yee_avg: whether to do a yee average (highly recommended)
    """

    def __init__(self, size: Size, spacing: GridSpacing,
                 wavelength: float = 1.55, eps: Union[float, np.ndarray] = 1,
                 bloch_phase: Union[Size, float] = 0.0, yee_avg: bool = True, name: str = 'mode'):

        super(ModeSolver, self).__init__(
            size=size,
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

        The WGM operator :math:`C(\\omega)` acts on the magnetic field
        :math:`\\mathbf{h}` of the form :code:`(hx, hy)`, which assumes cross-section translational x-symmetry:
        .. math::
            C \\mathbf{h}_m = \\lambda_m \\mathbf{h}_m,
        where :math:`0 \\leq m < M` for the :math:`M` modes with the largest wavenumbers
        (:math:`\\beta_m = \\sqrt{\\lambda_m}`).

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
        """Convert magnetic field :math:`\\mathbf{e}` to electric field :math:`\\mathbf{h}`.

        Usage is: :code:`h = mode.e2h(e)`, where :code:`e` is grid-shaped (not flattened)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            i \\omega \\mu \\mathbf{h} = \\nabla \\times \\mathbf{e}

        Returns:
            The h-field converted from the e-field.

        """
        return self.curl_e(beta)(self.reshape(e)) / (1j * self.k0)

    def h2e(self, h: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """Convert magnetic field :math:`\\mathbf{h}` to electric field :math:`\\mathbf{e}`.

        Usage is: :code:`e = mode.h2e(h)`, where :code:`h` is grid-shaped (not flattened)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:
        ..math::
            -i \\omega \\epsilon \\mathbf{e} = \\nabla \\times \\mathbf{h}.

        Returns:
            Function to convert h-field to e-field.

        """
        return self.curl_h(beta)(self.reshape(h)) / (1j * self.k0 * self.eps_t)

    def solve(self, num_modes: int = 6, beta_guess: Optional[Union[float, Size2]] = None,
              tol: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
        """FDFD waveguide mode solver

        Solve for waveguide modes (x-translational symmetry) by finding the eigenvalues of :math:`C`.

        .. math::
            C \\mathbf{h}_m = \\lambda_m \\mathbf{h}_m,

        where :math:`0 \\leq m < M` for the :math:`M` (`num_modes`) modes with the largest wavenumbers
        (:math:`\\beta_m = \\sqrt{\\lambda_m}`).

        Args:
            num_modes: Number of modes to return.
            beta_guess: Guess for propagation constant :math:`\beta` (the eigenvalue).
            tol: Tolerance of the mode eigensolver.

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
        return np.sqrt(eigvals[inds_sorted]), h * np.exp(-1j * np.angle(h[:1, :]))  # ensure phase doesn't change

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

    def block_design(self, waveguide: Box, wg_height: Optional[float] = None, sub_eps: float = 1, sub_height: float = 0,
                     coupling_gap: float = 0, block: Optional[Box] = None, sep: Size = (0, 0),
                     vertical: bool = False, rib_y: float = 0):
        """A helper function for designing a useful port or cross section that benefites from ModeSolver.

        Args:
            waveguide: The base waveguide material and size in the form of :code:`Box`.
            wg_height: The waveguide height.
            sub_eps: The substrate epsilon (defaults to air)
            sub_height: The height of the substrate (or the min height of the waveguide built on top of it)
            coupling_gap: The coupling gap specified means we get a pair of base blocks
            separated by :code:`coupling_gap`.
            block: Perturbing block.
            sep: Separation of the block from the base waveguide layer.
            vertical: Whether the perturbing block moves vertically, or laterally otherwise.
            rib_y: Rib section

        Returns:
            The resulting :code:`ModeSolver` with the modified :code:`eps` property.

        """
        if self.ndim == 1:
            raise NotImplementedError("Only implemented for 2d for now (most useful case).")
        if rib_y > 0:
            self.fill(rib_y + sub_height, waveguide.eps)
        self.fill(sub_height, sub_eps)
        wg_height = sub_height if wg_height is None else wg_height
        waveguide.align(self.center).valign(wg_height)
        sep = (sep, sep) if not isinstance(sep, Tuple) else sep
        d = coupling_gap / 2 + waveguide.size[0] / 2 if coupling_gap > 0 else 0
        waveguides = [waveguide.copy.translate(-d), waveguide.copy.translate(d)]
        blocks = []
        if vertical:
            blocks = [block.copy.align(waveguides[0]).valign(waveguides[0]).translate(dy=sep[0]),
                      block.copy.align(waveguides[1]).valign(waveguides[1]).translate(dy=sep[1])]
        elif block is not None:
            blocks = [block.copy.valign(wg_height).halign(waveguides[0], left=False).translate(-sep[0]),
                      block.copy.valign(wg_height).halign(waveguides[1]).translate(sep[1])]
        for wg in waveguides + blocks:
            self.set_eps((wg.center[0], wg.center[1], 0), (wg.size[0], wg.size[1], 0), wg.eps)
        return self

    def dispersion_sweep(self, wavelengths: np.ndarray, m: int = 6, pbar: Callable = None):
        """Dispersion sweep for cross sectional modes

        Args:
            wavelengths: Wavelengths :math:`\\lambda` of length :math:`L`
            m: Number of modes to solve
            pbar: Progress bar handle (to show progress using e.g. tqdm)

        Returns:
            A list of :math:`L` :code:`Modes` solution objects

        """
        solutions = []
        pbar = range if pbar is None else pbar
        for wavelength in pbar(wavelengths):
            solutions.append(self.solve(m, wavelength))
        return solutions


class ModeLibrary:
    """A data structure to contain the information about :math:`num_modes` cross-sectional modes

    Args:
        size: The size (1d or 2d) of the mode solver.
        spacing: The spacing of the mode solver.
        eps: The permittivity distribution for the mode solver.
        wavelength: The wavelength for the modes.
        num_modes: Number of modes that should be solved.
    """

    def __init__(self, size: Size, spacing: GridSpacing, eps: Union[float, np.ndarray],
                 wavelength: float = 1.55, num_modes: int = 1):
        self.solver = ModeSolver(
            size=size,
            spacing=spacing,
            eps=eps,
            wavelength=wavelength
        )
        self.ndim = self.solver.ndim
        self.betas, self.modes = self.solver.solve(num_modes)
        self.modes = self.modes
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
                                  -(mode - np.roll(mode, 1, axis=0)) / self.solver.cell_sizes[0])) / (
                               1j * self.solver.k0)
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
    def ns(self):
        """The refractive index for each mode :math:

        Returns:
            :math:`\\mathbf{n}`, an :code:`ndarray` for the refractive index of shape :code:`(M,)`
        """
        return self.betas / self.solver.k0

    @property
    def dbeta(self):
        return self.beta(0) - self.beta(1)

    @property
    def dn(self):
        return (self.beta(0) - self.beta(1)) / self.solver.k0

    @lru_cache()
    def te_ratio(self, mode_idx: int = 0):
        if self.ndim != 2:
            raise AttributeError("ndim must be 2, otherwise te_ratio is 1 or 0.")
        te_ratios = []
        habs = np.abs(self.h(mode_idx).squeeze())
        norms = np.asarray((np.linalg.norm(habs[0].flatten()), np.linalg.norm(habs[1].flatten())))
        te_ratios.append(norms[0] ** 2 / np.sum(norms ** 2))
        return np.asarray(te_ratios)

    def plot_sz(self, ax, mode_idx: int = 0, title: str = "Poynting", include_n: bool = False,
                title_size: float = 16, label_size=16):
        """Plot sz overlaid on the material

        Args:
            ax: Matplotlib axis handle.
            mode_idx: Mode index to plot.
            title: Title of the plot/subplot.
            include_n: Include the refractive index in the title.
            title_size: Fontsize of the title.
            label_size: Fontsize of the label.

        Returns:

        """
        if mode_idx > self.m - 1:
            raise ValueError("Out of range of number of solutions")
        plot_power_2d(ax, np.abs(self.sz(mode_idx).real), self.eps, spacing=self.solver.spacing[0])
        if include_n:
            ax.set_title(rf'{title}, $n_{mode_idx + 1} = {self.n(mode_idx):.4f}$', fontsize=title_size)
        else:
            ax.set_title(rf'{title}', fontsize=title_size)
        ax.text(x=0.9, y=0.9, s=rf'$s_z$', color='white', transform=ax.transAxes, fontsize=label_size)
        ratio = np.max((self.te_ratio(mode_idx), 1 - self.te_ratio(mode_idx)))
        polarization = "TE" if np.argmax((self.te_ratio(mode_idx), 1 - self.te_ratio(mode_idx))) > 0 else "TM"
        ax.text(x=0.05, y=0.9, s=rf'{polarization}[{ratio:.2f}]', color='white', transform=ax.transAxes)

    def plot_field(self, ax, idx: int = 0, axis: int = 1, use_h: bool = True, title: str = "Field",
                   include_n: bool = False, title_size: float = 16, label_size=16):
        """Plot field overlaid on the material.

        Args:
            ax: Matplotlib axis handle.
            idx: Mode index to plot.
            axis: Field axis to plot.
            use_h: Plot magnetic field :math:`\\mathbf{H}`.
            title: Title of the plot/subplot.
            include_n: Include the refractive index in the title.
            title_size: Fontsize of the title.
            label_size: Fontsize of the label.

        Returns:

        """
        field = self.h(mode_idx=0) if use_h else self.e(mode_idx=0)
        if idx > self.m - 1:
            ValueError("Out of range of number of solutions")
        if not (axis in (0, 1, 2)):
            ValueError(f"Axis expected to be (0, 1, 2) but got {axis}.")
        plot_field_2d(ax, field[idx][axis].real, self.eps, spacing=self.solver.spacing[0])
        if include_n:
            ax.set_title(rf'{title}, $n_{idx + 1} = {self.n(idx):.4f}$', fontsize=title_size)
        else:
            ax.set_title(rf'{title}', fontsize=title_size)
        ax.text(x=0.9, y=0.9, s=rf'$h_y$' if use_h else rf'$e_y$', color='black', transform=ax.transAxes,
                fontsize=label_size)
        ratio = np.max((self.te_ratio(idx), 1 - self.te_ratio(idx)))
        polarization = "TE" if np.argmax((self.te_ratio(idx), 1 - self.te_ratio(idx))) > 0 else "TM"
        ax.text(x=0.05, y=0.9, s=rf'{polarization}[{ratio:.2f}]', color='black', transform=ax.transAxes)

    def phase(self, length: float = 1):
        return self.solver.k0 * length * self.n()

    def overlap_fundamental(self, other_sol: "ModeLibrary"):
        return overlap(self.e(), self.h(), other_sol.e(), other_sol.h()) ** 2

    def place(self, mode_idx: int, grid: YeeGrid, center: Size, size: Size) -> np.ndarray:
        """Place at mode_idx in device with :math:`shape` and :math:`region`.

        Args:
            mode_idx: Mode index to place.
            grid: Finite-difference grid to place the mode.
            center: Specified center for placement.
            size: Specified size for placement.

        Returns:
            Places the mode into the provided grid at the requested center and size, with orientation of the mode
            automatically determined from the center and size provided.

        """
        region = grid.slice(center, size)
        if self.ndim == 2:
            x = np.zeros((3, *grid.shape), dtype=np.complex128)
            x[:, region[0], region[1], region[2]] = self.h(mode_idx)
        else:
            x = np.zeros(grid.shape, dtype=np.complex128)
            x[region[0], region[1]] = self.modes[mode_idx]
        return x

    def measure_fn(self, mode_idx: int = 0, use_jax: bool = False, tm_2d: bool = True):
        """Measure flux provided a mode indexed at :code:`mode_index`.

        Args:
            mode_idx: Mode index for the measurement.
            use_jax: Use jax.
            tm_2d: Use TM polarization (only relevant in the case of 2D simulations (i.e., 1D modes)).

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

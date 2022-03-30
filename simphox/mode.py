import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from .grid import YeeGrid
from .typing import Size, Spacing, Optional, Tuple, Union, Callable, Size2
from .utils import poynting_fn, Box
from .viz import plot_power_2d, plot_field_2d, plot_field_1d, hv_field_1d, hv_field_2d, hv_power_1d, hv_power_2d

try:  # pardiso (using Intel MKL) is much faster than scipy's solver
    from .mkl import feast_eigs
except OSError:  # if mkl isn't installed
    pass

try:
    from dphox.component import Pattern
    DPHOX_INSTALLED = True
except ImportError:
    DPHOX_INSTALLED = False

try:
    import holoviews as hv
    HOLOVIEWS_INSTALLED = True
except ImportError:
    HOLOVIEWS_INSTALLED = False


class ModeSolver(YeeGrid):
    """FDFD-based waveguide mode solver

    Notes:
        This class is capable of finding 1d or 2d cross-sectional modes, which are
        useful for defining sources and calculating the number of propagating modes that exist in multi-mode
        interferometers (MMIs).

        We can solve either the 1d or 2d case using 1 or 2 components of the field respectively.
        Note that in `simphox` units, we assume :math:`k_0 = \\frac{2\\pi}{\\lambda} = \\omega`,
        letting :math:`c = \\epsilon_0 = \\mu_0 = 1` for simplicity. We define the *wavenumber* :math`:\\beta_m`
        for mode :math:`m` to be the square root of the eigenvalue (hence the :math:`\beta^2` terms in the later
        equations) of :math:`C_{\\mathrm{1d}}` and :math:`C_{\\mathrm{2d}}` for the respective problems.

        For the 1d case, we consider the case where the simulation is a line cross-section of a 2d grid in the
        :math:`xy`-plane. In that case, we solve for the $z$-component of an :math:`h_z`-polarized mode of the form
        :math:`\\mathbf{h}_m = (0, 0, h_z(y)) e^{-i\\beta_m x}`. The solutions for :math:`h_z(y)` correspond to the
        simple equation:

        .. math::
            \\beta^2 h_z &= \\partial_y^2 h_z + k_0^2 \\epsilon_z h_z

            \\beta_m^2 \\mathbf{h}_{m} &= C_{\\mathrm{1d}} \\mathbf{h}_{m}

        For the 2d case, we cannot make this type of assumption.
        Instead we solve the frequency-domain Maxwell's equations for the case of :math:`z`-translation symmetry
        (here, we consider propagation along $z$ instead of $x$ to match convention).
        This time, we solve for an :math:`\\mathbf{h}`-field of the form
        :math:`\\mathbf{h}_m = \\mathbf{h}(x, y) e^{-i\\beta_m z}`.
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
        size: Tuple of size 1, or 2 representing the size in arbitrary units.
        spacing: Spacing (microns) between each pixel along each axis (MUST be in same units as `size`)
        wavelength: Wavelength for the mode solver.
        eps: Relative permittivity :math:`\\epsilon`.
    """

    def __init__(self, size: Size, spacing: Spacing, wavelength: float = 1.55,
                 eps: Union[float, np.ndarray] = 1, name: str = 'mode'):

        super(ModeSolver, self).__init__(
            size=size,
            spacing=spacing,
            eps=eps,
            pml=None,
            name=name
        )

        self.wavelength = wavelength

    @property
    def k0(self):
        return 2 * np.pi / self.wavelength

    @property
    def waveguide_mode_matrix(self) -> sp.spmatrix:
        """Build the WaveGuide Mode (WGM) operator (for 1D or 2D grid only)

        The WGM operator :math:`C(\\omega)` acts on the magnetic field
        :math:`\\mathbf{h}` of the form :code:`(hx, hy)`, which assumes cross-section translational x-symmetry:

        .. math::
            C \\mathbf{h}_m &= \\lambda_m \\mathbf{h}_m,

        where :math:`0 \\leq m < M` for the :math:`M` modes with the largest wavenumbers
        (:math:`\\beta_m = \\sqrt{\\lambda_m}`).

        Returns:
            Magnetic field operator :math:`C`.
        """

        if not self.ndim <= 2:
            raise AttributeError("Grid dimension must be 1 or 2")

        df, db = self.deriv_forward, self.deriv_backward

        if self.ndim == 2:
            eps = [e.flatten() for e in self.eps_t]
            eps_10 = sp.diags(np.hstack((eps[1], eps[0])))
            m1 = eps_10 * self.k0 ** 2
            m2 = eps_10 @ sp.vstack([-df[1], df[0]]) @ sp.diags(1 / eps[2]) @ sp.hstack([-db[1], db[0]])
            m3 = sp.vstack(db[:2]) @ sp.hstack(df[:2])
            return m1 + m2 + m3
        else:
            return sp.diags(self.eps.flatten()) * self.k0 ** 2 + df[0].dot(db[0])

    C = waveguide_mode_matrix  # C is the matrix for the guided mode eigensolver

    def e2h(self, e: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """Convert magnetic field :math:`\\mathbf{e}` to electric field :math:`\\mathbf{h}`.

        Usage is: :code:`h = mode.e2h(e)`, where :code:`e` is grid-shaped (not flattened)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:

        .. math::
            i \\omega \\mu \\mathbf{h} &= \\nabla \\times \\mathbf{e}

        Returns:
            The h-field converted from the e-field.

        """
        return self.curl_fn(beta=beta)(self.reshape(e)) / (1j * self.k0)

    def h2e(self, h: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """Convert magnetic field :math:`\\mathbf{h}` to electric field :math:`\\mathbf{e}`.

        Usage is: :code:`e = mode.h2e(h)`, where :code:`h` is grid-shaped (not flattened)

        Mathematically, this represents rearranging the Maxwell equation in the frequency domain:

        .. math::
            -i \\omega \\epsilon \\mathbf{e} = \\nabla \\times \\mathbf{h}.

        Returns:
            Function to convert h-field to e-field.

        """
        return self.curl_fn(of_h=True, beta=beta)(self.reshape(h)) / (1j * self.k0 * self.eps_t)

    def solve(self, max_num_modes: int = 6, beta_guess: Optional[Union[float, Size2]] = None,
              mode_guess: Optional[np.ndarray] = None, tol: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
        """FDFD waveguide mode solver

        Solve for waveguide modes (x-translational symmetry) by finding the eigenvalues of :math:`C`.

        .. math::
            C \\mathbf{h}_m &= \\lambda_m \\mathbf{h}_m,

        where :math:`0 \\leq m < M` for the :math:`M` (`num_modes`) modes with the largest wavenumbers
        (:math:`\\beta_m = \\pm \\sqrt{\\lambda_m}`).

        Args:
            max_num_modes: Maximum number of modes to return (less are returned if they correspond
                to an imaginary :math:`\\beta`).
            beta_guess: Guess for propagation constant :math:`\\beta` (the eigenvalue).
            mode_guess: Guess for the mode :math:`\\boldsymbol{h}`
            tol: Tolerance of the mode eigensolver.

        Returns:
            `num_modes` (:math:`M`) largest propagation constants (:math:`\\sqrt{\\lambda_m(C)}`)
            and corresponding modes (:math:`\\mathbf{h}_m`) of shape :code:`(num_modes, n)`.
        """

        df = self.deriv_forward
        if isinstance(beta_guess, float) or beta_guess is None:
            sigma = beta_guess ** 2 if beta_guess else (self.k0 * np.sqrt(np.max(self.eps))) ** 2
            eigvals, eigvecs = eigs(self.waveguide_mode_matrix, k=max_num_modes, sigma=sigma, tol=tol)
        elif isinstance(beta_guess, tuple):
            erange = beta_guess[0] ** 2, beta_guess[1] ** 2
            eigvals, eigvecs, _, _, _, _ = feast_eigs(self.waveguide_mode_matrix, erange=erange, k=max_num_modes)
        else:
            raise TypeError(f'Expected beta_guess to be None, float, or Tuple[float, float] but got {type(beta_guess)}')

        useful_modes = np.where(eigvals.real > 0)[0]
        eigvals = eigvals[useful_modes]
        eigvecs = eigvecs[:, useful_modes]

        inds_sorted = np.asarray(np.argsort(np.sqrt(eigvals.real))[::-1])
        eigvals = eigvals[inds_sorted]
        eigvecs = eigvecs[:, inds_sorted]

        if self.ndim > 1:
            hz = sp.hstack(df[:2]) @ eigvecs / (1j * np.sqrt(eigvals))
            h = np.vstack((eigvecs, hz))
        else:
            h = eigvecs

        h = h.T
        # max_phase_ref = np.array([h[i, idx] for i, idx in enumerate(np.argmax(np.abs(h), axis=1))])[:, np.newaxis]
        return np.sqrt(eigvals[useful_modes].real), h * np.exp(-1j * np.angle(h[:, :1]))

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
        max_num_modes: Maximum number of modes that should be solved.
    """

    def __init__(self, solver: ModeSolver, max_num_modes: int = 6):
        self.solver = solver
        self.max_num_modes = max_num_modes
        self._update_solve(self.solver.wavelength)

    def _update_solve(self, wavelength: float):
        self.ndim = self.solver.ndim
        self.solver.wavelength = wavelength
        self.betas, self.modes = self.solver.solve(self.max_num_modes)
        self.modes = self.modes
        self.eps = self.solver.eps
        self.num_modes = self.m = len(self.betas)
        self.o = np.zeros_like(self.modes[0])

    def update_wavelength(self, wavelength: float):
        """Update the wavelength for the mode solver.

        This method updates the wavelength for the mode solver and proceeds to solve at that new wavelength.
        This is useful for dispersion simulations.

        Args:
            wavelength: The new wavelength for the mode solver

        Returns:
            The current ModeLibrary object with the modified wavelength.

        """
        self._update_solve(wavelength)
        return self

    @classmethod
    def from_block_design(cls, size: Size, spacing: Spacing, waveguide: Box, num_modes: int = 6,
                          wavelength: float = 1.55, wg_height: Optional[float] = None,
                          sub_eps: float = 1, sub_height: float = 0, coupling_gap: float = 0,
                          block: Optional[Box] = None, sep: Size = (0, 0), vertical: bool = False, rib_y: float = 0):
        """A helper function for designing a useful port or cross section for a mode solver.

        Args:
            size: Size of the overall simulation.
            spacing: Spacing between points in the grid for the mode solver.
            waveguide: The base waveguide material and size in the form of :code:`Box`.
            wavelength: Wavelength for the mode solver.
            num_modes: Number of modes that should be solved.
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
            The resulting :code:`ModeLibrary` with the modified :code:`eps` property.

        """
        solver = ModeSolver(size, spacing, wavelength).block_design(
            waveguide, wg_height, sub_eps, sub_height, coupling_gap,
            block, sep, vertical, rib_y)
        return cls(solver, num_modes)

    def _check_num_modes(self, mode_idx: int):
        if mode_idx > self.m - 1:
            raise ValueError(f"Out of range of number of guided mode solutions {self.m}.")
        return mode_idx

    def h(self, mode_idx: int = 0, tm_2d: bool = True) -> np.ndarray:
        """Magnetic field :math:`\\mathbf{H}` for the mode of specified index

        Args:
            mode_idx: The mode index :math:`m \\leq M`
            tm_2d: If the mode is using a 1d distribution, this specifies if the mode is TM (otherwise TE)

        Returns:
            :math:`\\mathbf{H}_m`, an :code:`ndarray` of the form :code:`(3, X, Y)` for mode :math:`m \\leq M`

        """
        mode = self.modes[self._check_num_modes(mode_idx)]
        if self.ndim == 1:
            if tm_2d:
                mode = np.hstack((self.o, mode, self.o))
            else:
                mode = np.hstack((1j * self.betas[mode_idx] * mode, self.o,
                                  -(mode - np.roll(mode, 1, axis=0)) / self.solver.cells[0])) / (
                               1j * self.solver.k0)
        return self.solver.reshape(mode)

    def e(self, mode_idx: int = 0, tm_2d: bool = True) -> np.ndarray:
        """Electric field :math:`\\mathbf{E}` for the mode of specified index

        Args:
            mode_idx: The mode index :math:`m \\leq M`
            tm_2d: If the mode is using a 1d distribution, this specifies if the mode is TM (otherwise TE)

        Returns:
            :math:`\\mathbf{E}_m`, an :code:`ndarray` of shape :code:`(3, X, Y, Z)` for mode :math:`m \\leq M`

        """
        self._check_num_modes(mode_idx)
        if self.ndim == 2:
            return self.solver.h2e(self.h(mode_idx), self.betas[mode_idx])
        else:
            mode = self.modes[mode_idx]
            if tm_2d:
                mode = np.hstack((1j * self.betas[mode_idx] * mode, self.o,
                                  -(np.roll(mode, -1, axis=0) - mode) / self.solver.cells[0])) / (
                               1j * self.solver.k0 * self.solver.eps_t.flatten())
            else:
                mode = np.hstack((self.o, mode, self.o))
            return self.solver.reshape(mode)

    def sz(self, mode_idx: int = 0) -> np.ndarray:
        """Poynting vector :math:`\\mathbf{S}_z` for the mode of specified index

        Args:
            mode_idx: The mode index :math:`m \\leq M`

        Returns:
            :math:`\\mathbf{S}_{m, z}`, the z-component of Poynting vector (correspoding to power),
            of shape :code:`(X, Y)`

        """
        self._check_num_modes(mode_idx)
        return poynting_fn(2)(self.e(mode_idx), self.h(mode_idx)).squeeze()

    def beta(self, mode_idx: int = 0) -> float:
        """Fundamental mode propagation constant :math:`\\beta` for mode indexed by :code:`mode_idx`.

        Args:
            mode_idx: The mode index :math:`m \\leq M`

        Returns:
            :math:`\\beta_m` for mode :math:`m \\leq M`
        """
        return self.betas[self._check_num_modes(mode_idx)]

    def n(self, mode_idx: int = 0):
        """Effective index :math:`n` for mode indexed by :code:`mode_idx`.

        Returns:
            The effective index :math:`n`
        """
        return self.betas[self._check_num_modes(mode_idx)] / self.solver.k0

    @property
    def ns(self):
        """The refractive index for all modes corresponding to :code:`betas`.

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

    def te_ratio(self, mode_idx: int = 0):
        if self.ndim != 2:
            raise AttributeError("ndim must be 2, otherwise te_ratio is 1 or 0.")
        te_ratios = []
        habs = np.abs(self.h(mode_idx).squeeze())
        norms = np.asarray((np.linalg.norm(habs[0].flatten()), np.linalg.norm(habs[1].flatten())))
        te_ratios.append(norms[0] ** 2 / np.sum(norms ** 2))
        return np.asarray(te_ratios)

    def plot_power(self, ax, idx: int = 0, title: str = "Power", include_n: bool = True,
                   title_size: float = 16, label_size=16):
        """Plot sz overlaid on the material

        Args:
            ax: Matplotlib axis handle.
            idx: Mode index to plot.
            title: Title of the plot/subplot.
            include_n: Include the refractive index in the title.
            title_size: Fontsize of the title.
            label_size: Fontsize of the label.

        """
        if idx > self.m - 1:
            raise ValueError("Out of range of number of solutions")
        if include_n:
            ax.set_title(rf'{title}, $n_{idx + 1} = {self.n(idx):.4f}$', fontsize=title_size)
        else:
            ax.set_title(rf'{title}', fontsize=title_size)
        if self.ndim == 2:
            plot_power_2d(ax, np.abs(self.sz(idx).real), self.eps, spacing=self.solver.spacing[0])
            ax.text(x=0.9, y=0.9, s=rf'$s_z$', color='white', transform=ax.transAxes, fontsize=label_size)
            ratio = np.max((self.te_ratio(idx), 1 - self.te_ratio(idx)))
            polarization = "TE" if np.argmax((self.te_ratio(idx), 1 - self.te_ratio(idx))) > 0 else "TM"
            ax.text(x=0.05, y=0.9, s=rf'{polarization}[{ratio:.2f}]', color='white', transform=ax.transAxes)
        else:
            plot_field_1d(ax, np.abs(self.sz(idx).real), rf'Power',
                          self.eps, spacing=self.solver.spacing[0])

    def _get_field_component(self, idx: int = 0, axis: Union[int, str] = 1, use_h: bool = True):
        field = self.h(mode_idx=idx) if use_h else self.e(mode_idx=idx)
        if idx > self.m - 1:
            raise ValueError(f"Out of range of number of solutions {self.m}")
        if not (axis in (0, 1, 2, 'x', 'y', 'z')):
            raise ValueError(f"Axis expected to be (0, 1, 2) or ('x', 'y', 'z') but got {axis}.")
        a = ['x', 'y', 'z'][axis] if isinstance(axis, int) else axis
        axis = {'x': 0, 'y': 1, 'z': 2}[axis] if isinstance(axis, str) else axis
        return field[axis].squeeze(), rf'$h_{a}$' if use_h else rf'$e_{a}$'

    def plot_field(self, ax, idx: int = 0, axis: Union[int, str] = 1, use_h: bool = True, title: str = "Field",
                   include_n: bool = True, title_size: float = 16, label_size=16):
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

        """
        if include_n:
            ax.set_title(rf'{title}, $n_{idx + 1} = {self.n(idx):.4f}$', fontsize=title_size)
        else:
            ax.set_title(rf'{title}', fontsize=title_size)
        field, label = self._get_field_component(idx, axis, use_h)
        if self.ndim == 2:
            plot_field_2d(ax, field.real.squeeze(), self.eps, spacing=self.solver.spacing[0])
            ax.text(x=0.9, y=0.9, s=label, color='black', transform=ax.transAxes,
                    fontsize=label_size)
            ratio = np.max((self.te_ratio(idx), 1 - self.te_ratio(idx)))
            polarization = "TE" if np.argmax((self.te_ratio(idx), 1 - self.te_ratio(idx))) > 0 else "TM"
            ax.text(x=0.05, y=0.9, s=rf'{polarization}[{ratio:.2f}]', color='black', transform=ax.transAxes)
        else:
            plot_field_1d(ax, field[axis].real.squeeze(), rf'Field({label})', self.eps, spacing=self.solver.spacing[0])

    def phase(self, length: float = 1, mode_idx: int = 0):
        """Measure the phase delay propagated over a length

        Args:
            length: The length over which to propagate the mode
            mode_idx: The mode idx to propagate

        Returns:
            The aggregate phase delay over a length.

        """
        return self.solver.k0 * length * self.n(mode_idx)

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
            # Find the place axis (the poynting direction, where the size should be 0)
            place_axis = np.where(np.array(size) == 0)[0][0]

            # Find the reorientation of field axes based on place_axis
            # 0: (0, 1, 2) -> (2, 0, 1)
            # 1: (0, 1, 2) -> (0, 2, 1)
            # 2: (0, 1, 2) -> (0, 1, 2)
            axes = [
                np.asarray((2, 0, 1), dtype=int),
                np.asarray((0, 2, 1), dtype=int),
                np.asarray((0, 1, 2), dtype=int)
            ][place_axis]
            x = np.zeros((3, *grid.shape), dtype=np.complex128)
            x[(None,) + region] = self.h(mode_idx).transpose((0, *tuple(1 + axes)))
        else:
            x = np.zeros(grid.shape, dtype=np.complex128)
            x[region[:2]] = self.modes[mode_idx]
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

    def evolve(self, length: Union[float, np.ndarray], mode_weights: Tuple[float, ...] = (1,), use_h: bool = True):
        """Evolve a mode in time according to :code:`mode_weights`.

        Args:
            length: The length (or time) over which the mode is evolving. If a 1d array is provided,
                output the mode evaluated at all the lengths in the array.
            mode_weights: The mode weights.
            use_h: Use the h field as the mode profile.

        Returns:
            The evolved mode profile(s).

        """
        if use_h:
            f = sum([self.h(idx)[..., np.newaxis] * np.exp(1j * self.beta(idx) * length)[np.newaxis] * weight
                     for idx, weight in enumerate(mode_weights)])
        else:
            f = sum([self.e(idx)[..., np.newaxis] * np.exp(1j * self.beta(idx) * length)[np.newaxis] * weight
                     for idx, weight in enumerate(mode_weights)])
        return f

    def beat_length(self, idx0: int = 0, idx1: int = 1):
        return 2 * np.pi / (self.beta(idx0) - self.beta(idx1))

    def evolve_viz(self, max_length: float, mode_weights: Tuple[float, ...] = (1,), power: bool = True):
        """Use holoviews to dynamically visualize the evolution of a multimode field

        Args:
            max_length: Maximum length for the evolution of modes.
            mode_weights: Mode weights for the mode evolution.
            power: Visualize the power, otherwise visualize the field.

        Returns:

        """
        if not HOLOVIEWS_INSTALLED:
            raise ImportError("Holoviews not installed.")

        def _evolve(length: float):
            h = self.evolve(length, mode_weights)
            if power:
                e = self.evolve(length, mode_weights, use_h=False)
                s = poynting_fn(2)(e, h).squeeze()
                if self.ndim == 1:
                    return hv_power_1d(s, self.eps, self.solver.spacing)
                else:
                    return hv_power_2d(s, self.eps, self.solver.spacing[0])
            else:
                if self.ndim == 1:
                    return hv_field_1d(h[1], self.eps, self.solver.spacing)
                else:
                    return hv_field_2d(h[1], self.eps, self.solver.spacing[0])

        return hv.DynamicMap(_evolve, kdims=['length']).redim.values(
            length=np.linspace(0, max_length, int(max_length / self.solver.spacing[0]) + 1)
        )



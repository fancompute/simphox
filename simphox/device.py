import numpy as np
from .typing import Tuple, Dim2, Optional, Callable, List
from .fdfd import FDFD
from .utils import poynting_z, overlap
from .viz import plot_field_2d, plot_power_2d
from functools import lru_cache
import copy


class Material:
    def __init__(self, name: str, facecolor: Tuple[float, float, float] = None, eps: float = None):
        self.name = name
        self.eps = eps
        self.facecolor = facecolor

    def __str__(self):
        return self.name


SILICON = Material('Silicon', (0.3, 0.3, 0.3), 3.4784 ** 2)
POLYSILICON = Material('Poly-Si', (0.5, 0.5, 0.5), 3.4784 ** 2)
OXIDE = Material('Oxide', (0.6, 0, 0), 1.4442 ** 2)
NITRIDE = Material('Nitride', (0, 0, 0.7), 1.996 ** 2)
LS_NITRIDE = Material('Low-Stress Nitride', (0, 0.4, 1))
LT_OXIDE = Material('Low-Temp Oxide', (0.8, 0.2, 0.2), 1.4442 ** 2)
ALUMINUM = Material('Aluminum', (0, 0.5, 0))
ALUMINA = Material('Alumina', (0.2, 0, 0.2), 1.75)
ETCH = Material('Etch', (0, 0, 0))


class MaterialBlock:
    def __init__(self, dim: Dim2, material: Material):
        """Material block (substrate or waveguide)

        Args:
            dim: Dimension tuple of the form :code:`(x, y)` for the material block
            material: Material for the block
        """
        self.dim = dim
        self.material = material
        self.x = dim[0]
        self.y = dim[1]
        self.eps = self.material.eps


class Modes:
    def __init__(self, betas: np.ndarray, modes: np.ndarray, fdfd: FDFD):
        """A data structure to contain the information about :math:`M` cross-sectional modes

        Args:
            betas: Propagation constants :math:`\\beta` for :math:`M` modes
            modes: Magnetic field :math:`\mathbf{H}` for :math:`M` modes
            fdfd: The FDFD structure containing the epsilon and mesh information
        """
        self.betas = betas.real
        self.modes = modes
        self.fdfd = fdfd
        self.eps = fdfd.eps
        self.num_modes = self.m = len(self.betas)

    @lru_cache()
    def h(self, mode_idx: int = 0) -> np.ndarray:
        """Magnetic field :math:`\mathbf{H}` for the mode of specified index

        Args:
            mode_idx: The mode index :math:`m \\leq M`

        Returns:
            :math:`\mathbf{H}_m`, an :code:`ndarray` of the form :code:`(3, X, Y)` for mode :math:`m \\leq M`

        """
        return self.fdfd.reshape(self.modes[mode_idx])

    @lru_cache()
    def e(self, mode_idx: int = 0) -> np.ndarray:
        """Electric field :math:`\mathbf{E}` for the mode of specified index

        Args:
            mode_idx: The mode index :math:`m \\leq M`

        Returns:
            :math:`\mathbf{E}_m`, an :code:`ndarray` of shape :code:`(3, X, Y)` for mode :math:`m \\leq M`

        """
        return self.fdfd.h2e(self.h(mode_idx), self.betas[mode_idx])

    @lru_cache()
    def sz(self, mode_idx: int = 0) -> np.ndarray:
        """Poynting vector :math:`\mathbf{S}_z` for the mode of specified index

        Args:
            mode_idx: The mode index :math:`m \\leq M`

        Returns:
            :math:`\mathbf{S}_{m, z}`, the z-component of Poynting vector (correspoding to power),
            of shape :code:`(X, Y)`

        """
        return poynting_z(self.e(mode_idx), self.h(mode_idx)).squeeze()

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
        return self.betas[mode_idx] / self.fdfd.k0

    @property
    @lru_cache()
    def hs(self):
        """An array for the magnetic fields `\mathbf{H}` corresponding to all :math:`M` modes

        Returns:
           :math:`\mathbf{H}`, an :code:`ndarray` of shape :code:`(M, 3, X, Y)`
        """
        hs = []
        for mode in self.modes:
            hs.append(self.fdfd.reshape(mode))
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
            es.append(self.fdfd.h2e(h[..., np.newaxis], beta))
        return np.stack(es).squeeze()

    @property
    @lru_cache()
    def szs(self):
        """An array for the magnetic fields `\mathbf{S}_z` corresponding to all :math:`M` modes

        Returns:
           :math:`\mathbf{S}_z`, an :code:`ndarray` of shape :code:`(M, X, Y)`
        """
        szs = []
        for beta, e, h in zip(self.betas, self.es, self.hs):
            szs.append(poynting_z(e[..., np.newaxis], h[..., np.newaxis]))
        return np.stack(szs).squeeze()

    @property
    @lru_cache()
    def ns(self):
        return self.betas / self.fdfd.k0

    @property
    def dbeta(self):
        return self.beta(0) - self.beta(1)

    @property
    def dn(self):
        return (self.beta(0) - self.beta(1)) / self.fdfd.k0

    @property
    def te_ratios(self):
        te_ratios = []
        for h in self.hs:
            habs = np.abs(h.squeeze())
            norms = np.asarray((np.linalg.norm(habs[0].flatten()), np.linalg.norm(habs[1].flatten())))
            te_ratios.append(norms[0] ** 2 / np.sum(norms ** 2))
        return np.asarray(te_ratios)

    def s_matrix(self, other_modes: "Modes"):
        return np.asarray([
            [np.sum(poynting_z(e_o, h_i) + poynting_z(e_i, h_o)).real / np.sum(2 * poynting_z(e_i, h_i)).real
             for h_o, e_o in zip(other_modes.hs, other_modes.es)]
            for h_i, e_i in zip(self.hs, self.es)
        ])

    def fundamental_coeff(self, other_modes: "Modes"):
        e_i, h_i = self.e(), self.h()
        e_o, h_o = other_modes.e(), other_modes.h()
        return np.sum(poynting_z(e_o, h_i) + poynting_z(e_i, h_o)).real

    def plot_sz(self, ax, idx: int = 0, title: str = "Poynting", include_n: bool = False,
                title_size: float = 16, label_size=16):
        if idx > self.m - 1:
            ValueError("Out of range of number of solutions")
        plot_power_2d(ax, np.abs(self.sz(idx).real), self.eps, spacing=self.fdfd.spacing[0])
        if include_n:
            ax.set_title(rf'{title}, $n_{idx + 1} = {self.n(idx):.4f}$', fontsize=title_size)
        else:
            ax.set_title(rf'{title}', fontsize=title_size)
        ax.text(x=0.9, y=0.9, s=rf'$s_z$', color='white', transform=ax.transAxes, fontsize=label_size)
        ratio = np.max((self.te_ratios[idx], 1 - self.te_ratios[idx]))
        polarization = "TE" if np.argmax((self.te_ratios[idx], 1 - self.te_ratios[idx])) > 0 else "TM"
        ax.text(x=0.05, y=0.9, s=rf'{polarization}[{ratio:.2f}]', color='white', transform=ax.transAxes)

    def plot_field(self, ax, idx: int = 0, axis: int = 1, use_e: bool = False, title: str = "Field",
                   title_size: float=16, label_size=16, include_n: bool = False):
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
        if not (axis == 0 or axis == 1 or axis == 2):
            ValueError("Out of range of number of solutions")
        plot_field_2d(ax, field[idx][axis].real, self.eps, spacing=self.fdfd.spacing[0])
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
        return self.fdfd.k0 * length * self.n()

    def overlap_fundamental(self, other_sol: "Modes"):
        return overlap(self.e(), self.h(), other_sol.e(), other_sol.h()) ** 2


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
        self.nx = int(self.size[0] / spacing)
        self.ny = int(self.size[1] / spacing)
        self.wg_height = wg_height
        self.wg = wg
        self.sub = sub
        self.rib_y = rib_y

    def solve(self, eps: np.ndarray, m: int = 6, wavelength: float = 1.55) -> Modes:
        fdfd = FDFD(
            shape=(self.nx, self.ny),
            spacing=self.spacing,
            wavelength=wavelength,
            eps=eps
        )
        beta, modes = fdfd.wgm_solve(num_modes=m, beta_guess=fdfd.k0 * np.sqrt(self.wg.material.eps))
        solution = Modes(beta, modes, fdfd)
        return solution

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

        xr_wg = (center - int(wg.x / 2 / dx), center + int(wg.x / 2 / dx))
        yr_wg = slice(int(wg_y[0] / dx), int(wg_y[1] / dx))
        eps = np.ones((nx, ny))
        eps[:, :int(self.sub.y / dx)] = sub.material.eps
        eps[slice(*xr_wg), yr_wg] = wg.material.eps
        eps[:, int(self.sub.y / dx):int(self.sub.y / dx) + int(self.rib_y / dx)] = wg.material.eps

        if vert_ps is not None:
            ps_y = (wg.y + self.wg_height + sep, wg.y + self.wg_height + sep + vert_ps.y)
            xr_ps = slice(center - int(vert_ps.x / 2 / dx), center + int(vert_ps.x / 2 / dx))
            yr_ps = slice(int(ps_y[0] / dx), int(ps_y[1] / dx))
            eps[xr_ps, yr_ps] = vert_ps.material.eps

        if lat_ps is not None:
            xrps_l = slice(xr_wg[0] - int(lat_ps.x / dx) - int(sep / dx), xr_wg[0] - int(sep / dx))
            xrps_r = slice(xr_wg[1] + int(sep / dx), xr_wg[1] + int(sep / dx) + int(lat_ps.x / dx))
            eps[xrps_l, yr_wg] = lat_ps.material.eps
            eps[xrps_r, yr_wg] = lat_ps.material.eps

        return eps

    def coupled(self, gap: float, vert_ps: Optional[MaterialBlock] = None,
                lat_ps: Optional[MaterialBlock] = None,
                seps: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """Coupled-waveguide permittivity with an optional pair of phase shifter blocks placed above

        Args:
            gap: coupling gap for the interaction region
            ps: phase shifter :code:`MaterialBlock`
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
                pbar: Callable = None) -> List[Modes]:
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
                 pbar: Callable = None) -> List[Modes]:
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

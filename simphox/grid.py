import dataclasses
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .typing import Size, Size3, Spacing, Optional, List, Union, Dict, Op, Tuple
from .utils import curl_fn, yee_avg, fix_dataclass_init_docs, Box

try:
    DPHOX_IMPORTED = True
    from dphox.pattern import Pattern
except ImportError:
    DPHOX_IMPORTED = False


@fix_dataclass_init_docs
@dataclasses.dataclass
class Port:
    """Port to define where sources and measurements lie in photonic simulations.

    A port defines the center and angle/orientation in a design.

    Args:
        x: x position of the port
        y: y position of the port
        a: angle (orientation) of the port (in degrees)
        w: the width of the port (specified in design, mostly used for simulation)
        z: z position of the port (not specified in design, mostly used for simulation)
        h: the height of the port (not specified in design, mostly used for simulation)
    """
    x: float
    y: float
    a: float = 0
    w: float = 0
    z: float = 0
    h: float = 0

    def __post_init__(self):
        self.xy = (self.x, self.y)
        self.xya = (self.x, self.y, self.a)
        self.xyz = (self.x, self.y, self.z)
        self.center = np.array(self.xyz)

    @property
    def size(self):
        if np.mod(self.a, 90) != 0:
            raise ValueError(f"Require angle to be a multiple a multiple of 90 but got {self.a}")
        return np.array((self.w, 0, self.h)) if np.mod(self.a, 180) != 0 else np.array((0, self.w, self.h))


class Grid:
    def __init__(self, size: Size, spacing: Spacing, eps: Union[float, np.ndarray] = 1.0):
        """Grid object accomodating any electromagnetic simulation (FDFD, FDTD, BPM, etc.)

        Args:
            size: Tuple of size 1, 2, or 3 representing the size of the grid
            spacing: Spacing (microns) between each pixel along each axis (must be same dim as `grid_shape`)
            eps: Relative permittivity (
        """
        self.size = np.asarray(size)
        self.spacing = spacing * np.ones(len(size)) if isinstance(spacing, int) or isinstance(spacing, float) else np.asarray(spacing)
        self.ndim = len(size)
        if not self.ndim == self.spacing.size:
            raise AttributeError(f'Require size.size == ndim == spacing.size but got '
                                 f'{self.size.size} != {self.spacing.size}')
        self.shape = np.around(self.size / self.spacing).astype(int)
        self.shape3 = np.hstack((self.shape, np.ones((3 - self.ndim,), dtype=self.shape.dtype)))
        self.spacing3 = np.hstack((self.spacing, np.ones((3 - self.ndim,), dtype=self.spacing.dtype) * np.inf))
        self.size3 = np.hstack((self.size, np.zeros((3 - self.ndim,), dtype=self.size.dtype)))
        self.center = self.size3 / 2
        self.field_shape = (3, *self.shape3)

        self.n = np.prod(self.shape)
        self.eps: np.ndarray = np.ones(self.shape) * eps if not isinstance(eps, np.ndarray) else eps
        if not tuple(self.shape) == self.eps.shape:
            raise AttributeError(f'Require grid.shape == eps.shape but got '
                                 f'{self.shape} != {self.eps.shape}')

        self.cells = [(self.spacing[i] * np.ones((self.shape[i],)) if self.ndim > 1 else self.spacing * np.ones(self.shape))
                           if i < self.ndim else np.ones((1,)) for i in range(3)]
        self.pos = [np.hstack((0, np.cumsum(dx))) if dx.size > 1 else np.asarray((0,)) for dx in self.cells]
        self.components = []

        # used to handle special functions of waveguide-based components
        self.port: Dict[str, Port] = {}

    def fill(self, height: float, eps: float) -> "Grid":
        """Fill grid up to `height`, typically used for substrate + cladding epsilon settings

        Args:
            height: Maximum final dimension of the fill operation (`y` if 2D, `z` if 3D).
            eps: Relative permittivity to fill.

        Returns:
            The modified :code:`Grid` for chaining (:code:`self`)

        """
        if height > 0:
            self.eps[..., :int(height / self.spacing[-1])] = eps
        else:
            self.eps = np.ones_like(self.eps) * eps
        return self

    def add(self, component: "Pattern", eps: float, zmin: float = None, thickness: float = None) -> "Grid":
        """Add a component to the grid.

        Args:
            component: component to add
            eps: permittivity of the component being added (isotropic only, for now)
            zmin: minimum z extent of the component
            thickness: component thickness (`zmax = zmin + thickness`)

        Returns:
            The modified :code:`Grid` for chaining (:code:`self`)

        """
        b = component.bounds
        if not b[0] >= 0 and b[1] >= 0 and b[2] <= self.size[0] and b[3] <= self.size[1]:
            raise ValueError('The pattern must have min x, y >= 0 and max x, y less than size.')
        self.components.append(component)
        mask = component.mask(self.shape[:2], self.spacing)[:self.eps.shape[0], :self.eps.shape[1]]
        if self.ndim == 2:
            self.eps[mask == 1] = eps
        else:
            zidx = (int(zmin / self.spacing[0]), int((zmin + thickness) / self.spacing[1]))
            self.eps[mask == 1, zidx[0]:zidx[1]] = eps
        self.port = {port_name: Port(*port.xya, port.w, zmin + thickness / 2, thickness)
                     for port_name, port in component.port.items()}
        return self

    def set_eps(self, center: Size3, size: Size3, eps: float):
        """Set the region specified by :code:`center`, :code:`size` (in grid units) to :code:`eps`.

        Args:
            center: Center of the region.
            size: Size of the region.
            eps: Epsilon (relative permittivity) to set.

        Returns:
            The modified :code:`Grid` for chaining (:code:`self`)

        """
        s = self.slice(center, size, squeezed=True)
        eps_3d = self.eps.reshape(self.shape3)
        eps_3d[s] = eps
        self.eps = eps_3d.squeeze()
        return self

    def mask(self, center: Size3, size: Size3):
        """Given a size and center, this function defines a mask which sets pixels in the region corresponding to
        :code:`center` and :code:`size` to 1 and all other pixels to zero.

        Args:
            center: position of the mask in (x, y, z) in the units of the simulation (note: NOT in terms of array index)
            size: size of the mask box in (x, y, z) in the units of the simulation (note: NOT in terms of array index)

        Returns:
            The mask array of size :code:`grid.shape`.

        """
        s = self.slice(center, size, squeezed=True)
        mask = np.zeros(self.shape3)
        mask[s] = 1
        return mask.squeeze()

    def reshape(self, v: np.ndarray) -> np.ndarray:
        """A simple method to reshape flat 3d field array into the grid shape

        Args:
            v: vector of size :code:`3n` to rearrange into array of size :code:`(3, nx, ny, nz)`

        Returns:
            The reshaped array

        """
        return v.reshape((3, *self.shape3))

    def slice(self, center: Size3, size: Size3, squeezed: bool = True):
        """Pick a slide of this grid

        Args:
            center: center of the slice in (x, y, z) in the units of the simulation (note: NOT in terms of array index)
            size: size of the slice in (x, y, z) in the units of the simulation (note: NOT in terms of array index)
            squeezed: whether to squeeze the slice to the minimum dimension (the squeeze order is z, then y).

        Returns:
            The slices to access the array

        """
        # if self.ndim == 1:
        #     raise ValueError(f"Simulation dimension ndim must be 2 or 3 but got {self.ndim}.")
        if not len(size) == 3:
            raise ValueError(f"For simulation that is 3d, must provide size arraylike of size 3 but got {size}")
        if not len(center) == 3:
            raise ValueError(f"For simulation that is 3d, must provide center arraylike of size 3 but got {center}")

        c = np.around(np.asarray(center) / self.spacing3).astype(int)  # assume isotropic for now...
        shape = np.around(np.asarray(size) / self.spacing3).astype(int)

        s0, s1, s2 = shape[0] // 2, shape[1] // 2, shape[2] // 2
        c0 = c[0] if squeezed else slice(c[0], c[0] + 1)
        c1 = c[1] if squeezed else slice(c[1], c[1] + 1)
        c2 = c[2] if squeezed else slice(c[2], c[2] + 1)
        # if s0 == s1 == s2 == 0:
        #     raise ValueError(f"Require the size result in a nonzero-sized shape, but got a single point in the grid"
        #                      f"(i.e., the size {size} may be less than the spacing {self.spacing3})")
        return (slice(c[0] - s0, c[0] - s0 + shape[0]) if shape[0] > 0 else c0,
                slice(c[1] - s1, c[1] - s1 + shape[1]) if shape[1] > 0 else c1,
                slice(c[2] - s2, c[2] - s2 + shape[2]) if shape[2] > 0 else c2)

    def view_fn(self, center: Size3, size: Size3, use_jax: bool = True):
        """Return a function that views a field at specific region.

        The view function is specified by center and size in the grid. This is typically used for
        mode-based sources and measurements. Once a slice is found, the fields need to be reoriented
        such that the field components point in the right direction despite a change in axis assignment.
        This function will handle this logic automatically in 1d, 2d, and 3d cases.

        Args:
            center: Center of the region
            size: Size of the region
            use_jax: Use jax

        Returns:
            A view callable function that orients the field and finds the appropriate slice.

        """
        if np.count_nonzero(size) == 3:
            raise ValueError(f"At least one element of size must be zero, but got {size}")
        s = self.slice(center, size, squeezed=False)
        xp = jnp if use_jax else np

        # Find the view axis (the poynting direction)
        view_axis = 0
        for i in range(self.ndim):
            if size[i] == 0:
                view_axis = i

        # Find the reorientation of field axes based on view_axis
        # 0 -> (1, 2, 0)
        # 1 -> (0, 2, 1)
        # 2 -> (0, 1, 2)
        axes = [
            np.asarray((1, 2, 0), dtype=int),
            np.asarray((0, 2, 1), dtype=int),
            np.asarray((0, 1, 2), dtype=int)
        ][view_axis]

        def view(field):
            oriented_field = xp.stack(
                (field[axes[0]].reshape(self.shape3),
                 field[axes[1]].reshape(self.shape3),
                 field[axes[2]].reshape(self.shape3))
            )  # orient the field by axis (useful for mode calculation)
            return oriented_field[:, s[0], s[1], s[2]].transpose((0, *tuple(1 + axes)))

        return view

    def mask_fn(self, size: Size3, center: Optional[Size3] = None):
        """Given a box with :code:`size` and :code:`center`, return a function that sets pixels in :code:`rho`,
        where :code:`rho.shape == grid.eps.shape`, outside the box to :code:`eps`.
        This is important in inverse design to avoid modifying the material region near the source and measurement
        regions.

        Args:
            center: position of the mask in (x, y, z) in the units of the simulation (note: NOT in terms of array index)
            size: size of the mask box in (x, y, z) in the units of the simulation (note: NOT in terms of array index)

        Returns:
            The mask function

        """
        rho_init = self.eps
        center = self.center if center is None else center
        mask = self.mask(center, size)
        return lambda rho: jnp.array(rho_init) * (1 - mask) + rho * mask

    def block_design(self, waveguide: Box, wg_height: Optional[float] = None, sub_eps: float = 1,
                     sub_height: float = 0, gap: float = 0, block: Optional[Box] = None, sep: Size = (0, 0),
                     vertical: bool = False, rib_y: float = 0):
        """A helper function for designing a useful port or cross section for a mode solver.

        Args:
            waveguide: The base waveguide material and size in the form of :code:`Box`.
            wg_height: The waveguide height.
            sub_eps: The substrate epsilon (defaults to air)
            sub_height: The height of the substrate (or the min height of the waveguide built on top of it)
            gap: The coupling gap specified means we get a pair of base blocks
            separated by :code:`coupling_gap`.
            block: Perturbing block that is to be aligned either vertically or horizontally with waveguide (MEMS).
            sep: Separation of the block from the base waveguide layer.
            vertical: Whether the perturbing block moves vertically, or laterally otherwise.
            rib_y: Rib section.

        Returns:
            The resulting :code:`Grid` with the modified :code:`eps` property.

        """
        if rib_y > 0:
            self.fill(rib_y + sub_height, waveguide.eps)
        self.fill(sub_height, sub_eps)
        waveguide.align(self.center)
        if wg_height:
            waveguide.valign(wg_height)
        else:
            wg_height = waveguide.min[1]
        sep = (sep, sep) if not isinstance(sep, Tuple) else sep
        d = gap / 2 + waveguide.size[0] / 2 if gap > 0 else 0
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


class YeeGrid(Grid):
    def __init__(self, size: Size, spacing: Spacing, eps: Union[float, np.ndarray] = 1,
                 bloch_phase: Union[Size, float] = 0.0, pml: Optional[Size] = None, pml_sep: int = 5,
                 pml_params: Size3 = (4, -16, 1.0), name: str = 'simgrid'):
        """The base :code:`YeeGrid` class (adding things to :code:`Grid` like Yee grid support, Bloch phase,
        PML shape, etc.).

        Args:
            size: Tuple of size 1, 2, or 3 representing the size of the grid
            spacing: Spacing (microns) between each pixel along each axis (must be same dim as `grid_shape`)
            eps: Relative permittivity :math:`\\epsilon_r`
            bloch_phase: Bloch phase (generally useful for angled scattering sims)
            pml: Perfectly matched layer (PML) of thickness on both sides of the form :code:`(x_pml, y_pml, z_pml)`
            pml_sep: Specifies the number of pixels that any source must be placed away from a PML region.
            pml_params: The parameters of the form :code:`(exp_scale, log_reflectivity, pml_eps)`.
        """
        super(YeeGrid, self).__init__(size, spacing, eps)
        self.pml = pml
        self.pml_sep = pml_sep
        self.pml_shape = pml if pml is None else (np.asarray(pml, dtype=float) / self.spacing).astype(np.int)
        self.pml_params = pml_params
        self.name = name
        if self.pml_shape is not None:
            if np.any(self.pml_shape <= 3) or np.any(self.pml_shape >= self.shape // 2):
                raise AttributeError(f'PML shape must be more than 3 and less than half the shape on each axis.')
        if pml is not None and not len(self.pml_shape) == len(self.shape):
            raise AttributeError(f'Need len(pml_shape) == grid.shape,'
                                 f'got ({len(pml)}, {len(self.shape)}).')
        self.bloch = np.ones_like(self.shape) * np.exp(1j * np.asarray(bloch_phase)) if isinstance(bloch_phase, float) \
            else np.exp(1j * np.asarray(bloch_phase))
        if not len(self.bloch) == len(self.shape):
            raise AttributeError(f'Need bloch_phase.size == grid.shape,'
                                 f'got ({len(self.bloch)}, {len(self.shape)}).')
        self.dtype = np.float64 if pml is None and bloch_phase == 0 else np.complex128
        self._dxes = np.meshgrid(*self.cells, indexing='ij'), np.meshgrid(*self.cells, indexing='ij')

    def deriv(self, back: bool = False) -> List[sp.spmatrix]:
        """Calculate directional derivative.

        Args:
            back: Return backward derivative.

        Returns:
            Discrete directional derivative :code:`d` of the form :code:`(d_x, d_y, d_z)`

        """

        # account for 1d and 2d cases
        b = np.hstack((self.bloch, np.ones((3 - self.ndim,), dtype=self.bloch.dtype)))
        s = np.hstack((self.shape, np.ones((3 - self.ndim,), dtype=self.shape.dtype)))

        if back:
            # get backward derivative
            _, dx = self._dxes
            d = [sp.diags([1, -1, -np.conj(b[ax])], [0, -1, n - 1], shape=(n, n))
                 if n > 1 else 0 for ax, n in enumerate(s)]  # get single axis back-derivs
        else:
            # get forward derivative
            dx, _ = self._dxes
            d = [sp.diags([-1, 1, b[ax]], [0, 1, -n + 1], shape=(n, n))
                 if n > 1 else 0 for ax, n in enumerate(s)]  # get single axis forward-derivs
        d = [sp.kron(d[0], sp.eye(s[1] * s[2])).astype(np.complex128),
             sp.kron(sp.kron(sp.eye(s[0]), d[1]), sp.eye(s[2])).astype(np.complex128),
             sp.kron(sp.eye(s[0] * s[1]), d[2]).astype(np.complex128)]  # tile over the other axes using sp.kron
        d = [sp.diags(1 / dx[ax].ravel()) @ d[ax] for ax in range(len(s))]  # scale by dx (incl pml)

        return d

    @property
    def deriv_forward(self):
        """The forward derivative

        Returns:
            The forward derivative

        """
        return self.deriv()

    @property
    def deriv_backward(self):
        """The backward derivative

        Returns:
            The backward derivative

        """
        return self.deriv(back=True)

    def diff_fn(self, of_h: bool = False, use_jax: bool = False):
        """Return a function that takes the discrete derivative of a field in a functional manner based on grid.

        Args:
            of_h: Take the derivative of :math:`\\mathbf{H}`, otherwise :math:`\\mathbf{E}`.
            use_jax: Whether to use jax.

        Returns:
            The discrete derivative function
        """
        xp = jnp if use_jax else np
        dx = jnp.array(self._dxes[of_h]) if use_jax else self._dxes[of_h]
        if of_h:
            def _diff(f, a):
                return (f - xp.roll(f, 1, axis=a)) / dx[a]
        else:
            def _diff(f, a):
                return (xp.roll(f, -1, axis=a) - f) / dx[a]
        return _diff

    def curl_fn(self, beta: Optional[float] = None, of_h: bool = False, use_jax: bool = False) -> Op:
        """Get the function that computes curl of the electric field :math:`\\mathbf{E}`.

        Args:
            beta: Propagation constant in the z direction (note: x, y are the `cross section` axes).
            of_h: Whether to take the curl of h
            use_jax: Whether the returned function should use jax.

        Returns:
            A function that computes discretized curl :math:`\\nabla \\times \\mathbf{E}`.

        """
        diff_fn = self.diff_fn(of_h=of_h, use_jax=use_jax)
        return curl_fn(diff_fn, use_jax=use_jax, beta=beta)

    def pml_safe_placement(self, loc: Size3) -> Size3:
        """Specifies a source/ measurement placement that is safe from the PML region / edge of the simulation.

        Args:
            loc: Location of the form (x, y, z) to move safely away from the PML

        Returns:
            New x, y, z tuple that is safe from PML (at least one Yee grid point away from the pml region).

        """
        x, y, z = loc
        pml = (self.pml_shape + self.pml_sep) * self.spacing if self.pml_shape is not None else (0, 0)
        maxx, maxy = self.size[:2] - self.spacing[:2]
        new_x = min(max(x, pml[0]), maxx - pml[0])
        new_y = min(max(y, pml[1]), maxy - pml[1])
        return new_x, new_y, z

    @property
    @lru_cache()
    def eps_t(self):
        """This attribute specifies the grid-averaged epsilon in the grid.

        Returns:
            The grid-averaged epsilon.

        """
        return yee_avg(self.eps.reshape(self.shape3))

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import dataclasses

from .typing import Shape, Dim, Dim3, GridSpacing, Optional, List, Union, Dict, Tuple, Op
from .utils import curl_fn, yee_avg, fix_dataclass_init_docs


try:
    DPHOX_IMPORTED = True
    from dphox.component import Pattern
except ImportError:
    DPHOX_IMPORTED = False


@fix_dataclass_init_docs
@dataclasses.dataclass
class Port:
    """Port used in components in DPhox

    A port defines the center and angle/orientation in a design.

    Args:
        x: x position of the port
        y: y position of the port
        a: angle (orientation) of the port (in degrees)
        w: the width of the port (optional, specified in design, mostly used for simulation)
        z: z position of the port (optional)
        h: the height of the port (optional, not specified in design, mostly used for simulation)
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
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1.0):
        """Grid object accomodating any electromagnetic simulation strategy (FDFD, FDTD, BPM, etc.)

        Args:
            shape: Tuple of size 1, 2, or 3 representing the number of pixels in the grid
            spacing: Spacing (microns) between each pixel along each axis (must be same dim as `grid_shape`)
            eps: Relative permittivity
        """
        self.shape = np.asarray(shape, dtype=np.int)
        self.spacing = spacing * np.ones(len(shape)) if isinstance(spacing, int) or isinstance(spacing, float) else np.asarray(spacing)
        self.ndim = len(shape)
        self.shape3 = np.hstack((self.shape, np.ones((3 - self.ndim,), dtype=self.shape.dtype)))
        self.spacing3 = np.hstack((self.spacing, np.ones((3 - self.ndim,), dtype=self.spacing.dtype) * np.inf))

        if not self.ndim == self.spacing.size:
            raise AttributeError(f'Require shape.size == spacing.size but got '
                                 f'{self.shape.size} != {self.spacing.size}')
        self.n = np.prod(self.shape)
        self.eps: np.ndarray = np.ones(self.shape) * eps if not isinstance(eps, np.ndarray) else eps
        if not tuple(self.shape) == self.eps.shape:
            raise AttributeError(f'Require grid_shape == eps.shape but got '
                                 f'{self.shape} != {self.eps.shape}')
        self.size = self.spacing * self.shape
        self.cell_sizes = [(self.spacing[i] * np.ones((self.shape[i],)) if self.ndim > 1 else self.spacing * np.ones(self.shape))
                           if i < self.ndim else np.ones((1,)) for i in range(3)]
        self.pos = [np.hstack((0, np.cumsum(dx))) if dx.size > 1 else np.asarray((0,)) for dx in self.cell_sizes]
        self.components = []

        # used to handle special functions of waveguide-based components
        self.port: Dict[str, Port] = {}
        self.port_thickness = 0
        self.port_height = 0

    def _check_bounds(self, component) -> bool:
        b = component.bounds
        return b[0] >= 0 and b[1] >= 0 and b[2] <= self.size[0] and b[3] <= self.size[1]

    def fill(self, zmax: float, eps: float) -> "Grid":
        """Fill grid up to `zmax`, typically used for substrate + cladding epsilon settings

        Args:
            zmax: Maximum z (or final dimension) of the fill operation
            eps: Relative eps to fill

        Returns:
            The modified FDFD object (:code:`self`)

        """
        if zmax > 0:
            self.eps[..., :int(zmax / self.spacing[-1])] = eps
        else:
            self.eps = np.ones_like(self.eps) * eps
        return self

    def add(self, component: "Pattern", eps: float, zmin: float = None, thickness: float = None) -> "Grid":
        """Add a component to the grid

        Args:
            component: component to add
            eps: permittivity of the component being added (isotropic only, for now)
            zmin: minimum z extent of the component
            thickness: component thickness (`zmax = zmin + thickness`)

        Returns:
            The modified FDFD object (:code:`self`)

        """
        if not self._check_bounds(component):
            raise ValueError('The pattern is out of bounds')
        self.components.append(component)
        mask = component.mask(self.shape[:2], self.spacing)
        if self.ndim == 2:
            self.eps[mask == 1] = eps
        else:
            zidx = (int(zmin / self.spacing[0]), int((zmin + thickness) / self.spacing[1]))
            self.eps[mask == 1, zidx[0]:zidx[1]] = eps
        self.port = {port_name: Port(*port.xya, port.w, zmin + thickness / 2, thickness)
                     for port_name, port in component.port.items()}
        return self

    def reshape(self, v: np.ndarray) -> np.ndarray:
        """A simple method to reshape flat 3d vec array into the grid shape

        Args:
            v: vector of size `(3n,)` to rearrange into array of size `(3, n)`

        Returns:


        """
        return v.reshape((3, *self.shape3)) if v.ndim == 1 else v.flatten()

    def slice(self, center: Dim3, size: Dim3, squeezed: bool = True):
        """Pick a slide of this grid

        Args:
            center: position of the mode in (x, y, z) in the units of the simulation (note: NOT in terms of array index)
            size: position of the mode in (x, y, z) in the units of the simulation (note: NOT in terms of array index)
            squeezed: whether to squeeze the slice to the minimum dimension (the squeeze order is z, then y).

        Returns:
            The slices to access the array

        """
        if self.ndim == 1:
            raise ValueError(f"Simulation dimension ndim must be 2 or 3 but got {self.ndim}.")
        if not len(size) == 3:
            raise ValueError(f"For simulation that is 3d, must provide size arraylike of size 3 but got {size}")
        if not len(center) == 3:
            raise ValueError(f"For simulation that is 3d, must provide center arraylike of size 3 but got {center}")

        c = (np.asarray(center) / self.spacing3).astype(np.int)  # assume isotropic for now...
        shape = (np.asarray(size) / self.spacing3).astype(np.int)

        s0, s1, s2 = shape[0] // 2, shape[1] // 2, shape[2] // 2
        c0 = c[0] if squeezed else slice(c[0], c[0] + 1)
        c1 = c[1] if squeezed else slice(c[1], c[2] + 1)
        c2 = c[2] if squeezed else slice(c[2], c[2] + 1)
        if s0 == s1 == s2 == 0:
            raise ValueError(f"Require the size result in a nonzero-sized shape, but got a single point in the grid"
                             f"(i.e., the size {size} may be less than the spacing {self.spacing3})")
        return (slice(c[0] - s0, c[0] - s0 + shape[0]) if shape[0] > 0 else c0,
                slice(c[1] - s1, c[1] - s1 + shape[1]) if shape[1] > 0 else c1,
                slice(c[2] - s2, c[2] - s2 + shape[2]) if shape[2] > 0 else c2)

    def view_fn(self, center: Tuple[float, float, float], size: Tuple[float, float, float], use_jax: bool = True):
        """Return a function that views a field at specific region specified by center and size in the grid. This
        is used for mode measurements.

        Args:
            center: Center of the region
            size: Size of the region
            use_jax: Use jax
            squeezed: Whether to squeeze the array when viewing the fields

        Returns:
            A view callable function that orients the field and finds the appropriate slice

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
            np.asarray((1, 2, 0), dtype=np.int),
            np.asarray((0, 2, 1), dtype=np.int),
            np.asarray((0, 1, 2), dtype=np.int)
        ][view_axis]

        def view(field):
            oriented_field = xp.stack(
                (xp.atleast_3d(field[axes[0]]),
                 xp.atleast_3d(field[axes[1]]),
                 xp.atleast_3d(field[axes[2]]))
            )  # orient the field by axis (useful for mode calculation)
            return oriented_field[:, s[0], s[1], s[2]].transpose(0, *(1 + axes))

        return view

    def get_expand_3d_fn(self, use_jax=False):
        xp = jnp if use_jax else np
        if self.ndim == 1:
            return lambda x: xp.squeeze(x)[..., xp.newaxis, xp.newaxis]
        elif self.ndim == 2:
            return lambda x: xp.squeeze(x)[..., xp.newaxis]
        else:
            return lambda x: x


class YeeGrid(Grid):
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 bloch_phase: Union[Dim, float] = 0.0, pml: Optional[Union[int, Shape, Dim]] = None,
                 pml_params: Dim3 = (4, -16, 1.0), yee_avg: int = 1, name: str = 'simgrid'):
        """The base :code:`YeeGrid` class (adding things to :code:`Grid` like Yee grid support, Bloch phase,
        PML shape, etc.).

        Args:
            shape: Tuple of size 1, 2, or 3 representing the number of pixels in the grid
            spacing: Spacing (microns) between each pixel along each axis (must be same dim as `grid_shape`)
            eps: Relative permittivity :math:`\\epsilon_r`
            bloch_phase: Bloch phase (generally useful for angled scattering sims)
            pml: Perfectly matched layer (PML) of thickness on both sides of the form :code:`(x_pml, y_pml, z_pml)`
            pml_params: The parameters of the form :code:`(exp_scale, log_reflectivity, pml_eps)`.
            yee_avg: whether to do a yee average (highly recommended)
        """
        super(YeeGrid, self).__init__(shape, spacing, eps)
        self.pml_shape = np.asarray(pml, dtype=np.int) if isinstance(pml, tuple) else pml
        self.pml_shape = np.ones(self.ndim, dtype=np.int) * pml if isinstance(pml, int) else pml
        self.pml_params = pml_params
        self.yee_avg = yee_avg
        self.name = name
        self.field_shape = np.hstack((3, self.shape))
        if self.pml_shape is not None:
            if np.any(self.pml_shape <= 3) or np.any(self.pml_shape >= self.shape // 2):
                raise AttributeError(f'PML shape must be more than 3 and less than half the shape on each axis.')
        if pml is not None and not len(self.pml_shape) == len(self.shape):
            raise AttributeError(f'Need len(pml_shape) == len(grid_shape),'
                                 f'got ({len(pml)}, {len(self.shape)}).')
        self.bloch = np.ones_like(self.shape) * np.exp(1j * np.asarray(bloch_phase)) if isinstance(bloch_phase, float) \
            else np.exp(1j * np.asarray(bloch_phase))
        if not len(self.bloch) == len(self.shape):
            raise AttributeError(f'Need len(bloch_phase) == len(grid_shape),'
                                 f'got ({len(self.bloch)}, {len(self.shape)}).')
        self.dtype = np.float64 if pml is None and bloch_phase == 0 else np.complex128
        self._dxes = np.meshgrid(*self.cell_sizes, indexing='ij'), np.meshgrid(*self.cell_sizes, indexing='ij')

    def deriv(self, back: bool = False) -> List[sp.spmatrix]:
        """Calculate directional derivative

        Args:
            back: Return backward derivative

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
    def df(self):
        return self.deriv()

    @property
    def db(self):
        return self.deriv(back=True)

    def diff_fn(self, use_h: bool = False, use_jax: bool = False):
        xp = jnp if use_jax else np
        dx = jnp.array(self._dxes[use_h]) if use_jax else self._dxes[use_h]
        if use_h:
            def _diff(f, a):
                return (f - xp.roll(f, 1, axis=a)) / dx[a]
        else:
            def _diff(f, a):
                return (xp.roll(f, -1, axis=a) - f) / dx[a]
        return _diff

    def curl_e(self, beta: Optional[float] = None, use_jax: bool = False) -> Op:
        """Get the curl of the electric field :math:`\\mathbf{E}`

        Args:
            e: electric field :math:`\\mathbf{E}`
            beta: Propagation constant in the z direction (note: x, y are the `cross section` axes)

        Returns:
            The discretized curl :math:`\\nabla \\times \\mathbf{E}`

        """
        return curl_fn(self.diff_fn(use_h=False, use_jax=use_jax), use_jax=use_jax, beta=beta)

    def curl_h(self, beta: Optional[float] = None, use_jax: bool = False) -> Op:
        """Get the curl of the magnetic field :math:`\\mathbf{H}`

           Args:
               h: magnetic field :math:`\\mathbf{H}`
               beta: Propagation constant in the z direction (note: x, y are the `cross section` axes)

           Returns:
               The discretized curl :math:`\\nabla \times \mathbf{H}`

        """
        return curl_fn(self.diff_fn(use_h=True, use_jax=use_jax), use_jax=use_jax, beta=beta)

    def pml_safe_placement(self, x: float, y: float, z: float) -> Dim3:
        """ Specifies a source/ measurement placement that is safe from the PML region / edge of the simulation.

        Args:
            x: Input x location
            y: Input y location
            z: Input z location

        Returns:
            New x, y, z tuple that is safe from PML (at least one Yee grid point away from the pml region).

        """
        pml = (self.pml_shape + 1) * self.spacing if self.pml_shape is not None else (0, 0)
        maxx, maxy = self.size[:2]
        new_x = min(max(x, pml[0]), maxx - pml[0])
        new_y = min(max(y, pml[1]), maxy - pml[1])
        return new_x, new_y, z

    @property
    @lru_cache()
    def eps_t(self):
        expand_3d = self.get_expand_3d_fn()
        return yee_avg(expand_3d(self.eps), shift=self.yee_avg)

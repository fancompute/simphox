from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from dphox.component import Pattern, Callable, Port

from .typing import Shape, Dim, GridSpacing, Optional, List, Union, Dict
from .utils import curl_fn, yee_avg


class Grid:
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1.0):
        """Grid object accomodating any electromagnetic simulation strategy (FDFD, FDTD, BPM, etc.)

        Args:
            shape: Tuple of size 1, 2, or 3 representing the number of pixels in the grid
            spacing: Spacing (microns) between each pixel along each axis (must be same dim as `grid_shape`)
            eps: Relative permittivity
        """
        self.shape = np.asarray(shape, dtype=np.int)
        self.spacing = spacing * np.ones(len(shape)) if isinstance(spacing, float) else np.asarray(spacing)
        self.ndim = len(shape)
        self.shape3 = np.hstack((self.shape, np.ones((3 - self.ndim,), dtype=self.shape.dtype)))

        if not self.ndim == len(self.spacing):
            raise AttributeError(f'Require len(grid_shape) == len(grid_spacing) but got'
                                 f'({len(self.shape)}, {len(self.spacing)})')
        self.n = np.prod(self.shape)
        self.eps: np.ndarray = np.ones(self.shape) * eps if not isinstance(eps, np.ndarray) else eps
        if not tuple(self.shape) == self.eps.shape:
            raise AttributeError(f'Require grid_shape == eps.shape but got'
                                 f'({self.shape}, {self.eps.shape})')
        self.size = self.spacing * self.shape
        self.cell_sizes = [self.spacing[i] * np.ones((self.shape[i],))
                           if i < self.ndim else np.ones((1,)) for i in range(3)]
        self.pos = [np.hstack((0, np.cumsum(dx))) if dx.size > 1 else None for dx in self.cell_sizes]
        self.components = []

        # used to handle special functions of waveguide-based components
        self.port: Dict[str, Port] = {}
        self.port_w = None

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
        return self

    def add(self, component: Pattern, eps: float, zmin: float = None, thickness: float = None) -> "Grid":
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
        self.port = component.port
        self.port_w = component.config.get('waveguide_w', None)
        return self

    def reshape(self, v: np.ndarray) -> np.ndarray:
        """A simple method to reshape flat 3d vec array into the grid shape

        Args:
            v: vector of size `(3n,)` to rearrange into array of size `(3, n)`

        Returns:


        """
        return np.stack([split_v.reshape(self.shape3) for split_v in np.split(v, 3)]) if v.ndim == 1 else v.flatten()


class SimGrid(Grid):
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 bloch_phase: Union[Dim, float] = 0.0, pml: Optional[Union[int, Shape, Dim]] = None,
                 pml_eps: float = 1.0, yee_avg: int = 1, name: str = 'simgrid'):
        """The base :code:`SimGrid` class (adding things to :code:`Grid` like Yee grid support, Bloch phase,
        PML shape, etc.).

        Args:
            shape: Tuple of size 1, 2, or 3 representing the number of pixels in the grid
            spacing: Spacing (microns) between each pixel along each axis (must be same dim as `grid_shape`)
            eps: Relative permittivity :math:`\\epsilon_r`
            bloch_phase: Bloch phase (generally useful for angled scattering sims)
            pml: Perfectly matched layer (PML) of thickness on both sides of the form :code:`(x_pml, y_pml, z_pml)`
            pml_eps: The permittivity used to scale the PML (should probably assign to 1 for now)
            yee_avg: whether to do a yee average (highly recommended)
        """
        super(SimGrid, self).__init__(shape, spacing, eps)
        self.pml_shape = np.asarray(pml, dtype=np.int) if isinstance(pml, tuple) else pml
        self.pml_shape = np.ones(self.ndim, dtype=np.int) * pml if isinstance(pml, int) else pml
        self.pml_eps = pml_eps
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
        d = [sp.kron(d[0], sp.eye(s[1] * s[2])),
             sp.kron(sp.kron(sp.eye(s[0]), d[1]), sp.eye(s[2])),
             sp.kron(sp.eye(s[0] * s[1]), d[2])]  # tile over the other axes using sp.kron
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

    def curl_e(self, beta: Optional[float] = None, use_jax: bool = False) -> Callable[[np.ndarray], np.ndarray]:
        """Get the curl of the electric field :math:`\mathbf{E}`

        Args:
            e: electric field :math:`\mathbf{E}`
            beta: Propagation constant in the z direction (note: x, y are the `cross section` axes)

        Returns:
            The discretized curl :math:`\\nabla \times \mathbf{E}`

        """
        return curl_fn(self.diff_fn(use_h=False, use_jax=use_jax), use_jax=use_jax, beta=beta)

    def curl_h(self, beta: Optional[float] = None, use_jax: bool = False) -> Callable[[np.ndarray], np.ndarray]:
        """Get the curl of the magnetic field :math:`\mathbf{H}`

           Args:
               h: magnetic field :math:`\mathbf{H}`
               beta: Propagation constant in the z direction (note: x, y are the `cross section` axes)

           Returns:
               The discretized curl :math:`\\nabla \times \mathbf{H}`

        """
        return curl_fn(self.diff_fn(use_h=True, use_jax=use_jax), use_jax=use_jax, beta=beta)

    def pml_safe_placement(self, x: float, y: float, safe_threshold: float = 0.1):
        """ Specifies a new x and y that are safe from the PML region / esge of the simulation.

        Args:
            x: Input x location
            y: Input y location

        Returns:
            New x, y tuple that is safe from PML.

        """
        pml = self.pml_shape * self.spacing + safe_threshold
        maxx, maxy = self.size[:2]
        new_x = min(max(x, pml[0]), maxx - pml[0])
        new_y = min(max(y, pml[1]), maxy - pml[1])
        return new_x, new_y

    @property
    @lru_cache()
    def eps_t(self):
        return yee_avg(self.eps, shift=self.yee_avg) if self.yee_avg > 0 else np.stack((self.eps, self.eps, self.eps))

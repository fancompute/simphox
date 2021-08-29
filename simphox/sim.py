import jax
import numpy as np
import jax.numpy as jnp

from .grid import YeeGrid
from .mode import ModeSolver, ModeLibrary
from .typing import GridSpacing, Shape, Union, Dim, Optional, List, Tuple, Shape2, Dim2, Dict, Dim3, Callable, \
    MeasureInfo, Op, PortLabel

from .viz import get_extent_2d

try:
    HOLOVIEWS_IMPORTED = True
    import holoviews as hv
    from holoviews.streams import Pipe
    from holoviews import opts
    import panel as pn
except ImportError:
    HOLOVIEWS_IMPORTED = False

import dataclasses
import xarray as xr


@dataclasses.dataclass
class SimCrossSection:
    io: ModeLibrary
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]

    def place(self, mode_idx, grid) -> np.ndarray:
        return self.io.place(mode_idx, grid, self.center, self.size)

    def gaussian(self):
        raise NotImplementedError

    def cw(self):
        raise NotImplementedError


class SimGrid(YeeGrid):
    def __init__(self, shape: Shape, spacing: GridSpacing, eps: Union[float, np.ndarray] = 1,
                 bloch_phase: Union[Dim, float] = 0.0, pml: Optional[Union[int, Shape, Dim]] = None,
                 pml_params: Dim3 = (4, -16, 1.0), yee_avg: int = 1, use_jax: bool = False, name: str = 'simgrid'):
        """The base :code:`SimGrid` class (adding things to :code:`Grid` like Yee grid support, Bloch phase,
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
        super(SimGrid, self).__init__(shape, spacing, eps, bloch_phase, pml, pml_params, yee_avg, name)
        self.use_jax = use_jax

    def modes(self, center: Dim3, size: Dim3, wavelength: float = 1.55, num_modes: int = 1) -> SimCrossSection:
        """Eigenmode profile of a 2d or 3d :code:`SimGrid` object.

        Args:
            center: center tuple of the form :code:`(x, y, z)` (in sim units, NOT pixels)
            size: size of the source (in sim units, NOT pixels)
            wavelength: wavelength (arb. units, should match with spacing)
            num_modes: number of modes to find

        Returns:
            A Tuple of a Modes object and view function for measuring the fields.
        """
        mode_eps = np.atleast_3d(self.eps)[self.slice(center, size)].squeeze()
        modes = ModeLibrary(
            shape=mode_eps.shape, spacing=self.spacing[0], eps=mode_eps, wavelength=wavelength, num_modes=num_modes
        )
        return SimCrossSection(modes, center, size)

    def mode_source(self, center: Dim, size: Dim, wavelength: float = 1.55, mode_idx: int = 0):
        """For waveguide-related problems or shining light into a photonic port, an eigenmode source is used.

        Args:
            center: center tuple of the form :code:`(x, y, z)`
            size: size of the source
            axis: axis for normal vector of cross-section (one of :code:`(0, 1, 2)`)
            wavelength: wavelength (arb. units, should match with spacing)
            mode_idx: mode index for the eigenmode for source profile
            gpu: place source on the GPU

        Returns:
            Eigenmode source function and region (:code:`slice` object or mask)
        """
        sim_xs = self.modes(center, size, wavelength, num_modes=mode_idx + 1)
        return cw_source_fn(profile, wavelength, gpu), region
    #
    # def cw_source(self, profile: np.ndarray, wavelength: float, t: float, dt: float) -> np.ndarray:
    #     """CW source array
    #
    #     Args:
    #         profile: Profile :math:`\mathbf{\\Psi}`
    #         wavelength: Wavelength :mode:`\\lambda`
    #         t: total "on" time
    #         dt: time step size
    #
    #     Returns:
    #         CW source as an ndarray of size :code:`[t/dt, *source_shape]`
    #
    #     """
    #     return source(cw_source_fn(profile, wavelength), t, dt)
    #
    # def source(self, source_fn: Callable[[float], np.ndarray], t: float, dt: float) -> np.ndarray:
    #     """Source array given a source function
    #
    #     Args:
    #         source_fn: Source function
    #         t: total "on" time
    #         dt: time step size
    #
    #     Returns:
    #         ndarray of size :code:`[t/dt, *source_shape]`
    #
    #     """
    #     ts = np.linspace(0, t, int(t // dt) + 1)
    #     return np.asarray([source_fn(t) for t in ts])  # not the most efficient, but it'll do for now
    #
    # def cw_source_fn(self, profile: np.ndarray, wavelength: float) -> Callable[[float], np.ndarray]:
    #     """CW source function
    #
    #     Args:
    #         profile: Profile :math:`\mathbf{\\Psi}` (e.g. mode or TFSF) for the input source
    #         wavelength: Wavelength for CW source
    #         gpu: place source on the gpu
    #
    #     Returns:
    #         the CW source function of time
    #
    #     """
    #     profile = jnp.asarray(profile) if self.use_jax else profile
    #     xp = jnp if self.use_jax else np
    #     return lambda t: profile * xp.exp(-1j * 2 * xp.pi * t / wavelength)
    #
    # def gaussian_source(self, profiles: np.ndarray, pulse_width: float, center_wavelength: float, dt: float,
    #                     t0: float = None, linear_chirp: float = 0) -> np.ndarray:
    #     """Gaussian source array
    #
    #     Args:
    #         profiles: profiles defined at individual frequencies
    #         pulse_width: Gaussian pulse width
    #         center_wavelength: center wavelength
    #         dt: time step size
    #         t0: peak time (default to be central time step)
    #         linear_chirp: linear chirp coefficient (default to be 0)
    #
    #     Returns:
    #         the Gaussian source discretized in time
    #
    #     """
    #     k0 = 2 * np.pi / center_wavelength
    #     t = np.arange(profiles.shape[0]) * dt
    #     t0 = t[t.size // 2] if t0 is None else t0
    #     g = np.fft.fft(np.exp(1j * k0 * (t - t0)) * np.exp((-pulse_width + 1j * linear_chirp) * (t - t0) ** 2))
    #     src = np.fft.ifft(g * profiles, axis=0)
    #     return jnp.asarray(src) if self.use_jax else src
    #
    # def gaussian_source_fn(self, profiles: np.ndarray, pulse_width: float, center_wavelength: float, dt: float,
    #                        t0: float = None, linear_chirp: float = 0) -> Callable[[float], np.ndarray]:
    #     """Gaussian source function
    #
    #     Args:
    #         profiles: profiles defined at individual frequencies
    #         pulse_width: pulse width at individual frequencies
    #         center_wavelength: center wavelength
    #         dt: time step size
    #         t0: peak time (default to be central time step)
    #         linear_chirp: linear chirp coefficient (default to be 0)
    #
    #     Returns:
    #         the Gaussian source function of time
    #
    #     """
    #     src = self.gaussian_source(profiles, pulse_width, center_wavelength, dt, t0, linear_chirp)
    #     return lambda tt: src[tt // dt]

    def port_modes(self, excitation: List[Tuple[str, int]] = None,
                   profile_size_factor: float = 2) -> Dict[PortLabel, SimCrossSection]:
        """Profile for all the ports in the grid (always assumed to be along x or y axes!).

        Args:
            excitation: Dictionary mapping port to mode index for excitations
            profile_size_factor: Factor to rescale the mode view slice compared to the port

        Returns:
            A dictionary from mode to SimCrossSection containing the mode, and its center and size in this grid
            needed to reconstruct the source using :code:`mode.place`.

        """

        excitation = [(port, 0) for port in self.port] if excitation is None else excitation

        return {name: self.modes(center=self.pml_safe_placement(*p.xyz), size=p.size * profile_size_factor,
                                 num_modes=np.max([mode_idx
                                                   for port_name, mode_idx in excitation if port_name == name]) + 1)
                for name, p in self.port.items()}

    def port_source(self, source: Optional[Union[Dict[Tuple[str, int], float], Dict[str, float]]] = None,
                    profile_size_factor: float = 2, unidirectional: bool = True) -> np.ndarray:
        """Return a non-sparse source array based on the ports defined in the simulation grid.

        Args:
            source: Map each port and mode index to a weight to yield a weighted port source.
                If a dictionary is specified, it can be of the form :code:`{(port_name, mode_idx): weight}` or
                :code:`{port_name: weight}`, where in the latter case, a default mode index of 0 is used.
                A source is then created by summing the contributions from all of those ports.
            profile_size_factor: Factor to rescale the mode view slice compared to the port
            unidirectional: In FDFD, this specifies whether to send the mode in unidirectionally, determined
                using the port angle.

        Returns:
            The non-sparse source array that can be used as a source profile for either FDFD or FDTD

        """
        ports = list(self.port.keys())
        # if the source is a list of numbers, just assign the appropriate weight to the port's fundamental mode
        source = {(ports[0], 0): 1} if source is None else source
        source = {(port, 0): weight for port, weight in zip(ports, source)} if isinstance(source, tuple) else source
        source_library = self.port_modes(profile_size_factor=profile_size_factor)

        sources_to_sum = []
        for port_mode, weight in source.items():
            axis = int(np.mod(self.port[port_mode[0]].a, np.pi) == np.pi / 2)
            beta = source_library[port_mode[0]].io.betas[port_mode[1]]
            shift = 2 * (np.mod(self.port[port_mode[0]].a, 2 * np.pi) < np.pi) - 1
            src = source_library[port_mode[0]].place(port_mode[1], self) * weight
            src = np.roll(src, axis=axis, shift=2 * shift)
            if unidirectional:
                src += np.roll(src, axis=axis, shift=shift) * np.exp(-1j * self.spacing[axis] * beta - 1j * np.pi)
            sources_to_sum.append(src)

        return sum(sources_to_sum) if sources_to_sum else np.array([])

    def get_measure_fn(self, measure_info: Optional[MeasureInfo] = None,
                       profile_size_factor: float = 2, use_jax: bool = False, tm_2d: bool = True) -> Op:
        """Measure function: measure the fields using the Modes object provided for each port

        Args:
            measure_info: List of port name and mode index at that port
            profile_size_factor: Factor to rescale the mode view slice compared to the port
            use_jax: Whether to use jax in the measure function (relevant for simulations).
            tm_2d: Whether to use TM polarization (applies to the 2D case only).

        Returns:
            Callable function that gives port-wise measurements

        """
        # Set up the port profiles for measurement at each port (by default assumes single mode waveguides)
        measure_info = [(name, 0) for name in self.port] if measure_info is None else measure_info
        port_to_modes = self.port_modes(measure_info, profile_size_factor)
        ports = port_to_modes.keys()
        port_nums = np.arange(len(ports))
        angles = [self.port[port].a for port in ports]
        # We measure polarity, which is the orientation of the measurement interface
        # to determine whether the wave is entering or leaving the device
        # The polarity below assumes that ports are near edge of simulation.
        polarity = 2 * (np.mod(angles, 360) < 180).astype(np.int) - 1
        measure_fns = [port_to_modes[port_name].io.measure_fn(m, use_jax, tm_2d=tm_2d) for port_name, m in measure_info]
        view_fns = [self.view_fn(port_to_modes[port_name].center, port_to_modes[port_name].size, use_jax)
                    for port_name, _ in measure_info]

        xp = jnp if use_jax else np

        def measure_fn(fields):
            e, h = fields
            return xp.stack([measure_fns[i](view_fns[i](e), view_fns[i](h))[::polarity[i]] for i in port_nums]).T

        return measure_fn

    def get_fields_fn(self, src: Union[np.ndarray, Callable],
                      transform_fn: Optional[Callable] = None, tm_2d: bool = True) -> Callable:
        """Returns a function that yields the fields given a transform function and source.

        We first initialize the problem solver given two callable functions:

        1. A numpy array source :code:`src`
        2. The JAX-transformable transform function :code:`transform_fn` (e.g. transform) (identity if None)

        Args:
            src: source for the solver (either a callable for time domain or array for frequency domain)
            transform_fn: Transforms parameters to yield the epsilon function used by jax

        Returns:
            A solve function (2d or 3d based on defined :code:`ndim` specified for the instance of :code:`FDFD`)

        Returns:

        """
        raise NotImplementedError("A child class of SimGrid needs to implement get_fields_fn")

    def get_sim_fn(self, src: Union[np.ndarray, Callable], transform_fn: Optional[Callable] = None,
                   tm_2d: bool = True) -> Callable:
        """Returns a function that measures the sparams and fields.

        We first initialize the optimization problem solver given two callable functions:

        1. A numpy array source :code:`src`
        2. The JAX-transformable transform function :code:`transform_fn` (e.g. transform) (identity if None)

        We then extract the sparams using the port locations provided in this class.

        Args:
            src: source for the solver
            transform_fn: Transforms parameters to yield the epsilon function used by jax
            tm_2d: Whether to use TM polarization (applies to the 2D case only).

        Returns:
            A solve function (2d or 3d based on defined :code:`ndim` specified for the instance of :code:`FDFD`)

        """

        fields_fn = self.get_fields_fn(src, transform_fn, tm_2d=tm_2d)
        measure_fn = self.get_measure_fn(use_jax=True, tm_2d=tm_2d)

        @jax.jit
        def sim_fn(rho: jnp.ndarray):
            fields = fields_fn(rho)
            measurements = measure_fn(fields)
            return measurements, fields

        return sim_fn

    def get_sim_sparams_fn(self, port_name: Optional[str] = None, transform_fn: Optional[Callable] = None,
                           mode_idx: int = 0, profile_size_factor: int = 2,
                           measure_info: Optional[MeasureInfo] = None, tm_2d: bool = True) -> Callable:
        """Returns a function that measures the sparams and fields.

        We first initialize the optimization problem solver given a JAX-transformable transform function
        :code:`transform_fn` (e.g. transform) and the port, mode pair for the input source (used to normalize the
        output measurements to get the s params). We then extract the sparams using the port locations
        provided in this class.

        Args:
            port_name: Port name for the source
            mode_idx: Mode index for the source
            transform_fn: Transforms parameters to yield the epsilon function used by jax (identity if None)
            profile_size_factor: Profile size factor to rescale the port size to get mode size
            measure_info: Measurement info consisting of a list of port name and mode index pairs
            tm_2d: Whether to use TM polarization (applies to the 2D case only).

        Returns:
            A solve function (2d or 3d based on defined :code:`ndim` specified for the instance of :code:`FDFD`)

        """
        measure_info = [(name, 0) for name in self.port] if measure_info is None else measure_info
        source_info = (port_name, mode_idx) if port_name is not None else measure_info[0]
        fields_fn = self.get_fields_fn(src=self.port_source({source_info: 1}, profile_size_factor=profile_size_factor),
                                       transform_fn=transform_fn,
                                       tm_2d=tm_2d)
        measure_fn = self.get_measure_fn(use_jax=True, tm_2d=tm_2d)
        port_idx = measure_info.index(source_info)

        @jax.jit
        def sim_fn(rho: jnp.ndarray):
            fields = fields_fn(rho)
            s_out, s_in = measure_fn(fields)
            sparams = s_out / s_in[port_idx]
            return sparams, fields

        return sim_fn

    def viz_panel(self, img_width: float = 700) -> Tuple["pn.layout.Panel", Tuple["Pipe", "Pipe", "Pipe"]]:
        if not HOLOVIEWS_IMPORTED:
            raise ImportError("Holoviews not imported, so a viz panel cannot be generated")
        if not self.ndim == 2:
            raise NotImplementedError("Only implemented for ndim == 2!")
        extent = get_extent_2d(self.shape, self.spacing[0])
        aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
        bounds = (extent[0], extent[2], extent[1], extent[3])
        eps_norm = self.eps.T / np.max(self.eps.T)
        bounded_img = lambda data: hv.Image(data, bounds=bounds)
        eps_pipe = Pipe(data=[])
        eps_dmap = hv.DynamicMap(bounded_img, streams=[eps_pipe])
        field_pipe = Pipe(data=[])
        field_dmap = hv.DynamicMap(bounded_img, streams=[field_pipe])
        power_pipe = Pipe(data=[])
        power_dmap = hv.DynamicMap(bounded_img, streams=[power_pipe])
        eps_pipe.send(eps_norm)
        field_pipe.send(np.zeros_like(eps_norm))
        power_pipe.send(np.zeros_like(eps_norm))
        ed, fd, pd = (eps_dmap.opts(alpha=0.2, width=img_width, height=int(img_width / aspect), cmap='gray'),
                      field_dmap.opts(cmap='RdBu', width=img_width, height=int(img_width / aspect)),
                      power_dmap.opts(cmap='hot', width=img_width, height=int(img_width / aspect)))
        return pn.Row((fd * ed).opts(title=f'{self.name}: Fields (hz)'),
                      (pd * ed).opts(title=f'{self.name}: Power (|hz|²)')
                      ), (eps_pipe, field_pipe, power_pipe)

    def to_2d(self, wavelength: float = None,
              slab_x: Union[Shape2, Dim2] = None, slab_y: Union[Shape2, Dim2] = None) -> "SimGrid":
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
        # TODO(sunil): change to using port_profiles if possible
        # get slab index
        if not self.ndim == 3:
            raise RuntimeError("Require ndim = 3 for 2d variational effective index method.")
        if not wavelength:
            raise ValueError("Must specify a projection wavelength for the effective index method.")
        if slab_x is None and slab_y is None:
            if not self.port:
                raise ValueError('Must define x, y inputs since the port width and/or locations'
                                 'are not automatically discoverable.')
            port = list(self.port.values())[0]
            slab_x, slab_y, _ = self.pml_safe_placement(*port.xyz)
            if np.mod(port.a, np.pi) == 0:
                slab_x, slab_y = (int(slab_x / self.spacing[0]), (int((slab_y - port.w) / self.spacing[1]),
                                                                  int((slab_y + port.w) / self.spacing[1])))
            else:
                slab_x, slab_y = ((int((slab_x - port.w) / self.spacing[0]),
                                   int((slab_x + port.w) / self.spacing[0])), int(slab_y / self.spacing[1]))

        x_cen = slab_x if not isinstance(slab_x, tuple) else int((slab_x[0] + slab_x[1]) / 2)
        y_cen = slab_y if not isinstance(slab_y, tuple) else int((slab_y[0] + slab_y[1]) / 2)
        slab_mode_eps = self.eps[x_cen, y_cen]
        beta, slab_mode = ModeSolver(
            shape=slab_mode_eps.shape,
            spacing=self.spacing[-1],
            eps=slab_mode_eps,
            wavelength=wavelength
        ).profile(return_beta=True)
        eps_diff = self.eps - slab_mode_eps[np.newaxis, np.newaxis, :]
        eps_effective = (beta[0] / (2 * np.pi) * wavelength) ** 2 + eps_diff @ np.abs(slab_mode) ** 2 / np.sum(
            np.abs(slab_mode) ** 2)
        sim = SimGrid(
            shape=eps_effective.shape,
            spacing=self.spacing[:2],
            eps=eps_effective.real,
            pml=self.pml_shape[:2],
            name=self.name
        )
        sim.port = self.port
        return sim

    def decorate(self, sparams: np.ndarray, fields: np.ndarray) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Decorates the :code:`sparams` and :code:`fields` using :code:`xarray.DataArray`

        Args:
            sparams: The sparams resulting from a call to the returned callable from :code:`get_sim_sparams_fn`
            fields: The fields resulting from a call to the returned callable from :code:`get_sim_sparams_fn`

        Returns:
            The decorated :code:`sparams` and :code:`fields`.

        """
        decorated_sparams = xr.DataArray(
            data=sparams,
            dims=["port"],
            coords={
                "port": list(self.port.keys())
            }
        )
        e, h = fields
        decorated_e = xr.DataArray(
            data=e,
            dims=["direction", "x", "y", "z"],
            coords={
                "direction": ["x", "y", "z"],
                "x": self.pos[0][:-1] if self.pos[0].size > 1 else [0],
                "y": self.pos[1][:-1] if self.pos[1].size > 1 else [0],
                "z": self.pos[2][:-1] if self.pos[2].size > 1 is not None else [0]
            }
        )

        # shift h by half yee cell
        decorated_h = xr.DataArray(
            data=h,
            dims=["direction", "x", "y", "z"],
            coords={
                "direction": ["x", "y", "z"],
                "x": self.pos[0][:-1] + 0.5 * self.spacing3[0] if self.pos[0].size > 1 else [0],
                "y": self.pos[1][:-1] + 0.5 * self.spacing3[1] if self.pos[1].size > 1 else [0],
                "z": self.pos[2][:-1] + 0.5 * self.spacing3[2] if self.pos[2].size > 1 is not None else [0]
            }
        )
        return decorated_sparams, decorated_e, decorated_h

    def fidelity(self, desired_sparams: Union[Dict[Tuple[str, int], np.complex128], Dict[str, np.complex128]],
                 measure_info: List[Tuple[str, int]] = None) -> Callable:
        """ Returns the fidelity for the sparams.

        Args:
            desired_sparams: The desired sparams, provided in dictionary form mapping port to relative magnitude;
                if not an ndarray and/or not normalized, it is converted to a normalized ndarray.
            measure_info: Measurement info consisting of a list of port name and mode index pairs (used to index s)

        Returns:
            The fidelity based on the desired sparams :code:`s`.

        """

        measure_info = [(name, 0) for name in self.port] if measure_info is None else measure_info
        s = np.zeros(len(measure_info), dtype=np.complex128)
        for port, weight in desired_sparams.items():
            key = (port, 0) if isinstance(port, str) else port
            s[measure_info.index(key)] = weight
        s = jnp.array(s / np.linalg.norm(s))

        def obj(sparams_fields: Tuple[jnp.ndarray, jnp.ndarray]):
            sparams, fields = sparams_fields
            return -jnp.abs(s @ sparams) ** 2, jax.lax.stop_gradient((sparams, fields))

        return obj

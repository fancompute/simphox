import jax.numpy as jnp
import xarray as xr

try:
    DPHOX_IMPORTED = True
    from dphox.device import Device
    from dphox.pattern import Pattern
except ImportError:
    DPHOX_IMPORTED = False

try:
    DPHOX_IMPORTED = True
    from dphox.device import Device
    from dphox.pattern import Pattern
except ImportError:
    DPHOX_IMPORTED = False

from ..fdfd import FDFD
from ..typing import Callable, Iterable, List, Optional, Size, Union


class Component:
    def __init__(self, structure: Union["Pattern", "Device"],
                 model: Union[xr.DataArray, Callable[[jnp.ndarray], xr.DataArray]], name: str):
        """A component in a circuit will have some structure that can be simulated
        (a pattern or device defined in DPhox), a model, and a name string.

        Args:
            structure: Structure of the device.
            model: Model of the device (in terms of wavelength).
            name: Name of the component (string representing the model name).
        """
        self.structure = structure
        self.model = model
        self.name = name

    @classmethod
    def from_fdfd(cls, pattern: "Pattern", core_eps: float, clad_eps: float, spacing: float,
                  wavelengths: Iterable[float],
                  boundary: Size, pml: float, name: str, in_ports: Optional[List[str]] = None,
                  out_ports: Optional[List[str]] = None, component_t: float = 0, component_zmin: Optional[float] = None,
                  rib_t: float = 0, sub_z: float = 0, height: float = 0, bg_eps: float = 1,
                  profile_size_factor: int = 3,
                  pbar: Optional[Callable] = None):
        """From FDFD, this classmethod produces a component model based on a provided pattern
        and simulation attributes (currently configured for scalar photonics problems).

        Args:
            pattern: component provided by DPhox
            core_eps: core epsilon
            clad_eps: clad epsilon
            spacing: spacing required
            wavelengths: wavelengths
            boundary: boundary size around component
            pml: PML size (see :code:`FDFD` class for details)
            name: component name
            in_ports: input ports
            out_ports: output ports
            height: height for 3d simulation
            sub_z: substrate minimum height
            component_zmin: component height (defaults to substrate_z)
            component_t: component thickness
            rib_t: rib thickness for component (partial etch)
            bg_eps: background epsilon (usually 1 or air)
            profile_size_factor: profile size factor (multiply port size dimensions to get mode dimensions at each port)
            pbar: Progress bar (e.g. TQDM in a notebook which can be a valuable progress indicator).

        Returns:
            Initialize a component which contains a structure (for port specificication and visualization purposes)
            and model describing the component behavior.

        """
        sparams = []

        iterator = wavelengths if pbar is None else pbar(wavelengths)
        for wl in iterator:
            fdfd = FDFD.from_pattern(
                component=pattern,
                core_eps=core_eps,
                clad_eps=clad_eps,
                spacing=spacing,
                height=height,
                boundary=boundary,
                pml=pml,
                component_t=component_t,
                component_zmin=component_zmin,
                wavelength=wl,
                rib_t=rib_t,
                sub_z=sub_z,
                bg_eps=bg_eps,
                name=f'{name}_{wl}um'
            )
            sparams_wl = []
            for port in fdfd.port:
                s, _ = fdfd.get_sim_sparams_fn(port, profile_size_factor=profile_size_factor)(fdfd.eps)
                sparams_wl.append(s)
            sparams.append(sparams_wl)

        model = xr.DataArray(
            data=sparams,
            dims=["wavelengths", "in_ports", "out_ports"],
            coords={
                "wavelengths": wavelengths,
                "in_ports": in_ports,
                "out_ports": out_ports
            }
        )

        return cls(pattern, model=model, name=name)


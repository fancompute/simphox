from .grid import Grid, YeeGrid
from .fdfd import FDFD
from .fdtd import FDTD
from .mode import ModeLibrary, ModeSolver
from .utils import Box, Material, SILICON, POLYSILICON, NITRIDE, OXIDE, TEST_INF, AIR

from jax.config import config
config.update("jax_enable_x64", True)

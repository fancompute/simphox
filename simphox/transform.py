import jax.numpy as jnp
import numpy as np

from jax.scipy.signal import convolve as conv
from skimage.draw import disk
from .typing import Union, Dim2, Tuple, List


class Box:
    def __init__(self, size: Dim2, spacing: float, min: Dim2 = (0.0, 0.0), reverse: Tuple[bool, bool] = (False, False)):
        """Helper class for quickly generating functions for design region placements

        Args:
            size: size of box
            min: min x and min y of box
            spacing: spacing for pixelation
        """
        self.min = min
        self.size = size
        self.spacing = spacing
        self.reverse = reverse

    @property
    def max(self):
        return self.min[0] + self.size[0], self.min[1] + self.size[1]

    @property
    def min_i(self):
        return int(self.min[0] / self.spacing), int(self.min[1] / self.spacing)

    @property
    def max_i(self):
        return int(self.max[0] / self.spacing), int(self.max[1] / self.spacing)

    @property
    def shape(self):
        return self.max_i[0] - self.min_i[0], self.max_i[1] - self.min_i[1]

    @property
    def center(self):
        return self.min[0] + self.size[0] / 2, self.min[1] + self.size[1] / 2

    @property
    def slice(self):
        return slice(self.min_i[0], self.max_i[0], -1 if self.reverse[0] else 1), \
               slice(self.min_i[1], self.max_i[1], -1 if self.reverse[1] else 1)

    @property
    def copy(self):
        return deepcopy(self)

    def mask(self, array: Union[np.ndarray, jnp.ndarray]):
        mask = np.zeros_like(array)
        mask[self.slice[0], self.slice[1]] = 1.0
        return mask

    def rot90(self) -> "Box":
        self.size = (self.size[1], self.size[0])
        return self

    def flip_x(self) -> "Box":
        self.reverse = (not self.reverse[0], self.reverse[1])
        return self

    def flip_y(self) -> "Box":
        self.reverse = (self.reverse[0], not self.reverse[1])
        return self

    def flip_xy(self) -> "Box":
        self.reverse = (not self.reverse[0], not self.reverse[1])
        return self

    def translate(self, dx: float = 0, dy: float = 0) -> "Box":
        self.min = (self.min[0] + dx, self.min[1] + dy)
        return self

    def align(self, c: Union["Box", Tuple[float, float]]) -> "Box":
        center = c.center if isinstance(c, Box) else c
        self.translate(center[0] - self.center[0], center[1] - self.center[1])
        return self

    def halign(self, c: Union["Box", float], left: bool = True, opposite: bool = False):
        x = self.min[0] if left else self.max[0]
        p = c if isinstance(c, float) or isinstance(c, int) \
            else (c.min[0] if left and not opposite or opposite and not left else c.max[0])
        self.translate(dx=p - x)
        return self

    def valign(self, c: Union["Box", float], bottom: bool = True, opposite: bool = False):
        y = self.min[1] if bottom else self.max[1]
        p = c if isinstance(c, float) or isinstance(c, int) \
            else (c.min[1] if bottom and not opposite or opposite and not bottom else c.max[1])
        self.translate(dy=p - y)
        return self


def get_smooth_fn(beta: float, radius: float, eta: float = 0.5):
    """Using the sigmoid function and convolutional kernel provided in jax, we return a function that
        effectively binarizes the design respectively and smooths the density parameters.

    Args:
        beta: A multiplicative factor in the tanh function to effectively define how binarized the design should be
        radius: The radius of the convolutional kernel for smoothing
        eta: The average value of the design

    Returns:
        The smoothing function

    """
    rr, cc = disk((radius, radius), radius + 1)
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float)
    kernel[rr, cc] = 1
    kernel = kernel / kernel.sum()

    def smooth(rho: jnp.ndarray):
        rho = conv(rho, kernel, mode='same')
        return jnp.divide(jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta)),
                          jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))

    return smooth


def get_symmetry_fn(ortho_x: bool = False, ortho_y: bool = False, diag_p: bool = False, diag_n: bool = False):
    """Get the array-based reflection symmetry function based on orthogonal or diagonal axes.

    Args:
        ortho_x: symmetry along x-axis (axis 0)
        ortho_y: symmetry along y-axis (axis 1)
        diag_p: symmetry along positive ([1,  1] plane) diagonal (shape of params must be square)
        diag_n: symmetry along negative ([1, -1] plane) diagonal (shape of params must be square)

    Returns:
        The symmetry function
    """
    identity = (lambda x: x)
    diag_n_fn = (lambda x: (x + x.T) / 2) if diag_p else identity
    diag_p_fn = (lambda x: (x + x[::-1, ::-1].T) / 2) if diag_n else identity
    ortho_x_fn = (lambda x: (x + x[::-1]) / 2) if ortho_x else identity
    ortho_y_fn = (lambda x: (x + x[:, ::-1]) / 2) if ortho_y else identity
    return lambda x: diag_p_fn(diag_n_fn(ortho_x_fn(ortho_y_fn(x))))


def get_mask_fn(rho_init: jnp.ndarray, box: Union[Box, List[Box]]):
    """Given an initial param set, this function defines the box region(s) where the params are allowed to change

    Args:
        rho_init: initial rho definition
        box: Box (or list of boxes) defines position and orientation of the design region(s)

    Returns:
        The mask function

    """
    mask = box.mask(rho_init) if isinstance(box, Box) else (np.sum([b.mask(rho_init) for b in box]) > 0).astype(np.float)
    return lambda rho: jnp.array(rho_init) * (1 - mask) + rho * mask

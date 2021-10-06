import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import convolve as conv
from skimage.draw import disk

from .typing import Union, List
from .utils import Box


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
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    kernel[rr, cc] = 1
    kernel = kernel / kernel.sum()

    def smooth(rho: jnp.ndarray):
        rho = conv(rho, kernel, mode='same')
        return jnp.divide(jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta)),
                          jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))

    return smooth


def get_symmetry_fn(ortho_x: bool = False, ortho_y: bool = False, diag_p: bool = False, diag_n: bool = False,
                    avg: bool = False):
    """Get the array-based reflection symmetry function based on orthogonal or diagonal axes.

    Args:
        ortho_x: symmetry along x-axis (axis 0)
        ortho_y: symmetry along y-axis (axis 1)
        diag_p: symmetry along positive ([1,  1] plane) diagonal (shape of params must be square)
        diag_n: symmetry along negative ([1, -1] plane) diagonal (shape of params must be square)
        avg: Whether the symmetry should take the average (applies to ortho symmetries ONLY)

    Returns:
        The overall symmetry function
    """
    identity = (lambda x: x)
    diag_n_fn = (lambda x: (x + x.T) / 2) if diag_p else identity
    diag_p_fn = (lambda x: (x + x[::-1, ::-1].T) / 2) if diag_n else identity
    if avg:
        ortho_x_fn = (lambda x: (x + x[::-1]) / 2) if ortho_x else identity
        ortho_y_fn = (lambda x: (x + x[:, ::-1]) / 2) if ortho_y else identity
    else:
        ortho_x_fn = (lambda x: x.at[-(x.shape[0] // 2 + 1):, :].set(x[:x.shape[0] // 2 + 1:, :][::-1, :])) if ortho_x else identity
        ortho_y_fn = (lambda x: x.at[:, -(x.shape[1] // 2 + 1):].set(x[:, :x.shape[1] // 2 + 1][:, ::-1])) if ortho_y else identity
    return lambda x: diag_p_fn(diag_n_fn(ortho_x_fn(ortho_y_fn(x))))


def get_mask_fn(rho_init: jnp.ndarray, box: Union[Box, List[Box]]):
    """Given an initial param set, this function defines the box region(s) where the params are allowed to change.

    Args:
        rho_init: initial rho definition
        box: Box (or list of boxes) defines position and orientation of the design region(s)

    Returns:
        The mask function

    """
    mask = box.mask(rho_init) if isinstance(box, Box) else (sum([b.mask(rho_init) for b in box]) > 0).astype(np.float)
    return lambda rho: jnp.array(rho_init) * (1 - mask) + rho * mask

"""This file is for back-of-the-envelope calculations / figures based on them."""

import numpy as np


def binary_svd_depth_size(n, k, interport_distance=25, device_length=200, loss_db: float = 0.3):
    """Binary SVD architecture depth and size.

    Args:
        n: Number of inputs.
        k: Number of outputs (generally want :math:`k << n`).
        interport_distance: Distance between the ports of the device.
        device_length: Overall device length.
        loss_db: Loss in dB for the circuit.

    Returns:
        Information about number of layers, length, height, footprint, loss, etc.

    """
    num_input_layers = np.ceil(np.log2(n))
    num_unitary_layers = k * np.ceil(np.log2(n))
    attenuate_and_unitary_layer = k + 1
    num_waveguides_height = np.ceil(n / k) * n
    layers = num_input_layers + num_unitary_layers + attenuate_and_unitary_layer
    height = num_waveguides_height * interport_distance
    length = device_length * layers
    return {
        'layers': layers,
        'length (cm)': length / 1e4,
        'height (cm)': height / 1e4,
        'footprint (cm^2)': length * height / 1e8,
        'loss (dB)': -layers * loss_db + 10 * np.log10(1 / np.ceil(n / k)) + 10 * np.log10(1 / n) - 3
    }


def rectangular_depth_size(n, interport_distance=25, device_length=200, loss_db: float = 0.3, svd: bool = True):
    """Rectangular architecture depth and size.


    Args:
        n: The number of outputs of the binary tree
        interport_distance: Distance between each port in the network (include the phase shifters)
        device_length: Length of the device
        loss_db: Loss of the device
        svd: Whether to use an SVD architecture (doubles the number of layers in the architecture)

    Returns:
        Information about number of layers, length, height, footprint, loss, etc.

    """
    num_input_layers = np.ceil(np.log2(n))
    num_unitary_layers = n * (1 + svd) + svd
    num_waveguides_height = n
    layers = num_input_layers + num_unitary_layers
    height = num_waveguides_height * interport_distance
    length = device_length * layers
    return {
        'layers': layers,
        'length (cm)': length / 1e4,
        'height (cm)': height / 1e4,
        'footprint (cm^2)': length * height / 1e8,
        'loss (dB)': -layers * loss_db - 10 * np.log10(n) - 3 * svd
    }


def binary_equiv_cascade_size(n, n_equiv, interport_distance=25, device_length=200, loss_db: float = 0.3):
    """Binary architecture cascade size.

    Args:
        n: The number of outputs of the binary tree
        n_equiv: Find the k (number of inputs) required to match the flops of n_equiv x n_equiv matrix
        interport_distance: Distance between each port in the network (include the phase shifters)
        device_length: Length of the device
        loss_db: Loss of the device

    Returns:
        Information about number of layers, length, height, footprint, loss, etc.

    """
    k = np.ceil(n_equiv ** 2 / n)
    num_input_layers = np.ceil(np.log2(n))
    num_unitary_layers = k * np.ceil(np.log2(n))
    attenuate_and_unitary_layer = k + 1
    num_waveguides_height = n
    layers = num_input_layers + num_unitary_layers + attenuate_and_unitary_layer
    height = num_waveguides_height * interport_distance
    length = device_length * layers
    return {
        'layers': layers,
        'length (cm)': length / 1e4,
        'height (cm)': height / 1e4,
        'footprint (cm^2)': length * height / 1e8,
        'loss (dB)': -layers * loss_db + 10 * np.log10(1 / n)
    }
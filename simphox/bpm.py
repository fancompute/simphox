from typing import Tuple

import numpy as np
from scipy.linalg import solve_banded

from .sim import SimGrid
from .typing import Shape, Size, Spacing, Optional, Union, Size3


class BPM(SimGrid):
    def __init__(self, size: Size, spacing: Spacing, eps: Union[float, np.ndarray] = 1,
                 wavelength: float = 1.55, bloch_phase: Union[Size, float] = 0.0,
                 pml: Optional[Union[Shape, Size]] = None, pml_params: Size3 = (4, -16, 1, 5),
                 yee_avg: bool = True, no_grad: bool = True, not_implemented: bool = True):

        if not_implemented:  # this is just to avoid annoying pycharm linting (TODO: remove this when fixed)
            raise NotImplementedError("This class is still WIP")

        self.wavelength = wavelength
        self.k0 = 2 * np.pi / self.wavelength  # defines the units for the simulation!
        self.no_grad = no_grad

        super(BPM, self).__init__(
            size=size,
            spacing=spacing,
            eps=eps,
            bloch_phase=bloch_phase,
            pml=pml,
            pml_params=pml_params
        )

        if self.ndim == 1:
            raise ValueError(f"Simulation dimension ndim must be 2 or 3 but got {self.ndim}.")
        self.init()


    def init(self, center: Tuple[float, ...] = None, shape: Tuple[float, ...] = None, axis: int = 0):
        # initial scalar fields for fdtd
        center = (0, self.shape[1] // 2, self.shape[2] // 2) if center is None else center
        shape = self.eps[0].shape if shape is None else shape
        self.x = center[0]
        self.beta, _, self.e, self.h = mode_profile(self, center=center, size=shape, axis=axis)

    def adi_polarized(self, te: bool = True):
        """The ADI step for beam propagation method based on https://publik.tuwien.ac.at/files/PubDat_195610.pdf

        Returns:

        """
        d, _ = self._dxes
        if self.ndim == 3:
            s, e = d[1], d[0]
            n, w = np.roll(s, 1, axis=1), np.roll(e, 1, axis=0)
            n[0], w[0], s[-1], e[-1] = 0, 0, 0, 0  # set to zero to make life easy later

            a_x = np.tile(2 / (w * (e + w)).flatten(), 2)
            c_x = np.tile(2 / (e * (e + w)).flatten(), 2)
            a_y = np.tile(2 / (n * (n + s)).flatten(), 2)
            c_y = np.tile(2 / (s * (s + n)).flatten(), 2)

            eps = self.eps[self.x, :, :]
            e = self.e[1, self.x, :, :] if te else self.e[0, self.x, :, :]
            h = self.h[0, self.x, :, :] if te else self.h[1, self.x, :, :]
            phi = np.stack(e.flatten(), h.flatten())

            if te:
                eps_e = np.roll(eps, 1, axis=0)
                eps_w = np.roll(eps, -1, axis=0)
                a_x *= np.hstack(((2 * eps_w / (eps + eps_w)).flatten(), (2 * eps / (eps + eps_w)).flatten()))
                c_x *= np.hstack(((2 * eps_e / (eps + eps_e)).flatten(), (2 * eps / (eps + eps_e)).flatten()))
            else:
                eps_n = np.roll(eps, -1, axis=1)
                eps_s = np.roll(eps, 1, axis=1)
                a_y *= np.hstack(((2 * eps_n / (eps + eps_n)).flatten(), (2 * eps / (eps + eps_n)).flatten()))
                c_y *= np.hstack(((2 * eps_s / (eps + eps_s)).flatten(), (2 * eps / (eps + eps_s)).flatten()))

            b_x = -(c_x + a_x)
            b_y = -(a_y + c_y)

            if te:
                adjustment = -4 / (e * w).flatten()
                b_x = np.hstack(adjustment, np.zeros_like(adjustment)) - b_x
            else:
                adjustment = -4 / (n * s).flatten()
                b_y = np.hstack(adjustment, np.zeros_like(adjustment)) - b_y

            # ADI algorithm

            b_x += (self.k0 ** 2 * eps.flatten() - self.beta ** 2) / 2
            b_y += (self.k0 ** 2 * eps.flatten() - self.beta ** 2) / 2
            t_x = np.vstack([-a_x, -b_x - 4 * 1j * self.beta / self.spacing[-1], -c_x])
            t_y = np.vstack([-a_y, -b_y - 4 * 1j * self.beta / self.spacing[-1], -c_y])
            d_x = np.roll(phi, -1) * a_y + phi * b_y + np.roll(phi, 1) * c_y
            phi_x = solve_banded((1, 1), t_x, d_x)
            d_y = np.roll(phi, -1) * a_x + phi_x * b_x + np.roll(phi_x, 1) * c_x
            new_phi = solve_banded((1, 1), t_y, d_y)
            if te:
                self.e[1, self.x, :, :].flat, self.h[0, self.x, :, :].flat = np.hsplit(new_phi, 2)
            else:
                self.e[0, self.x, :, :].flat, self.h[1, self.x, :, :].flat = np.hsplit(new_phi, 2)
            self.x += 1

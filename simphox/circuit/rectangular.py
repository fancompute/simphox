import numpy as np

try:
    DPHOX_IMPORTED = True
    from dphox.device import Device
    from dphox.pattern import Pattern
except ImportError:
    DPHOX_IMPORTED = False

from ..typing import Callable
from .coupling import CouplingNode
from .forward import ForwardMesh


def checkerboard_to_param(checkerboard: np.ndarray, units: int):
    param = np.zeros((units, units // 2))
    if units % 2:
        param[::2, :] = checkerboard.T[::2, :-1:2]
    else:
        param[::2, :] = checkerboard.T[::2, ::2]
    param[1::2, :] = checkerboard.T[1::2, 1::2]
    return param


def get_alpha_checkerboard(n: int):
    """Get the sensitivity index for each of the nodes in a rectangular architecture.

    The sensitivity values are arranged in a checkerboard form for easy spatial mapping to
    coupling nodes in a rectangular architecture.

    Args:
        n: The number of inputs in the rectangular architecture

    Returns:

    """
    def rectangular_alpha(length: int, parity_odd: bool = False):
        odd_nums = list(length + 1 - np.flip(np.arange(1, length + 1, 2), axis=0))
        even_nums = list(length + 1 - np.arange(2, 2 * (length - len(odd_nums)) + 1, 2))
        nums = np.asarray(odd_nums + even_nums)
        if parity_odd:
            nums = nums[::-1]
        return nums
    alpha_checkerboard = np.zeros((n, n))
    diagonal_length_to_sequence = [rectangular_alpha(i, bool(n % 2)) for i in range(1, n + 1)]
    for i in range(n - 1):
        for j in range(n):
            if (i + j) % 2 == 0:
                if j > i:
                    diagonal_length = n - np.abs(i - j)
                elif i > 0 and j < i:
                    diagonal_length = n - np.abs(i - j) - 1
                else:
                    diagonal_length = n - 1
                alpha_checkerboard[i, j] = 1 if diagonal_length == 1 else \
                    diagonal_length_to_sequence[int(diagonal_length) - 1][min(i, j)]
    return alpha_checkerboard


def grid_common_mode_flow(external_phases: np.ndarray, gamma: np.ndarray):
    """In a grid mesh (e.g., triangular, rectangular meshes), phases may need to be re-arranged.
     This is achieved using a procedure called "common mode flow" where common modes are shifted
     throughout the mesh until phases are correctly set.

    Args:
        external_phases: external phases in the grid mesh
        gamma: input phase shifts

    Returns:
        new external phases shifts and new gamma resulting

    """
    units, num_layers = external_phases.shape
    phase_shifts = np.hstack((external_phases, gamma[:, np.newaxis])).T
    new_phase_shifts = np.zeros_like(external_phases.T)

    for i in range(num_layers):
        start_idx = i % 2
        end_idx = units - (i + units) % 2

        # calculate upper and lower phases
        upper_phase = phase_shifts[i][start_idx:end_idx][::2]
        lower_phase = phase_shifts[i][start_idx:end_idx][1::2]
        upper_phase = np.mod(upper_phase, 2 * np.pi)
        lower_phase = np.mod(lower_phase, 2 * np.pi)

        # upper - lower
        new_phase_shifts[i][start_idx:end_idx][::2] = upper_phase - lower_phase

        # lower_phase is now the common mode for all phase shifts in this layer
        phase_shifts[i] -= new_phase_shifts[i]

        # shift the phases to the next layer in parallel
        phase_shifts[i + 1] += np.mod(phase_shifts[i], 2 * np.pi)
    new_gamma = np.mod(phase_shifts[-1], 2 * np.pi)
    return np.mod(new_phase_shifts.T, 2 * np.pi), new_gamma


def rectangular(u: np.ndarray, pbar: Callable = None):
    """Get a rectangular architecture for the unitary matrix :code:`u` using the Clements decomposition.

    Args:
        u: The unitary matrix
        pbar: The progress bar for the clements decomposition (useful for larger unitaries)

    Returns:

    """
    u_hat = u.copy()
    n = u.shape[0]
    # odd and even layer dimensions
    theta_checkerboard = np.zeros_like(u, dtype=np.float64)
    phi_checkerboard = np.zeros_like(u, dtype=np.float64)
    phi_checkerboard = np.hstack((np.zeros((n, 1)), phi_checkerboard))
    iterator = pbar(range(n - 1)) if pbar else range(n - 1)
    for i in iterator:
        if i % 2:
            for j in range(i + 1):
                pairwise_index = n + j - i - 2
                target_row, target_col = n + j - i - 1, j
                theta = np.arctan2(np.abs(u_hat[target_row - 1, target_col]), np.abs(u_hat[target_row, target_col])) * 2
                phi = np.angle(u_hat[target_row, target_col]) - np.angle(u_hat[target_row - 1, target_col])
                left = CouplingNode(n=n, top=pairwise_index, bottom=pairwise_index + 1)
                # Need an equivalent of differential internal phase shift to invert the matrix properly.
                u_hat = left.phase_matrix(-theta / 2, -theta / 2) @ left.mzi_node_matrix(theta, phi) @ u_hat
                theta_checkerboard[pairwise_index, -j - 1] = theta
                phi_checkerboard[pairwise_index, -j - 1] = -phi - theta / 2 + np.pi
                phi_checkerboard[pairwise_index + 1, -j - 1] = -theta / 2 + np.pi
        else:
            for j in range(i + 1):
                pairwise_index = i - j
                target_row, target_col = n - j - 1, i - j
                theta = np.arctan2(np.abs(u_hat[target_row, target_col + 1]), np.abs(u_hat[target_row, target_col])) * 2
                phi = np.angle(-u_hat[target_row, target_col]) - np.angle(u_hat[target_row, target_col + 1])
                right = CouplingNode(n=n, top=pairwise_index, bottom=pairwise_index + 1)
                u_hat = u_hat @ right.mzi_node_matrix(theta, phi).conj().T
                theta_checkerboard[pairwise_index, j] = theta
                phi_checkerboard[pairwise_index, j] = phi

    diag_phases = np.angle(np.diag(u_hat))
    theta = checkerboard_to_param(theta_checkerboard, n)
    alpha_checkerboard = get_alpha_checkerboard(n)
    if n % 2:
        phi_checkerboard[:, :-1] += np.fliplr(np.diag(diag_phases))
    else:
        phi_checkerboard[:, 1:] += np.fliplr(np.diag(diag_phases))

    # Run the common mode flow algorithm to move phase front to the last layer of the mesh
    phi, gamma = grid_common_mode_flow(external_phases=phi_checkerboard[:, :-1], gamma=phi_checkerboard[:, -1])
    phi = checkerboard_to_param(phi, n)
    alpha = checkerboard_to_param(alpha_checkerboard, n)

    # Set up the rectangular mesh nodes
    nodes = []
    thetas = np.array([])
    phis = np.array([])
    node_id = 0
    for i in range(n):
        num_to_interfere = theta.shape[1] - (i % 2) * (1 - n % 2)
        nodes += [CouplingNode(node_id=node_id + j, n=n, column=i,
                               top=2 * j + i % 2, bottom=2 * j + 1 + i % 2,
                               alpha=1, beta=alpha[i, j])
                  for j in range(num_to_interfere)]
        thetas = np.hstack([thetas, theta[i, :num_to_interfere]])
        phis = np.hstack([phis, phi[i, :num_to_interfere]])
        node_id += num_to_interfere

    unit = ForwardMesh(nodes)
    unit.params = thetas, phis, gamma

    return unit


def rectangular_rows(rectangular_mesh: ForwardMesh):
    n = rectangular_mesh.n
    rows = [[] for _ in range(n - 1)]
    for i in range(n - 1):
        for j in range(i + 1):
            pairwise_index = n + j - i - 2 if i % 2 else i - j
            rows[i].append(CouplingNode(n=n, top=n - pairwise_index - 2, bottom=n - pairwise_index - 1,
                                        column=n - j - 1 if i % 2 else j))
    return [ForwardMesh(row).column_ordered for row in rows]


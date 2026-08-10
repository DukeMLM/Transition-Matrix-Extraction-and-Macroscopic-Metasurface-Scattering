"""Deliverable 1: aggregate per-atom T-matrices into one array system.

Two aggregation modes:

(a) Finite array (Foldy-Lax).  For sites i with T-matrices T^i at positions
    R_i, the self-consistent exciting coefficients a^i satisfy

        a^i - sum_{j != i} A(R_j - R_i) T^j a^j = a_inc^i

    which is ONE global block matrix  M x = b  with  M = I - Ablk diag(T),
    x = (a^1 ... a^N),  b = (a_inc^1 ... a_inc^N).  Scattered coefficients
    are f^i = T^i a^i.

(b) Infinite periodic array (normal incidence, k_par = 0).  All cells are
    identical, so the block system collapses to the unit cell:

        a = a_inc + C T a   =>   a = (I - C T)^{-1} a_inc,   f = T a

    with C the lattice sum of translation operators.  The effective per-cell
    ("array") T-matrix is  T_eff = T (I - C T)^{-1},  f = T_eff a_inc.
"""
import numpy as np

from tmatrix.aggregation.vswf import RegularProjector
from tmatrix.aggregation.translate import translation_matrix, translation_shells, rotate_inplane


def build_finite_system(k, T_list, positions, modes, r0, quad):
    """Global block matrix M = I - Ablk diag(T) for the Foldy-Lax system.

    T_list: list of (n, n) per-site T-matrices (or a single matrix reused).
    positions: (N, 3) site positions.  For in-plane (z = const) arrays the
    translation operators are built from one projection per distinct radius
    plus the azimuthal rotation identity.
    Returns M (N*n x N*n) and the list of per-site T.
    """
    positions = np.asarray(positions, dtype=float)
    N = len(positions)
    n = modes.n
    if isinstance(T_list, np.ndarray) and T_list.ndim == 2:
        T_list = [T_list] * N
    proj = RegularProjector(modes, quad)
    in_plane = np.ptp(positions[:, 2]) < 1e-12

    M = np.eye(N * n, dtype=complex)
    if in_plane:
        # one projection per distinct displacement radius, then rotate
        d_all = positions[None, :, :2] - positions[:, None, :2]
        r_all = np.round(np.linalg.norm(d_all, axis=2), 9)
        radii = np.unique(r_all[r_all > 1e-12])
        A_r = translation_shells(k, radii, modes, r0, quad, projector=proj)
        lookup = {r: A_r[s] for s, r in enumerate(radii)}
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = d_all[i, j]
                phi = np.arctan2(d[1], d[0])
                A = rotate_inplane(lookup[r_all[i, j]], phi, modes)
                M[i * n:(i + 1) * n, j * n:(j + 1) * n] -= A @ T_list[j]
    else:
        cache = {}
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = positions[j] - positions[i]
                key = tuple(np.round(d, 12))
                if key not in cache:
                    cache[key] = translation_matrix(
                        k, d, modes, r0, quad, projector=proj)
                M[i * n:(i + 1) * n, j * n:(j + 1) * n] -= cache[key] @ T_list[j]
    return M, T_list


def solve_finite(M, T_list, a_inc_blocks):
    """Solve the global system; returns per-site exciting a^i and scattered f^i."""
    N = len(T_list)
    n = a_inc_blocks.shape[1]
    b = a_inc_blocks.reshape(N * n)
    x = np.linalg.solve(M, b)
    a = x.reshape(N, n)
    f = np.stack([T_list[i] @ a[i] for i in range(N)])
    return a, f


def solve_periodic(T, C, a_inc):
    """Unit-cell solve for the infinite array: returns (a, f)."""
    n = T.shape[0]
    a = np.linalg.solve(np.eye(n, dtype=complex) - C @ T, a_inc)
    return a, T @ a


def effective_array_T(T, C):
    """T_eff with f = T_eff a_inc for the periodic array (per unit cell)."""
    n = T.shape[0]
    return T @ np.linalg.inv(np.eye(n, dtype=complex) - C @ T)

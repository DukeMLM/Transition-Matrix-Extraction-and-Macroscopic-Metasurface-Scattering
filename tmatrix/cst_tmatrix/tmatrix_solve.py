"""Assemble the multi-illumination linear system and solve for T.

Following NAMM Eq. 61: with incident coefficient columns [A] and scattered
coefficient columns [F] (one column per illumination), the T-matrix is the
least-squares solution of F = T A:
    T = F A^H (A A^H)^{-1}   (right pseudoinverse; computed via lstsq).
"""

from __future__ import annotations

import numpy as np


def solve_tmatrix(A: np.ndarray, F: np.ndarray, rcond: float | None = None):
    """Least-squares solve T from F = T A.

    A : (N, K) incident coefficient columns (K illuminations)
    F : (N, K) scattered coefficient columns
    Returns (T, diag) where diag holds 'residual' (relative Frobenius),
    'cond' (condition number of A), 'rank'.
    """
    A = np.asarray(A)
    F = np.asarray(F)
    if A.shape != F.shape:
        raise ValueError(f"shape mismatch {A.shape} vs {F.shape}")
    # F = T A  <=>  A^T T^T = F^T : standard lstsq per column of T^T
    Tt, res, rank, sv = np.linalg.lstsq(A.T, F.T, rcond=rcond)
    T = Tt.T
    resid = np.linalg.norm(T @ A - F) / max(np.linalg.norm(F), 1e-300)
    diag = {
        "residual": float(resid),
        "cond": float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
        "rank": int(rank),
        "n_illuminations": A.shape[1],
        "n_modes": A.shape[0],
    }
    return T, diag


def passivity_check(T: np.ndarray):
    """S = I + 2T must have all singular values <= 1 (passive scatterer);
    equality for lossless.  Returns (max_singular_value, unitarity_error)."""
    S = np.eye(T.shape[0]) + 2 * T
    sv = np.linalg.svd(S, compute_uv=False)
    unitarity = float(np.max(np.abs(sv - 1.0)))
    return float(sv.max()), unitarity


def reciprocity_check(T: np.ndarray, lmax: int):
    """Relative deviation of T from the reciprocity constraint
        T[(p',n',m'),(p,n,m)] = (-1)^{m+m'} T[(p,n,-m),(p',n',-m')].

    Any scatterer made of reciprocal media (no magnetized plasmas / biased
    ferrites) must satisfy this to within extraction accuracy — a violation
    at O(1) flags an extraction/convention bug, not physics.  The sign
    factor was pinned NUMERICALLY against Saxon's amplitude reciprocity
    (e2.F(k2<-k1;e1) = e1.F(-k1<-(-k2);e2)) built from the locked
    plane-wave/farfield primitives; see tests/test_reciprocity_coated.py.

    Returns ||T - R(T)|| / ||T|| (Frobenius).
    """
    from .vswf import mode_list, block_index, n_modes

    N = n_modes(lmax)
    if T.shape != (N, N):
        raise ValueError(f"T shape {T.shape} != ({N}, {N})")
    pol, ns, ms = mode_list(lmax)
    half = N // 2
    neg = np.array([(0 if p == 0 else half) + block_index(n, -m)
                    for p, n, m in zip(pol, ns, ms)])
    sign = (-1.0) ** (ms[:, None] + ms[None, :])
    R = sign * T[np.ix_(neg, neg)].T
    return float(np.linalg.norm(T - R) / max(np.linalg.norm(T), 1e-300))


def _diag_to_full(lmax: int, TM: np.ndarray, TN: np.ndarray):
    from .vswf import block_index, n_modes

    N = n_modes(lmax)
    half = N // 2
    d = np.zeros(N, dtype=complex)
    for n in range(1, lmax + 1):
        for mm in range(-n, n + 1):
            d[block_index(n, mm)] = TM[n - 1]
            d[half + block_index(n, mm)] = TN[n - 1]
    return np.diag(d)


def sphere_reference_T(lmax: int, x: float, m: complex | None):
    """Full (N x N) diagonal reference T-matrix of a sphere for validation."""
    from .mie import sphere_tmatrix_diagonal

    TM, TN = sphere_tmatrix_diagonal(lmax, x, m)
    return _diag_to_full(lmax, TM, TN)


def coated_sphere_reference_T(lmax: int, x_core: float, x_shell: float,
                              m_core: complex, m_shell: complex):
    """Full (N x N) reference T-matrix of a coated (core-shell) sphere.
    Size parameters and index contrasts as in mie.coated_sphere_tmatrix_diagonal
    (package convention: loss = negative imaginary part)."""
    from .mie import coated_sphere_tmatrix_diagonal

    TM, TN = coated_sphere_tmatrix_diagonal(lmax, x_core, x_shell,
                                            m_core, m_shell)
    return _diag_to_full(lmax, TM, TN)

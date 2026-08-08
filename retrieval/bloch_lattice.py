"""Bloch-phased periodic lattice sums (retrieval deliverable 1).

Computes, for an in-plane square lattice (pitch p) and in-plane Bloch vector
k_par = k (sin th cos ph, sin th sin ph):

    C(k, k_par) = sum_{R != 0} A(R) e^{+i k_par . R}

with the same Gaussian-taper + Richardson-in-1/Rc^2 machinery as
translate.lattice_sum_C, which is recovered exactly at k_par = 0 (the code
path is the same arithmetic: the Bloch factors evaluate to exactly 1+0j).

Bloch sign (normative, INVERSE_TMATRIX_FROM_FLOQUET.md par. 2): POSITIVE,
e^{+i k_par . R}.  With the e^{+i k.r} spatial convention of the aggregation
package, the site at R sees the incident coefficients a_inc e^{+i k_par . R},
so summing the response back at the origin cell carries the same positive
phase (verified against aggregate.build_finite_system's conventions).

Structure and reuse contract
----------------------------
The shell matrices A_s = A(r_s x_hat) produced by translate.translation_shells
depend ONLY on (k, radii, modes, r0, quad) -- not on k_par and not on the
taper set.  They are therefore computed once per (k, pitch, r_max) and reused
across incidence angles and taper sets through the `shells_cache` dict
accepted by lattice_sum_C_bloch (essential downstream: precompute_C reuses a
single shell set for ~13 angles per frequency).

Only the assembly differs from translate.assemble_shell_sum: the per-shell
azimuthal phase sums become per-site sums

    P_s(dm) = w(r_s) sum_{R in shell s} e^{i dm phi_R} e^{+i k_par . R},
    w(r) = exp(-r^2/Rc^2),

with the sites R = r_s (cos phi_R, sin phi_R, 0) enumerated by
translate.square_lattice_shells and the site matrix given by the rotation
identity A(R)[mu, nu] = A_s[mu, nu] e^{i (m_nu - m_mu) phi_R}
(= translate.rotate_inplane; identity verified in test_translate.py).  The
Richardson extrapolation over the kRc list is unchanged.

Convergence warning (doc par. 2): Bloch phases slow the propagating-channel
shell-sum convergence -- the controlling parameter becomes k (1 - sin th) Rc,
the distance to the grazing Rayleigh anomaly.  Scale kRc >~ 10/(1 - sin th)
at oblique incidence and re-run Richardson-stability checks at the largest
k_par (see test_bloch_lattice.py tests (c) and (d)).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "aggregation"))

import numpy as np

from translate import translation_shells, square_lattice_shells


def build_shells(k, pitch, modes, r0, quad, r_max, projector=None, chunk=48):
    """Enumerate square-lattice shells with 0 < |R| <= r_max and compute their
    translation matrices A_s = A(r_s x_hat).

    Returns (radii, angles, A_s): radii (nshell,) ascending, angles a list of
    per-shell site-angle arrays phi_R, A_s (nshell, n, n) complex.  The result
    depends only on (k, pitch, modes, r0, quad, r_max): it is independent of
    the Bloch vector and of the taper set, and is the expensive part of any
    lattice sum -- cache it (see lattice_sum_C_bloch's shells_cache).
    """
    radii, angles = square_lattice_shells(pitch, r_max)
    A_s = translation_shells(k, radii, modes, r0, quad,
                             projector=projector, chunk=chunk)
    return radii, angles, A_s


def slice_shells(radii, angles, A_s, pitch, r_max):
    """Restrict a shell set built at some r_max' >= r_max to exactly the
    shells a fresh square_lattice_shells(pitch, r_max) call would keep.

    square_lattice_shells keeps integer sites with r2 * pitch**2 <= r_max**2;
    the same float test is replicated here on the (exactly recoverable)
    integer r2 of each shell, so the sliced set is identical to a fresh
    enumeration.  radii are ascending, so the kept shells are a prefix.
    Useful for convergence studies: build one superset, then evaluate smaller
    taper sets on prefix slices at zero extra projection cost.
    """
    r2 = np.rint((np.asarray(radii) / pitch) ** 2).astype(np.int64)
    keep = r2 * pitch ** 2 <= r_max ** 2
    K = int(np.count_nonzero(keep))
    return radii[:K], angles[:K], A_s[:K]


def site_phase_sums(radii, angles, modes, k_par):
    """Taper-independent per-shell Bloch phase sums.

    Q[s, i] = sum_{R in shell s} e^{i dms[i] phi_R} e^{+i k_par . R},
    R = r_s (cos phi_R, sin phi_R).  Returns (dms, Q) with dms the sorted
    unique values of m_nu - m_mu and Q of shape (nshell, len(dms)).
    """
    kx, ky = float(k_par[0]), float(k_par[1])
    dm = modes.m[None, :] - modes.m[:, None]      # (mu, nu): m_nu - m_mu
    dms = np.unique(dm)
    Q = np.empty((len(radii), len(dms)), dtype=complex)
    for s, (r, ph) in enumerate(zip(radii, angles)):
        bloch = np.exp(1j * r * (kx * np.cos(ph) + ky * np.sin(ph)))
        Q[s] = (bloch[:, None] * np.exp(1j * np.outer(ph, dms))).sum(axis=0)
    return dms, Q


def assemble_shell_sum_bloch(A_s, radii, angles, modes, Rcs, k_par,
                             chunk=2048):
    """Combine shell matrices with tapered Bloch-phased per-site sums and
    Richardson-extrapolate over the taper lengths Rcs (length units).

    C(Rc) = sum_s A_s[mu, nu] * P_s(dm[mu, nu]),
    P_s(dm) = w(r_s) sum_{R in shell s} e^{i dm phi_R} e^{+i k_par . R},

    then Lagrange (Richardson) extrapolation of C(Rc) to 1/Rc^2 -> 0 over the
    Rcs list, exactly as translate.assemble_shell_sum.  At k_par = (0, 0) the
    Bloch factors are exactly 1+0j and the arithmetic reduces to
    translate.assemble_shell_sum's (shells with taper weight w < 1e-16
    contribute exact zeros, matching translate's skip).

    The einsum is chunked over shells (`chunk`) so the (nshell, n, n) phase
    array is never materialized in full; this only regroups the shell sum and
    is numerically equivalent at rounding level.
    """
    dm = modes.m[None, :] - modes.m[:, None]      # (mu, nu): m_nu - m_mu
    dms, Q = site_phase_sums(radii, angles, modes, k_par)
    dm_idx = np.searchsorted(dms, dm)
    Cs = []
    for Rc in Rcs:
        w = np.exp(-(np.asarray(radii) / Rc) ** 2)
        w[w < 1e-16] = 0.0                        # translate's shell skip
        P = w[:, None] * Q                        # (nshell, ndms)
        C_rc = np.zeros((modes.n, modes.n), dtype=complex)
        for i0 in range(0, len(radii), chunk):
            sl = slice(i0, min(i0 + chunk, len(radii)))
            C_rc += np.einsum("smn,smn->mn", A_s[sl], P[sl][:, dm_idx])
        Cs.append(C_rc)
    x = 1.0 / np.asarray(Rcs, dtype=float) ** 2
    C = np.zeros_like(Cs[0])
    for i in range(len(Rcs)):
        L = 1.0
        for j in range(len(Rcs)):
            if j != i:
                L *= x[j] / (x[j] - x[i])
        C += L * Cs[i]
    return C


def lattice_sum_C_bloch(k, pitch, modes, r0, quad, k_par,
                        kRc=(10.0, 14.0, 20.0), r_max_factor=3.5,
                        projector=None, shells_cache=None):
    """Richardson-extrapolated Bloch lattice sum
    C(k, k_par) = sum_{R != 0} A(R) e^{+i k_par . R} for a square lattice.

    Parameters
    ----------
    k : float
        Embedding-medium wavenumber (rad / length unit).
    pitch : float
        Square-lattice pitch (length units).
    modes : vswf.ModeBasis
    r0, quad : projection sphere radius and quadrature, as in translate.
    k_par : array-like, length 2 (or 3 with zero z component)
        In-plane Bloch vector (kx, ky) = k sin th (cos ph, sin ph).
        Sign convention: POSITIVE phase e^{+i k_par . R} (doc par. 2).
        k_par = (0, 0) reproduces translate.lattice_sum_C exactly.
    kRc : sequence of float
        Dimensionless taper lengths k*Rc for the Richardson extrapolation.
        At oblique incidence scale as kRc >~ 10/(1 - sin th) (doc par. 2).
    r_max_factor : float
        Shell cutoff r_max = r_max_factor * max(Rc), as in translate.
    projector : vswf.RegularProjector, optional
        Reused across calls if provided (only consulted on a cache miss).
    shells_cache : dict, optional
        Mutable cache of shell sets keyed by (k, pitch, r_max).  Pass the
        same dict across calls to reuse the expensive translation_shells
        build for every incidence angle and taper set sharing (k, r_max).
        The cache assumes fixed (modes, r0, quad, projector) -- use one dict
        per configuration.

    Returns
    -------
    C : (modes.n, modes.n) complex ndarray
    """
    k_par = np.asarray(k_par, dtype=float).ravel()
    if k_par.size == 3:
        if abs(k_par[2]) > 1e-12 * max(1.0, abs(k)):
            raise ValueError("k_par must be in-plane (z component ~ 0)")
        k_par = k_par[:2]
    elif k_par.size != 2:
        raise ValueError("k_par must have 2 (or 3, in-plane) components")
    kRc = np.asarray(kRc, dtype=float)
    Rcs = kRc / k
    r_max = r_max_factor * Rcs.max()
    key = (round(float(k), 12), round(float(pitch), 12),
           round(float(r_max), 9))
    if shells_cache is not None and key in shells_cache:
        radii, angles, A_s = shells_cache[key]
    else:
        radii, angles, A_s = build_shells(k, pitch, modes, r0, quad, r_max,
                                          projector=projector)
        if shells_cache is not None:
            shells_cache[key] = (radii, angles, A_s)
    return assemble_shell_sum_bloch(A_s, radii, angles, modes, Rcs, k_par)

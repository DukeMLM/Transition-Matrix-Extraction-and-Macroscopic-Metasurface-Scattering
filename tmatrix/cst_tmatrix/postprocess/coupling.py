"""Multiple-scattering (Foldy-Lax) coupling between extracted T-matrices.

The single-scatterer T-matrix in `library/` describes an ISOLATED particle.
A metasurface is an array, and its specular response is not the isolated
response: each site is driven by the external field PLUS the field scattered
by every other site.  This module supplies that coupling, which is the
missing physics behind the residual disagreement documented in
`postprocess/metasurface.py` (its `specular_s_parameters` uses the isolated
T) and item 1 of docs/COMPARISON_padilla_tmatrix.md Sec. 5.

Foldy-Lax for a cluster of scatterers at positions r_i with T-matrices T_i:

    b_i = T_i ( a_i + sum_{j != i} C(r_i - r_j) b_j )

where b_i are outgoing coefficients about site i, a_i is the external
incident field expanded about site i, and C(d) is the vector translation-
addition operator taking OUTGOING coefficients about a source centre to
REGULAR coefficients about a centre displaced by d.  Rearranged, this is a
linear system solved directly by `foldy_lax_solve`.

Construction of C(d)
--------------------
C(d) is built NUMERICALLY from this package's own validated field
primitives (vswf.evaluate_EH / vswf.project_surface_field): evaluate each
outgoing basis mode about the origin on a sphere centred at d, then project
that field onto regular modes about d.  This is the translation-addition
theorem evaluated rather than derived.

The alternative -- closed-form Cruzan/Stein coefficients via Gaunt
coefficients and Wigner 3j symbols -- carries exactly the class of
normalization/phase-convention risk this package has repeatedly been bitten
by (the Mie conjugation, the S-parameter sign, and the rotation convention
in postprocess/symmetry.py, where the naive D(+alpha) is O(1) wrong).  The
numerical route inherits whatever conventions the package's own primitives
already use, so it cannot disagree with them.  The same rationale is spelled
out in postprocess/symmetry.mirror_representation.

Choosing the projection radius rho
----------------------------------
rho is the radius of the sphere on which the re-expansion is performed about
the RECEIVING centre.  It must satisfy rho < |d| so the sphere excludes the
source singularity.  Critically it must NOT be scaled with the separation:
a regular VSWF expansion on a sphere of radius rho carries angular content
up to ~k*rho, so truncating at lmax is only faithful while k*rho <~ lmax.
Measured field-reproduction error at k=1, lmax=8, with rho = |d|/2:

    |d| = 12 (k*rho =  6)  ->  1.9e-3
    |d| = 20 (k*rho = 10)  ->  1.8e-2
    |d| = 40 (k*rho = 20)  ->  6.6e-1     <- unusable

The fix is physical rather than numerical: the re-expanded field is only
ever needed where the RECEIVING particle actually sits, i.e. out to its own
circumscribing radius r_circ -- and lmax is already chosen (Wiscombe) to
resolve exactly that region.  With rho = r_circ the same measurement gives
1e-5..1e-6 for well-separated sites, independent of |d|.  So the API takes
`r_circ`, not a fraction of the separation.

Separation limit
----------------
Accuracy degrades as circumscribing spheres approach (measured at k=1):

    r_circ = 2, |d| =  8  (|d| = 2.0 * 2*r_circ)  ->  1.5e-3
    r_circ = 2, |d| = 15  (|d| = 3.8 * 2*r_circ)  ->  2.4e-6
    r_circ = 3, |d| =  8  (|d| = 1.3 * 2*r_circ)  ->  6.3e-2

`check_pair_separation` reports this rather than letting it corrupt results
silently.  Overlapping circumscribing spheres are outside the validity of
the VSWF multiple-scattering formulation altogether and must be treated as a
single scatterer.

A note on cond(C)
-----------------
The condition number of C is enormous (1e10-1e19) because outgoing modes of
high order have huge amplitude at distance (h_l(kr) grows steeply with l),
so the columns span many orders of magnitude.  This is a property of the
basis, not an error: C is only ever applied to physically-decaying
coefficient vectors, and the measured field reproduction above stays at
1e-5..1e-6 regardless.  Do not "fix" it by regularizing C.
"""

from __future__ import annotations

import numpy as np

from ..vswf import (evaluate_EH, project_surface_field, n_modes,
                    plane_wave_coefficients)
from ..quadrature import quadrature_for_monitor, unit_vectors


def check_pair_separation(separation: float, r_circ_i: float,
                          r_circ_j: float, safety: float = 2.0) -> str | None:
    """None if a pair is comfortably separated for VSWF multiple scattering,
    otherwise a warning string.

    Hard requirement: separation > r_circ_i + r_circ_j (non-overlapping
    circumscribing spheres).  Practical requirement for the truncated
    translation operator: a further factor `safety` beyond that.
    """
    touch = r_circ_i + r_circ_j
    if separation <= touch:
        return (f"circumscribing spheres OVERLAP: separation "
                f"{separation:.4g} <= r_i + r_j = {touch:.4g}. VSWF "
                f"multiple scattering is not valid here at all -- the two "
                f"particles must be extracted as ONE scatterer.")
    if separation < safety * touch:
        return (f"separation {separation:.4g} is only "
                f"{separation / touch:.2f}x the touching distance "
                f"(r_i + r_j = {touch:.4g}). The truncated translation "
                f"operator loses accuracy as circumscribing spheres "
                f"approach (measured ~1e-3 at 2x, ~6e-2 at 1.3x, vs ~1e-6 "
                f"at 4x). Raise lmax or treat the pair as one scatterer.")
    return None


def translation_operator(lmax: int, k: float, d, r_circ: float,
                         oversample: float = 1.0,
                         quad_margin: int = 6) -> np.ndarray:
    """C(d): OUTGOING coefficients about the origin -> REGULAR coefficients
    about a centre displaced by `d`.

    d : (3,) Cartesian displacement, same length units as 1/k.
    r_circ : circumscribing radius of the RECEIVING scatterer -- the radius
        of the projection sphere.  See the module docstring: this must track
        the particle size, never a fraction of the separation.  Must be
        < |d| (and in practice well under it; see check_pair_separation).
    """
    d = np.asarray(d, dtype=float).reshape(3)
    dist = float(np.linalg.norm(d))
    if dist <= 0:
        raise ValueError("translation displacement must be non-zero")
    if not (0 < r_circ < dist):
        raise ValueError(
            f"projection radius r_circ={r_circ:.4g} must satisfy "
            f"0 < r_circ < |d|={dist:.4g} (the sphere has to exclude the "
            f"source singularity)")

    theta, phi, w, _ = quadrature_for_monitor(lmax, k, r_circ,
                                              margin=quad_margin,
                                              oversample=oversample)
    local = r_circ * unit_vectors(theta, phi)       # about the shifted centre
    pts_global = local + d                          # about the source centre

    N = n_modes(lmax)
    C = np.zeros((N, N), dtype=complex)
    for j in range(N):
        e_j = np.zeros(N, dtype=complex)
        e_j[j] = 1.0
        E, ZH = evaluate_EH(e_j, lmax, "outgoing", k, pts_global)
        C[:, j] = project_surface_field(E, lmax, "regular", k, r_circ,
                                        theta, phi, w, H=ZH)
    return C


def plane_wave_coefficients_at(theta_i: float, phi_i: float, pol: str,
                               lmax: int, k: float, r_site) -> np.ndarray:
    """Incident plane-wave coefficients expanded about a site at `r_site`.

    A plane wave is a translation eigenfunction, so this is the origin-centred
    expansion times a scalar phase.  The phase convention follows
    vswf.plane_wave_field (E = e_hat exp(-j k khat.r)); it is verified
    numerically against a direct projection in the tests (5.7e-15) rather
    than assumed.
    """
    a0 = plane_wave_coefficients(theta_i, phi_i, pol, lmax)
    k_hat = unit_vectors(np.atleast_1d(theta_i), np.atleast_1d(phi_i))[0]
    r_site = np.asarray(r_site, dtype=float).reshape(3)
    return np.exp(-1j * k * float(k_hat @ r_site)) * a0


def _as_radii(r_circ, M):
    r = np.atleast_1d(np.asarray(r_circ, dtype=float))
    if r.size == 1:
        r = np.repeat(r, M)
    if r.size != M:
        raise ValueError(f"r_circ must be scalar or length {M}, got {r.size}")
    return r


def coupling_matrix(lmax: int, k: float, positions, r_circ,
                    oversample: float = 1.0, warn: bool = True):
    """All pairwise translation operators for a cluster.

    Returns {(i, j): C(r_i - r_j)} for i != j -- the operator re-expanding
    site j's outgoing field as an incident field about site i.  The
    projection radius for entry (i, j) is site i's own r_circ, since site i
    is the receiver.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    M = len(positions)
    radii = _as_radii(r_circ, M)
    out = {}
    warned = set()
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            d = positions[i] - positions[j]
            sep = float(np.linalg.norm(d))
            if warn and (j, i) not in warned:
                msg = check_pair_separation(sep, radii[i], radii[j])
                if msg is not None:
                    print(f"WARNING: sites {i},{j}: {msg}")
                    warned.add((i, j))
            out[(i, j)] = translation_operator(lmax, k, d, r_circ=radii[i],
                                               oversample=oversample)
    return out


def foldy_lax_solve(T_list, positions, a_list, k: float, lmax: int,
                    r_circ, oversample: float = 1.0, C: dict | None = None,
                    warn: bool = True):
    """Self-consistent outgoing coefficients for a finite cluster.

    T_list : list of (N, N) T-matrices, one per site (may be the same array
        repeated for identical particles).
    positions : (M, 3) site centres, same length units as 1/k.
    a_list : list of (N,) external incident coefficients about each site
        (see plane_wave_coefficients_at).
    r_circ : circumscribing radius, scalar or per site.
    C : optional precomputed dict from coupling_matrix() -- pass it to reuse
        the (expensive) translation operators across many incidences at the
        same geometry and frequency.

    Returns (b_list, info) where b_list[i] is site i's outgoing coefficient
    vector INCLUDING all multiple scattering.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    M = len(positions)
    N = n_modes(lmax)
    if len(T_list) != M or len(a_list) != M:
        raise ValueError(f"T_list ({len(T_list)}), positions ({M}) and "
                         f"a_list ({len(a_list)}) must agree in length")
    if C is None:
        C = coupling_matrix(lmax, k, positions, r_circ,
                            oversample=oversample, warn=warn)

    Amat = np.zeros((M * N, M * N), dtype=complex)
    rhs = np.zeros(M * N, dtype=complex)
    for i in range(M):
        sl_i = slice(i * N, (i + 1) * N)
        Amat[sl_i, sl_i] = np.eye(N)
        rhs[sl_i] = T_list[i] @ a_list[i]
        for j in range(M):
            if i == j:
                continue
            Amat[sl_i, j * N:(j + 1) * N] = -T_list[i] @ C[(i, j)]
    b = np.linalg.solve(Amat, rhs)
    b_list = [b[i * N:(i + 1) * N] for i in range(M)]

    resid = np.linalg.norm(Amat @ b - rhs) / max(np.linalg.norm(rhs), 1e-300)
    info = {"residual": float(resid), "n_sites": M, "n_modes": N}
    return b_list, info


def single_scattering_solve(T_list, a_list):
    """Born / isolated-scatterer result: b_i = T_i a_i, no coupling.

    The limit Foldy-Lax must reduce to as the sites separate, and the
    approximation currently used by metasurface.specular_s_parameters.
    """
    return [T @ a for T, a in zip(T_list, a_list)]


def effective_site_tmatrix(T, Omega):
    """T_eff = [I - T Omega]^-1 T -- the per-site T-matrix dressed by array
    coupling, given the lattice sum Omega = sum_{R != 0} C(R) e^{i k_par . R}.

    Substituting T_eff for T in metasurface.specular_s_parameters upgrades
    that calculation from the isolated-scatterer approximation to the
    self-consistent array response.

    NOTE: computing Omega for an INFINITE periodic lattice is a separate
    problem -- the direct sum is only conditionally convergent and requires
    Ewald summation, which is NOT implemented here.  `foldy_lax_solve`
    covers finite clusters, where the sum is finite and no such treatment is
    needed; a finite patch of increasing size is the available route to the
    periodic limit today.
    """
    T = np.asarray(T)
    N = T.shape[0]
    return np.linalg.solve(np.eye(N) - T @ np.asarray(Omega), T)

"""Point-group symmetry checks for extracted T-matrices.

A scatterer invariant under a geometric operation g (rotation about the
z-axis, or a mirror plane containing z) forces its T-matrix to commute with
the representation D(g) of that operation on the VSWF mode basis:

    T @ D(g) == D(g) @ T

(Waterman, "Symmetry, unitarity, and geometry in electromagnetic
scattering," Phys. Rev. D 3, 825 (1971) -- the same paper already cited in
docs/manual.tex Sec. "Diagnostics" for the passivity/unitarity results).
Proof sketch: if D(g) a is the incident-coefficient vector of the
g-transformed illumination, g-invariance of the scatterer means its
scattered response is the g-transform of the original response, i.e.
D(g)(T a) = T(D(g) a) for every a, hence T D(g) = D(g) T.

Two representations are provided:

- rotation_z_representation(): proper rotations about z act diagonally on
  the mode basis (m -> m, phase e^{i m alpha}) -- closed form, no ambiguity.
- mirror_representation(): mirror planes containing z are IMPROPER
  operations that mix m -> -m and couple the electric/magnetic (N/M)
  blocks.  Rather than trust a hand-derived sign convention for this --
  this package is explicit elsewhere (the reciprocity sign in
  tmatrix_solve.py, the Mie-conjugation bug in the manual's Validation
  section) that this class of sign error is invisible to internal
  consistency tests -- D(sigma) is built NUMERICALLY from the package's own
  validated field-evaluation/projection primitives (vswf.evaluate_EH /
  vswf.project_surface_field): a regular VSWF basis mode is evaluated at
  the geometrically mirrored points, the field VECTORS are transformed by
  the standard true-vector/axial-vector mirror rule for (E, H), and the
  result is re-projected onto the basis.  D(sigma)**2 == I is checked
  internally (mirroring twice must return the original field) as a
  consistency test of the construction itself, independent of any
  scatterer -- analogous to the involution check on to_treams_convention in
  tests/test_storage_treams.py.
"""

from __future__ import annotations

import numpy as np

from ..vswf import mode_list, n_modes, evaluate_EH, project_surface_field
from ..quadrature import gauss_legendre_sphere


# ----------------------------------------------------------------------------
# Representations of symmetry operations on the VSWF mode basis
# ----------------------------------------------------------------------------

def rotation_z_representation(lmax: int, alpha: float) -> np.ndarray:
    """D(R) for a proper rotation by `alpha` (radians) about the z-axis.

    Diagonal, eigenvalue e^{i m alpha} on mode (pol, n, m) -- independent of
    pol and n, since VSWFs carry their entire azimuthal dependence through
    e^{i m phi}."""
    _, _, ms = mode_list(lmax)
    return np.diag(np.exp(1j * ms * alpha))


def _mirror_matrix(phi0: float) -> np.ndarray:
    """Cartesian reflection matrix for the vertical mirror plane containing
    the z-axis at azimuth `phi0` (elementary 2D reflection about the line
    at angle phi0 in the xy-plane, z unaffected)."""
    c, s = np.cos(2 * phi0), np.sin(2 * phi0)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, 1.0]])


def mirror_representation(lmax: int, phi0: float = 0.0, k: float = 1.3,
                          oversample: float = 1.5,
                          involution_tol: float = 1e-8) -> np.ndarray:
    """D(sigma) for the vertical mirror plane at azimuth `phi0` (radians),
    built numerically -- see module docstring.

    k, oversample : construction parameters only (an arbitrary regular-wave
        argument and a quadrature safety margin); they do not describe any
        physical scatterer and the result is independent of them up to
        numerical noise.
    involution_tol : raises RuntimeError if D(sigma)**2 deviates from the
        identity by more than this (relative Frobenius, per sqrt(N)) --
        a bug in this construction, not a scatterer property.
    """
    N = n_modes(lmax)
    P = _mirror_matrix(phi0)
    theta, phi, w, _ = gauss_legendre_sphere(lmax, oversample=oversample)
    pts = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                    np.cos(theta)], axis=-1)
    pts_mirrored = pts @ P             # P symmetric: P @ p == p @ P per row

    D = np.zeros((N, N), dtype=complex)
    for j in range(N):
        e_j = np.zeros(N, dtype=complex)
        e_j[j] = 1.0
        E, H = evaluate_EH(e_j, lmax, "regular", k, pts_mirrored)
        E_pullback = E @ P             # true vector:  P . E(P r)
        H_pullback = -(H @ P)          # axial vector: -P . H(P r)
        D[:, j] = project_surface_field(E_pullback, lmax, "regular", k, 1.0,
                                        theta, phi, w, H=H_pullback)

    resid = np.linalg.norm(D @ D - np.eye(N)) / np.sqrt(N)
    if resid > involution_tol:
        raise RuntimeError(
            f"mirror_representation self-check failed: ||D^2 - I||/sqrt(N) "
            f"= {resid:.2e} (mirroring twice must return the original "
            f"field). This is a bug in the numerical construction, not a "
            f"property of any scatterer.")
    return D


# ----------------------------------------------------------------------------
# Selection-rule / conformance checks
# ----------------------------------------------------------------------------

def check_symmetry(T: np.ndarray, D: np.ndarray) -> float:
    """Relative Frobenius violation of T @ D == D @ T.

    A scatterer invariant under the operation represented by D must satisfy
    this to within extraction noise (residual/reciprocity level, per
    docs/manual.tex Sec. "Diagnostics"); an O(1) value means the scatterer
    does not have this symmetry, or flags an extraction/alignment problem
    if it should.
    """
    comm = T @ D - D @ T
    return float(np.linalg.norm(comm) / max(np.linalg.norm(T), 1e-300))



# ----------------------------------------------------------------------------
# Standalone CAD/geometry point-group detector
# ----------------------------------------------------------------------------
#
# Independent of the T-matrix machinery above: this operates purely on a
# triangulated surface (an STL export of the solid) and answers "what point
# group does this geometry actually have?" numerically, rather than by
# reading off the build script's parametrization.  It is the natural source
# of the `n_fold` / `mirror_phi0_deg` arguments to point_group_report()
# above for scatterers not authored by a script in this repo (a template
# project, a colleague's CAD file, ...) -- for a scripted `build=` callable
# whose parametric symmetry is already obvious (e.g. "4 identical spokes at
# 90 degree intervals"), this detector is confirmation, not a prerequisite.
#
# Scope: axial point groups (Cn / Cnv) about a GIVEN axis, default z -- the
# layer-normal convention used throughout this package for planar/extruded
# metasurface unit cells (docs/manual.tex Sec. "Simulation setup").  This is
# deliberately narrower than a general 3D point-group search (which would
# also need to locate the axis itself, handle non-axial groups, Sn, sigma_h,
# in-plane C2' axes, ...); axis_is_principal() gives a sanity check that the
# assumed axis is at least consistent with the mass distribution of the
# solid, so a badly-wrong axis choice does not pass silently.

def load_stl_ascii(path) -> np.ndarray:
    """Parse an ASCII STL file into (Ntri, 3, 3) triangle vertex arrays.

    CST's STL object (Model3D.STL, .WriteCAD) writes ASCII STL by default;
    this avoids adding a binary-STL / mesh-library dependency for a format
    this simple to parse directly."""
    verts = []
    tri = []
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            toks = line.split()
            if len(toks) == 4 and toks[0] == "vertex":
                tri.append([float(x) for x in toks[1:4]])
                if len(tri) == 3:
                    verts.append(tri)
                    tri = []
    if not verts:
        raise ValueError(f"No triangles found in {path} -- not an ASCII "
                         f"STL file, or empty geometry?")
    return np.array(verts, dtype=float)


def surface_points(triangles: np.ndarray, dedup_tol: float = 1e-6
                   ) -> np.ndarray:
    """Deduplicated (Nverts, 3) point cloud from triangle vertices -- the
    boundary sample used for symmetry-operation matching."""
    pts = triangles.reshape(-1, 3)
    # snap to a grid at dedup_tol and drop duplicates (STL repeats shared
    # vertices once per adjacent triangle)
    key = np.round(pts / dedup_tol).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(first)]


def axis_is_principal(points: np.ndarray, axis=(0.0, 0.0, 1.0),
                      rel_tol: float = 1e-3) -> bool:
    """Sanity check for the assumed rotation axis: True iff `axis` is (near)
    an eigenvector of the point cloud's second-moment tensor about its
    centroid.  An extruded/planar solid has its thickness (layer-normal)
    direction as a sharply distinct principal axis regardless of the
    in-plane shape, so this holds for the intended use case even when the
    in-plane shape has no rotational symmetry at all; it fails only when
    `axis` genuinely is not a symmetry-relevant direction of the solid."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    c = points - points.mean(axis=0)
    M = c.T @ c
    w, v = np.linalg.eigh(M)
    proj = v.T @ axis                  # component of axis along each eigvec
    # axis is "an eigenvector" if it is (anti)parallel to one eigvec, i.e.
    # its component onto the OTHER two is small relative to its own norm
    off = np.sort(np.abs(proj))[:2]    # two smallest-magnitude components
    return bool(np.linalg.norm(off) < rel_tol)


def _rotation_matrix_z(alpha: float) -> np.ndarray:
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _matches_operation(points: np.ndarray, R: np.ndarray, tol: float) -> bool:
    """True iff applying R to every point of the cloud maps it back onto
    itself (bijective nearest-neighbor match within `tol`)."""
    from scipy.spatial import cKDTree
    centroid = points.mean(axis=0)
    c = points - centroid
    transformed = c @ R.T
    tree = cKDTree(c)
    d, idx = tree.query(transformed)
    return bool(np.max(d) < tol and len(set(idx.tolist())) == len(c))


def _auto_tol(points: np.ndarray, factor: float = 0.5) -> float:
    """Tolerance scaled to the mesh's own point spacing (median nearest-
    neighbor distance) -- avoids a hardcoded absolute tolerance that would
    be wrong for a geometry ten times larger or smaller."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    d, _ = tree.query(points, k=2)
    return factor * float(np.median(d[:, 1]))


def detect_axial_point_group(stl_path, axis=(0.0, 0.0, 1.0),
                             n_candidates=(2, 3, 4, 5, 6, 8), tol=None):
    """Detect the axial point group (Cn / Cnv) of the solid in `stl_path`
    about `axis` (default z).

    Returns a dict:
      rotation_order   : highest n in n_candidates for which Cn holds (1 if
                         none do)
      mirror_planes_deg: azimuths (degrees, in [0, 180)) of detected
                         vertical mirror planes, tested at k*180/n for
                         k = 0..n-1 once `rotation_order` = n is known
                         (the only azimuths a Cn-symmetric solid CAN have
                         mirrors at)
      label             : 'C1' / 'Cn' / 'Cnv'
      axis_is_principal : sanity check, see axis_is_principal()
      tol               : tolerance actually used
    """
    triangles = load_stl_ascii(stl_path)
    points = surface_points(triangles)
    if tol is None:
        tol = _auto_tol(points)

    n_best = 1
    for n in sorted(n_candidates):
        if _matches_operation(points, _rotation_matrix_z(2 * np.pi / n), tol):
            n_best = n

    mirrors = []
    for k in range(n_best):
        phi0 = k * np.pi / n_best
        P = _mirror_matrix(phi0)
        if _matches_operation(points, P, tol):
            mirrors.append(float(np.degrees(phi0)))

    label = f"C{n_best}" if n_best > 1 else "C1"
    if mirrors:
        label += "v"
    return {"rotation_order": n_best, "mirror_planes_deg": mirrors,
           "label": label, "axis_is_principal": axis_is_principal(points, axis),
           "tol": tol, "n_points": len(points)}


# ----------------------------------------------------------------------------
# Symmetry-reduced illumination: one solve -> |G| illumination columns
# ----------------------------------------------------------------------------
#
# A g-invariant scatterer satisfies T D(g) = D(g) T, so for any solved
# illumination with measured (a, f = T a),
#
#     T (D(g) a) = D(g) (T a) = D(g) f
#
# i.e. (D(g) a, D(g) f) is a VALID extra illumination column requiring no
# extra solve.  With |G| = 8 for C4v, one CST solve yields up to 8 columns,
# cutting the solve count for a full extraction by up to 8x.
#
# The direction bookkeeping below is NOT needed to make the algebra work (any
# D commuting with T generates a valid column, and the polarization sign
# cancels because it multiplies a and f alike).  It exists so the orbit can be
# checked for DEGENERACY: if g maps a direction onto itself, D(g) a is
# parallel to a and the "extra" column carries no new rank.  Those must be
# detected and skipped, or the reduced illumination set silently under-
# determines T.
#
# Transformation conventions below were established EMPIRICALLY (see the
# module docstring's rationale for not hand-deriving this class of sign):
#   rotation:  a(theta, phi + alpha, pol) = D(-alpha) a(theta, phi, pol)
#              -- equivalently D(beta) a(theta, phi) = a(theta, phi - beta).
#              The naive D(+alpha) is WRONG (O(1) error), D(-alpha) is 3e-16.
#   mirror:    D(sigma_phi0) a(theta, phi, pol)
#                  = s(pol) * a(theta, 2*phi0 - phi, pol),
#              s = +1 for theta-pol, -1 for phi-pol  (verified to 7e-15).

def point_group_operations(lmax: int, n_fold: int = 1,
                           mirror_phi0_deg=None) -> list:
    """Full element list of the axial point group C_n / C_nv on the VSWF
    basis.

    Returns [(label, D, spec), ...] where `spec` is ("rot", beta) or
    ("mirror", phi0_rad), used by orbit_directions() to track which physical
    illumination each generated column corresponds to.

    n_fold : order of the principal rotation axis (1 = no rotational symmetry).
    mirror_phi0_deg : azimuths (degrees) of vertical mirror planes, e.g. the
        `mirror_planes_deg` field returned by detect_axial_point_group().
        For a genuine C_nv group there are n such planes at k*180/n.
    """
    ops = []
    for j in range(max(int(n_fold), 1)):
        beta = 2.0 * np.pi * j / max(int(n_fold), 1)
        label = "E" if j == 0 else f"C{n_fold}^{j}"
        ops.append((label, rotation_z_representation(lmax, beta),
                    ("rot", beta)))
    if mirror_phi0_deg is not None:
        phis = ([mirror_phi0_deg] if np.isscalar(mirror_phi0_deg)
                else list(mirror_phi0_deg))
        for deg in phis:
            phi0 = np.radians(float(deg))
            ops.append((f"sigma({deg:g})", mirror_representation(lmax, phi0),
                        ("mirror", phi0)))
    return ops


def _wrap(phi):
    """Azimuth into [0, 2*pi)."""
    return float(np.mod(phi, 2.0 * np.pi))


def orbit_directions(theta: float, phi: float, pol: str, ops: list):
    """Physical illuminations generated by applying each op to (theta, phi,
    pol).

    Returns [(label, theta, phi_new, pol, sign), ...] in the same order as
    `ops`, where `sign` is the scalar relating D(g) a to the plane-wave
    coefficients of the listed direction (it cancels in T and is reported
    only for completeness).
    """
    out = []
    for label, _D, spec in ops:
        kind, val = spec
        if kind == "rot":
            out.append((label, theta, _wrap(phi - val), pol, 1.0))
        else:
            sign = 1.0 if pol == "theta" else -1.0
            out.append((label, theta, _wrap(2.0 * val - phi), pol, sign))
    return out


def orbit_is_degenerate(theta: float, phi: float, pol: str, ops: list,
                        tol_deg: float = 1.0) -> bool:
    """True if two ops send this illumination to the same physical direction.

    Such an orbit yields fewer independent columns than len(ops); the caller
    should either skip the direction or account for the true multiplicity
    rather than assume a full |G|-fold gain.
    """
    dirs = orbit_directions(theta, phi, pol, ops)
    tol = np.radians(tol_deg)
    seen = []
    for _lab, th, ph, _pol, _s in dirs:
        v = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                      np.cos(th)])
        for u in seen:
            if np.linalg.norm(v - u) < tol:
                return True
        seen.append(v)
    return False


def expand_columns_by_symmetry(A: np.ndarray, F: np.ndarray, ops: list):
    """Grow solved illumination columns into their full symmetry orbits.

    A, F : (N, n_solved) measured incident / scattered coefficient columns.
    Returns (A_exp, F_exp) each (N, n_solved * len(ops)).

    This is exact for a scatterer with the given symmetry: every generated
    column satisfies f = T a with the SAME T as the solved columns.  It adds
    no information the symmetry did not already imply -- that is precisely
    why it is free -- but it supplies the extra linearly independent columns
    the least-squares solve needs to pin down all of T.
    """
    A = np.asarray(A)
    F = np.asarray(F)
    A_blocks = [D @ A for _lab, D, _spec in ops]
    F_blocks = [D @ F for _lab, D, _spec in ops]
    return np.concatenate(A_blocks, axis=1), np.concatenate(F_blocks, axis=1)


def symmetrize_tmatrix(T: np.ndarray, ops: list) -> np.ndarray:
    """Group-average projection of T onto the G-invariant subspace:

        T_sym = (1/|G|) sum_g D(g) T D(g)^-1

    Two uses: (1) build an exactly-symmetric synthetic T for testing;
    (2) denoise a measured T, since extraction noise is generically NOT
    G-invariant and is suppressed by the averaging while the true
    G-invariant signal is untouched.  D is unitary here, so D^-1 = D^H.
    """
    acc = np.zeros_like(np.asarray(T, dtype=complex))
    for _lab, D, _spec in ops:
        acc += D @ T @ D.conj().T
    return acc / max(len(ops), 1)


def project_onto_symmetry(T: np.ndarray, ops: list):
    """Group-average projection of T with the removed fraction as an error
    estimate.  Returns (T_projected, removed_fraction).

    For a scatterer whose symmetry has been verified independently
    (geometrically, and by comparing the commutator violation of an
    existing matrix with its reciprocity deviation), the true T is
    invariant under the group, so the non-invariant part of a measured T
    is extraction error:

        removed_fraction = ||T - P(T)||_F / ||T||_F ,
        P(T) = (1/|G|) sum_g D(g) T D(g)^{-1}.

    P is the orthogonal projection onto the invariant subspace; it sets
    symmetry-forbidden elements to zero AND averages symmetry-related
    allowed elements, which reduces their error as well -- preferable to
    zeroing forbidden elements alone, which discards only part of the
    information the symmetry provides.

    Two qualifications.  First, apply this only to a verified symmetry;
    projecting onto a symmetry the object does not have replaces physics
    with an assumption.  Second, examine the diagnostics of the raw
    matrix first: a systematic error with a symmetric component (for
    example a misplaced monitor) is only partially removed by P, and the
    projected matrix then looks better than the data are.  Store the raw
    matrix; apply the projection in post-processing and record the
    removed fraction alongside.
    """
    T = np.asarray(T, dtype=complex)
    P = symmetrize_tmatrix(T, ops)
    removed = float(np.linalg.norm(T - P)
                    / max(np.linalg.norm(T), 1e-300))
    return P, removed


def reduced_illumination_directions(theta_i, phi_i, ops: list,
                                    n_columns_needed: int,
                                    tol_deg: float = 1.0):
    """Pick the subset of candidate directions actually worth solving.

    Selection is farthest-point greedy: after the first non-degenerate
    candidate, each subsequent pick maximizes the minimum angular
    distance to every direction already covered by the accumulated
    orbits.  The reason: the axial point-group
    operations all PRESERVE the polar angle theta, so orbits only ever add
    azimuthal copies -- an in-order walk of a Fibonacci direction list
    (which sweeps from the pole) selects a polar cap, and the resulting
    orbit-expanded least-squares system is rank-deficient in practice
    (measured: cond 4.2e5, rank 38/48 at lmax=4/C4v; the extracted T passed
    noise-free synthetic tests yet was non-m-diagonal, non-reciprocal and
    non-passive on real ~1e-3-noise CST data, while the plane-wave-observable
    extinction stayed correct).  Farthest-point spreading the SAME number of
    solves dropped the measured cond to 24.

    Directions with a degenerate orbit (on-axis, in a mirror plane) are
    skipped: they cannot deliver the full |G|-fold column gain.  Picking
    stops once the accumulated orbit count reaches `n_columns_needed`
    (counting BOTH polarizations per direction, matching the pipeline's
    illumination convention).

    Returns (indices, n_columns_expected).
    """
    th_arr = np.asarray(theta_i, dtype=float)
    ph_arr = np.asarray(phi_i, dtype=float)
    per_dir = 2 * len(ops)          # two polarizations, |G| ops each
    tol = np.radians(tol_deg)

    def uvec(t, p):
        return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p),
                         np.cos(t)])

    candidates = [i for i in range(len(th_arr))
                  if not orbit_is_degenerate(float(th_arr[i]),
                                             float(ph_arr[i]),
                                             "theta", ops, tol_deg)]
    picked, covered = [], []
    while candidates and len(picked) * per_dir < n_columns_needed:
        if not covered:
            i_best = candidates[0]
        else:
            def min_dist(i):
                v = uvec(float(th_arr[i]), float(ph_arr[i]))
                return min(float(np.linalg.norm(v - u)) for u in covered)
            i_best = max(candidates, key=min_dist)
            if min_dist(i_best) < tol:
                break                    # every candidate already covered
        picked.append(i_best)
        candidates.remove(i_best)
        for _lab, th2, ph2, _p, _s in orbit_directions(float(th_arr[i_best]),
                                                        float(ph_arr[i_best]),
                                                        "theta", ops):
            covered.append(uvec(th2, ph2))
    return np.array(sorted(picked), dtype=int), len(picked) * per_dir


def point_group_report(T: np.ndarray, lmax: int, n_fold: int | None = None,
                       mirror_phi0_deg=None) -> dict:
    """Convenience diagnostic: violation ratio for a rotational order and/or
    one or more mirror planes.

    n_fold : if given, checks the C_n rotation by 2*pi/n_fold about z.
    mirror_phi0_deg : if given, a mirror-plane azimuth (degrees) or list of
        azimuths; each is checked with mirror_representation.
    Returns {label: relative_violation}.
    """
    report = {}
    if n_fold:
        D = rotation_z_representation(lmax, 2 * np.pi / n_fold)
        report[f"C{n_fold}"] = check_symmetry(T, D)
    if mirror_phi0_deg is not None:
        phis = ([mirror_phi0_deg] if np.isscalar(mirror_phi0_deg)
               else list(mirror_phi0_deg))
        for deg in phis:
            D = mirror_representation(lmax, np.radians(deg))
            report[f"sigma_v({deg:g} deg)"] = check_symmetry(T, D)
    return report

"""Spherical grids and quadrature rules.

Two distinct roles:
1. Projection quadrature (Gauss-Legendre in theta x uniform in phi): exact for
   spherical harmonics up to a known degree — used to project fields on a
   sphere (near field or farfield) onto vector spherical harmonics.
2. Illumination direction sets (Fibonacci sphere): well-spread incidence
   directions for the multiple-plane-wave excitation of the scatterer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


def gauss_legendre_sphere(lmax: int, oversample: float = 1.0):
    """Product quadrature on the unit sphere, exact for spherical-harmonic
    integrands up to degree 2*lmax (sufficient to project fields containing
    degrees <= lmax against conjugated harmonics of degree <= lmax).

    Returns
    -------
    theta : (Nt*Np,) polar angles in [0, pi]
    phi   : (Nt*Np,) azimuthal angles in [0, 2pi)
    w     : (Nt*Np,) weights such that sum(w * f) approximates
            ∫_0^2pi ∫_0^pi f(theta, phi) sin(theta) dtheta dphi
    shape : (Nt, Np) underlying grid shape (theta-major ordering)
    """
    n_theta = int(np.ceil((2 * lmax + 1) / 2 * oversample)) + 1
    n_phi = int(np.ceil((2 * lmax + 1) * oversample)) + 1

    x, wx = leggauss(n_theta)          # nodes on (-1, 1) = cos(theta)
    theta_1d = np.arccos(x[::-1])      # increasing theta
    wtheta = wx[::-1]

    phi_1d = np.arange(n_phi) * 2 * np.pi / n_phi
    wphi = np.full(n_phi, 2 * np.pi / n_phi)

    theta = np.repeat(theta_1d, n_phi)
    phi = np.tile(phi_1d, n_theta)
    w = np.repeat(wtheta, n_phi) * np.tile(wphi, n_theta)
    return theta, phi, w, (n_theta, n_phi)


def quadrature_for_monitor(lmax: int, k: float, radius: float,
                           margin: int = 6, oversample: float = 1.0):
    """Sphere quadrature sized for projecting fields sampled on a monitor
    sphere of given radius.

    The scattered field is band-limited to lmax, but the INCIDENT plane wave
    on the sphere has angular content up to ~ Wiscombe(k*radius).  Exact
    projection of a degree-n_field field against degree-lmax conjugated
    harmonics needs quadrature exactness to degree n_field + lmax, i.e.
    gauss_legendre_sphere(ceil((n_field + lmax)/2)).
    """
    x = k * radius
    n_field = int(np.ceil(x + 4.05 * x ** (1 / 3) + 2)) + margin
    lmax_q = int(np.ceil((n_field + lmax) / 2))
    return gauss_legendre_sphere(max(lmax_q, lmax), oversample=oversample)


def _lebedev_builtin():
    """The three classical Lebedev rules with exactly known rational
    weights (precision 3, 5, 7).  Points are generated from the octahedral
    orbits; weights are normalized to sum to 1 (multiply by 4 pi for the
    solid-angle integral).  Higher-order rules can be supplied as data
    files (see lebedev_directions)."""
    s = 1.0 / np.sqrt(2.0)
    c = 1.0 / np.sqrt(3.0)
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                         [0, 0, 1], [0, 0, -1]], dtype=float)
    edges = np.array([[sx * s, sy * s, 0] for sx in (1, -1) for sy in (1, -1)]
                     + [[sx * s, 0, sz * s] for sx in (1, -1) for sz in (1, -1)]
                     + [[0, sy * s, sz * s] for sy in (1, -1) for sz in (1, -1)])
    corners = np.array([[sx * c, sy * c, sz * c] for sx in (1, -1)
                        for sy in (1, -1) for sz in (1, -1)])
    rules = {
        6: (vertices, np.full(6, 1.0 / 6.0)),
        14: (np.vstack([vertices, corners]),
             np.concatenate([np.full(6, 1.0 / 15.0),
                             np.full(8, 3.0 / 40.0)])),
        26: (np.vstack([vertices, edges, corners]),
             np.concatenate([np.full(6, 1.0 / 21.0),
                             np.full(12, 4.0 / 105.0),
                             np.full(8, 27.0 / 840.0)])),
    }
    return rules


LEBEDEV_PRECISION = {6: 3, 14: 5, 26: 7}


def _xyz_to_angles(xyz):
    theta = np.arccos(np.clip(xyz[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2 * np.pi)
    return theta, phi


def verify_spherical_rule(theta, phi, w, precision, tol=1e-10):
    """Check that a spherical rule with weights summing to 1 integrates all
    spherical harmonics Y_lm, 1 <= l <= precision, to zero (equivalent to
    exactness for polynomials of the given degree, since Y_00 is handled by
    the weight sum).  Returns the maximum error."""
    from scipy.special import sph_harm
    if abs(np.sum(w) - 1.0) > tol:
        return np.inf
    err = 0.0
    for l in range(1, precision + 1):
        for m in range(-l, l + 1):
            val = np.sum(w * sph_harm(m, l, phi, theta))
            err = max(err, abs(val))
    return err


def lebedev_directions(n_min: int, data_dir=None):
    """Smallest available Lebedev rule with at least n_min points.

    Built-in rules: 6, 14, and 26 points (precision 3, 5, 7).  Larger
    rules are read from text files named ``lebedev_*.txt`` in ``data_dir``
    (default: a ``lebedev_rules`` folder next to this module), in the
    commonly distributed three-column format: azimuth and colatitude in
    degrees, weight (normalized to 1).  Every rule — built-in or loaded —
    is verified by integrating spherical harmonics before use.

    Returns (theta, phi, weights, n_points).
    """
    available = {}
    for npts, (xyz, w) in _lebedev_builtin().items():
        th, ph = _xyz_to_angles(xyz)
        available[npts] = (th, ph, w, LEBEDEV_PRECISION[npts])
    data_dir = Path(data_dir) if data_dir else \
        Path(__file__).resolve().parent / "lebedev_rules"
    if data_dir.is_dir():
        for f in sorted(data_dir.glob("lebedev_*.txt")):
            try:
                dat = np.loadtxt(f)
                ph = np.radians(dat[:, 0] % 360.0)
                th = np.radians(dat[:, 1])
                w = dat[:, 2] / np.sum(dat[:, 2])
                prec = int(f.stem.split("_")[1])
                available[len(w)] = (th, ph, w, prec)
            except Exception as e:                        # noqa: BLE001
                raise ValueError(f"Could not parse Lebedev file {f}: {e}")
    counts = sorted(n for n in available if n >= n_min)
    if not counts:
        raise ValueError(
            f"No Lebedev rule with >= {n_min} points is available "
            f"(built-in: 6, 14, 26). Place standard rule files "
            f"lebedev_<precision>.txt in {data_dir} for larger sets, or "
            f"use the 'fibonacci' scheme, which supports any count.")
    th, ph, w, prec = available[counts[0]]
    err = verify_spherical_rule(th, ph, w, prec)
    if err > 1e-8:
        raise ValueError(
            f"Lebedev rule with {counts[0]} points failed the exactness "
            f"check (error {err:.2e} at precision {prec}); the rule data "
            f"are corrupted.")
    return th, ph, w, counts[0]


def illumination_directions(scheme: str, n_dirs: int):
    """Incidence directions for the T-matrix illumination set.

    scheme = "fibonacci": exactly n_dirs quasi-uniform, pole-free points.
    scheme = "lebedev":   the smallest verified Lebedev rule with at least
    n_dirs points (the returned count may be larger).
    """
    if scheme == "fibonacci":
        th, ph = fibonacci_directions(n_dirs)
        return th, ph
    if scheme == "lebedev":
        th, ph, _, _ = lebedev_directions(n_dirs)
        return th, ph
    raise ValueError(f"Unknown illumination scheme '{scheme}' "
                     f"(use 'fibonacci' or 'lebedev')")


def fibonacci_directions(n: int, hemisphere: bool = False):
    """n quasi-uniform unit vectors on the sphere (golden-spiral lattice).

    Returns (theta, phi) arrays of shape (n,). Directions avoid the exact
    poles (useful: theta=0 is a coordinate singularity for polarization
    definitions in CST and for VSH evaluation).
    """
    i = np.arange(n) + 0.5
    cos_theta = 1 - 2 * i / n
    if hemisphere:
        cos_theta = 1 - i / n
    golden = (1 + 5**0.5) / 2
    phi = (2 * np.pi * i / golden) % (2 * np.pi)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return theta, phi


def unit_vectors(theta, phi):
    """Cartesian unit vectors (N,3) from spherical angles."""
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    return np.stack([st * cp, st * sp, ct], axis=-1)


def spherical_basis(theta, phi):
    """Orthonormal spherical basis vectors at (theta, phi).

    Returns (r_hat, theta_hat, phi_hat), each (N,3) Cartesian.
    """
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    r_hat = np.stack([st * cp, st * sp, ct], axis=-1)
    theta_hat = np.stack([ct * cp, ct * sp, -st], axis=-1)
    phi_hat = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1)
    return r_hat, theta_hat, phi_hat

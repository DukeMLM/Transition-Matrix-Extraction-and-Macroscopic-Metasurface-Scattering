"""Bloch lattice coupling C for a general 2-D Bravais lattice.

The validated pipeline computes C(k, k_par) = sum_{R != 0} A(R) e^{+i k_par.R}
for a SQUARE lattice by Gaussian-tapered shell sums with Richardson
extrapolation in 1/Rc^2 (retrieval/bloch_lattice.py, verified against a
brute-force per-site sum to 1.3e-15).  The assembly step there is already
lattice-agnostic -- it consumes only radii, per-site azimuths and the
in-plane rotation identity -- so extending it to a rectangular or oblique
lattice needs nothing but a different shell enumeration
(Lattice2D.shells).  This module is that thin extension, and it deliberately
reuses `bloch_lattice.assemble_shell_sum_bloch` rather than reimplementing
the arithmetic.

Why this is NOT yet the M2 deliverable
--------------------------------------
The taper is a Gaussian of length Rc, and the Richardson extrapolation in
1/Rc^2 assumes many sites inside Rc.  With the campaign's 2 um pitch and
kRc = 10..20 that is hundreds of sites.  With a diffractive cell of ~30 um
pitch, kRc = 10 puts Rc BELOW one pitch: the taper annihilates the sum before
it has summed anything, and the extrapolation is meaningless.  Convergence
also degrades near every Rayleigh threshold, and a diffractive cell has many.

`converged_C` therefore refuses rather than guesses: it requires a declared
number of sites inside the smallest taper and a stability check across taper
sets, and reports why it declined.  A large diffractive cell will fail that
check, which is exactly the proposal's argument for an Ewald implementation
at M2 -- at which point this module becomes the independent second
implementation that Gate D requires.

Bloch sign is the repository's normative POSITIVE convention, inherited from
bloch_lattice: C = sum A(R) e^{+i k_par . R}.
"""
import numpy as np

from tmatrix.aggregation.translate import translation_shells, make_quad
from tmatrix.aggregation.vswf import RegularProjector
from tmatrix.retrieval.bloch_lattice import assemble_shell_sum_bloch

from .lattice import Lattice2D


def lattice_sum_C(lattice, k, modes, r0, quad, k_par,
                  kRc=(10.0, 14.0, 20.0), r_max_factor=3.5,
                  projector=None, shells_cache=None):
    """C(k, k_par) for `lattice`, same conventions as
    bloch_lattice.lattice_sum_C_bloch (which it reproduces exactly for a
    square lattice -- gated in test_fastfull_core.py).

    shells_cache : optional dict keyed by (k, lattice id, r_max); the
        translation-operator build is the expensive part and depends on
        neither k_par nor the taper set.
    """
    k_par = np.asarray(k_par, dtype=float).ravel()
    if k_par.size == 3:
        if abs(k_par[2]) > 1e-12 * max(1.0, abs(k)):
            raise ValueError("k_par must be in-plane")
        k_par = k_par[:2]
    if k_par.size != 2:
        raise ValueError("k_par must have 2 in-plane components")
    kRc = np.asarray(kRc, dtype=float)
    Rcs = kRc / k
    r_max = r_max_factor * Rcs.max()

    key = (round(float(k), 12), id(lattice), round(float(r_max), 9))
    if shells_cache is not None and key in shells_cache:
        radii, angles, A_s = shells_cache[key]
    else:
        radii, angles = lattice.shells(r_max)
        if projector is None:
            projector = RegularProjector(modes, quad)
        if len(radii) and r0 >= radii.min():
            raise ValueError(
                "projection sphere r0 = %g exceeds the nearest lattice "
                "distance %g" % (r0, radii.min()))
        A_s = translation_shells(k, radii, modes, r0, quad,
                                 projector=projector)
        if shells_cache is not None:
            shells_cache[key] = (radii, angles, A_s)
    return assemble_shell_sum_bloch(A_s, radii, angles, modes, Rcs, k_par)


def shell_statistics(lattice, k, kRc=(10.0, 14.0, 20.0), r_max_factor=3.5):
    """Cost / feasibility numbers for the tapered sum, without computing it.

    Returns n_sites, n_radii (= number of translation-operator projections),
    r_max, and `sites_in_min_taper`: how many lattice sites lie inside the
    SMALLEST taper length Rc = min(kRc)/k.  That last number is the one that
    decides whether the tapered sum can converge at all.
    """
    Rcs = np.asarray(kRc, dtype=float) / k
    r_max = r_max_factor * Rcs.max()
    radii, angles = lattice.shells(r_max)
    n_sites = int(sum(len(a) for a in angles))
    inside = int(sum(len(a) for r, a in zip(radii, angles) if r <= Rcs.min()))
    return dict(n_sites=n_sites, n_radii=int(len(radii)), r_max=float(r_max),
                Rc_min=float(Rcs.min()), Rc_max=float(Rcs.max()),
                sites_in_min_taper=inside)


def converged_C(lattice, k, modes, r0, quad, k_par,
                kRc_sets=((10.0, 14.0, 20.0), (14.0, 20.0, 28.0)),
                r_max_factor=3.5, min_sites_in_taper=60, rtol=1e-3,
                projector=None):
    """C with an explicit convergence verdict; refuses instead of guessing.

    Two independent gates, both required:
      (a) FEASIBILITY -- at least `min_sites_in_taper` sites inside the
          smallest taper of the first set, else the Gaussian kills the sum
          before it has summed anything and the 1/Rc^2 extrapolation is
          meaningless;
      (b) STABILITY -- the Richardson results of two different taper sets
          agree to `rtol` in relative Frobenius norm.

    Returns (C, info); C is None when a gate fails, and info['reasons'] says
    which.  A large diffractive cell is EXPECTED to fail (a); that is the
    proposal's case for Ewald at M2, not a bug here.
    """
    reasons = []
    stats = shell_statistics(lattice, k, kRc_sets[0], r_max_factor)
    if stats["sites_in_min_taper"] < min_sites_in_taper:
        reasons.append("only %d sites inside the smallest taper Rc = %.3g um "
                       "(need %d); the cell is too large for the tapered "
                       "real-space sum -- use Ewald (M2)"
                       % (stats["sites_in_min_taper"], stats["Rc_min"],
                          min_sites_in_taper))
        return None, dict(stats=stats, reasons=reasons, converged=False)

    if projector is None:
        projector = RegularProjector(modes, quad)
    cache = {}
    Cs = [lattice_sum_C(lattice, k, modes, r0, quad, k_par, kRc=s,
                        r_max_factor=r_max_factor, projector=projector,
                        shells_cache=cache)
          for s in kRc_sets]
    rel = max(np.linalg.norm(Cs[i] - Cs[0]) / np.linalg.norm(Cs[0])
              for i in range(1, len(Cs))) if len(Cs) > 1 else 0.0
    info = dict(stats=stats, taper_rel_spread=float(rel), reasons=reasons)
    if rel > rtol:
        reasons.append("taper sets disagree by %.3e > %.1e" % (rel, rtol))
        info["converged"] = False
        return None, info
    info["converged"] = True
    return Cs[0], info


def default_quad(n_theta=16, n_phi=32):
    return make_quad(n_theta, n_phi)

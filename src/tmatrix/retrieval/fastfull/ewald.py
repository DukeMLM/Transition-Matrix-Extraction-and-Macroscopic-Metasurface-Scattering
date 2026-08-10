"""Ewald lattice coupling C for a general 2-D Bravais lattice (milestone M2).

M1 identified this as the blocker: the repository's tapered real-space Bloch
sum (`tmatrix.retrieval.bloch_lattice`, generalized in `coupling.py`) needs
many lattice sites inside a Gaussian taper of length Rc, and a diffractive
coding cell has a pitch LARGER than the taper.  `coupling.converged_C`
correctly refuses there.  This module supplies the Ewald-summed alternative
through `treams`,
which is also the independent second implementation the proposal's Gate D
asks for ("preferably the repository implementation and an Ewald/treams
calculation").

What the bridge turned out to be
--------------------------------
`treams.sw.translate_periodic(k, kpar, a, [[0,0,0]], (l, m, pol),
poltype="parity")` returns exactly the repository's

    C(k, k_par) = sum_{R != 0} A(R) e^{+i k_par . R}

with NO transpose, conjugation, polarization-index flip or Bloch sign
change.  Measured against the validated tapered square-lattice sum
(pitch 2 um, lambda 12 um):

    theta = 0            relative max deviation 1.7e-07
    theta = 30, phi = 22.5                      1.2e-06

Those residuals are the TAPERED method's Richardson extrapolation error, not
Ewald's: the Ewald sum is exact up to its eta split, which is verified to
1e-13 below.  The polarization convention (treams parity pol = 1 is the
tmat.h5 electric mode, the repository uses ELECTRIC = 0) provably does not
matter for C: for an in-plane displacement the translation operator has
A_ee = A_mm and A_em = A_me, so swapping both indices leaves it invariant.
That identity is gated rather than assumed.

The eta split
-------------
`eta` divides the sum between real and reciprocal space.  treams estimates it
when eta = 0, and that estimate is reliable: it agrees with eta in
[0.5, 1.0] to ~1e-13 on every cell tested.  Large eta is NOT safe -- at
eta = 1.8 the oblique 13x19 um cell moves by 1.6e-1, because the real-space
part then needs far more terms than treams retains.  `converged_C` therefore
brackets eta around the automatic value and REFUSES if the bracket disagrees,
rather than trusting a single evaluation.

Speed and regime
----------------
About 13 ms per 30x30 C, versus minutes for the tapered sum on the cells
where that one still works.  The coupling is also far weaker on a coding
cell than on the campaign's 2 um cell: |C|max is 1.88 for the M1 winner
(78 um^2) against 25817 at pitch 2 um, four orders of magnitude.  That is
what makes the M1 screening Jacobian close to the real one -- see
`lattice_dressing_strength`.
"""
import numpy as np

from treams import sw

from tmatrix import treams_compat

treams_compat.apply()


# ------------------------------------------------------------------ the sum

def _modes_tuple(modes, pol_flip=False):
    pol = (1 - modes.pol) if pol_flip else modes.pol
    return (np.asarray(modes.l, np.int32), np.asarray(modes.m, np.int32),
            np.asarray(pol, np.int32))


def lattice_sum_C(lattice, k, modes, k_par, eta=0, pol_flip=False):
    """Ewald C(k, k_par) in the repository's convention and mode order.

    Parameters
    ----------
    lattice : tmatrix.retrieval.fastfull.lattice.Lattice2D  (or anything with
        a (2, 2) `.A`)
    k : float          embedding wavenumber, rad/um
    modes : vswf.ModeBasis
    k_par : (2,) in-plane Bloch vector, rad/um.  Sign convention is the
        repository's POSITIVE one, e^{+i k_par . R}; verified against
        bloch_lattice.lattice_sum_C_bloch, not assumed.
    eta : Ewald split; 0 asks treams for its automatic estimate (recommended
        -- see the module docstring; large eta is not safe).
    pol_flip : swap the polarization index before the call.  Provided only so
        that the gate can demonstrate C is invariant under it.

    Returns (n, n) complex.
    """
    A = np.asarray(getattr(lattice, "A", lattice), dtype=float)
    if A.shape != (2, 2):
        raise ValueError("expected a 2-D lattice with (2, 2) primitive rows")
    k_par = np.asarray(k_par, dtype=float).ravel()
    if k_par.size == 3:
        if abs(k_par[2]) > 1e-12 * max(1.0, abs(k)):
            raise ValueError("k_par must be in-plane")
        k_par = k_par[:2]
    if k_par.size != 2:
        raise ValueError("k_par must have 2 in-plane components")
    M = sw.translate_periodic(k, k_par, A, [[0.0, 0.0, 0.0]],
                              _modes_tuple(modes, pol_flip),
                              poltype="parity", eta=eta)
    return np.asarray(M).reshape(modes.n, modes.n)


def converged_C(lattice, k, modes, k_par, eta_bracket=(0.5, 0.7, 1.0),
                rtol=1e-8, return_info=False):
    """C with an explicit eta-stability verdict; refuses instead of guessing.

    The automatic eta is evaluated first, then the bracket.  All must agree
    to `rtol` in relative max-norm, else C is None and `info['reasons']` says
    so.  The bracket deliberately stays at or below eta = 1: treams' real-
    space part loses accuracy above that (1.6e-1 at eta = 1.8 on an oblique
    cell), so a wider bracket would fail for a reason that has nothing to do
    with the physics.
    """
    C0 = lattice_sum_C(lattice, k, modes, k_par, eta=0)
    scale = float(np.abs(C0).max())
    devs = {}
    reasons = []
    if not np.isfinite(C0).all():
        reasons.append("automatic eta produced non-finite entries")
    for e in eta_bracket:
        Ce = lattice_sum_C(lattice, k, modes, k_par, eta=e)
        d = float(np.abs(Ce - C0).max() / scale) if scale > 0 else np.inf
        devs[float(e)] = d
        if not np.isfinite(Ce).all():
            reasons.append("eta = %g produced non-finite entries" % e)
        elif d > rtol:
            reasons.append("eta = %g deviates by %.3e > %.1e" % (e, d, rtol))
    info = dict(eta_deviations=devs, abs_max=scale, reasons=reasons,
                converged=not reasons)
    if reasons:
        return (None, info) if return_info else None
    return (C0, info) if return_info else C0


# ------------------------------------------------------------- diagnostics

def lattice_dressing_strength(C, T0):
    """How far the lattice pushes the problem away from the Born limit.

    Returns spectral norms of C T0 and T0 C and the resulting bound on the
    Neumann series (I - C T0)^{-1}: below 1 the series converges and the
    forward map is a mild perturbation of the linear C = 0 model that M1's
    screening Jacobian assumes; well above 1 the map is strongly nonlinear.

    This is the number that decides how much of M1's C = 0 screening survives
    into M2.
    """
    CT = np.linalg.norm(C @ T0, 2)
    TC = np.linalg.norm(T0 @ C, 2)
    return dict(norm_CT=float(CT), norm_TC=float(TC),
                born_valid=bool(max(CT, TC) < 1.0),
                neumann_bound=(float(1.0 / (1.0 - max(CT, TC)))
                               if max(CT, TC) < 1.0 else np.inf))


def rayleigh_scan(lattice, k, modes, f_bloch, scales, kz_min_frac=0.0):
    """|C| and eta stability as the Bloch vector walks toward a threshold.

    Reports, per scale factor applied to the fractional Bloch vector, the
    Wood margin, |C|max, and the eta spread -- so that a design sitting close
    to a Rayleigh anomaly can be recognised as such rather than silently
    returning a poorly converged C.
    """
    from .lattice import enumerate_orders
    f = np.asarray(f_bloch, dtype=float)
    rows = []
    for s in scales:
        kpar = lattice.bloch(*(f * float(s)))
        o = enumerate_orders(lattice, k, k_bloch=kpar,
                             kz_min_frac=kz_min_frac, wood_margin=0.0)
        C, info = converged_C(lattice, k, modes, kpar, return_info=True)
        rows.append(dict(scale=float(s), wood=o.wood_margin_actual(),
                         n_prop=o.n_prop, abs_max=info["abs_max"],
                         converged=info["converged"],
                         eta_spread=max(info["eta_deviations"].values())))
    return rows

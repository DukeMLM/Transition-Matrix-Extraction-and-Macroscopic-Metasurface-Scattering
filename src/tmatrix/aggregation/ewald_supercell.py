"""Ewald pair-resolved Bloch sums W_st for a heterogeneous supercell.

`supercell.block_lattice_sums` computes the manual's Eq. (48) with the
repository's own Gaussian-taper + Richardson machinery.  That method is
validated for the *one-atom* lattice, and its sub-lattice decomposition is
exact in the sense that

    sum_t W_st  ==  C_p     (to 1e-15, checked in
                             tests/aggregation/test_supercell.py)

but the *individual* blocks are shifted sub-lattice sums whose period is M^(1/2)
times coarser, so a taper of length Rc contains proportionally fewer sites and
the 1/Rc^2 series has not settled: at the default kRc = (10, 14, 20) the blocks
are only ~4e-2 converged even though their sum is exact.  Manual section 6.5.3
anticipates exactly this ("for shifted sublattices these values are only
starting choices ... a validated Ewald or reciprocal-space method is an
alternative").

This module supplies that alternative.  `treams.sw.translate_periodic` accepts a
list of shift vectors `rs` and returns the full block matrix in one call; the
repository has already gated its convention against the tapered sum for the
one-atom case (`tmatrix.retrieval.fastfull.ewald`): no transpose, no conjugation, no
Bloch-sign change, and the parity polarization index may be flipped or not
without changing C for in-plane displacements.

Block layout.  treams returns an (M n) x (M n) matrix ordered by (position,
mode); it is reshaped to (M, M, n, n) with `W[s, t]` = target s, source t --
the same indexing `supercell.solve_supercell` expects.  The direction is fixed
by test, not by assumption: `selfsum_residual` below checks `sum_t W_st == C_p`
(which a transposed block layout fails), `tests/aggregation/test_supercell.py`
checks the M = 1 reduction against `translate.lattice_sum_C` and the
four-identical-atom equivalence, and `tests/aggregation/test_supercell.py
--taper` compares block by block against the repository's own tapered sum.
"""
import numpy as np

import treams  # noqa: F401  (imported for its side effect of loading the gufuncs)
from treams import sw

from tmatrix import treams_compat

treams_compat.apply()


def block_lattice_sums_ewald(k, a1, a2, rho, modes, kpar=(0.0, 0.0), eta=0):
    """W[s, t] of manual Eq. (48) by Ewald summation.  Shape (M, M, n, n)."""
    rho = np.atleast_2d(np.asarray(rho, float))
    if rho.shape[1] == 2:
        rho = np.column_stack([rho, np.zeros(len(rho))])
    M, n = len(rho), modes.n
    kpar = np.asarray(kpar, float).ravel()[:2]
    A = np.array([np.asarray(a1, float)[:2], np.asarray(a2, float)[:2]])
    pidx = np.repeat(np.arange(M, dtype=np.int32), n)
    l = np.tile(np.asarray(modes.l, np.int32), M)
    m = np.tile(np.asarray(modes.m, np.int32), M)
    # repository ELECTRIC = 0 / MAGNETIC = 1; treams parity pol = 1 is electric
    pol = np.tile(np.asarray(1 - modes.pol, np.int32), M)
    W = np.asarray(sw.translate_periodic(k, kpar, A, rho, (pidx, l, m, pol),
                                         poltype="parity", eta=eta))
    return W.reshape(M, n, M, n).transpose(0, 2, 1, 3)


def converged_W(k, a1, a2, rho, modes, kpar=(0.0, 0.0),
                eta_bracket=(0.5, 0.7, 1.0), rtol=1e-8, return_info=False):
    """W with an explicit eta-stability verdict; refuses rather than guessing.

    Same policy as `tmatrix.retrieval.fastfull.ewald.converged_C`: the
    automatic split is evaluated first and must agree with a bracket at or
    below eta = 1, where treams' real-space part is reliable.
    """
    W0 = block_lattice_sums_ewald(k, a1, a2, rho, modes, kpar, eta=0)
    scale = float(np.abs(W0).max())
    devs, reasons = {}, []
    if not np.isfinite(W0).all():
        reasons.append("automatic eta produced non-finite entries")
    for e in eta_bracket:
        We = block_lattice_sums_ewald(k, a1, a2, rho, modes, kpar, eta=e)
        d = float(np.abs(We - W0).max() / scale) if scale > 0 else np.inf
        devs[float(e)] = d
        if not np.isfinite(We).all():
            reasons.append(f"eta = {e:g} produced non-finite entries")
        elif d > rtol:
            reasons.append(f"eta = {e:g} deviates by {d:.3e} > {rtol:.1e}")
    info = dict(eta_deviations=devs, abs_max=scale, reasons=reasons,
                converged=not reasons)
    if reasons:
        return (None, info) if return_info else None
    return (W0, info) if return_info else W0


def selfsum_residual(k, a1, a2, rho, modes, primitive_a1, primitive_a2,
                     kpar=(0.0, 0.0)):
    """||sum_t W_st - C_primitive|| / ||C_primitive||, the sub-lattice identity.

    Exact whenever the M basis positions tile the primitive lattice, and the
    single strongest end-to-end check on the block bookkeeping: it fails if the
    block layout, the self-term exclusion or the offsets are wrong.
    """
    W = block_lattice_sums_ewald(k, a1, a2, rho, modes, kpar)
    C = block_lattice_sums_ewald(k, primitive_a1, primitive_a2, [(0.0, 0.0)],
                                 modes, kpar)[0, 0]
    return float(np.abs(W.sum(axis=1) - C[None]).max() / np.abs(C).max())

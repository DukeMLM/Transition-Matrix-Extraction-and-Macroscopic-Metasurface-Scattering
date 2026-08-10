"""Small array reductions shared by the gates, tests and report writers.

Each of these was previously copy-pasted into between two and six modules,
occasionally with a different zero-guard.  The versions here take the most
defensive variant of each, so importing them can only make a caller more
robust, never less.
"""
import numpy as np

_TINY = 1e-300          # smallest denominator we will divide by


def maxabs(x):
    """max |x| as a plain float; accepts anything array-like."""
    return float(np.abs(np.asarray(x)).max())


def rel_err(a, b):
    """|a - b| / |b|, 2-norm, guarded against a zero reference."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), _TINY))


def rel_frob(A, B):
    """|A - B|_F / |B|_F.  Same quantity as rel_err; kept under the name the
    matrix-valued gates use so their output tables stay readable."""
    return rel_err(A, B)


def masked_rel_frob(A, B, floor_frac=1e-6):
    """Relative Frobenius difference over the entries that carry the norm.

    Restricted to |A| > floor_frac * max|A|, which keeps a sea of
    near-cancelling small entries from dominating the ratio.  Returns
    (value, n_entries_kept).
    """
    A = np.asarray(A)
    B = np.asarray(B)
    mask = np.abs(A) > floor_frac * np.abs(A).max()
    if not mask.any():
        return 0.0, 0
    return (float(np.linalg.norm((A - B)[mask]) /
                  max(np.linalg.norm(A[mask]), _TINY)), int(mask.sum()))


def richardson(values, Rcs):
    """Lagrange extrapolation of a Gaussian-tapered lattice sum to 1/Rc^2 -> 0.

    `values` are the tapered sums at cutoff radii `Rcs`; the extrapolation is
    in x = 1/Rc^2, matching translate.assemble_shell_sum and
    bloch_lattice.assemble_shell_sum_bloch exactly.
    """
    x = 1.0 / np.asarray(Rcs, dtype=float) ** 2
    out = np.zeros_like(values[0])
    for i in range(len(Rcs)):
        L = 1.0
        for j in range(len(Rcs)):
            if j != i:
                L *= x[j] / (x[j] - x[i])
        out = out + L * values[i]
    return out


def nearest_idx(data, lam_target):
    """Index of the stored frequency closest to `lam_target` um.

    `data` is anything exposing `wavelength_um` (TMatrixData does), or a bare
    wavelength array.
    """
    lam = getattr(data, "wavelength_um", data)
    return int(np.argmin(np.abs(np.asarray(lam) - lam_target)))

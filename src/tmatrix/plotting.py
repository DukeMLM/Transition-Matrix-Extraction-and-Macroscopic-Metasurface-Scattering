"""Matplotlib setup and the two figure helpers every plot script needs.

Importing this module selects the Agg backend, which is what all of these
scripts want -- they run headless and write PNGs.  Do that here, once, instead
of the three-line `matplotlib.use("Agg")` dance at the top of each script.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402  (must follow use())

from .units import C_UM_THZ, f_thz, lam_um     # noqa: E402

__all__ = ["plt", "thz_axis", "parabola_min", "dips"]


def thz_axis(ax, label="Frequency (THz)"):
    """Add a top axis in THz to a plot whose x axis is wavelength in um."""
    top = ax.secondary_xaxis("top", functions=(f_thz, lam_um))
    top.set_xlabel(label)
    return top


def parabola_min(lam, y):
    """Sub-sample wavelength of a minimum of y(lam), fitted in frequency.

    These sweeps are sampled uniformly in frequency, and a resonance line is
    symmetric in f, not in lambda -- fitting the parabola in lambda biases the
    peak toward the long-wavelength side by several percent.
    """
    lam = np.asarray(lam, dtype=float)
    y = np.asarray(y, dtype=float)
    i = int(np.argmin(y))
    if i in (0, len(y) - 1):
        return float(lam[i])
    f0, f1, f2 = C_UM_THZ / lam[i - 1:i + 2]
    y0, y1, y2 = y[i - 1:i + 2]
    a, b = f1 - f0, f1 - f2
    den = a * (y1 - y2) - b * (y1 - y0)
    if abs(den) < 1e-300:
        return float(lam[i])
    return float(C_UM_THZ / (f1 - 0.5 * (a * a * (y1 - y2)
                                         - b * b * (y1 - y0)) / den))


def dips(lam, s21, n=3, sep=1.5):
    """Up to n local minima of |S21|, deepest first, at least `sep` um apart.

    Returns [(lam_um, depth), ...] sorted by wavelength, each wavelength
    refined by parabola_min.
    """
    lam = np.asarray(lam, dtype=float)
    y = np.abs(np.asarray(s21))
    loc = [i for i in range(1, len(y) - 1)
           if y[i] <= y[i - 1] and y[i] <= y[i + 1]]
    loc.sort(key=lambda i: y[i])
    out = []
    for i in loc:
        if all(abs(lam[i] - l) > sep for l, _ in out):
            out.append((parabola_min(lam[i - 1:i + 2], y[i - 1:i + 2]),
                        float(y[i])))
        if len(out) == n:
            break
    return sorted(out)

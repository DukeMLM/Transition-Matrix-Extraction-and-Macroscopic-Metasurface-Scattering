"""The speed of light, in the two unit conventions this repository uses.

Both were previously spelled out by hand in ten different modules, in two
conventions that differ by a factor of 1e6 and are easy to mistake for each
other.  Import them from here instead.

  C0_M_S     SI, for anything read straight out of a tmat.h5 (frequency in Hz)
  C_UM_THZ   the working convention everywhere else: wavelength in um and
             frequency in THz, so that f = C_UM_THZ / lam
"""
import numpy as np

C0_M_S = 299792458.0        # m/s
C_UM_THZ = 299.792458       # um * THz  (== C0_M_S * 1e-6)

_EPS = 1e-9                 # guards the reciprocal against a zero entry


def f_thz(lam_um):
    """Frequency in THz from wavelength in um (elementwise, safe at 0)."""
    return C_UM_THZ / np.maximum(np.asarray(lam_um, dtype=float), _EPS)


def lam_um(f_thz_):
    """Wavelength in um from frequency in THz (elementwise, safe at 0)."""
    return C_UM_THZ / np.maximum(np.asarray(f_thz_, dtype=float), _EPS)


def lam_um_from_hz(f_hz):
    """Wavelength in um from frequency in Hz (the tmat.h5 frequency axis)."""
    return C0_M_S / np.asarray(f_hz, dtype=float) * 1e6


def k0_rad_per_um(lam_um_):
    """Vacuum wavenumber in rad/um."""
    return 2 * np.pi / np.asarray(lam_um_, dtype=float)

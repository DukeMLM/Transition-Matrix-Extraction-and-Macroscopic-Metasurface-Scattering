"""Plane-wave excitation control.

Convention mapping: cst_tmatrix.vswf.plane_wave_field defines the wave as
E = e^ exp(-j k k^.r), propagating ALONG k^(theta_i, phi_i), with e^ the
theta^ or phi^ unit vector at the propagation direction.  CST's PlaneWave
takes Normal = propagation direction and EVector = linear polarization
vector; CST also uses the e^{+j omega t} engineering convention, so the
mapping is direct.  Any residual amplitude/phase convention of CST cancels
in the self-calibrating separation (vswf.separate_surface_field).
"""

from __future__ import annotations

import numpy as np

from ..quadrature import spherical_basis
from .session import ProjectHandle


def _f(x) -> str:
    return f'"{x:.12g}"'


def plane_wave_vectors(theta_i: float, phi_i: float, pol: str):
    """(k_hat, e_hat) Cartesian unit vectors for the given incidence."""
    r_hat, th_hat, ph_hat = spherical_basis(np.atleast_1d(theta_i),
                                            np.atleast_1d(phi_i))
    e_hat = th_hat[0] if pol == "theta" else ph_hat[0]
    return r_hat[0], e_hat


def set_plane_wave(h: ProjectHandle, theta_i: float, phi_i: float, pol: str):
    """(Re)define the linear plane wave.  PlaneWave is a single global
    excitation, and the caption is deliberately CONSTANT: per PRACTICES §2,
    re-emitting a block whose caption matches the previous block's caption
    REPLACES it — so the illumination loop keeps exactly one plane-wave
    block in the history instead of accumulating one per solve."""
    k_hat, e_hat = plane_wave_vectors(theta_i, phi_i, pol)
    h.add_history(
        "define plane wave properties", f"""With PlaneWave
  .Reset
  .Normal {_f(k_hat[0])}, {_f(k_hat[1])}, {_f(k_hat[2])}
  .EVector {_f(e_hat[0])}, {_f(e_hat[1])}, {_f(e_hat[2])}
  .Polarization "Linear"
  .Store
End With""")

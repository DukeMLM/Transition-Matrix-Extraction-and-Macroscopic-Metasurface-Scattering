"""T-matrix -> forward-scattering amplitude -> periodic-array specular
S-parameters (docs/manual.tex, Sec. "Planned extensions" / "Metasurface
post-processing").

Implements the manual's forward-scattering-theorem formula:

    S21 = 1 - (j 2 pi) / (k^2 A cos(theta_inc)) * e_inc^* . F(k_inc)
    S11 =   - (j 2 pi) / (k^2 A cos(theta_inc)) * e_ref^* . F(k_ref)

with F the SINGLE SCATTERER's far-field scattering amplitude (evaluate via
vswf.evaluate_farfield of the outgoing coefficients f = T a), A the
unit-cell area, and k_ref the specular direction (theta_ref = pi -
theta_inc, phi_ref = phi_inc, for an array in the z = 0 plane).

TWO CONVENTION FIXES, found in sequence, both because the manual's formula
is written for a scattering amplitude f(theta,phi) with units of LENGTH in
the "standard" e^{-i omega t} optical-theorem convention (E_sca ~ E0
f(theta,phi) e^{ikr}/r), while this package works in e^{+j omega t}:

1. MAGNITUDE (units). This package's own F (vswf.evaluate_farfield /
   vswf.project_farfield, "E_sca -> F e^{-jkr}/(kr)") differs from
   f_standard by one power of k. Missing this made S11 five orders of
   magnitude too large against a real periodic CST run (see
   examples/periodic_unit_cell_check.py) -- caught by that external
   comparison; nothing internal to this module could have.
2. SIGN. |f_standard| = |F_package| / k is NOT enough -- switching time
   convention conjugates the phasor, and f_standard = -F_package / k (not
   +F_package / k). Missing the sign passed the magnitude-level CST
   comparison (both numbers looked like "same ballpark, few-% error") but
   is DECISIVELY wrong: it makes |S11|^2 + |S21|^2 grow smoothly above 1
   with increasing scattering strength -- unphysical gain from a manifestly
   passive scatterer. Confirmed and fixed via a fully offline, CST-
   independent check: feed the EXACT Mie sphere reference T-matrix
   (tmatrix_solve.sphere_reference_T, independently verified passive via
   passivity_check, maxsv(S) == 1.00000 to 5 digits at every tested x) into
   forward_scattering_amplitude, and compare Im(F_package)/k against the
   optical theorem's Im(f_standard) = k/(4 pi) * sigma_ext, with sigma_ext
   from mie.efficiencies (the SAME independently-validated Mie reference
   Qext used in tests/test_vswf.py). The ratio was exactly -1.0000 at every
   size parameter tested (x = 0.1 to 1.0) -- i.e. f_standard = -F_package/k,
   not +F_package/k. With the sign corrected, the sphere reference respects
   |S11|^2 + |S21|^2 <= 1 almost exactly in the weak-scattering limit
   (0.982-0.988 for x <= 0.2) and degrades GRACEFULLY, not catastrophically,
   as x grows -- exactly the signature of a correct leading-order formula
   meeting the edge of its own validity, not a bug. The lesson: a smooth,
   monotonic, sign-of-one-kind error that GROWS with scattering strength is
   the signature of a wrong sign/convention, not of numerical noise (mesh,
   extraction quality) -- those degrade less cleanly. When in doubt, prefer
   an offline, analytically-exact check (Mie + the optical theorem, both
   already independently validated elsewhere in this package) over trusting
   an external FEM comparison's magnitude alone -- magnitude-only agreement
   is exactly what let this sign error hide the first time.

ISOLATED-SCATTERER APPROXIMATION: a = a_inc (the bare incident plane wave)
is fed into T with NO correction for inter-element coupling across the
lattice, and no higher-order terms beyond this leading-order theorem. The
manual also names Foldy-Lax equations with translation-addition operators
as the general array formalism; the periodic-array specialization of that
(Ewald-summed lattice Green's functions) is a substantially larger
undertaking and is NOT implemented here.

Against the first real design (spoke-and-wheel, wl=15.0094 um, 2x2 um cell,
5 deg incidence -- examples/periodic_unit_cell_check.py), |S11|^2+|S21|^2
sits ABOVE 1 (1.017-1.148) across the ENTIRE swept band (15-37 THz), never
dropping below 1 even at the weakest tested point -- worth being explicit
about, since "it degrades gracefully at strong scattering" undersells what
"never crosses 1" looks like at a glance, and DOES warrant the extra
scrutiny below rather than being waved off.

Ruled out, each via an offline, CST-independent test additional to the sign
check above (all in tests/test_metasurface.py):
- Extraction quality: the extraction's own incident_deviation_max
  diagnostic is worst at 15 THz (0.37) and best at 37 THz (0.033) -- the
  OPPOSITE trend from the unitarity violation (smallest at 15 THz, largest
  at 37 THz), which instead tracks ||T|| exactly. Not the driver.
- Higher-multipole content: zeroing out every mode above the dipole
  (n<=1 only) reproduces the full-T violation almost exactly (1.0164 of
  1.0170 at 15 THz) -- n=2,3 content is 2-3 orders of magnitude smaller
  and barely moves the number. It's entirely a dipole-term effect.
- Magnetoelectric (bianisotropic) coupling: the dipole block's M-N and
  N-M cross-quadrants are ~200x smaller than the M-M/N-N diagonal blocks
  -- negligible, not the driver.
- M-type vs. N-type asymmetry in evaluate_farfield: this design's dipole
  is magnetic (M-type/TE) dominated (|T_MM| / |T_NN| ~ 7), unlike every x
  in the original sign-check sweep, which was electric (N-type/TM)
  dominated throughout (a plain dielectric Mie sphere's b_1 essentially
  never exceeds a_1 at modest x). Re-ran the optical-theorem check with
  the Mie a_l/b_l coefficients SWAPPED into each other's slot (Qext is
  symmetric under that swap, so it's still the right target) -- exact
  match (ratio 1.0000) from x=0.1 (magnetic:electric ~500:1) to x=1.3.
  evaluate_farfield has no M/N asymmetry.

What DOES explain it: matching by |T_dipole| MAGNITUDE (not by size
parameter x) against that same swapped, magnetic-dominated Mie reference,
the reference shows the identical pattern -- sum^2 crosses ABOVE 1 once
|T_dipole| exceeds roughly 0.005-0.006, and the real design's dipole never
gets weaker than 0.0064 anywhere in its swept band. At matched magnitude
the two also land at comparable violation levels (reference ~1.01 vs.
design's actual 1.017 at the weak end; reference ~1.22 vs. 1.148 at the
strong end) -- same order, same trend, not an exact match (the two have a
different loss/reactance ratio in their dipole term, so exact numerical
agreement isn't expected), but consistent enough to support "this design's
dipole moment, across this whole extraction band, is simply too strong for
the leading-order theorem to stay under 1" over "there is a further bug."
That is evidence, not proof -- a genuinely independent check (Foldy-Lax
coupling, or extracting a materially weaker/smaller design and rechecking
that it DOES cross below 1) would settle it more firmly.

Two normalization differences the comparison against a real CST periodic
run must still account for -- see examples/periodic_unit_cell_check.py:

1. Reference plane: this module references phase to the scatterer's own
   origin (z=0, matching vswf.plane_wave_field's "zero phase at origin").
   CST's Floquet-port S-parameters are referenced to the PORT planes, which
   sit some finite distance from the structure -- S21's PHASE differs by
   the corresponding propagation factor e^{-/+jk*(port separation)}, which
   this module does not add back in.
2. Polarization convention: e_inc and e_ref are each defined in the LOCAL
   (theta_hat, phi_hat) basis at their own propagation direction, exactly
   as vswf.plane_wave_field already defines a "theta"- or "phi"-polarized
   wave at any direction. Which CST Floquet mode index (1 or 2) this
   corresponds to is a CST-side convention (TE/TM vs. scan-plane geometry)
   that examples/periodic_unit_cell_check.py resolves empirically by
   comparing against both, rather than assuming one.
"""

from __future__ import annotations

import numpy as np

from ..vswf import evaluate_farfield, plane_wave_coefficients


def forward_scattering_amplitude(T: np.ndarray, lmax: int, theta_i: float,
                                 phi_i: float, pol: str,
                                 theta_eval: float | None = None,
                                 phi_eval: float | None = None):
    """(F_th, F_ph) scattering amplitude (see vswf.evaluate_farfield) for
    unit plane-wave incidence (theta_i, phi_i, pol) [vswf.plane_wave_field
    convention], evaluated at (theta_eval, phi_eval) -- default: the
    incidence direction itself (forward scattering, for S21)."""
    a = plane_wave_coefficients(theta_i, phi_i, pol, lmax)
    f = T @ a
    te = theta_i if theta_eval is None else theta_eval
    pe = phi_i if phi_eval is None else phi_eval
    F_th, F_ph = evaluate_farfield(f, lmax, te, pe)
    return complex(F_th[0]), complex(F_ph[0])


def specular_s_parameters(T: np.ndarray, lmax: int, k: float,
                          period_area: float, theta_i: float, phi_i: float,
                          pol_i: str = "theta"):
    """S11, S21 for an infinite periodic array (unit-cell area
    `period_area`, in the same length units as 1/k) of this scatterer,
    under plane-wave incidence (theta_i, phi_i, pol_i) -- isolated-
    scatterer approximation, see module docstring.

    theta_i measured as in vswf.plane_wave_field (propagation direction);
    avoid theta_i very near 0/pi, a coordinate singularity for the
    theta/phi polarization basis (as elsewhere in this package).
    """
    if abs(np.sin(theta_i)) < 1e-6:
        raise ValueError("theta_i too close to the pole (0 or pi) -- the "
                         "theta/phi polarization basis is singular there; "
                         "use a small off-normal angle instead.")
    theta_r, phi_r = np.pi - theta_i, phi_i

    F_th_i, F_ph_i = forward_scattering_amplitude(T, lmax, theta_i, phi_i,
                                                   pol_i)
    F_th_r, F_ph_r = forward_scattering_amplitude(
        T, lmax, theta_i, phi_i, pol_i, theta_eval=theta_r, phi_eval=phi_r)

    F_i_dot = F_th_i if pol_i == "theta" else F_ph_i
    F_r_dot = F_th_r if pol_i == "theta" else F_ph_r

    pref = -1j * 2 * np.pi / (k ** 2 * period_area * np.cos(theta_i))
    S21 = 1.0 + pref * F_i_dot
    S11 = pref * F_r_dot
    return S11, S21

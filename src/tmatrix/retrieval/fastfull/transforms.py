"""Flux-normalized incoming / outgoing Floquet transforms A and W.

This module is where CST's power-normalized Floquet port waves are put into
the repository's unit-field VSWF convention.  The proposal (par. 5) is
explicit that "angular vectors alone are not the physical measurement
operator": the order-dependent sqrt(|k_z|) factors, the cell area, the wave
impedance, the propagation direction and the port reference-plane phase all
have to be in A and W or the rank/SNR numbers are meaningless.

Derivation of the normalization
-------------------------------
A plane wave  E = E0 e_hat exp(i k_hat . r)  in a medium of impedance Z0
carries time-averaged power through one unit cell of area `area`

    P = (1/2) |E0|^2 / Z0 * cos(theta) * area,      cos(theta) = |k_z| / k.

A CST Floquet port mode is power normalized: unit modal amplitude carries
unit power.  Hence unit CST amplitude corresponds to the field amplitude

    alpha = sqrt( 2 Z0 k / (area |k_z|) )                            (1)

so the regular-VSWF coefficients of an incoming unit-amplitude channel c are

    A[:, c] = s_c * alpha_c * plane_wave_coeffs(k_hat_c, e_hat_c, modes).  (2)

For the outgoing side, a periodic sheet whose per-cell outgoing coefficients
are f radiates, into the Floquet channel c' of in-plane wavevector q and
out-of-plane wavenumber k_z, the plane-wave field amplitude

    E_out(c') = (2 pi i / (area |k_z|)) e_hat_{c'}^dagger . F(k_hat_{c'})  (3)

with F the single-cell far-field amplitude of vswf.far_field_amplitude.
(3) is exactly the prefactor of sparams_oblique.jones_blocks, written with
cos(theta) = |k_z| / k substituted, and is therefore convention-locked to the
validated specular pipeline.  Dividing by alpha_{c'} converts the field back
to a CST power amplitude, giving

    W[c', :] = s_{c'} * (2 pi i / (area |k_z_{c'}| alpha_{c'})) * G[c', :]  (4)

where G[c', nu] = e_hat_{c'}^dagger . FF_nu(k_hat_{c'}) and FF_nu is the
per-mode far-field vector (farfield_basis below, which reproduces
vswf.far_field_amplitude exactly by construction).

Z0 CANCELS: A carries sqrt(Z0), W carries 1/sqrt(Z0), so every entry of
W T A is impedance independent.  Z0 = 1 is therefore used, and the flux
factor is stored as nu = sqrt(k / (area |k_z|)) with alpha = sqrt(2) nu.

Sanity anchor (gated in test_fastfull_core.py): for a cell in which only one
order propagates, W T_eff A + S_empty reproduces sparams_oblique's S11 / S21
Jones blocks to roundoff, because there alpha_in = alpha_out and the product
of the two normalizations collapses back to 2 pi i / (k * area * cos theta).

TE / TM gauge (the measured CST convention)
-------------------------------------------
The par.-2 polarization basis evaluates e_TM SEPARATELY for each propagation
direction, and the two are transversally opposite:

    e_TM(theta, phi, d) = (d cos(theta) cos(phi), d cos(theta) sin(phi),
                           -sin(theta)).

A CST Floquet port mode is one fixed transverse pattern per (order,
polarization), used unchanged whichever way the wave travels through it.
Modelling that fixed pattern as the d = +1 member gives the per-use sign

    s(TE, d) = +1,        s(TM, d) = d                                (5)

and S_cst[a, b] = s_b^in s_a^out S_phys[a, b].  Rule (5) reproduces, entry
for entry, the mapping table in the sparams_oblique docstring: for
illumination with d_in = +1 every S21 entry maps with +, while the TM
RECEIVE row of S11 (whose outgoing wave has d = -1) maps with -1.  That is
the sign whose omission took the campaign's chi^2_red from 658.9 to 2.49
(retrieval/HANDOFF.md, deembed.py "S11 TM-ROW SIGN").

Two caveats, both deliberate:

* Rule (5) fixes the reference to d = +1 for BOTH ports.  For down-going
  illumination (d_in = -1, the campaign's case) it therefore predicts the
  sign on the TM *column* of the reflection block rather than its row.  The
  two readings differ only by a per-mode gauge s_TM(g) = +/-1, which flips
  TE-TM cross terms and leaves every co-polar entry -- the only ones the
  campaign could measure, its real cross-pol being ~1e-4 -- invariant.  The
  residual freedom is exposed as `mode_gauge` and must be closed by the M3
  port-mode field export, not by assumption.
* A per-port reference tied to each port's inward normal is RULED OUT by the
  campaign: it would put the same sign on the transmission block, which
  measurably does not carry one.

Reference plane
---------------
A and W are built at the VSWF origin (the wheel mid-plane).  CST reports at
its port planes; `port_plane_phase` supplies the diagonal de-embedding phase
exp(i |k_z| L/2) per channel, so that the caller can move measured data to
the mid-plane (proposal par. 9.1 step 4).  Nothing here silently assumes a
reference plane: with the default L = 0 the phase is the identity.
"""
import numpy as np

from tmatrix.aggregation.vswf import ModeBasis, MAGNETIC, plane_wave_coeffs, xz_vectors
from tmatrix.retrieval.sparams_oblique import pol_basis, khat_from_angles

from .lattice import ChannelSet, OrderSet, TE, TM, ZMAX, ZMIN

Z0 = 1.0        # cancels exactly between A and W; see module docstring

GAUGE_PHYSICAL = "physical"
GAUGE_CST = "cst"


# ------------------------------------------------------------- far-field basis

def farfield_basis(k, modes, rhat_pts):
    """Per-mode far-field vectors FF[nu, p, :] at unit directions rhat_pts.

    Defined so that  far_field_amplitude(k, f, modes, rhat)
                     == np.tensordot(f, farfield_basis(k, modes, rhat),
                                     axes=(0, 0))
    exactly (same arithmetic, contraction pulled out); gated in
    test_fastfull_core.py.
    """
    pts = np.atleast_2d(np.asarray(rhat_pts, dtype=float))
    nrm = np.linalg.norm(pts, axis=1)
    if np.any(np.abs(nrm - 1.0) > 1e-9):
        raise ValueError("far-field directions must be unit vectors")
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    xz = xz_vectors(modes, theta, phi)
    FF = np.empty((modes.n, len(pts), 3), dtype=complex)
    for i in range(modes.n):
        l, m, p = int(modes.l[i]), int(modes.m[i]), int(modes.pol[i])
        X, Z = xz[(l, m)]
        FF[i] = ((-1j) ** (l + 1)) * X if p == MAGNETIC else ((-1j) ** l) * Z
    return FF / k


# --------------------------------------------------------------- flux factors

def plane_wave_coeffs_batch(khat, ehat, modes):
    """Batched vswf.plane_wave_coeffs: (n_modes, n_dirs) regular coefficients.

    Same arithmetic with the angular tables built once for all directions
    instead of once per direction (the design search evaluates thousands of
    candidate cells).  Gated against plane_wave_coeffs entry by entry in
    test_fastfull_core.py.
    """
    khat = np.atleast_2d(np.asarray(khat, dtype=float))
    ehat = np.atleast_2d(np.asarray(ehat, dtype=complex))
    khat = khat / np.linalg.norm(khat, axis=1, keepdims=True)
    theta = np.arccos(np.clip(khat[:, 2], -1.0, 1.0))
    phi = np.arctan2(khat[:, 1], khat[:, 0])
    xz = xz_vectors(modes, theta, phi)
    a = np.empty((modes.n, len(khat)), dtype=complex)
    for i in range(modes.n):
        l, m, p = int(modes.l[i]), int(modes.m[i]), int(modes.pol[i])
        X, Z = xz[(l, m)]
        if p == MAGNETIC:
            a[i] = 4 * np.pi * (1j ** l) * (np.conj(X) * ehat).sum(axis=1)
        else:
            a[i] = 4 * np.pi * (1j ** (l - 1)) * (np.conj(Z) * ehat).sum(axis=1)
    return a


def flux_nu(k, area, kz):
    """nu = sqrt(k / (area |k_z|)); the CST field amplitude is sqrt(2 Z0) nu."""
    kz = np.asarray(kz, dtype=float)
    if np.any(~np.isfinite(kz)) or np.any(kz <= 0):
        raise ValueError("flux normalization needs propagating channels "
                         "(finite k_z > 0)")
    return np.sqrt(float(k) / (float(area) * kz))


def gauge_signs(channels, gauge=GAUGE_CST, mode_gauge=None):
    """(s_in, s_out) per channel for the requested TE/TM gauge.

    gauge = 'cst'      : s(TE) = +1, s(TM) = propagation direction  (eq. 5)
    gauge = 'physical' : s = +1 everywhere (sparams_oblique's own basis)

    mode_gauge : optional (n_channels,) array of +/-1, the residual per-port-
        mode orientation freedom.  It multiplies BOTH s_in and s_out of a
        channel, so it acts on S as S -> D S D and can never change a
        co-polar diagonal entry (deembed.py's label-hypothesis family).
    """
    n = channels.n
    if gauge == GAUGE_PHYSICAL:
        s_in = np.ones(n)
        s_out = np.ones(n)
    elif gauge == GAUGE_CST:
        is_tm = channels.pol == TM
        s_in = np.where(is_tm, channels.direction_in, 1.0).astype(float)
        s_out = np.where(is_tm, channels.direction_out, 1.0).astype(float)
    else:
        raise ValueError("gauge must be %r or %r"
                         % (GAUGE_PHYSICAL, GAUGE_CST))
    if mode_gauge is not None:
        mg = np.asarray(mode_gauge, dtype=float).ravel()
        if mg.size != n or not np.all(np.isin(mg, (-1.0, 1.0))):
            raise ValueError("mode_gauge must be (n_channels,) of +/-1")
        s_in = s_in * mg
        s_out = s_out * mg
    return s_in, s_out


def _channel_vectors(channels, which):
    """(khat, e_hat) arrays (n, 3) for the incoming or outgoing use."""
    if which not in ("in", "out"):
        raise ValueError("which must be 'in' or 'out'")
    d = channels.direction_in if which == "in" else channels.direction_out
    khat = np.empty((channels.n, 3))
    ehat = np.empty((channels.n, 3))
    for i in range(channels.n):
        th, ph, di = float(channels.theta[i]), float(channels.phi[i]), \
            float(d[i])
        khat[i] = khat_from_angles(th, ph, di)
        ehat[i] = pol_basis(th, ph, di)[int(channels.pol[i])]
    return khat, ehat


def check_channel_kinematics(channels, atol=1e-10):
    """Worst |k * k_hat_inplane - q| over channels (a convention gate).

    Confirms that the (theta, phi, direction) triple handed to the par.-2
    polarization basis really carries the Floquet order's in-plane
    wavevector, and that incoming and outgoing uses share it.
    """
    worst = 0.0
    for which in ("in", "out"):
        khat, _ = _channel_vectors(channels, which)
        q_pred = channels.k * khat[:, :2]
        worst = max(worst, float(np.abs(q_pred - channels.q).max()))
        kz_pred = channels.k * np.abs(khat[:, 2])
        worst = max(worst, float(np.abs(kz_pred - channels.kz).max()))
    return worst


# --------------------------------------------------------------- A and W

def build_A(k, channels, modes, area=None, gauge=GAUGE_CST, mode_gauge=None):
    """Incoming transform A (n_modes, n_channels), eq. (2).

    Column c holds the regular-VSWF coefficients, about the VSWF origin, of
    the incident field of unit CST modal amplitude in channel c.
    """
    area = channels.orders.lattice.area if area is None else float(area)
    nu = flux_nu(k, area, channels.kz)
    s_in, _ = gauge_signs(channels, gauge, mode_gauge)
    khat, ehat = _channel_vectors(channels, "in")
    A = plane_wave_coeffs_batch(khat, ehat, modes)
    return A * (s_in * np.sqrt(2.0 * Z0) * nu)[None, :]


def build_W(k, channels, modes, area=None, gauge=GAUGE_CST, mode_gauge=None):
    """Outgoing transform W (n_channels, n_modes), eq. (4).

    Row c' projects the per-cell outgoing VSWF coefficients onto the CST
    modal amplitude radiated into channel c'.
    """
    area = channels.orders.lattice.area if area is None else float(area)
    nu = flux_nu(k, area, channels.kz)
    _, s_out = gauge_signs(channels, gauge, mode_gauge)
    khat, ehat = _channel_vectors(channels, "out")
    FF = farfield_basis(k, modes, khat)              # (n_modes, n_ch, 3)
    # (2 pi i / (area |kz|)) / alpha, alpha = sqrt(2 Z0) nu
    pref = (2j * np.pi / (area * np.asarray(channels.kz, dtype=float))
            / (np.sqrt(2.0 * Z0) * nu))
    W = np.einsum("ncj,cj->cn", FF, np.conj(ehat))
    return W * (s_out * pref)[:, None]


def empty_modal_S(channels):
    """Direct (structure-free) modal operator at the VSWF-origin reference
    plane: unit transmission to the opposite port, same order, same
    polarization; zero reflection.

    Identical in both gauges: for a transmitted wave the outgoing direction
    equals the incoming one, so the two TM gauge signs multiply to +1.
    """
    n = channels.n
    S = np.zeros((n, n), dtype=complex)
    key_out = {}
    for c in range(n):
        key_out[(int(channels.side[c]), int(channels.order_index[c]),
                 int(channels.pol[c]))] = c
    for c in range(n):
        partner = key_out.get((-int(channels.side[c]),
                               int(channels.order_index[c]),
                               int(channels.pol[c])))
        if partner is None:
            raise RuntimeError("channel set is not port-symmetric")
        S[partner, c] = 1.0
    return S


def port_plane_phase(channels, L_port):
    """Diagonal exp(i |k_z| L_port / 2) per channel.

    CST reports at port planes a distance L_port apart, symmetric about the
    VSWF origin.  Measured data are moved to the mid-plane by dividing the
    incoming and the outgoing amplitude by this phase each; the empty-cell
    operator then becomes the identity of `empty_modal_S`.  L_port = 0 gives
    the identity, so nothing is assumed by default.
    """
    return np.exp(1j * np.asarray(channels.kz, dtype=float) * 0.5
                  * float(L_port))


# --------------------------------------------------------------- forward map

def t_effective(T0, C):
    """T_eff = T0 (I - C T0)^{-1} = (I - T0 C)^{-1} T0 (aggregate's form)."""
    n = T0.shape[0]
    return T0 @ np.linalg.inv(np.eye(n, dtype=complex) - C @ T0)


def deembed_lattice(T_eff, C, return_diag=False):
    """Solve (I + T_eff C) T0 = T_eff for the isolated T0 (proposal par. 5).

    The T_eff^{-1} form is deliberately NOT used: exact symmetry can make
    T_eff singular.  With return_diag, also reports sigma_min(I + T_eff C)
    and the resulting error-amplification bound, which the proposal requires
    to be published at every retained frequency (Gate D).
    """
    n = T_eff.shape[0]
    M = np.eye(n, dtype=complex) + T_eff @ C
    T0 = np.linalg.solve(M, T_eff)
    if not return_diag:
        return T0
    sv = np.linalg.svd(M, compute_uv=False)
    return T0, dict(sigma_min=float(sv.min()), sigma_max=float(sv.max()),
                    cond=float(sv.max() / sv.min()))


def modal_S(W, T_eff, A, S_empty=None):
    """Full modal Floquet S = S_empty + W T_eff A (proposal par. 5)."""
    S = W @ T_eff @ A
    return S if S_empty is None else S_empty + S


def scattered_S(W, T0, C, A):
    """Convenience: W T_eff(T0, C) A, the de-embedded scattered block."""
    return W @ t_effective(T0, C) @ A


# ------------------------------------------------------------- generic track

def generic_track_metrics(A, W, rcond=1e-12):
    """rank(A), rank(W), kappa(A), kappa(W) after flux normalization.

    The proposal's Gate A for the generic algebraic branch: both transforms
    must span all 30 VSWFs, preferably with kappa <= 10; anything above 30
    needs investigation (par. 10 Gate A).
    """
    out = {}
    for nm, M in (("A", A), ("W", W)):
        sv = np.linalg.svd(M, compute_uv=False)
        n_min = min(M.shape)
        tol = rcond * sv.max() if sv.size else 0.0
        out["sv_" + nm] = sv
        out["rank_" + nm] = int((sv > tol).sum())
        out["sigma_min_" + nm] = float(sv[n_min - 1]) if sv.size >= n_min \
            else 0.0
        out["sigma_max_" + nm] = float(sv.max()) if sv.size else 0.0
        out["kappa_" + nm] = (float(sv.max() / sv[n_min - 1])
                              if sv.size >= n_min and sv[n_min - 1] > 0
                              else np.inf)
    out["full_rank"] = (out["rank_A"] >= A.shape[0]
                        and out["rank_W"] >= W.shape[1])
    return out


def unit_field_transforms(k, channels, modes, area=None, gauge=GAUGE_CST,
                          mode_gauge=None):
    """A and W WITHOUT the flux factors (unit incident field amplitude).

    Provided only so that the proposal's par. 6 preliminary result -- which
    quoted kappa(A) = kappa(W) = 4.5 for the *unit-field angular* transforms
    -- can be reproduced and contrasted with the physical, flux-normalized
    conditioning.  Never use these for a retrieval.
    """
    area = channels.orders.lattice.area if area is None else float(area)
    s_in, s_out = gauge_signs(channels, gauge, mode_gauge)
    khat_i, ehat_i = _channel_vectors(channels, "in")
    khat_o, ehat_o = _channel_vectors(channels, "out")
    FF = farfield_basis(k, modes, khat_o)
    A = plane_wave_coeffs_batch(khat_i, ehat_i, modes) * s_in[None, :]
    W = np.einsum("ncj,cj->cn", FF, np.conj(ehat_o)) * s_out[:, None]
    return A, W

"""Per-frequency Levenberg-Marquardt driver for the inverse T-matrix
retrieval (checklist item 5 of INVERSE_TMATRIX_FROM_FLOQUET.md, doc par. 4).

Per frequency index ifreq the driver solves

    min over real t   sum_i w_i |S_meas,i - S_pred,i(T0(t))|^2
                      (+ tikhonov * ||t||^2   when tikhonov > 0)
    with  T0(t) = parametrize.unpack(t, B)    (C4v x reciprocity subspace)

using scipy.optimize.least_squares on the stacked real/imag residual.  The
residual is EXACTLY ForwardModel.pack_S(S_pred, w) - ForwardModel.pack_S(
S_meas, w) (all real parts first, then all imaginary parts; both scaled by
sqrt(w) per COMPLEX observable), so the squared 2-norm of the residual
equals the doc-par.-4 objective sum_i w_i |dS_i|^2 by the pack_S identity
validated in forward.py's selftest.

Optimizer (doc par. 4): method 'lm' by default with the Born seed t0 = 0
(rho(T0 C) = 0.17-0.32 in-band at the smoke frequencies -> near-linear).
When tikhonov > 0 the default switches to 'trf' and the residual vector is
augmented with rows sqrt(tikhonov) * t, pinning null-space directions
toward 0 (the par.-4 prescription for unobservable directions); any caller
override goes through **lsq_kwargs (method, xtol, ftol, gtol, ...).

JACOBIAN -- deviation from the doc's "finite differences are fine",
justified by measurement: with scipy/MINPACK internal finite differences
the 136-parameter full-basis fit did NOT converge -- measured > 21900
residual calls (> 160 LM iterations at 137 FD evaluations each, ~19 ms
per call => > 7 min, still running; the singular-value spread of J spans
~6 decades and FD noise keeps the gradient test unsatisfied).  The driver
therefore uses an EXACT ANALYTIC Jacobian by default (AnalyticJacobian
below): the forward map is affine in the outgoing coefficients,
S = const + L f, and the resolvent derivative has the closed form

    f_b = T x_b,  x_b = (I - C T)^-1 a_b
    dS(row, b) / dT = [L (I - T C)^-1]_row  (x)  x_b        (push-through)

so one Jacobian costs two 30x30 solves + one 4-rhs transpose solve + an
einsum per angle (~50 ms full-basis, all 13 angles) instead of 137
predict() calls, and is exact (no FD noise; LM then converges in ~10
iterations).  Safeguards: (1) the row map L is built by calling the SAME
verified primitives (pol_basis, khat_from_angles, far_field_amplitude)
and the incident coefficients come from sparams_oblique itself at T = 0;
(2) at construction the engine must reproduce fm.predict at the reference
T to <= validate_tol (1e-9) -- this pins L and the x-solve (measured
~1e-16); (3) the resolvent-derivative calculus is cross-checked against
an INDEPENDENT central-finite-difference Jacobian (observability.jacobian
at FD step 1e-8) in test_fit_smoke.py -- measured column-wise agreement
5.2e-8 / 7.2e-9 at the smoke frequencies.  Pass jac='2-point' (or any
scipy jac spec) through **lsq_kwargs to fall back to finite differences.

CAMPAIGN CONVENTION: direction = -1 (incidence from +z travelling down,
the CST/treams illumination) is the DEFAULT in this module and everything
built on it; forward.py's low-level default is +1, so direction is always
passed explicitly.  direction stays a parameter throughout.

Exactness checks (assert, not just report): T0_hat = unpack(t_hat, B) lies
in the symmetry subspace BY CONSTRUCTION (every B_k is an eigenvector of
the full projector), so after each fit the driver verifies

    max|P(T0_hat) - T0_hat|   <= check_tol * max(1, max|T0_hat|)
    max|Rec(T0_hat) - T0_hat| <= check_tol * max(1, max|T0_hat|)

and RAISES on violation (roundoff-level ~1e-15 expected; check_tol=1e-10).
Passivity max SV(I + 2 T0_hat) is computed and reported (post-check per
doc par. 4, not enforced).

Measured timings (cst_inference env, this machine, ifreq 32, direction -1,
campaign 13 angles): predict 1.51 ms/angle (19.6 ms / residual
evaluation); analytic Jacobian ~6-10 ms (full 68-basis); bright-basis fit
0.3-6 s (15-270 residual calls), observable-restricted full fit 15-40 s
(600-1600 calls incl. multistart; numbers re-measured and reported by
test_fit_smoke.py / synthetic_test.py).

MEASURED IDENTIFIABILITY STRUCTURE (campaign-13 specular data; central to
every gate interpretation downstream -- see synthetic_test.py):
  * At the Born point (T = 0) the exact Jacobian has rank 58 of 136 real
    parameters (singular values >= ~0.1, then a clean drop to ~1e-13):
    only 29 of the 68 complex directions are visible at LINEAR order.
  * Linearized at the reference T the spectrum fills in through the
    quadratic lattice coupling but spans ~16 decades (s_max/s_min ~ 6e16,
    numerically rank ~82/136 at 1e-6 s_max): the full 68-basis is NOT
    identifiable from this data at any noise level, and an unregularized
    full fit wanders on a near-degenerate manifold (measured: data matched
    to 7.4e-7 while ||dt||/||t|| = 0.93).  observable_subbasis() below
    restricts the parametrization to the observable span -- the exact
    (hard) form of the doc-par.-4 "null-space directions are not fitted,
    Tikhonov toward 0 keeps them pinned".
  * The bright-10 basis IS well conditioned at the reference linearization
    (s_w down to only ~7 at sigma = 3e-3 weighting; 20/20 directions
    observable at campaign and starter sets, 4/20 at normal-only), but a
    bright-only fit of the PHYSICAL target carries an irreducible MODEL
    ERROR: sub-threshold entries (|T| < 1e-3 |T|max, e.g. the
    (3,-+3)<->(1,+-1) content with |dS/dT| ~ 70-800) produce specular
    signal at the 1e-3 level (measured maxdS = 1.2e-3 at 8 um) that the
    bright span cannot represent; absorbed into the bright coefficients it
    biases the SMALL bright entries by orders of their own size.  These
    are physics findings of the observability program, not code defects:
    the closed-loop machinery itself is exact (test_fit_smoke identity
    recovers in-span targets to ~1e-13).

Public API:
    fit_frequency(fm, ifreq, B, S_meas, angle_indices, ...) -> result dict
    fit_band(fm, B, S_meas_stack, freq_indices, ...)        -> (results, skipped)
    AnalyticJacobian(fm, ifreq, B, angle_indices, ...)      -> exact-J engine
    compare_to_reference(T_hat_stack, T_tgt_stack, mask, ...) -> metrics dict
    entry_errors(T_hat, T_tgt, entries, peak)   -> light per-entry errors
    classify_entries(modes)                     -> class masks (n, n)
    ANGLE_SETS / resolve_angles(spec)           -> named angle subsets
    mode_label / entry_label                    -> "(l,m,P)" strings
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "aggregation"))
sys.path.insert(0, HERE)

import numpy as np
from scipy.optimize import least_squares

from vswf import ELECTRIC, far_field_amplitude         # noqa: E402
from sparams_oblique import (khat_from_angles, pol_basis,
                             sparams_oblique)          # noqa: E402
import parametrize as par                              # noqa: E402

# Campaign convention (doc par. 7): incidence from +z travelling down.
DIRECTION_DEFAULT = -1

# Named subsets of the normative 17-angle table (precompute_C.ANGLES_DEG):
#   campaign 13  = indices 0-12  (theta {0,15,30,45,60} x phi {0,22.5,45})
#   starter 5    = {(0,0),(30,0),(30,22.5),(60,0),(60,22.5)} (doc par. 7)
#   treams 4     = indices 13-16 ({20,40} x {0,45}, validation-only)
ANGLE_SETS = {
    "campaign": list(range(13)),
    "normal-only": [0],
    "starter": [0, 4, 5, 10, 11],
    "treams": [13, 14, 15, 16],
    "all17": list(range(17)),
}


def resolve_angles(spec):
    """Angle-index list from a name in ANGLE_SETS, a comma string, or a
    sequence of ints.  None -> campaign 13."""
    if spec is None:
        return list(ANGLE_SETS["campaign"])
    if isinstance(spec, str):
        if spec in ANGLE_SETS:
            return list(ANGLE_SETS[spec])
        return [int(tok) for tok in spec.split(",") if tok.strip()]
    return [int(a) for a in spec]


def _maxabs(x):
    return float(np.abs(x).max())


# ------------------------------------------------------------------ labels

def mode_label(modes, i):
    """Short mode label, e.g. '3-3E' for (l=3, m=-3, electric)."""
    p = "E" if modes.pol[i] == ELECTRIC else "M"
    return "%d%+d%s" % (modes.l[i], modes.m[i], p)


def entry_label(modes, i, j):
    """Entry label '(l,m,P)x(l',m',P')' for T[i, j]."""
    return "%sx%s" % (mode_label(modes, i), mode_label(modes, j))


def classify_entries(modes):
    """Boolean (n, n) class masks over T-matrix ENTRY positions.

    All classes except 'c4_violating' are intersected with the C4 selection
    rule (m - m') mod 4 == 0 (entries outside it are annihilated by the
    projector and carry no fit parameters).  Classes:
      dipole  : l = l' = 1                     (the par.-6.3 gate block)
      quad22  : l = l' = 2, |m| = |m'| = 2     (the theta=0-dark quadrupole)
      even_m  : m, m' both even                (all strictly theta=0-dark)
      odd_m   : m, m' both odd
      m33_11  : {|m|, |m'|} = {3, 1}           (doc par. 4: visible at theta=0)
      m33_33  : |m| = |m'| = 3                 (doc par. 4: visible at theta=0)
    """
    M1, M2 = modes.m[:, None], modes.m[None, :]
    L1, L2 = modes.l[:, None], modes.l[None, :]
    conf = np.mod(M1 - M2, 4) == 0
    return dict(
        c4_conforming=conf,
        c4_violating=~conf,
        dipole=(L1 == 1) & (L2 == 1) & conf,
        quad22=(L1 == 2) & (L2 == 2) & (np.abs(M1) == 2)
               & (np.abs(M2) == 2) & conf,
        even_m=(np.mod(M1, 2) == 0) & (np.mod(M2, 2) == 0) & conf,
        odd_m=(np.mod(M1, 2) != 0) & (np.mod(M2, 2) != 0) & conf,
        m33_11=(((np.abs(M1) == 3) & (np.abs(M2) == 1))
                | ((np.abs(M1) == 1) & (np.abs(M2) == 3))) & conf,
        m33_33=(np.abs(M1) == 3) & (np.abs(M2) == 3) & conf,
    )


# -------------------------------------------------- symmetry-check helpers

_SYM_CACHE = {}


def sym_ops(modes):
    """(Ds, perm, sign) for the exactness checks, cached per ModeBasis."""
    key = (id(modes), modes.n)
    if key not in _SYM_CACHE:
        Ds, _ = par.c4v_group(modes)
        perm, sign = par.reciprocity_perm_sign(modes)
        _SYM_CACHE[key] = (Ds, perm, sign)
    return _SYM_CACHE[key]


# ------------------------------------------------- analytic Jacobian engine

class AnalyticJacobian:
    """Exact Jacobian d packed(S) / d t of the par.-2 forward map.

    The specular Jones blocks are AFFINE in the per-cell outgoing
    coefficients: S[blk, a, b] = delta_ab delta_{blk,S21} + L[blk, a] . f_b
    with L the fixed projection rows (prefactor x receive-polarization x
    per-mode far-field), and f_b = T x_b, x_b = (I - C T)^-1 a_b.  The
    derivative along a T-perturbation E is (push-through identity)

        dS[blk, a, b] = L[blk, a] . (I - T C)^-1 . E . x_b,

    so with W = L (I - T C)^-1 (one 4-rhs transpose solve per angle) the
    Jacobian w.r.t. the COMPLEX basis coefficient z_k is W B_k x_b, and the
    real-parameter columns follow from holomorphy (column k+nb = i x
    column k in packed form).

    Construction safeguards (see module docstring): L is assembled from the
    same verified primitives sparams_oblique uses (pol_basis /
    khat_from_angles / far_field_amplitude, per-mode unit vectors), a_inc
    comes from sparams_oblique itself evaluated at T = 0, and the affine
    model must reproduce fm.predict at the reference T of this frequency to
    <= validate_tol, else construction RAISES.  The derivative calculus is
    additionally FD-cross-checked in test_fit_smoke.py.

    Row/packing layout is EXACTLY ForwardModel.pack_S: complex observable
    index c = ((angle*2 + blk)*2 + a)*2 + b, packed rows = [Re(all c),
    Im(all c)], both scaled by sqrt(weights) per complex observable.
    """

    def __init__(self, fm, ifreq, B, angle_indices=None,
                 direction=DIRECTION_DEFAULT, validate_tol=1e-9):
        angles = (resolve_angles(angle_indices)
                  if angle_indices is not None
                  else [int(a) for a in np.nonzero(fm.have[ifreq])[0]])
        miss = [a for a in angles if not fm.have[ifreq, a]]
        if miss:
            raise KeyError("C not cached for ifreq=%d, angles %s"
                           % (ifreq, miss))
        self.fm = fm
        self.ifreq = int(ifreq)
        self.angles = list(angles)
        self.direction = int(direction)
        self.B = np.asarray(B)
        nb = self.B.shape[0]
        n = fm.modes.n
        k = fm.k[ifreq]
        self.C_sel = np.array([fm.C[ifreq, a] for a in angles])
        self.a_inc = np.empty((len(angles), 2, n), dtype=complex)
        self.L = np.empty((len(angles), 2, 2, n), dtype=complex)
        zhat = np.array([0.0, 0.0, 1.0])
        for j, ia in enumerate(angles):
            th = np.deg2rad(fm.theta_deg[ia])
            ph = np.deg2rad(fm.phi_deg[ia])
            khat_t = khat_from_angles(th, ph, direction)
            khat_r = khat_t - 2.0 * khat_t[2] * zhat
            pref = 2j * np.pi / (k * fm.A_cell * abs(khat_t[2]))
            e_t = pol_basis(th, ph, direction)
            e_r = pol_basis(th, ph, -direction)
            rhat = np.stack([khat_t, khat_r])
            G = np.empty((n, 2, 3), dtype=complex)   # [mode, (t, r), xyz]
            for nu in range(n):
                e_nu = np.zeros(n, dtype=complex)
                e_nu[nu] = 1.0
                G[nu] = far_field_amplitude(k, e_nu, fm.modes, rhat)
            for a in range(2):
                # blk 0 = S11 (reflected direction, e_r); 1 = S21 (e_t)
                self.L[j, 0, a] = pref * (np.conj(e_r[a]) @ G[:, 1, :].T)
                self.L[j, 1, a] = pref * (np.conj(e_t[a]) @ G[:, 0, :].T)
            res0 = sparams_oblique(k, fm.A_cell, fm.modes,
                                   np.zeros((n, n), dtype=complex),
                                   self.C_sel[j], th, ph, direction)
            self.a_inc[j] = res0["a_inc"]
        # construction gate: affine model == verified forward at T_ref
        T_ref = fm.data.T[ifreq]
        S_lin = self.predict_affine(T_ref)
        S_ref = fm.predict(T_ref, ifreq, angles, direction)
        self.validation_resid = float(np.abs(S_lin - S_ref).max())
        if self.validation_resid > validate_tol:
            raise AssertionError(
                "AnalyticJacobian affine model disagrees with fm.predict "
                "at the reference T: max|dS| = %.3e > %.1e"
                % (self.validation_resid, validate_tol))
        self._I = np.eye(n, dtype=complex)
        self._nb = nb

    def predict_affine(self, T):
        """S stack from the affine assembly S = const + L f (validation)."""
        n_sel = len(self.angles)
        n = self.fm.modes.n
        S = np.zeros((n_sel, 2, 2, 2), dtype=complex)
        I = np.eye(n, dtype=complex)
        for j in range(n_sel):
            X = np.linalg.solve(I - self.C_sel[j] @ T, self.a_inc[j].T)
            F = T @ X                                   # (n, 2) outgoing
            S[j] = np.einsum("uan,nb->uab", self.L[j], F)
            S[j, 1, 0, 0] += 1.0
            S[j, 1, 1, 1] += 1.0
        return S

    def jac_packed(self, t, weights=None):
        """Exact J (16*n_sel [+0], 2*nb) at parameter vector t (real)."""
        T = par.unpack(t, self.B)
        n_sel = len(self.angles)
        nb = self._nb
        Jc = np.empty((n_sel, 2, 2, 2, nb), dtype=complex)
        for j in range(n_sel):
            C = self.C_sel[j]
            X = np.linalg.solve(self._I - C @ T, self.a_inc[j].T)  # (n, 2)
            A = self._I - T @ C
            W = np.linalg.solve(A.T, self.L[j].reshape(4, -1).T).T  # (4, n)
            Q = np.einsum("rm,kmn->rkn", W, self.B)
            Jc[j] = np.einsum("rkn,nb->rbk", Q, X).reshape(2, 2, 2, nb)
        m8 = 8 * n_sel
        Jc = Jc.reshape(m8, nb)
        if weights is not None:
            sw = np.sqrt(np.broadcast_to(
                np.asarray(weights, dtype=float),
                (n_sel, 2, 2, 2))).ravel()
            Jc = Jc * sw[:, None]
        J = np.empty((2 * m8, 2 * nb))
        J[:m8, :nb] = Jc.real
        J[m8:, :nb] = Jc.imag
        J[:m8, nb:] = -Jc.imag
        J[m8:, nb:] = Jc.real
        return J


def observable_subbasis(engine, t_lin=None, s_cutoff_rel=1e-4):
    """Complex sub-basis of span{B} restricted to the OBSERVABLE directions
    of the exact Jacobian at t_lin (default: the Born point t = 0).

    The packed real Jacobian encodes the holomorphic complex Jacobian
    dS/dz (nb complex columns); its complex SVD right vectors V with
    s_j >= s_cutoff_rel * s_max span the observable coefficient
    directions, and the returned basis is B_obs[j] = sum_k V[k, j] B_k
    (complex matrices; parametrize.pack/unpack handle complex bases -- the
    Frobenius-Hermitian orthonormality is inherited from V).  Fitting in
    B_obs is the exact form of the doc-par.-4 null-space pinning: the
    unobservable directions are not fitted and stay at 0.

    Returns (B_obs (r, n, n) complex, s (nb,) singular values, V (nb, r)).
    """
    nb = engine.B.shape[0]
    if t_lin is None:
        t_lin = np.zeros(2 * nb)
    J = engine.jac_packed(t_lin)
    m8 = J.shape[0] // 2
    Jc = J[:m8, :nb] + 1j * J[m8:, :nb]
    U, s, Vh = np.linalg.svd(Jc, full_matrices=False)
    keep = s >= s_cutoff_rel * (s[0] if s.size else 1.0)
    V = Vh.conj().T[:, keep]
    B_obs = np.tensordot(V.T, engine.B, axes=(1, 0))
    return B_obs, s, V


# ------------------------------------------------------------- fit driver

def fit_frequency(fm, ifreq, B, S_meas, angle_indices=None, weights=None,
                  t0=None, direction=DIRECTION_DEFAULT, tikhonov=0.0,
                  check_tol=1e-10, engine=None, **lsq_kwargs):
    """LM fit of the subspace T-matrix at one frequency (doc par. 4).

    Parameters
    ----------
    fm : forward.ForwardModel
    ifreq : int                    frequency index into the tmat.h5 grid
    B : (nb, n, n) real            subspace basis (parametrize; bright/full)
    S_meas : (n_sel, 2, 2, 2) complex
        Measured Jones blocks in the normative layout
        [angle, block 0=S11/1=S21, receive a, incident b], 0=TE/1=TM,
        matching angle_indices order.
    angle_indices : sequence of int or ANGLE_SETS name or None
        Rows of the normative 17-angle table (None -> all cached angles at
        ifreq).  Every requested angle must be cached (KeyError otherwise).
    weights : None, scalar, or array broadcastable to S_meas.shape
        Per-COMPLEX-observable weights w_i = 1/sigma_i^2 (pack_S semantics).
    t0 : (2*nb,) real or None      seed; None -> Born seed 0 (doc par. 4)
    direction : +-1                DEFAULT -1 (campaign convention)
    tikhonov : float >= 0
        Adds rows sqrt(tikhonov)*t to the residual (objective term
        tikhonov*||t||^2), pinning unobservable directions toward the seed
        0; switches the default method to 'trf'.
    check_tol : float              tolerance of the exactness assertions
    engine : AnalyticJacobian or None
        Prebuilt exact-Jacobian engine (must match fm/ifreq/B/angles/
        direction) for reuse across repeated fits; None -> built here.
    **lsq_kwargs                   forwarded to scipy least_squares
        ('method' overrides the default choice; jac='2-point' falls back
        to finite differences instead of the analytic engine)

    Returns dict with keys:
      t_hat, T0_hat, S_pred, S_meas, dS (complex per observable), abs_dS,
      objective (= sum_i w_i |dS_i|^2), objective_tik (tikhonov*||t||^2),
      cost_scipy, success/status/message/nfev/njev/optimality (convergence),
      passivity_max_sv (max SV(I + 2 T0_hat)), projector_resid,
      reciprocity_resid (both asserted <= check_tol * scale),
      wall_s, n_calls, ms_per_residual, method, angle_indices, ifreq,
      direction, tikhonov.
    """
    angle_indices = (resolve_angles(angle_indices)
                     if angle_indices is not None
                     else [int(a) for a in np.nonzero(fm.have[ifreq])[0]])
    missing = [a for a in angle_indices if not fm.have[ifreq, a]]
    if missing:
        raise KeyError("C not cached for ifreq=%d, angle indices %s"
                       % (ifreq, missing))
    B = np.asarray(B)
    nb = B.shape[0]
    npar = 2 * nb
    n_sel = len(angle_indices)
    S_meas = np.asarray(S_meas, dtype=complex)
    if S_meas.shape != (n_sel, 2, 2, 2):
        raise ValueError("S_meas shape %s != expected %s"
                         % (S_meas.shape, (n_sel, 2, 2, 2)))

    packed_meas = fm.pack_S(S_meas, weights)          # cached once
    sqrt_tik = float(np.sqrt(tikhonov)) if tikhonov > 0.0 else 0.0

    n_calls = [0]

    def residual(t):
        n_calls[0] += 1
        r = fm.predict_packed(par.unpack(t, B), ifreq, angle_indices,
                              weights, direction) - packed_meas
        if sqrt_tik:
            r = np.concatenate([r, sqrt_tik * t])
        return r

    if t0 is None:
        t0 = np.zeros(npar)
    t0 = np.asarray(t0, dtype=float)
    if t0.shape != (npar,):
        raise ValueError("t0 must have shape (%d,)" % npar)

    method = lsq_kwargs.pop("method", "lm" if tikhonov == 0.0 else "trf")
    n_rows = 16 * n_sel + (npar if sqrt_tik else 0)
    if method == "lm" and n_rows < npar:
        raise ValueError(
            "method 'lm' needs at least as many residual rows (%d) as "
            "parameters (%d); use method='trf' and/or tikhonov > 0 "
            "(e.g. the normal-only angle set with the bright basis)"
            % (n_rows, npar))

    jac_spec = lsq_kwargs.pop("jac", "analytic")
    if jac_spec == "analytic":
        if engine is None:
            engine = AnalyticJacobian(fm, ifreq, B, angle_indices,
                                      direction)
        if (engine.ifreq != ifreq or engine.angles != list(angle_indices)
                or engine.direction != int(direction)
                or engine.B.shape != B.shape):
            raise ValueError("engine does not match this fit's "
                             "ifreq/angles/direction/basis")

        if sqrt_tik:
            tik_block = np.sqrt(tikhonov) * np.eye(npar)

            def jac(t):
                return np.vstack([engine.jac_packed(t, weights), tik_block])
        else:
            def jac(t):
                return engine.jac_packed(t, weights)
    else:
        jac = jac_spec                      # scipy FD spec ('2-point', ...)

    t_start = time.time()
    res = least_squares(residual, t0, jac=jac, method=method, **lsq_kwargs)
    wall = time.time() - t_start

    t_hat = res.x
    T0_hat = par.unpack(t_hat, B)
    S_pred = fm.predict(T0_hat, ifreq, angle_indices, direction)
    dS = S_pred - S_meas
    if weights is None:
        w_arr = np.ones(S_meas.shape)
    else:
        w_arr = np.broadcast_to(np.asarray(weights, dtype=float),
                                S_meas.shape)
    objective = float(np.sum(w_arr * np.abs(dS) ** 2))
    objective_tik = float(tikhonov * np.dot(t_hat, t_hat)) if sqrt_tik \
        else 0.0

    # exact-by-construction symmetry: assert, don't just report
    Ds, perm, sign = sym_ops(fm.modes)
    scale = max(1.0, _maxabs(T0_hat))
    proj_resid = _maxabs(
        par.apply_full_projector(T0_hat, Ds, perm, sign) - T0_hat)
    rec_resid = _maxabs(par.apply_rec(T0_hat, perm, sign) - T0_hat)
    if proj_resid > check_tol * scale:
        raise AssertionError(
            "fitted T0 left the C4v x reciprocity subspace: "
            "max|P(T0)-T0| = %.3e > %.1e (impossible by construction -- "
            "corrupted basis?)" % (proj_resid, check_tol * scale))
    if rec_resid > check_tol * scale:
        raise AssertionError(
            "fitted T0 violates reciprocity: max|Rec(T0)-T0| = %.3e > %.1e"
            % (rec_resid, check_tol * scale))

    return dict(
        t_hat=t_hat, T0_hat=T0_hat, S_pred=S_pred, S_meas=S_meas, dS=dS,
        abs_dS=np.abs(dS), objective=objective, objective_tik=objective_tik,
        cost_scipy=float(res.cost), success=bool(res.success),
        status=int(res.status), message=res.message, nfev=int(res.nfev),
        njev=int(res.njev) if res.njev is not None else None,
        optimality=float(res.optimality),
        passivity_max_sv=par.passivity_max_sv(T0_hat[None]),
        projector_resid=proj_resid, reciprocity_resid=rec_resid,
        wall_s=wall, n_calls=int(n_calls[0]),
        ms_per_residual=1e3 * wall / max(n_calls[0], 1),
        method=method, jac="analytic" if jac_spec == "analytic" else
        str(jac_spec),
        engine_validation_resid=(engine.validation_resid
                                 if jac_spec == "analytic" else None),
        angle_indices=list(angle_indices), ifreq=int(ifreq),
        direction=int(direction), tikhonov=float(tikhonov),
    )


def fit_frequency_multistart(fm, ifreq, B, S_meas, angle_indices=None,
                             n_starts=3, seed=0, start_scale=None,
                             **kwargs):
    """Born seed + (n_starts - 1) random-seed restarts; returns the
    best-(objective + tikhonov-term) result (the doc-par.-9 local-minima
    mitigation: Born-dark directions enter S quadratically, so the Born
    basin can have sign-branch traps for weak targets).

    start_scale: rms scale of the random seeds (default: ||t_hat|| of the
    Born-seed solution / sqrt(n_params), floored at 1e-6).  Restarts are
    skipped when the Born solution already reaches objective <= 1e-20.
    """
    r0 = fit_frequency(fm, ifreq, B, S_meas, angle_indices, **kwargs)
    total0 = r0["objective"] + r0["objective_tik"]
    cands = [r0]
    if total0 > 1e-20:
        npar = 2 * np.asarray(B).shape[0]
        rng = np.random.default_rng([seed, int(ifreq), npar])
        scl = start_scale or max(np.linalg.norm(r0["t_hat"])
                                 / np.sqrt(npar), 1e-6)
        for _ in range(max(n_starts - 1, 0)):
            t0 = scl * rng.standard_normal(npar)
            cands.append(fit_frequency(fm, ifreq, B, S_meas, angle_indices,
                                       t0=t0, **kwargs))
    best = min(cands, key=lambda r: r["objective"] + r["objective_tik"])
    best["multistart_objectives"] = [r["objective"] + r["objective_tik"]
                                     for r in cands]
    best["multistart_pick"] = int(np.argmin(best["multistart_objectives"]))
    return best


def fit_band(fm, B, S_meas_stack, freq_indices, angle_indices=None,
             weights=None, t0=None, direction=DIRECTION_DEFAULT,
             tikhonov=0.0, verbose=True, **lsq_kwargs):
    """Loop fit_frequency over freq_indices with a progress line each.

    S_meas_stack : dict {ifreq: (n_sel, 2, 2, 2)} or array
        (len(freq_indices), n_sel, 2, 2, 2) aligned with freq_indices.
    Frequencies whose requested angles are not all cached are SKIPPED
    gracefully (so the same driver serves the currently-partial cache and
    the eventual full-band run).

    Returns (results, skipped): results = {ifreq: fit_frequency dict},
    skipped = [(ifreq, reason), ...].
    """
    angles = (resolve_angles(angle_indices) if angle_indices is not None
              else None)
    results = {}
    skipped = []
    for pos, ifreq in enumerate(freq_indices):
        ifreq = int(ifreq)
        want = angles if angles is not None else \
            [int(a) for a in np.nonzero(fm.have[ifreq])[0]]
        miss = [a for a in want if not fm.have[ifreq, a]]
        if miss or not want:
            reason = ("angles %s not cached" % miss) if miss else "no angles"
            skipped.append((ifreq, reason))
            if verbose:
                print("[fit_band] ifreq %2d lam=%6.2f um: SKIP (%s)"
                      % (ifreq, fm.lam_um[ifreq], reason), flush=True)
            continue
        S_meas = (S_meas_stack[ifreq] if isinstance(S_meas_stack, dict)
                  else S_meas_stack[pos])
        r = fit_frequency(fm, ifreq, B, S_meas, want, weights=weights,
                          t0=t0, direction=direction, tikhonov=tikhonov,
                          **dict(lsq_kwargs))
        results[ifreq] = r
        if verbose:
            print("[fit_band] ifreq %2d lam=%6.2f um: obj=%.4e nfev=%4d "
                  "wall=%6.2f s  passivity=%.4f  %s"
                  % (ifreq, fm.lam_um[ifreq], r["objective"], r["nfev"],
                     r["wall_s"], r["passivity_max_sv"],
                     "ok" if r["success"] else "NOT CONVERGED"), flush=True)
    return results, skipped


# -------------------------------------------------------- error metrics

def entry_errors(T_hat, T_tgt, entries, peak):
    """Light per-entry errors for one frequency (used per noise trial).

    entries : (nE, 2) int index pairs;  peak : (nE,) per-entry band-peak
    |T| normalization (from the full 49-frequency reference).
    Returns (rel_pointwise, rel_peaknorm), each (nE,):
        rel_pointwise = |dT_e| / |T_tgt_e|   (inf where T_tgt_e = 0, dT != 0)
        rel_peaknorm  = |dT_e| / peak_e      (robust near entry zeros)
    """
    ii, jj = entries[:, 0], entries[:, 1]
    dT = np.abs(T_hat[ii, jj] - T_tgt[ii, jj])
    den = np.abs(T_tgt[ii, jj])
    rel_pw = np.where(den > 0, dT / np.where(den > 0, den, 1.0),
                      np.where(dT > 0, np.inf, 0.0))
    pk = np.where(peak > 0, peak, 1.0)
    rel_pk = np.where(peak > 0, dT / pk, np.where(dT > 0, np.inf, 0.0))
    return rel_pw, rel_pk


def compare_to_reference(T_hat_stack, T_tgt_stack, mask, peak_ref_stack=None):
    """Bright-entry comparison metrics for the synthetic/real gates.

    T_hat_stack, T_tgt_stack : (nf_sel, n, n) complex, frequency-aligned.
    mask : (n, n) bool                     bright-entry mask
    peak_ref_stack : optional stack used ONLY for the per-entry band-peak
        normalization max_f |T[f, i, j]| (pass the FULL 49-frequency
        reference fm.data.T; default: T_tgt_stack, i.e. peaks over the
        fitted subset).

    Returns dict:
      entries       (nE, 2)  index pairs (row-major argwhere order)
      peak          (nE,)    band-peak |T| per entry (from peak_ref_stack)
      rel_pointwise (nf_sel, nE)   |dT|/|T_tgt| per freq (inf-guarded)
      rel_peaknorm  (nf_sel, nE)   |dT|/peak per freq
      peak_pos      (nE,)    argmax_f |T_tgt| within the FITTED subset
      rel_at_peak   (nE,)    pointwise rel at that per-entry peak frequency
      band_max_pointwise, band_max_peaknorm  (nE,)  max over fitted freqs
      frob_rel      (nf_sel,)   ||dT[mask]||_F / ||T_tgt[mask]||_F per freq
      max_rel_peaknorm, max_rel_at_peak      scalars (gate numbers)
    """
    T_hat_stack = np.asarray(T_hat_stack)
    T_tgt_stack = np.asarray(T_tgt_stack)
    if peak_ref_stack is None:
        peak_ref_stack = T_tgt_stack
    entries = np.argwhere(mask)
    ii, jj = entries[:, 0], entries[:, 1]
    peak = np.abs(np.asarray(peak_ref_stack)[:, ii, jj]).max(axis=0)

    nf = T_hat_stack.shape[0]
    rel_pw = np.empty((nf, len(entries)))
    rel_pk = np.empty((nf, len(entries)))
    frob = np.empty(nf)
    for f in range(nf):
        rel_pw[f], rel_pk[f] = entry_errors(T_hat_stack[f], T_tgt_stack[f],
                                            entries, peak)
        num = np.linalg.norm((T_hat_stack[f] - T_tgt_stack[f])[mask])
        den = np.linalg.norm(T_tgt_stack[f][mask])
        frob[f] = num / den if den > 0 else (np.inf if num > 0 else 0.0)
    peak_pos = np.abs(T_tgt_stack[:, ii, jj]).argmax(axis=0)
    rel_at_peak = rel_pw[peak_pos, np.arange(len(entries))]
    return dict(
        entries=entries, peak=peak, rel_pointwise=rel_pw,
        rel_peaknorm=rel_pk, peak_pos=peak_pos, rel_at_peak=rel_at_peak,
        band_max_pointwise=rel_pw.max(axis=0),
        band_max_peaknorm=rel_pk.max(axis=0), frob_rel=frob,
        max_rel_peaknorm=float(rel_pk.max()),
        max_rel_at_peak=float(rel_at_peak.max()),
    )


def format_compare(cmp, modes, top=None):
    """Per-entry table (label, peak |T|, rel at peak, band-max metrics)."""
    lines = ["  %-12s %9s %12s %12s %12s"
             % ("entry", "peak|T|", "rel@peak", "bandmax_pw", "bandmax_pk")]
    order = np.argsort(cmp["band_max_peaknorm"])[::-1]
    if top is not None:
        order = order[:top]
    for e in order:
        i, j = cmp["entries"][e]
        lines.append("  %-12s %9.2e %12.3e %12.3e %12.3e"
                     % (entry_label(modes, i, j), cmp["peak"][e],
                        cmp["rel_at_peak"][e],
                        cmp["band_max_pointwise"][e],
                        cmp["band_max_peaknorm"][e]))
    return "\n".join(lines)


if __name__ == "__main__":
    print(__doc__)
    print("Self-test: python test_fit_smoke.py")

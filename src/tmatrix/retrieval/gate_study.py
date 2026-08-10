"""Gate recalibration study for the multi-angle-Floquet -> T-matrix retrieval
(adjudication of INVERSE_TMATRIX_FROM_FLOQUET.md par. 6.2 / 6.3 / 8).

This module introduces NO new physics.  It re-uses forward.ForwardModel,
fit.fit_frequency / fit.AnalyticJacobian / fit.observable_subbasis /
fit.entry_errors / fit.classify_entries, parametrize.* and
observability.svd_resolution / fold_complex / entry_heatmap /
subspace_basis_for_entries.  Everything is synthetic: the "measured" data
is the forward model evaluated at the reference tmat.h5 T (no CST, no
license, no new solver).

WHAT EACH STUDY MEASURES -- AND WHAT IT DOES NOT
------------------------------------------------
Study 1 (span ladder).  MEASURES: for each candidate fit subspace (bright
  masks at |T| thresholds 1e-3 / 3e-4 / 1e-4 / 3e-5 of the global |T|max,
  plus the SVD-observable restriction of the full 68-dim symmetry basis at
  the reference linearization) its entry count, complex rank, orbit
  structure, the EXACT structural identity "the Frobenius projection of
  T_proj onto the span reproduces T_proj on the span's own mask entries",
  the resulting forward model-error floor max|S(T_proj) - S(P_span T_proj)|
  and its ratio to sigma, and the per-entry S-leverage
  |dS/dT_e|_max * |T_proj[e]| that says WHICH entries create that floor.
  DOES NOT MEASURE: what a fit actually achieves (Study 2); the leverage is
  a first-order quantity at the base point T_proj, not a finite-excursion
  statement.  For the observable restriction the Frobenius-projection floor
  is reported but is NOT a fit floor (the SVD restriction is not defined by
  entry support, so orthogonal projection discards physical content that a
  fit would instead re-absorb); it is printed flagged, and the fit residual
  in Study 2 is the number to use for that span.

Study 2 (estimator ladder, noise-free).  MEASURES: for each span x
  estimator x data variant x seed, the achieved objective, max|dS|, the
  excursion factor ||t_hat||/||t_true||, passivity max SV(I + 2T), and
  per-entry-class errors (both |dT|/band-peak|T| and |dT|/|T_true|) against
  the physical target.  Estimators: E0 unweighted/unregularized (the
  protocol synthetic_test.py used), E1 weighted, E2 isotropic Tikhonov at
  the physical prior, E3 weighted + isotropic Tikhonov, E4 orbit-scaled
  (diagonal) Tikhonov in an orbit-pure basis -- in an ORACLE-PRIMED variant
  (tau_k = |z_k^true|) and a REALIZABLE two-step variant (tau_k from a
  first-pass E3 fit).  Also: the basis-invariance measurement (SVD-
  orthonormal vs orbit-pure bright basis under the SAME isotropic penalty
  -- mathematically a no-op, so any difference is a Born-seed BASIN
  difference, not an estimator difference) and the linear shrinkage-bias
  decomposition of the strongest E-dipole entry over the weighted-Jacobian
  SVD directions.  DOES NOT MEASURE: noise response (Study 3), and it does
  not prove global optimality of any fit -- LM is a local method and
  max_nfev caps are reported per row (`capped` flag).

Study 3 (noise ladder).  MEASURES: per-class mean / median / 90th
  percentile of the per-trial MAX entry error over `--trials` seeded
  complex-Gaussian noise realizations at sigma in `--sigmas`, using the
  synthetic_test.py noise convention n = (sigma/sqrt 2)(g_re + i g_im) per
  COMPLEX observable, alongside the deterministic (noise-free) bias of the
  same protocol so bias vs noise can be separated.  DOES NOT MEASURE: any
  real CST noise -- sigma is the 3e-3 magnitude-only placeholder ladder
  until the par.-7 complex closure measures the true floor.

Study 4 (angle-set ladder).  MEASURES: per angle set, the CST cost
  (distinct (theta,phi) structure runs, one empty run per distinct theta),
  the number of observable complex directions of the chosen span at the
  PHYSICAL prior (observability.svd_resolution with prior_scale = tau,
  sigma = 3e-3) with per-direction res_c, and a Study-3-style noise ladder
  at sigma = 3e-3.  DOES NOT MEASURE: angles outside the 17 cached rows of
  precompute_C.ANGLES_DEG -- no uncached angle can be evaluated here.

Study 5 (gate proposal).  Emits results/GATE_STUDY.md: the measured tables,
  a proposed replacement for the doc's par.-6.2 / 6.3 / 8 gates as per-class
  numeric thresholds with a stated margin, an observability precondition
  (entries below a res_c cutoff are REPORTED, not gated), and a "what
  changed and why" section.  It is a PROPOSAL derived from synthetic data
  at the smoke frequencies only; it is not a measurement of real hardware.

Conventions (inherited, not re-derived -- see retrieval/HANDOFF.md):
  direction = -1 (campaign illumination); Jones index 0 = TE, 1 = TM; block
  0 = S11, 1 = S21; angle table = precompute_C.ANGLES_DEG (17 rows);
  sigma = 3e-3 is a PLACEHOLDER (magnitude-only closure) until the par.-7
  normal-incidence complex closure measures the complex floor.

CLI:
    python gate_study.py --freqs 32,48
    python gate_study.py --freqs 32,48 --quick          # smoke run
    python gate_study.py --freqs all --trials 20 --no-figs
Options: --sigmas, --angles, --direction, --seed, --trials, --quick,
--no-figs, --cont-span, --max-nfev.
"""
import argparse
import os
import sys
import time


import numpy as np                                       # noqa: E402
from scipy.optimize import least_squares                 # noqa: E402

from tmatrix.numerics import maxabs

from tmatrix.retrieval.forward import ForwardModel                         # noqa: E402
from tmatrix.retrieval import parametrize as par # noqa: E402
from tmatrix.retrieval import fit as fitmod # noqa: E402
from tmatrix.retrieval import observability as obs # noqa: E402

from tmatrix.paths import RETRIEVAL_RESULTS

RESULTS_DIR = str(RETRIEVAL_RESULTS)
SIGMA_REF = 3e-3                  # PLACEHOLDER (magnitude-only closure)
THRESHOLDS = (1e-3, 3e-4, 1e-4, 3e-5)
OBS_CUTOFF = 1e-4                 # s/s_max cutoff of the observable span
RES_C_GATE = 0.5                  # res_c above which a direction is "observable"

# Extended angle candidates (subsets of the 17 CACHED rows only).
#   ext_phi   = starter 5 + (30,45) + (60,45): completes phi at the theta
#               already paid for -> +2 structure runs, +0 empty runs.
#   ext_theta = campaign 13 + (20,0) + (40,0): denser in theta on the phi=0
#               mirror plane -> +2 structure runs, +2 empty runs.
EXT_ANGLE_SETS = {
    "ext_phi": [0, 4, 5, 6, 10, 11, 12],
    "ext_theta": list(range(13)) + [13, 15],
}

CLASS_NAMES = ("all25", "dipole", "dipole_odd", "dipole_even", "quad22",
               "even_m", "m33_11", "m33_33")


# ------------------------------------------------------------------ utils


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def sub(title):
    print("\n--- %s ---" % title, flush=True)


def class_selectors(modes, entries):
    """Boolean selectors over the bright-entry list (fit.classify_entries
    intersected with the entry list); 'all25' selects everything."""
    cls = fitmod.classify_entries(modes)
    sel = {}
    for name in ("dipole", "quad22", "even_m", "odd_m", "m33_11", "m33_33"):
        sel[name] = np.array([bool(cls[name][i, j]) for i, j in entries])
    sel["dipole_odd"] = sel["dipole"] & sel["odd_m"]
    sel["dipole_even"] = sel["dipole"] & sel["even_m"]
    sel["all25"] = np.ones(len(entries), dtype=bool)
    return sel


def class_stats(vals, sel):
    """{class: (max, median)} of a per-entry vector."""
    out = {}
    for name in CLASS_NAMES:
        s = sel[name]
        if s.any():
            v = np.asarray(vals)[s]
            out[name] = (float(np.nanmax(v)), float(np.nanmedian(v)))
        else:
            out[name] = (np.nan, np.nan)
    return out


def _cn(n):
    return n.replace("dipole", "dip").replace("_", "")


def _v(x):
    return "  n/a   " if not np.isfinite(x) else "%.2e" % x


def fmt_class_line(tag, st):
    return ("      %-9s " % tag) + " ".join(
        "%s=%s" % (_cn(n), _v(st[n][0])) for n in CLASS_NAMES)


def cst_run_counts(fm, idx):
    """(n_structure_runs, n_empty_runs) implied by an angle-index list.
    theta = 0 collapses to a single (0,0) pair; the empty cell depends only
    on k_z, so one empty run per distinct theta (doc par. 7)."""
    pairs, thetas = set(), set()
    for a in idx:
        th = float(fm.theta_deg[a])
        ph = 0.0 if th == 0.0 else float(fm.phi_deg[a])
        pairs.add((th, ph))
        thetas.add(th)
    return len(pairs), len(thetas)


def all_angle_sets():
    d = dict(fitmod.ANGLE_SETS)
    d.update(EXT_ANGLE_SETS)
    return d


# ------------------------------------------------------- orbit-pure basis

def orbit_pure_basis(B68, mask, modes, tol=1e-10):
    """Orbit-pure real basis of span{P(E_ij) : (i,j) in mask}: ONE
    normalized direction per position orbit (Klein four-group generated by
    sigma_v conjugation and reciprocity; parametrize.bright_orbits).

    The orbits have DISJOINT entry supports (the C4 average is diagonal in
    position; sigma_v and reciprocity permute positions), so the normalized
    P(E_rep) are automatically Frobenius-orthonormal -- asserted here, not
    assumed.  Orbits annihilated by the projector (C4-violating, or killed
    by the reciprocity sign) drop out, so rank <= number of orbits.

    Returns (B_orb (r, n, n) real, orbits_kept, diag) with diag carrying the
    Gram residual, the support-overlap count and the per-orbit norms.
    """
    perm = par.mflip_perm(modes)
    orbits, closed = par.bright_orbits(mask, perm)
    n = modes.n
    rows, kept, norms = [], [], []
    for o in orbits:
        i, j = o[0]
        E = np.zeros((n, n))
        E[i, j] = 1.0
        P = par.unpack(par.pack(E, B68), B68)
        if maxabs(P.imag) > 1e-12:
            raise AssertionError("P(E_ij) is not real (basis corrupted)")
        P = P.real
        nrm = float(np.linalg.norm(P))
        norms.append(nrm)
        if nrm <= tol:
            continue
        rows.append(P / nrm)
        kept.append(o)
    B_orb = np.array(rows)
    r = B_orb.shape[0]
    flat = B_orb.reshape(r, -1)
    gram = maxabs(flat @ flat.T - np.eye(r))
    supp = (np.abs(B_orb) > 1e-14).sum(axis=0)
    overlap = int((supp > 1).sum())
    if gram > 1e-10:
        raise AssertionError("orbit-pure basis is not orthonormal: "
                             "max|Gram - I| = %.3e" % gram)
    if overlap:
        raise AssertionError("orbit-pure basis has %d overlapping-support "
                             "entries (orbits should be disjoint)" % overlap)
    return B_orb, kept, dict(gram_resid=gram, support_overlap=overlap,
                             orbit_norms=np.array(norms),
                             mask_closed=bool(closed))


def span_residual(B_a, B_b):
    """||B_a - Proj_{span(B_b)} B_a||_F (both real, rows orthonormal)."""
    A = np.asarray(B_a).reshape(np.asarray(B_a).shape[0], -1)
    Bm = np.asarray(B_b).reshape(np.asarray(B_b).shape[0], -1)
    return float(np.linalg.norm(A - (A @ Bm.T) @ Bm))


# ------------------------------------------- fits with a diagonal prior

def fit_diag(fm, ifreq, B, S_meas, angles, engine, weights=None,
             tik_vec=None, t0=None, direction=-1, max_nfev=500,
             tol=1e-12, method="lm"):
    """LM fit with a PER-DIRECTION (diagonal) Tikhonov prior.

    fit.fit_frequency only accepts a SCALAR tikhonov, so E4 lives here.
    The machinery is otherwise identical: residual =
    fm.predict_packed(unpack(t, B), ...) - pack_S(S_meas, w), augmented
    with rows sqrt(d) * t, and the Jacobian is engine.jac_packed(t, w)
    stacked with diag(sqrt(d)).  tik_vec is per COMPLEX direction; the real
    parameter vector t = [Re z; Im z] gets d = concat(tik_vec, tik_vec).
    """
    B = np.asarray(B)
    nb = B.shape[0]
    npar = 2 * nb
    packed_meas = fm.pack_S(np.asarray(S_meas, dtype=complex), weights)
    if tik_vec is None:
        d = np.zeros(npar)
    else:
        tv = np.asarray(tik_vec, dtype=float)
        if tv.shape != (nb,):
            raise ValueError("tik_vec must have length nb = %d" % nb)
        d = np.concatenate([tv, tv])
    sq = np.sqrt(d)
    use_tik = bool(np.any(d > 0))
    Dblk = np.diag(sq)
    ncalls = [0]

    def residual(t):
        ncalls[0] += 1
        r = fm.predict_packed(par.unpack(t, B), ifreq, angles, weights,
                              direction) - packed_meas
        return np.concatenate([r, sq * t]) if use_tik else r

    def jac(t):
        J = engine.jac_packed(t, weights)
        return np.vstack([J, Dblk]) if use_tik else J

    t0 = np.zeros(npar) if t0 is None else np.asarray(t0, dtype=float)
    tstart = time.time()
    res = least_squares(residual, t0, jac=jac, method=method, xtol=tol,
                        ftol=tol, gtol=tol, max_nfev=max_nfev)
    wall = time.time() - tstart
    T0 = par.unpack(res.x, B)
    S_pred = fm.predict(T0, ifreq, angles, direction)
    dS = S_pred - np.asarray(S_meas, dtype=complex)
    w_arr = (np.ones(dS.shape) if weights is None else
             np.broadcast_to(np.asarray(weights, dtype=float), dS.shape))
    return dict(t_hat=res.x, T0_hat=T0, dS=dS,
                objective=float(np.sum(w_arr * np.abs(dS) ** 2)),
                objective_tik=float(np.dot(d * res.x, res.x)),
                nfev=int(res.nfev), success=bool(res.success), wall_s=wall,
                passivity_max_sv=par.passivity_max_sv(T0[None]),
                capped=bool(res.nfev >= max_nfev))


def norm_result(r, max_nfev):
    """Uniform view over fit.fit_frequency and fit_diag results."""
    return dict(t_hat=r["t_hat"], T0_hat=r["T0_hat"],
                objective=r["objective"],
                maxdS=float(np.abs(r["dS"]).max()), nfev=int(r["nfev"]),
                wall_s=float(r["wall_s"]),
                passivity=float(r["passivity_max_sv"]),
                capped=bool(r.get("capped", r["nfev"] >= max_nfev)))


def predicted_entry_rms(engine, B, t_true, tik_diag, weights, entries):
    """Linear-theory per-entry |dT| rms of the (possibly diagonal-)Tikhonov
    estimator at the truth linearization: sqrt(|bias|^2 + var).

    Same construction as synthetic_test.py's step-3 theory gate (which
    passes with median measured/predicted 0.81-1.93): with the weighted
    Jacobian Jw the packed residual noise has covariance (1/2) I, so
    Cov(t_hat) = (1/2) A^-1 Jw^T Jw A^-1 with A = Jw^T Jw + diag(tik), and
    bias(t_hat) = -A^-1 diag(tik) t_true.  Returns (pred, bias, var).
    """
    B = np.asarray(B)
    Jw = engine.jac_packed(t_true, weights)
    npar = t_true.size
    A = Jw.T @ Jw + np.diag(tik_diag)
    Ainv = np.linalg.inv(A)
    Cov = 0.5 * Ainv @ (Jw.T @ Jw) @ Ainv
    bias_t = -Ainv @ (tik_diag * t_true)
    nb = npar // 2
    pred = np.empty(len(entries))
    bias = np.empty(len(entries))
    var = np.empty(len(entries))
    for e, (i, j) in enumerate(entries):
        col = np.asarray(B[:, i, j], dtype=complex)
        # T[i,j] = sum_k (t_k + i t_{k+nb}) B_k[i,j]
        gr = np.concatenate([col.real, -col.imag])
        gi = np.concatenate([col.imag, col.real])
        b_e = complex(gr @ bias_t, gi @ bias_t)
        v_e = float(gr @ Cov @ gr + gi @ Cov @ gi)
        bias[e] = abs(b_e)
        var[e] = v_e
        pred[e] = np.sqrt(abs(b_e) ** 2 + v_e)
    return pred, bias, var


# ==================================================================== S1

def build_spans(fm, ifreq, B68, eng68, T_proj, angles, direction, sigma,
                thresholds=THRESHOLDS, obs_cutoff=OBS_CUTOFF, verbose=True):
    """Study 1: the span ladder with the structural-identity assertion."""
    modes = fm.modes
    spans = []
    S_proj = fm.predict(T_proj, ifreq, angles, direction)

    def add(name, kind, B, mask, thr, extra):
        B = np.asarray(B)
        T_span = par.unpack(par.pack(T_proj, B), B)
        S_span = fm.predict(T_span, ifreq, angles, direction)
        dS = S_proj - S_span
        if mask is not None:
            num = maxabs(T_span[mask] - T_proj[mask])
            den = max(maxabs(T_proj[mask]), 1e-300)
            ident = num / den
        else:
            m25 = spans[0]["mask"] if spans else None
            ident = (maxabs(T_span[m25] - T_proj[m25])
                     / max(maxabs(T_proj[m25]), 1e-300)
                     if m25 is not None else np.nan)
        t_true = par.pack(T_span, B)
        tau = float(np.linalg.norm(t_true) / np.sqrt(t_true.size))
        d = dict(name=name, kind=kind, threshold=thr, B=B, mask=mask,
                 rank=B.shape[0], T_span=T_span, t_true=t_true, tau=tau,
                 ident_rel=float(ident), floor_max=maxabs(dS),
                 floor_rms=float(np.sqrt((np.abs(dS) ** 2).mean())),
                 n_entries=(int(mask.sum()) if mask is not None else -1))
        d["floor_over_sigma"] = d["floor_max"] / sigma
        d.update(extra)
        spans.append(d)
        return d

    for thr in thresholds:
        Bb, info = par.bright_orbit_basis(B68, fm.data.T, threshold=thr,
                                          modes=modes)
        B_orb, orbits_kept, odiag = orbit_pure_basis(B68, info["mask"], modes)
        if B_orb.shape[0] != Bb.shape[0]:
            raise AssertionError(
                "orbit-pure rank %d != SVD-span rank %d at threshold %.1e"
                % (B_orb.shape[0], Bb.shape[0], thr))
        sres = span_residual(B_orb, Bb)
        if sres > 1e-10:
            raise AssertionError("orbit-pure basis does not span the same "
                                 "space (resid %.3e)" % sres)
        d = add("bright%.0e" % thr, "bright", Bb, info["mask"], thr,
                dict(n_orbits=info["n_orbits"],
                     n_orbits_c4=info["n_orbits_c4_conforming"],
                     B_orbit=B_orb, orbits=orbits_kept,
                     orbit_gram=odiag["gram_resid"],
                     orbit_support_overlap=odiag["support_overlap"],
                     span_resid_orbit_vs_svd=sres))
        # STRUCTURAL IDENTITY -- raise, do not merely report
        if d["ident_rel"] > 1e-12:
            raise AssertionError(
                "structural identity VIOLATED for span %s: the projection "
                "of T_proj onto the span differs from T_proj on the span's "
                "own mask entries by %.3e relative (> 1e-12).  Any bright-"
                "entry recovery failure would then be a representability "
                "problem, contradicting the verified structural fact."
                % (d["name"], d["ident_rel"]))

    B_obs, s_ref, V_obs = fitmod.observable_subbasis(
        eng68, t_lin=par.pack(T_proj, B68), s_cutoff_rel=obs_cutoff)
    add("obs68", "obs", B_obs, None, obs_cutoff,
        dict(n_orbits=-1, n_orbits_c4=-1, B_orbit=None,
             s_ref=s_ref, V_obs=V_obs))

    if verbose:
        sub("Study 1: span ladder (ifreq %d, lam %.2f um, %d angles, "
            "sigma = %.1e)" % (ifreq, fm.lam_um[ifreq], len(angles), sigma))
        print("  %-11s %6s %6s %7s %7s %11s %11s %11s %8s"
              % ("span", "entries", "rank", "orbits", "orb_C4",
                 "ident_rel", "floor_max", "floor_rms", "flr/sig"))
        for d in spans:
            print("  %-11s %6s %6d %7s %7s %11.2e %11.3e %11.3e %8.3f"
                  % (d["name"],
                     d["n_entries"] if d["n_entries"] >= 0 else "-",
                     d["rank"],
                     d["n_orbits"] if d["n_orbits"] >= 0 else "-",
                     d["n_orbits_c4"] if d["n_orbits_c4"] >= 0 else "-",
                     d["ident_rel"], d["floor_max"], d["floor_rms"],
                     d["floor_over_sigma"]))
        print("  [PASS] structural identity asserted <= 1e-12 relative on "
              "every MASK-DEFINED span (worst %.2e): the bright spans can "
              "represent their own bright content EXACTLY, so every bright-"
              "entry failure below is an ESTIMATOR/INFORMATION problem."
              % max(d["ident_rel"] for d in spans if d["mask"] is not None))
        print("  [NOTE] obs68 is NOT mask-defined: its ident_rel is measured "
              "on the bright-25 entries and is LARGE by construction (the "
              "SVD restriction is chosen by data leverage, not entry "
              "support), so its 'floor' row is a Frobenius-projection "
              "number, NOT a fit floor -- use the Study-2 max|dS| for it.")
        for d in spans:
            if d["kind"] == "bright":
                print("  orbit-pure check %-11s: rank %2d == SVD rank, "
                      "max|Gram-I| = %.1e, support overlaps = %d, "
                      "||B_orb - P_span(B_svd) B_orb||_F = %.1e"
                      % (d["name"], d["rank"], d["orbit_gram"],
                         d["orbit_support_overlap"],
                         d["span_resid_orbit_vs_svd"]))
    return spans


def leverage_table(fm, ifreq, angles, T_proj, direction, spans,
                   mask25, top=15, verbose=True):
    """Per-entry S-leverage |dS/dT_e|_max * |T_proj[e]| at base T_proj."""
    modes = fm.modes
    conf = fitmod.classify_entries(modes)["c4_conforming"]
    supp = conf & (np.abs(T_proj) > 1e-14 * maxabs(T_proj))
    entries = np.argwhere(supp)
    _, summ = obs.entry_sensitivity(fm, ifreq, entries, angles, T_proj,
                                    direction)
    sens = summ["max_abs"]
    amp = np.abs(T_proj[entries[:, 0], entries[:, 1]])
    lev = sens * amp
    order = np.argsort(lev)[::-1]
    in25 = np.array([bool(mask25[i, j]) for i, j in entries])
    if verbose:
        sub("Study 1b: S-leverage |dS/dT_e|_max * |T_proj[e]| "
            "(base T_proj, %d C4-conforming support entries)" % len(entries))
        print("  %-16s %9s %11s %11s %11s %6s"
              % ("entry", "|T_proj|", "|dS/dT|max", "leverage", "cumfrac",
                 "in25"))
        tot = lev.sum()
        cum = 0.0
        for e in order[:top]:
            cum += lev[e]
            i, j = entries[e]
            print("  %-16s %9.2e %11.3e %11.3e %11.3f %6s"
                  % (fitmod.entry_label(modes, i, j), amp[e], sens[e],
                     lev[e], cum / tot, "yes" if in25[e] else "NO"))
        print("  total leverage %.3e; bright-25 share %.3f, "
              "sub-threshold share %.3f"
              % (tot, lev[in25].sum() / tot, lev[~in25].sum() / tot))
        print("  per-span capture of the leverage NOT in the span's mask "
              "(this is what creates the model-error floor):")
        for d in spans:
            if d["mask"] is None:
                continue
            inm = np.array([bool(d["mask"][i, j]) for i, j in entries])
            out_lev = lev[~inm]
            if out_lev.size:
                k = np.argmax(out_lev)
                oi, oj = entries[~inm][k]
                worst = "%s (%.2e)" % (fitmod.entry_label(modes, oi, oj),
                                       out_lev[k])
            else:
                worst = "-"
            print("    %-11s captures %.4f of total leverage; missed "
                  "leverage %.3e; worst missed entry %s"
                  % (d["name"], lev[inm].sum() / tot, out_lev.sum(), worst))
    return dict(entries=entries, sens=sens, amp=amp, lev=lev, in25=in25)


# ==================================================================== S2

def run_estimator(fm, ifreq, span, S_meas, T_tgt, entries, peak, sel,
                  angles, direction, engine, est, seed_mode, sigma,
                  max_nfev, tol=1e-12, tik_vec=None, B_use=None):
    """One (span, estimator, data variant, seed) fit -> summary dict."""
    B = span["B"] if B_use is None else B_use
    t_true = par.pack(par.unpack(par.pack(T_tgt, B), B), B)
    tau = float(np.linalg.norm(t_true) / np.sqrt(t_true.size))
    tau = max(tau, 1e-12)
    t0 = t_true if seed_mode == "truth" else None
    w = 1.0 / sigma ** 2
    if est in ("E0", "E1", "E2", "E3"):
        kw = {}
        if est in ("E1", "E3"):
            kw["weights"] = w
        if est in ("E2", "E3"):
            kw["tikhonov"] = 1.0 / tau ** 2
            kw["method"] = "lm"
        r = fitmod.fit_frequency(fm, ifreq, B, S_meas, angles, t0=t0,
                                 direction=direction, engine=engine,
                                 max_nfev=max_nfev, xtol=tol, ftol=tol,
                                 gtol=tol, **kw)
        out = norm_result(r, max_nfev)
    elif est.startswith("E4"):
        r = fit_diag(fm, ifreq, B, S_meas, angles, engine, weights=w,
                     tik_vec=tik_vec, t0=t0, direction=direction,
                     max_nfev=max_nfev, tol=tol)
        out = norm_result(r, max_nfev)
    else:
        raise ValueError(est)
    pw, pk = fitmod.entry_errors(out["T0_hat"], T_tgt, entries, peak)
    out.update(est=est, seed=seed_mode, span=span["name"],
               excursion=float(np.linalg.norm(out["t_hat"])
                               / max(np.linalg.norm(t_true), 1e-300)),
               rel_peaknorm=pk, rel_pointwise=pw,
               pk_stats=class_stats(pk, sel), pw_stats=class_stats(pw, sel),
               tau=tau)
    return out


def print_row(r):
    print("  %-11s %-10s %-5s obj=%.3e maxdS=%.2e |t|/|t*|=%7.3f "
          "passiv=%6.3f nfev=%4d%s wall=%5.1fs"
          % (r["span"], r["est"], r["seed"], r["objective"], r["maxdS"],
             r["excursion"], r["passivity"], r["nfev"],
             "*" if r["capped"] else " ", r["wall_s"]), flush=True)
    print(fmt_class_line("pk max", r["pk_stats"]))
    print(fmt_class_line("pw max", r["pw_stats"]))


def bias_decomposition(engine, B, t_true, tik, weights, entry, modes,
                       top=6, verbose=True):
    """Linear shrinkage-bias decomposition of Re T[entry] over the SVD
    directions of the WEIGHTED Jacobian:
        b = - sum_j [tik/(s_j^2+tik)] (g . v_j)(v_j . t_true).
    Distinguishes 'genuinely unobservable' from 'the isotropic prior
    imports bias from dark directions carrying large true coefficients'."""
    i, j = entry
    Jw = engine.jac_packed(t_true, weights)
    U, s, Vt = np.linalg.svd(Jw, full_matrices=False)
    V = Vt.T
    nb = np.asarray(B).shape[0]
    g = np.concatenate([np.asarray(B)[:, i, j], np.zeros(nb)])  # Re part
    shrink = tik / (s ** 2 + tik)
    gv = g @ V
    vt = V.T @ t_true
    contrib = -shrink * gv * vt
    total = float(contrib.sum())
    order = np.argsort(np.abs(contrib))[::-1][:top]
    if verbose:
        print("  bias decomposition of Re T[%s] (isotropic tik, "
              "sqrt(tik) = %.3g, weighted J at truth): total bias = %.3e"
              % (fitmod.entry_label(modes, i, j), np.sqrt(tik), total))
        print("      %4s %11s %10s %11s %12s %12s"
              % ("j", "s_j", "shrink", "g.v_j", "v_j.t_true", "contrib"))
        for jj in order:
            print("      %4d %11.4g %10.4f %11.4f %12.4e %12.4e"
                  % (jj, s[jj], shrink[jj], gv[jj], vt[jj], contrib[jj]))
    return dict(total=total, s=s, shrink=shrink, gv=gv, vt=vt,
                contrib=contrib)


# ==================================================================== S3

def _est_kwargs(est, sigma, tau, n_rows, npar):
    """scipy/fit kwargs for the E0-E3 estimator family (see Study 2)."""
    kw = {}
    if est in ("E1", "E3"):
        kw["weights"] = 1.0 / sigma ** 2
    if est in ("E2", "E3"):
        kw["tikhonov"] = 1.0 / tau ** 2
        kw["method"] = "lm"
    elif n_rows < npar:
        kw["method"] = "trf"       # 'lm' needs rows >= params
    return kw


class SpanCtx:
    """Per-(basis, angle-set) cache of everything a fit at ANY frequency
    needs: the analytic-Jacobian engine, the projected target, its packed
    coefficients, the isotropic prior scale tau and the per-orbit prior
    scales tau_k, and the noise-free measured stack S(P68(T_ref[f])).

    This is what makes frequency-CONTINUATION affordable: a continuation
    sweep touches several frequencies, and each of them needs its own
    engine, which is the expensive object (doc: 'reuse ONE AnalyticJacobian
    per (basis, angle-set, frequency)').
    """

    def __init__(self, fm, B68, B, key, angles, direction):
        self.fm = fm
        self.B68 = B68
        self.B = np.asarray(B)
        self.key = key
        self.angles = list(angles)
        self.direction = int(direction)
        self._f = {}

    def at(self, f):
        f = int(f)
        if f not in self._f:
            fm = self.fm
            Tp = par.unpack(par.pack(fm.data.T[f], self.B68), self.B68)
            t_true = par.pack(par.unpack(par.pack(Tp, self.B), self.B),
                              self.B)
            nb = self.B.shape[0]
            z = t_true[:nb] + 1j * t_true[nb:]
            tau_k = np.maximum(np.abs(z),
                               1e-6 * max(np.abs(z).max(), 1e-300))
            self._f[f] = dict(
                T_proj=Tp, t_true=t_true, tau_k=tau_k,
                tau=max(float(np.linalg.norm(t_true)
                              / np.sqrt(t_true.size)), 1e-12),
                engine=fitmod.AnalyticJacobian(fm, f, self.B, self.angles,
                                               self.direction),
                S_base=fm.predict(Tp, f, self.angles, self.direction))
        return self._f[f]


def _chain_freqs(fm, ifreq, angles, chain_len):
    """Contiguous frequency chain of length <= chain_len centred on ifreq,
    restricted to frequencies whose requested angles are all cached."""
    h = (int(chain_len) - 1) // 2
    fs = [f for f in range(ifreq - h, ifreq + h + 1)
          if 0 <= f < fm.nf and all(fm.have[f, a] for a in angles)]
    return fs if ifreq in fs else [ifreq]


def single_fit(ctx, f, est, S, t0, sigma, max_nfev, tol):
    """One local LM fit at frequency f in ctx's basis/angle set.

    E4o  -- diagonal prior tik_k = 1/|z_k^true|^2 from ctx (ORACLE; only
            realizable in the QA-gate use case where a candidate T is
            supplied).
    E4r  -- REALIZABLE two-step at THIS frequency: an isotropic-prior fit
            first, then tau_k = |z_k| of that fit (floored) as the diagonal
            prior, warm-started from it.  Nothing here uses the truth
            except the scalar tau of the isotropic step, which is the
            doc-par.-4 'physical prior' convention already used by
            synthetic_test.py.
    """
    c = ctx.at(f)
    B = ctx.B
    nb = B.shape[0]
    npar = 2 * nb
    if est == "E4o":
        return fit_diag(ctx.fm, f, B, S, ctx.angles, c["engine"],
                        weights=1.0 / sigma ** 2,
                        tik_vec=1.0 / c["tau_k"] ** 2, t0=t0,
                        direction=ctx.direction, max_nfev=max_nfev, tol=tol)
    if est == "E4r":
        r1 = fitmod.fit_frequency(
            ctx.fm, f, B, S, ctx.angles, t0=t0, weights=1.0 / sigma ** 2,
            tikhonov=1.0 / c["tau"] ** 2, method="lm",
            direction=ctx.direction, engine=c["engine"],
            max_nfev=max_nfev, xtol=tol, ftol=tol, gtol=tol)
        z1 = r1["t_hat"][:nb] + 1j * r1["t_hat"][nb:]
        tk = np.maximum(np.abs(z1), 1e-3 * max(np.abs(z1).max(), 1e-30))
        r2 = fit_diag(ctx.fm, f, B, S, ctx.angles, c["engine"],
                      weights=1.0 / sigma ** 2, tik_vec=1.0 / tk ** 2,
                      t0=r1["t_hat"], direction=ctx.direction,
                      max_nfev=max_nfev, tol=tol)
        r2["nfev"] = int(r1["nfev"]) + int(r2["nfev"])
        r2["capped"] = bool(r2["capped"] or r1["nfev"] >= max_nfev)
        return r2
    if est.startswith("E4"):
        raise ValueError("unknown diagonal-prior estimator %r" % est)
    if est == "E5":
        # continuation-native prior: handled by protocol_trial, which
        # supplies z_prev; without one it degenerates to E3.
        est = "E3"
    kw = _est_kwargs(est, sigma, c["tau"], 16 * len(ctx.angles), npar)
    return fitmod.fit_frequency(ctx.fm, f, B, S, ctx.angles, t0=t0,
                                direction=ctx.direction, engine=c["engine"],
                                max_nfev=max_nfev, xtol=tol, ftol=tol,
                                gtol=tol, **kw)


def protocol_trial(ctx, ifreq, est, seed_mode, sigma, seed, tag, tr,
                   chain_len, max_nfev, tol, noise_scale=1.0):
    """One realization of a full PROTOCOL = (basis, estimator, SEED MODE).

    seed_mode:
      'born'  -- single fit at ifreq from t = 0 (the doc-par.-4 Born seed);
      'truth' -- single fit at ifreq seeded at the projected target (the
                 ACHIEVABILITY BOUND, not a protocol);
      'cont'  -- FREQUENCY CONTINUATION: a contiguous chain centred on
                 ifreq is swept in BOTH directions, each fit seeded with
                 the previous frequency's t_hat (the first fit of each
                 sweep uses the Born seed), and the result kept at ifreq is
                 the one with the smaller total objective.  Nothing in this
                 uses the truth: it is realizable on real data, and it is
                 what a real 49-frequency campaign sweep does anyway.

    Noise is drawn per FREQUENCY with key [seed, f, tag, sigma, trial], so
    a 'born' trial and the ifreq step of a 'cont' trial see the SAME noise
    realization -- the seed comparison is paired.
    """
    def data(f):
        c = ctx.at(f)
        if noise_scale == 0.0:
            return c["S_base"]
        rng = np.random.default_rng([seed, int(f), tag,
                                     int(round(sigma * 1e12)), tr])
        n = (rng.standard_normal(c["S_base"].shape)
             + 1j * rng.standard_normal(c["S_base"].shape)) \
            * (noise_scale * sigma / np.sqrt(2.0))
        return c["S_base"] + n

    nb = ctx.B.shape[0]

    def e5_fit(f, S, t0, z_prev):
        """E5: diagonal prior whose per-orbit scale comes from the PREVIOUS
        FREQUENCY's estimate.  Fully realizable (a sweep has it for free)
        and far better conditioned than same-frequency empirical Bayes,
        which has to estimate the prior from the very fit it regularizes."""
        c = ctx.at(f)
        if z_prev is None:
            return single_fit(ctx, f, "E3", S, t0, sigma, max_nfev, tol)
        tk = np.maximum(np.abs(z_prev),
                        1e-3 * max(np.abs(z_prev).max(), 1e-30))
        return fit_diag(ctx.fm, f, ctx.B, S, ctx.angles, c["engine"],
                        weights=1.0 / sigma ** 2, tik_vec=1.0 / tk ** 2,
                        t0=t0, direction=ctx.direction, max_nfev=max_nfev,
                        tol=tol)

    if seed_mode in ("born", "truth"):
        t0 = ctx.at(ifreq)["t_true"] if seed_mode == "truth" else None
        return single_fit(ctx, ifreq, est, data(ifreq), t0, sigma,
                          max_nfev, tol), 1, [int(ifreq)]
    fs = _chain_freqs(ctx.fm, ifreq, ctx.angles, chain_len)
    best = None
    nfits = 0
    for order in (fs, fs[::-1]):
        t0 = None
        z_prev = None
        for f in order:
            if est == "E5":
                r = e5_fit(f, data(f), t0, z_prev)
            else:
                r = single_fit(ctx, f, est, data(f), t0, sigma, max_nfev,
                               tol)
            t0 = r["t_hat"]
            z_prev = t0[:nb] + 1j * t0[nb:]
            nfits += 1
            if f == ifreq:
                tot = r["objective"] + r.get("objective_tik", 0.0)
                if best is None or tot < best[0]:
                    best = (tot, r)
    return best[1], nfits, fs


def noise_trials(ctx, ifreq, entries, peak, sel, sigma, trials, seed, est,
                 seed_mode="born", tag=0, chain_len=5, max_nfev=300,
                 tol=1e-10, noise_scale=1.0):
    """`trials` seeded realizations of one protocol (synthetic_test noise
    convention n = (sigma/sqrt 2)(g_re + i g_im) per COMPLEX observable)."""
    nE = len(entries)
    pk = np.full((trials, nE), np.nan)
    pw = np.full((trials, nE), np.nan)
    objs = np.full(trials, np.nan)
    pas = np.full(trials, np.nan)
    exc = np.full(trials, np.nan)
    ncap = 0
    nfits = 0
    c = ctx.at(ifreq)
    T_tgt = c["T_proj"]
    nrm_true = max(float(np.linalg.norm(c["t_true"])), 1e-300)
    chain = [ifreq]
    for tr in range(trials):
        r, nf, chain = protocol_trial(ctx, ifreq, est, seed_mode, sigma,
                                      seed, tag, tr, chain_len, max_nfev,
                                      tol, noise_scale)
        rr = norm_result(r, max_nfev)
        pw[tr], pk[tr] = fitmod.entry_errors(rr["T0_hat"], T_tgt, entries,
                                             peak)
        objs[tr] = rr["objective"]
        pas[tr] = rr["passivity"]
        exc[tr] = np.linalg.norm(rr["t_hat"]) / nrm_true
        ncap += int(rr["capped"])
        nfits += nf
    return dict(pk=pk, pw=pw, obj=objs, passivity=pas, n_capped=ncap,
                excursion=exc, n_fits=nfits, chain=chain)


def trial_class_stats(pk, sel):
    """{class: (mean, median, p90)} of the per-trial MAX error in class."""
    out = {}
    for name in CLASS_NAMES:
        s = sel[name]
        if not s.any():
            out[name] = (np.nan, np.nan, np.nan)
            continue
        per_trial = np.nanmax(pk[:, s], axis=1)
        out[name] = (float(np.mean(per_trial)), float(np.median(per_trial)),
                     float(np.percentile(per_trial, 90)))
    return out


def fmt_trial_line(tag, st, which=0):
    return ("      %-11s " % tag) + " ".join(
        "%s=%s" % (_cn(n), _v(st[n][which])) for n in CLASS_NAMES)


def fmt_gain_line(st, st_zero, which=0):
    """Gain over the shrink-to-zero estimator (>1 = better than returning
    nothing; <=1 = the protocol is worthless for that class)."""
    return ("      %-11s " % "gain/zero") + " ".join(
        "%s=%s" % (_cn(n), _v(st_zero[n][0] / st[n][which])
                   if np.isfinite(st[n][which]) and st[n][which] > 0
                   else np.nan)
        for n in CLASS_NAMES)


# ==================================================================== main

def run_frequency(fm, ifreq, B68, args, store):
    modes = fm.modes
    direction = args.direction
    sigma = SIGMA_REF
    angles = fitmod.resolve_angles(args.angles)
    T_ref = fm.data.T[ifreq]
    T_proj = par.unpack(par.pack(T_ref, B68), B68)
    mask25 = par.bright_mask(fm.data.T, 1e-3)
    entries = np.argwhere(mask25)
    peak = np.abs(fm.data.T[:, entries[:, 0], entries[:, 1]]).max(axis=0)
    sel = class_selectors(modes, entries)
    Tmax_global = maxabs(fm.data.T)
    out = dict(ifreq=ifreq, lam_um=float(fm.lam_um[ifreq]), entries=entries,
               peak=peak, mask25=mask25, angle_indices=np.array(angles),
               direction=direction, sigma=sigma, Tmax_global=Tmax_global)

    hr("ifreq %d  (lam = %.3f um)  angles=%s (%d)  direction %+d  "
       "sigma = %.1e" % (ifreq, fm.lam_um[ifreq], args.angles, len(angles),
                         direction, sigma))
    print("bright-25 band-peak |T|: min %.2e  median %.2e  max %.2e; "
          "global |T|max = %.3e"
          % (peak.min(), np.median(peak), peak.max(), Tmax_global))

    engines = {}

    def engine_for(name, B, aset_key, aset):
        key = (name, aset_key)
        if key not in engines:
            engines[key] = fitmod.AnalyticJacobian(fm, ifreq, B, aset,
                                                   direction)
        return engines[key]

    # ---------------------------------------------------------- Study 1
    t_s1 = time.time()
    eng68 = engine_for("B68", B68, args.angles, angles)
    spans = build_spans(fm, ifreq, B68, eng68, T_proj, angles, direction,
                        sigma)
    lev = leverage_table(fm, ifreq, angles, T_proj, direction, spans,
                         mask25, top=15)
    t_s1 = time.time() - t_s1
    print("  [wall] Study 1: %.1f s" % t_s1)
    out["s1_names"] = np.array([d["name"] for d in spans])
    out["s1_rank"] = np.array([d["rank"] for d in spans])
    out["s1_n_entries"] = np.array([d["n_entries"] for d in spans])
    out["s1_n_orbits"] = np.array([d["n_orbits"] for d in spans])
    out["s1_n_orbits_c4"] = np.array([d["n_orbits_c4"] for d in spans])
    out["s1_ident_rel"] = np.array([d["ident_rel"] for d in spans])
    out["s1_floor_max"] = np.array([d["floor_max"] for d in spans])
    out["s1_floor_rms"] = np.array([d["floor_rms"] for d in spans])
    out["s1_tau"] = np.array([d["tau"] for d in spans])
    out["s1_lev_entries"] = lev["entries"]
    out["s1_lev"] = lev["lev"]
    out["s1_lev_sens"] = lev["sens"]
    out["s1_lev_in25"] = lev["in25"]

    # ---------------------------------------------------------- Study 2
    t_s2 = time.time()
    sub("Study 2: estimator ladder (noise-free).  '*' after nfev = hit "
        "max_nfev (%d) -- NOT converged" % args.max_nfev)
    print("  estimators: E0 unweighted/tik0 | E1 weighted(sigma=%.0e)/tik0 "
          "| E2 unweighted/isotropic tik=1/tau^2 | E3 weighted+isotropic "
          "| E4o orbit-pure basis + DIAGONAL tik_k=1/|z_k^true|^2 "
          "(ORACLE-PRIMED) | E4r same but tau_k from a %d-round "
          "empirical-Bayes iteration seeded by E3 (REALIZABLE)"
          % (sigma, args.eb_rounds))
    print("  NOTE on E4o: knowing |z_k^true| is NOT realizable for a blind "
          "retrieval, but it IS realizable in the doc-par.-1 use case #1 "
          "(QA gate for an externally supplied tmat.h5): there a candidate "
          "T is given, and its own |z_k| legitimately supply the prior "
          "scale.  E4o numbers therefore bound the QA-gate application, "
          "not the blind-retrieval application.")
    S_proj = fm.predict(T_proj, ifreq, angles, direction)
    S_raw = fm.predict(T_ref, ifreq, angles, direction)
    rows = []
    for d in spans:
        eng = engine_for(d["name"], d["B"], args.angles, angles)
        d["engine"] = eng
        ests = ["E0", "E1", "E2", "E3"]
        for est in ests:
            for seed_mode in ("born", "truth"):
                if seed_mode == "truth" and est not in ("E0", "E3"):
                    continue                    # trimmed: see docstring
                r = run_estimator(fm, ifreq, d, S_proj, T_proj, entries,
                                  peak, sel, angles, direction, eng, est,
                                  seed_mode, sigma, args.max_nfev)
                r["data"] = "S(T_proj)"
                rows.append(r)
                print_row(r)
        for est in ("E0", "E3"):                # raw-reference variant
            r = run_estimator(fm, ifreq, d, S_raw, T_ref, entries, peak,
                              sel, angles, direction, eng, est, "born",
                              sigma, args.max_nfev)
            r["data"] = "S(T_ref)"
            r["span"] = d["name"] + "/raw"
            rows.append(r)
            print_row(r)

    # ---- E4: orbit-scaled prior (oracle + realizable), bright spans only
    print("")
    for d in spans:
        if d["kind"] != "bright":
            continue
        Bo = d["B_orbit"]
        engo = engine_for(d["name"] + "_orb", Bo, args.angles, angles)
        d["engine_orbit"] = engo
        t_true_o = par.pack(par.unpack(par.pack(T_proj, Bo), Bo), Bo)
        nb = Bo.shape[0]
        z_true = t_true_o[:nb] + 1j * t_true_o[nb:]
        floor = 1e-6 * max(np.abs(z_true).max(), 1e-300)
        tau_k = np.maximum(np.abs(z_true), floor)
        d["tau_k_oracle"] = tau_k
        r = run_estimator(fm, ifreq, d, S_proj, T_proj, entries, peak, sel,
                          angles, direction, engo, "E4o", "born", sigma,
                          args.max_nfev, tik_vec=1.0 / tau_k ** 2, B_use=Bo)
        r["data"] = "S(T_proj)"
        r["span"] = d["name"] + "/orb"
        rows.append(r)
        print_row(r)
        # REALIZABLE: iterated empirical-Bayes.  Round 0 = isotropic E3;
        # each further round re-estimates tau_k = |z_k| from the previous
        # fit.  Nothing here uses the truth.
        tau_iso = float(np.linalg.norm(t_true_o) / np.sqrt(t_true_o.size))
        tau_k2 = None
        for rnd in range(args.eb_rounds):
            if rnd == 0:
                rr = fitmod.fit_frequency(
                    fm, ifreq, Bo, S_proj, angles, weights=1.0 / sigma ** 2,
                    tikhonov=1.0 / tau_iso ** 2, method="lm",
                    direction=direction, engine=engo,
                    max_nfev=args.max_nfev, xtol=1e-12, ftol=1e-12,
                    gtol=1e-12)
                t_prev = rr["t_hat"]
            else:
                rr = fit_diag(fm, ifreq, Bo, S_proj, angles, engo,
                              weights=1.0 / sigma ** 2,
                              tik_vec=1.0 / tau_k2 ** 2,
                              direction=direction, max_nfev=args.max_nfev,
                              tol=1e-12)
                t_prev = rr["t_hat"]
            z1 = t_prev[:nb] + 1j * t_prev[nb:]
            tau_k2 = np.maximum(np.abs(z1),
                                1e-3 * max(np.abs(z1).max(), 1e-30))
            _, pk_r = fitmod.entry_errors(par.unpack(t_prev, Bo), T_proj,
                                          entries, peak)
            print("      E4r round %d: all-25 pk max %.3e (tau_k rms "
                  "%.2e)" % (rnd, pk_r.max(),
                             float(np.sqrt((tau_k2 ** 2).mean()))))
        d["tau_k_2step"] = tau_k2
        r = run_estimator(fm, ifreq, d, S_proj, T_proj, entries, peak, sel,
                          angles, direction, engo, "E4r", "born", sigma,
                          args.max_nfev, tik_vec=1.0 / tau_k2 ** 2,
                          B_use=Bo)
        r["data"] = "S(T_proj)"
        r["span"] = d["name"] + "/orb"
        rows.append(r)
        print_row(r)
        print("      tau_k oracle vs realizable: max ratio %.2f, median "
              "ratio %.2f (realizable / oracle)"
              % ((tau_k2 / tau_k).max(), np.median(tau_k2 / tau_k)))

    # ---- basis-invariance measurement (SVD vs orbit-pure, same iso tik)
    print("")
    d0 = spans[0]
    rng = np.random.default_rng([args.seed, ifreq, 4242])
    noise = (rng.standard_normal(S_proj.shape)
             + 1j * rng.standard_normal(S_proj.shape)) * (sigma / np.sqrt(2))
    S_noisy = S_proj + noise
    tau0 = d0["tau"]
    kw = dict(weights=1.0 / sigma ** 2, tikhonov=1.0 / tau0 ** 2,
              method="lm", direction=direction, max_nfev=args.max_nfev,
              xtol=1e-12, ftol=1e-12, gtol=1e-12)
    ra = fitmod.fit_frequency(fm, ifreq, d0["B"], S_noisy, angles,
                              engine=d0["engine"], **kw)
    rb = fitmod.fit_frequency(fm, ifreq, d0["B_orbit"], S_noisy, angles,
                              engine=d0["engine_orbit"], **kw)
    dab = maxabs(ra["T0_hat"] - rb["T0_hat"])
    scale = max(maxabs(ra["T0_hat"]), maxabs(rb["T0_hat"]))
    print("  BASIS-INVARIANCE measurement (same noisy data, same isotropic "
          "tik = 1/tau^2, SVD bright basis vs orbit-pure basis):")
    print("    max|T_svd - T_orbit| = %.3e vs scale %.3e (relative %.2e); "
          "objectives %.4e / %.4e"
          % (dab, scale, dab / scale, ra["objective"], rb["objective"]))
    print("    Both are real-orthonormal bases of the SAME real span, and "
          "||t||^2 is invariant under the real orthogonal change of basis, "
          "so the regularized MINIMIZER is identical: any difference here "
          "is a Born-seed BASIN difference (different local minimum), NOT "
          "an estimator difference.")
    out["s2_basis_inv_absdiff"] = dab
    out["s2_basis_inv_scale"] = scale
    out["s2_basis_inv_obj"] = np.array([ra["objective"], rb["objective"]])

    # ---- bias decomposition of the strongest E-dipole diagonal
    print("")
    dip_idx = np.nonzero(sel["dipole"])[0]
    e_strong = dip_idx[np.argmax(peak[dip_idx])]
    ent_strong = tuple(int(x) for x in entries[e_strong])
    bd = bias_decomposition(d0["engine"], d0["B"], d0["t_true"],
                            1.0 / tau0 ** 2, 1.0 / sigma ** 2, ent_strong,
                            modes)
    print("      -> linear shrinkage bias is %.3e = %.3f%% of that entry's "
          "band-peak |T| (%.2e).  Compare with the MEASURED error of the "
          "same entry under E3 below; the remainder is nonlinearity / wrong "
          "basin, NOT shrinkage."
          % (abs(bd["total"]), 100 * abs(bd["total"]) / peak[e_strong],
             peak[e_strong]))
    out["s2_bias_entry"] = np.array(ent_strong)
    out["s2_bias_total"] = bd["total"]
    out["s2_bias_s"] = bd["s"]
    out["s2_bias_contrib"] = bd["contrib"]

    t_s2 = time.time() - t_s2
    print("\n  [wall] Study 2: %.1f s (%d fits)" % (t_s2, len(rows)))
    for key in ("span", "est", "seed", "data"):
        out["s2_%s" % key] = np.array([r[key] for r in rows])
    for key in ("objective", "maxdS", "excursion", "passivity", "nfev",
                "wall_s", "capped"):
        out["s2_%s" % key] = np.array([r[key] for r in rows])
    out["s2_rel_peaknorm"] = np.array([r["rel_peaknorm"] for r in rows])
    out["s2_rel_pointwise"] = np.array([r["rel_pointwise"] for r in rows])

    # ---- noise-free "winner" (reported, but NOT the selection criterion)
    realizable = [r for r in rows
                  if r["seed"] == "born" and r["data"] == "S(T_proj)"
                  and r["est"] != "E4o"]
    nf_best = min(realizable, key=lambda r: r["pk_stats"]["all25"][0])
    oracle = [r for r in rows if r["est"] == "E4o"]
    nf_oracle = (min(oracle, key=lambda r: r["pk_stats"]["all25"][0])
                 if oracle else None)
    print("  noise-free best REALIZABLE row: span %s, %s (all-25 max "
          "%.3e).  This is NOT used to select the campaign protocol -- a "
          "noise-free winner can be an overfitter; Study 3 screens at "
          "sigma = %.0e instead."
          % (nf_best["span"], nf_best["est"],
             nf_best["pk_stats"]["all25"][0], sigma))
    if nf_oracle is not None:
        print("  noise-free best ORACLE-PRIMED row (labelled, not "
              "realizable): span %s, %s (all-25 max %.3e)"
              % (nf_oracle["span"], nf_oracle["est"],
                 nf_oracle["pk_stats"]["all25"][0]))
    out["s2_noisefree_best"] = np.array([nf_best["span"], nf_best["est"]])

    # ---------------------------------------------------------- Study 2b
    # frequency-continuation seeding (E6) vs Born seed, C-clean loop
    t_s6 = time.time()
    sub("Study 2b (E6): frequency-continuation seeding vs Born seed, "
        "C-clean loop (target EXACTLY in the bright-10 span)")
    cont = continuation_study(fm, ifreq, B68, spans[0], entries, peak, sel,
                              angles, direction, args)
    t_s6 = time.time() - t_s6
    print("  [wall] Study 2b: %.1f s" % t_s6)
    for k, v in cont.items():
        out["s6_%s" % k] = v

    # ------------------------------------------------- Study 3 screening
    t_s3 = time.time()
    sub("Study 3a: protocol SCREENING at sigma = %.0e (%d trials each; "
        "selection criterion = mean over trials of the per-trial max "
        "all-25 peak-normalized error).  Only WEIGHTED protocols are "
        "screened, so the linear error-bar theory used for the gates "
        "applies to the winner." % (sigma, args.screen_trials))
    # SHRINK-TO-ZERO reference: any protocol that does not beat this is
    # worthless (a strongly regularized fit can "win" by returning ~0).
    pk_zero = np.abs(T_proj[entries[:, 0], entries[:, 1]]) / peak
    st_zero = class_stats(pk_zero, sel)
    print("  ZERO-ESTIMATOR reference (T_hat = 0; any protocol must beat "
          "this to be worth anything):")
    print(fmt_class_line("zero max", st_zero))
    out["s3a_zero_pk"] = pk_zero
    # candidate = (label, span, basis, key, estimator, seed_mode, n_trials)
    ctxs = {}

    def ctx_for(key, B, aset=None):
        aset = angles if aset is None else aset
        k = (key, tuple(aset))
        if k not in ctxs:
            ctxs[k] = SpanCtx(fm, B68, B, key, aset, direction)
        return ctxs[k]

    cand = []
    for d in spans:
        cand.append(("%s x E3" % d["name"], d, d["B"], d["name"], "E3",
                     "born", args.screen_trials))
        if d["kind"] == "bright":
            cand.append(("%s/orb x E4r" % d["name"], d, d["B_orbit"],
                         d["name"] + "_orb", "E4r", "born",
                         args.screen_trials))
            cand.append(("%s/orb x E4o [ORACLE]" % d["name"], d,
                         d["B_orbit"], d["name"] + "_orb", "E4o", "born",
                         args.screen_trials))
    # --- CONTINUATION half (doc: seeding is a protocol dimension) ---
    cont_spans = [d for d in spans if d["kind"] == "bright"
                  and d["name"] in ("bright1e-03", "bright3e-05")]
    for d in cont_spans:
        cand.append(("%s x E3 +CONT" % d["name"], d, d["B"], d["name"],
                     "E3", "cont", args.cont_screen_trials))
        cand.append(("%s/orb x E5 +CONT" % d["name"], d, d["B_orbit"],
                     d["name"] + "_orb", "E5", "cont",
                     args.cont_screen_trials))
        cand.append(("%s/orb x E4o +CONT [ORACLE]" % d["name"], d,
                     d["B_orbit"], d["name"] + "_orb", "E4o", "cont",
                     args.cont_screen_trials))
    d10 = spans[0]
    cand.append(("bright1e-03 x E0 +CONT", d10, d10["B"], d10["name"],
                 "E0", "cont", args.cont_screen_trials))
    cand.append(("bright1e-03 x E2 +CONT", d10, d10["B"], d10["name"],
                 "E2", "cont", args.cont_screen_trials))
    cand.append(("bright1e-04/orb x E5 +CONT", spans[2], spans[2]["B_orbit"],
                 spans[2]["name"] + "_orb", "E5", "cont",
                 args.cont_screen_trials))
    print("  CONTINUATION protocols ('+CONT') sweep a %d-frequency "
          "contiguous chain centred on this frequency in BOTH directions, "
          "seeding each fit from the previous frequency's t_hat and "
          "keeping the smaller-objective result at this frequency.  "
          "Nothing in that uses the truth.  E4r (Born half) is the "
          "REALIZABLE per-frequency two-step; **E5** is a "
          "continuation-native REALIZABLE protocol introduced here: the "
          "per-orbit prior scale tau_k at frequency f is |z_k| of the "
          "PREVIOUS frequency's fit (isotropic E3 at the first step of a "
          "sweep), which a real 49-frequency sweep supplies for free and "
          "which -- unlike same-frequency empirical Bayes -- does not "
          "estimate the prior from the very fit it regularizes.  Only E4o "
          "uses the truth.  TRIMMED (stated): the continuation half uses "
          "%d trials (vs %d for the Born half), covers spans "
          "{bright1e-03, bright1e-04, bright3e-05} rather than all four, "
          "and excludes obs68, whose regularized fit is information-"
          "limited rather than basin-limited (its truth-seeded and "
          "Born-seeded rows agree to within noise in the previous run)."
          % (args.cont_chain, args.cont_screen_trials, args.screen_trials))
    screen = []
    for si, (label, d, Bx, key, est, smode, ntr) in enumerate(cand):
        ctx = ctx_for(key, Bx)
        tr = noise_trials(ctx, ifreq, entries, peak, sel, sigma, ntr,
                          args.seed, est, seed_mode=smode, tag=900 + si,
                          chain_len=args.cont_chain,
                          max_nfev=args.trial_nfev)
        st = trial_class_stats(tr["pk"], sel)
        exc = float(np.nanmean(tr["excursion"]))
        screen.append((label, st["all25"][0], st["dipole"][0], st, si, exc,
                       tr["n_capped"], ntr))
        print("  %-32s all25 %.3e (gain %5.2f) | dipole %.3e (gain %5.2f) "
              "| |t|/|t*| %.3f | capped %d/%d fits"
              % (label, st["all25"][0],
                 st_zero["all25"][0] / max(st["all25"][0], 1e-300),
                 st["dipole"][0],
                 st_zero["dipole"][0] / max(st["dipole"][0], 1e-300),
                 exc, tr["n_capped"], tr["n_fits"]), flush=True)
    screen.sort(key=lambda x: x[1])
    # SELECTION: REALIZABLE only, non-degenerate (mean excursion in
    # [0.3, 3]), and -- new -- it must BEAT THE ZERO ESTIMATOR on the
    # dipole class, otherwise it carries no information worth gating.
    real = [s for s in screen if "ORACLE" not in s[0]]
    ok = [s for s in real if 0.3 <= s[5] <= 3.0]
    beats = [s for s in ok
             if st_zero["dipole"][0] / max(s[2], 1e-300) > 1.0]
    if beats:
        win = min(beats, key=lambda s: s[1])
    elif ok:
        win = min(ok, key=lambda s: s[1])
        print("  [WARN] no realizable candidate BEATS the zero estimator "
              "on the dipole class; selecting the smallest all-25 error "
              "among the non-degenerate ones.  Its gate thresholds are "
              "upper bounds on |T|, not measurements of it.")
    else:
        win = min(real, key=lambda s: s[1])
        print("  [WARN] NO realizable candidate has a mean "
              "||t_hat||/||t_true|| in [0.3, 3]: every one of them either "
              "collapses toward zero or blows up.")
    win_label, _, _, _, win_i, win_exc = win[:6]
    best_lbl, bd_, Bb_, keyb_, estb_, smodeb_, _ = cand[win_i]
    beat = st_zero["dipole"][0] / max(win[2], 1e-300)
    print("  -> SELECTED (realizable, non-degenerate): %s   (mean "
          "excursion %.3f, dipole gain/zero %.2f -- %s)"
          % (win_label, win_exc, beat,
             "BEATS the zero estimator" if beat > 1.0 else
             "does NOT beat the zero estimator: its Study-5 thresholds "
             "are UPPER BOUNDS on |T|, not measurements"))
    out["s3a_labels"] = np.array([s[0] for s in screen])
    out["s3a_all25_mean"] = np.array([s[1] for s in screen])
    out["s3a_dipole_mean"] = np.array([s[2] for s in screen])
    out["s3a_excursion"] = np.array([s[5] for s in screen])
    out["s3a_capped"] = np.array([s[6] for s in screen])
    out["s3a_selected"] = win_label
    out["s3a_cont_chain"] = args.cont_chain

    # ---------------------------------------------------------- Study 3
    sub("Study 3b: noise ladder (per-class MEAN / MEDIAN / P90 of the "
        "per-trial MAX peak-normalized entry error)")
    print(fmt_class_line("zero max", st_zero)
          + "     <- shrink-to-zero reference (sigma-independent)")
    # combo = (label, ctx, est, seed_mode, n_trials, max_nfev)
    combos = []
    d10 = spans[0]
    combos.append(("baseline bright1e-03 x E0 (Born, doc protocol)",
                   ctx_for(d10["name"], d10["B"]), "E0", "born",
                   min(args.trials, args.baseline_trials), args.trial_nfev))
    ctx_win = ctx_for(keyb_, Bb_)
    n_sel = min(args.trials, args.sel_trials)
    combos.append(("best-realizable " + best_lbl, ctx_win, estb_, smodeb_,
                   n_sel, args.win_nfev))
    if smodeb_ == "cont":
        combos.append(("seed contrast: " + best_lbl.replace(" +CONT", "")
                       + " (BORN)", ctx_win, estb_, "born",
                       min(args.trials, args.aux_trials), args.win_nfev))
    combos.append(("achievability bound (TRUTH-SEEDED) " + best_lbl,
                   ctx_win, estb_, "truth",
                   min(args.trials, args.aux_trials), args.win_nfev))
    # QA-gate bound: best oracle protocol from the screening
    orc = [s for s in screen if "ORACLE" in s[0]]
    if orc:
        ow = min(orc, key=lambda s: s[1])
        olab, od, oB, okey, oest, osm, _ = cand[ow[4]]
        combos.append(("QA-gate bound (ORACLE) " + olab,
                       ctx_for(okey, oB), oest, osm,
                       min(args.trials, args.aux_trials), args.win_nfev))
    print("  TRIAL BUDGET (grid reduced for runtime -- stated, not hidden):"
          " the headline 'best-realizable' row uses %d trials (of the "
          "requested --trials %d) at max_nfev = %d, raised from the "
          "screening's %d per the 'no threshold from an unconverged fit' "
          "rule; the unregularized baseline %d; the seed-contrast, "
          "truth-seeded and oracle rows %d each.  The trial count is "
          "reduced BECAUSE the cap was raised: the selected protocol is a "
          "%d-complex-dimension fit whose LM iteration does not terminate "
          "on this data, so each trial costs ~%dx a bright-span fit."
          % (n_sel, args.trials, args.win_nfev, args.trial_nfev,
             min(args.trials, args.baseline_trials),
             min(args.trials, args.aux_trials), Bb_.shape[0],
             max(1, args.win_nfev // 50)))
    s3 = {}
    for ci, (label, ctxc, est, smode, ntr, nfev) in enumerate(combos):
        print("\n  %s   (%d trials, seed=%s, max_nfev=%d)"
              % (label, ntr, smode, nfev))
        det = noise_trials(ctxc, ifreq, entries, peak, sel, sigma, 1,
                           args.seed, est, seed_mode=smode, tag=50 + ci,
                           chain_len=args.cont_chain, max_nfev=nfev,
                           noise_scale=0.0)
        st_det = class_stats(det["pk"][0], sel)
        print(fmt_class_line("bias(0)", st_det))
        for sg in args.sigma_list:
            tr = noise_trials(ctxc, ifreq, entries, peak, sel, sg, ntr,
                              args.seed, est, seed_mode=smode,
                              tag=100 + ci, chain_len=args.cont_chain,
                              max_nfev=nfev)
            st = trial_class_stats(tr["pk"], sel)
            print("    sigma=%.0e  (capped %d of %d fits, mean passivity "
                  "%.3f, mean |t|/|t*| %.3f, chain %s)"
                  % (sg, tr["n_capped"], tr["n_fits"],
                     np.nanmean(tr["passivity"]),
                     np.nanmean(tr["excursion"]),
                     tr["chain"] if smode == "cont" else "-"))
            print(fmt_trial_line("mean", st, 0))
            print(fmt_trial_line("median", st, 1))
            print(fmt_trial_line("p90", st, 2))
            print(fmt_gain_line(st, st_zero, 0))
            s3[(label, sg)] = st
            out["s3_pk_%d_%s" % (ci, ("%.0e" % sg).replace("-", "m"))] = \
                tr["pk"]
            out["s3_capped_%d_%s" % (ci, ("%.0e" % sg).replace("-", "m"))] \
                = np.array([tr["n_capped"], tr["n_fits"]])
        out["s3_bias_%d" % ci] = det["pk"][0]
        out["s3_label_%d" % ci] = label
        out["s3_ntrials_%d" % ci] = ntr
    out["s3_sigmas"] = np.array(args.sigma_list)
    out["s3_n_combos"] = len(combos)
    t_s3 = time.time() - t_s3
    print("\n  [wall] Study 3: %.1f s" % t_s3)

    # ---------------------------------------------------------- Study 4
    t_s4 = time.time()
    label4 = "best-realizable " + best_lbl
    B4, key4, est4, smode4 = Bb_, keyb_, estb_, smodeb_
    sub("Study 4: angle-set ladder (selected protocol %s, sigma = %.0e, "
        "%d trials)" % (label4, sigma, min(args.trials, args.angle_trials)))
    protos = [("selected realizable: %s" % best_lbl, B4, key4, est4,
               smode4)]
    if smode4 == "cont":
        protos.append(("seed contrast: same protocol, BORN seed", B4, key4,
                       est4, "born"))
    a4 = angle_set_study(fm, ifreq, B68, T_proj, entries, peak, sel,
                         direction, args, sigma, protos,
                         B_ref=spans[0]["B"], st_zero=st_zero)
    t_s4 = time.time() - t_s4
    print("  [wall] Study 4: %.1f s" % t_s4)
    for k, v in a4.items():
        out["s4_%s" % k] = v

    # ------------------------------------------- per-entry gate inputs
    t_s5 = time.time()
    sub("Study 5 inputs: per-entry information floor and predicted error "
        "bar (selected protocol %s, angle set %s, sigma = %.0e).  NOTE: "
        "the seed mode does not enter linear theory -- the local estimator "
        "at this frequency is the same, so pred_sigma below applies to the "
        "selected protocol with or without continuation."
        % (label4, args.angles, sigma))
    Bg = B4
    cg = ctx_win.at(ifreq)
    engg = cg["engine"]
    t_true_g = cg["t_true"]
    if est4.startswith("E4"):
        tikd = np.concatenate([1.0 / cg["tau_k"] ** 2,
                               1.0 / cg["tau_k"] ** 2])
    else:
        tikd = np.full(t_true_g.size, 1.0 / cg["tau"] ** 2)
    pred, bias_e, var_e = predicted_entry_rms(engg, Bg, t_true_g, tikd,
                                              1.0 / sigma ** 2, entries)
    Jg = engg.jac_packed(t_true_g)
    tau_pr = float(np.linalg.norm(t_true_g) / np.sqrt(t_true_g.size))
    svg = obs.svd_resolution(Jg, sigma=sigma, prior_scale=tau_pr)
    res_c = obs.fold_complex(svg["res"])
    Hmap = obs.entry_heatmap(res_c, Bg)
    res_entry = np.array([Hmap[i, j] / max(
        (np.abs(np.asarray(Bg)[:, i, j]) ** 2).sum(), 1e-300)
        for i, j in entries])
    # reference: the same resolution measure on the doc's current bright-10
    d10 = spans[0]
    J10 = d10["engine"].jac_packed(d10["t_true"])
    sv10 = obs.svd_resolution(J10, sigma=sigma, prior_scale=d10["tau"])
    H10 = obs.entry_heatmap(obs.fold_complex(sv10["res"]), d10["B"])
    res10 = np.array([H10[i, j] / max(
        (np.abs(np.asarray(d10["B"])[:, i, j]) ** 2).sum(), 1e-300)
        for i, j in entries])
    c4v = np.array([not bool(fitmod.classify_entries(modes)
                             ["c4_conforming"][i, j]) for i, j in entries])
    model_err_T = maxabs(T_proj - par.unpack(t_true_g, Bg))
    print("  off-span model error in T units: max|T_proj - P_span T_proj| "
          "= %.3e (span %s)" % (model_err_T, key4))
    print("  %-16s %10s %10s %10s %10s %10s %7s %7s %s"
          % ("entry", "peak|T|", "1%peak", "pred_sig", "|bias|",
             "sqrt(var)", "res", "res10", "gateable"))
    gateable = np.zeros(len(entries), dtype=bool)
    for e in range(len(entries)):
        i, j = entries[e]
        g = bool(res_entry[e] >= RES_C_GATE) and not c4v[e]
        gateable[e] = g
        print("  %-16s %10.3e %10.3e %10.3e %10.3e %10.3e %7.3f %7.3f %s"
              % (fitmod.entry_label(modes, i, j), peak[e], 0.01 * peak[e],
                 pred[e], bias_e[e], np.sqrt(var_e[e]), res_entry[e],
                 res10[e], "yes" if g else
                 ("NO (C4-violating)" if c4v[e] else "NO")))
    print("  -> %d of %d bright entries have entry-resolution res >= %.2f "
          "under the SELECTED protocol/angle set; %d of %d under the "
          "doc's current bright-10 span (column res10).  The rest are "
          "REPORTED, not gated."
          % (int((res_entry >= RES_C_GATE).sum()), len(entries), RES_C_GATE,
             int((res10 >= RES_C_GATE).sum()), len(entries)))
    print("  -> %d of the 25 'bright' entries are C4-VIOLATING (they are "
          "the file's own 0.2-0.4%% symmetry noise): every symmetry-"
          "constrained span annihilates them, so they can only ever be "
          "returned as 0 and must be dropped from any gate list."
          % int(c4v.sum()))
    n_imposs = int((0.01 * peak < pred).sum())
    print("  -> for %d of %d bright entries the doc's '1%% of the entry' "
          "target is BELOW the estimator's own predicted error bar: the "
          "par.-6.2 gate is unattainable IN PRINCIPLE for them under this "
          "protocol" % (n_imposs, len(entries)))
    out["s5_res10"] = res10
    out["s5_c4_violating"] = c4v
    out["s5_pred"] = pred
    out["s5_bias"] = bias_e
    out["s5_var"] = var_e
    out["s5_res_entry"] = res_entry
    out["s5_gateable"] = gateable
    out["s5_model_err_T"] = model_err_T
    t_s5 = time.time() - t_s5
    print("  [wall] Study 5 inputs: %.1f s" % t_s5)

    out["walls"] = np.array([t_s1, t_s2, t_s6, t_s3, t_s4, t_s5])
    print("\n  [wall] ifreq %d total: %.1f s  (S1 %.0f, S2 %.0f, S2b %.0f, "
          "S3 %.0f, S4 %.0f, S5 %.0f)"
          % (ifreq, t_s1 + t_s2 + t_s6 + t_s3 + t_s4 + t_s5, t_s1, t_s2,
             t_s6, t_s3, t_s4, t_s5))
    store[ifreq] = dict(rows=rows, spans=[
        {k: v for k, v in d.items()
         if k not in ("B", "B_orbit", "engine", "engine_orbit", "V_obs")}
        for d in spans], selected=label4, screen=screen, s3=s3,
        s4=a4, cont=cont, out=out, sel=sel, entries=entries, peak=peak,
        pred=pred, res_entry=res_entry, res10=res10, c4v=c4v,
        gateable=gateable, st_zero=st_zero,
        model_err_T=model_err_T, Tmax_global=Tmax_global,
        bias_total=bd["total"], bias_entry=ent_strong,
        basis_inv=(dab, scale), nf_best=nf_best)
    return out


def continuation_study(fm, ifreq, B68, span10, entries, peak, sel, angles,
                       direction, args):
    """E6: frequency-continuation seeding on the C-clean loop.

    C-clean = data synthesized from T_bp = P_bright(P68(T_ref)) at EACH
    frequency, so the target lies exactly in the fitted span and any
    residual error is purely a landscape/basin effect (synthetic_test.py's
    'Born reachability' measurement).  We walk a contiguous run of
    frequencies toward ifreq from below and from above, seeding each fit
    with the previous frequency's t_hat, and compare with the Born seed at
    ifreq itself.  Unweighted, unregularized (E0) so the comparison isolates
    the SEED.
    """
    B10 = span10["B"]
    nspan = args.cont_span
    tol = dict(xtol=1e-12, ftol=1e-12, gtol=1e-12)

    def clean_target(f):
        Tp = par.unpack(par.pack(fm.data.T[f], B68), B68)
        return par.unpack(par.pack(Tp, B10), B10)

    def born_at(f):
        Tb = clean_target(f)
        S = fm.predict(Tb, f, angles, direction)
        eng = fitmod.AnalyticJacobian(fm, f, B10, angles, direction)
        r = fitmod.fit_frequency(fm, f, B10, S, angles, direction=direction,
                                 engine=eng, max_nfev=args.max_nfev, **tol)
        _, pk = fitmod.entry_errors(r["T0_hat"], Tb, entries, peak)
        return r, pk, Tb, eng

    r0, pk0, Tb0, eng0 = born_at(ifreq)
    print("  Born seed at ifreq %d: obj = %.3e, all-25 max peaknorm = %.3e "
          "(nfev %d)" % (ifreq, r0["objective"], pk0.max(), r0["nfev"]))

    res = dict(born_pk=pk0, born_obj=r0["objective"])
    for name, step in (("up", +1), ("down", -1)):
        f_start = ifreq - step * nspan
        chain = [f for f in range(f_start, ifreq + step, step)
                 if 0 <= f < fm.nf]
        chain = [f for f in chain if all(fm.have[f, a] for a in angles)]
        if len(chain) < 2 or chain[-1] != ifreq:
            print("  continuation %-4s: SKIP (frequencies %s not all "
                  "cached)" % (name, chain))
            continue
        t_seed = None
        objs = []
        for f in chain:
            Tb = clean_target(f)
            S = fm.predict(Tb, f, angles, direction)
            eng = fitmod.AnalyticJacobian(fm, f, B10, angles, direction)
            r = fitmod.fit_frequency(fm, f, B10, S, angles, t0=t_seed,
                                     direction=direction, engine=eng,
                                     max_nfev=args.max_nfev, **tol)
            t_seed = r["t_hat"]
            objs.append(r["objective"])
        _, pk = fitmod.entry_errors(r["T0_hat"], clean_target(ifreq),
                                    entries, peak)
        print("  continuation %-4s (%s -> %d, %d steps): obj = %.3e, "
              "all-25 max peaknorm = %.3e   [Born was %.3e]"
              % (name, chain[0], ifreq, len(chain), r["objective"],
                 pk.max(), pk0.max()))
        res["cont_%s_pk" % name] = pk
        res["cont_%s_obj" % name] = r["objective"]
        res["cont_%s_chain" % name] = np.array(chain)
    best_cont = min([res[k].max() for k in res if k.endswith("_pk")]
                    + [pk0.max()])
    print("  -> best over {Born, continuation-up, continuation-down}: "
          "all-25 max peaknorm = %.3e (Born alone %.3e)"
          % (best_cont, pk0.max()))
    res["best_pk"] = best_cont
    return res


def angle_set_study(fm, ifreq, B68, T_proj, entries, peak, sel, direction,
                    args, sigma, protocols, B_ref=None, st_zero=None):
    """Study 4: CST cost, observable directions, and the sigma = 3e-3 noise
    ladder per angle set.

    `protocols` = [(tag, B, key, est, seed_mode), ...]; the FIRST one is the
    selected protocol and drives the reported `errs` array.  Observability
    (`n_obs`, `res_c`) is a property of the basis + angle set + prior and is
    seed-INDEPENDENT; the error columns are protocol-dependent, which is why
    both the selected protocol and (for a continuation winner) its Born twin
    are run.  B_ref: a second span whose observability is reported for
    reference (the doc's bright-10)."""
    B = protocols[0][1]
    sets = ["campaign", "starter", "all17", "ext_phi", "ext_theta"]
    table = all_angle_sets()
    names, nstruct, nempty, nobs, errs = [], [], [], [], []
    res_store = {}
    nobs_ref = []
    print("  candidate extended sets (justification): ext_phi = starter-5 "
          "+ (30,45) + (60,45) -- completes phi at theta already paid for, "
          "so +2 structure runs and +0 empty runs; ext_theta = campaign-13 "
          "+ (20,0) + (40,0) -- denser in theta on the phi=0 mirror plane "
          "(+2 structure, +2 empty).  Both use only CACHED rows of "
          "precompute_C.ANGLES_DEG.")
    print("  %-10s %5s %8s %7s %7s %9s   per-class mean of per-trial max "
          "peaknorm" % ("set", "n_ang", "n_struct", "n_empty", "n_obs",
                        "sum res_c"))
    for name in sets:
        aset = sorted(table[name])
        miss = [a for a in aset if not fm.have[ifreq, a]]
        if miss:
            print("  %-10s SKIP (angles %s not cached)" % (name, miss))
            continue
        eng = fitmod.AnalyticJacobian(fm, ifreq, B, aset, direction)
        T_span = par.unpack(par.pack(T_proj, B), B)
        t_true = par.pack(T_span, B)
        tau = float(np.linalg.norm(t_true) / np.sqrt(t_true.size))
        J = eng.jac_packed(t_true)
        sv = obs.svd_resolution(J, sigma=sigma, prior_scale=tau)
        res_c = obs.fold_complex(sv["res"])
        ns, ne = cst_run_counts(fm, aset)
        S_base = fm.predict(T_proj, ifreq, aset, direction)
        ntr = min(args.trials, args.angle_trials)
        names.append(name)
        nstruct.append(ns)
        nempty.append(ne)
        nobs.append(int((res_c >= RES_C_GATE).sum()))
        res_store[name] = res_c
        nref = -1
        if B_ref is not None:
            engr = fitmod.AnalyticJacobian(fm, ifreq, B_ref, aset,
                                           direction)
            t_r = par.pack(par.unpack(par.pack(T_proj, B_ref), B_ref), B_ref)
            tau_r = float(np.linalg.norm(t_r) / np.sqrt(t_r.size))
            rc_r = obs.fold_complex(obs.svd_resolution(
                engr.jac_packed(t_r), sigma=sigma,
                prior_scale=tau_r)["res"])
            nref = int((rc_r >= RES_C_GATE).sum())
            res_store["ref_" + name] = rc_r
        nobs_ref.append(nref)
        print("  %-10s %5d %8d %7d %7d %9.3f  (bright-10 ref: %d/%d obs)"
              % (name, len(aset), ns, ne, nobs[-1], res_c.sum(), nref,
                 len(rc_r) if B_ref is not None else 0))
        print("      res_c: min %.3f  median %.3f  max %.3f  (%d of %d "
              ">= %.2f; full array in the npz)"
              % (res_c.min(), np.median(res_c), res_c.max(), nobs[-1],
                 len(res_c), RES_C_GATE))
        for pi, (ptag, Bp, keyp, estp, smodep) in enumerate(protocols):
            ctxp = SpanCtx(fm, B68, Bp, keyp, aset, direction)
            tr = noise_trials(ctxp, ifreq, entries, peak, sel, sigma, ntr,
                              args.seed, estp, seed_mode=smodep,
                              tag=500 + pi, chain_len=args.cont_chain,
                              max_nfev=args.win_nfev)
            st = trial_class_stats(tr["pk"], sel)
            if pi == 0:
                errs.append([st[c][0] for c in CLASS_NAMES])
            print("    [%s]  (capped %d of %d fits)"
                  % (ptag, tr["n_capped"], tr["n_fits"]))
            print(fmt_trial_line("mean", st, 0))
            print(fmt_trial_line("p90", st, 2))
            if st_zero is not None:
                print(fmt_gain_line(st, st_zero, 0))
    if names:
        print("\n  marginal information gain per extra CST solve "
              "(reference = starter):")
        if "starter" in names:
            i0 = names.index("starter")
            base_cost = nstruct[i0] + nempty[i0]
            base_obs = nobs[i0]
            base_err = errs[i0][0]
            for i, nm in enumerate(names):
                dc = (nstruct[i] + nempty[i]) - base_cost
                if dc <= 0:
                    continue
                print("    %-10s +%2d solves -> +%d observable dirs "
                      "(%.3f dirs/solve); all-25 mean err %.3e -> %.3e "
                      "(%.1f%% of starter)"
                      % (nm, dc, nobs[i] - base_obs,
                         (nobs[i] - base_obs) / dc, base_err, errs[i][0],
                         100 * errs[i][0] / base_err))
    return dict(names=np.array(names), n_struct=np.array(nstruct),
                n_empty=np.array(nempty), n_obs=np.array(nobs),
                n_obs_ref=np.array(nobs_ref), errs=np.array(errs),
                **{"res_c_%s" % k: v for k, v in res_store.items()})


# ------------------------------------------------------------- figures

def make_figures(store, args):
    from tmatrix.plotting import plt
    paths = []
    for ifreq, S in store.items():
        out = S["out"]
        sig = out["s3_sigmas"]
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
        for ci in range(out["s3_n_combos"]):
            label = out["s3_label_%d" % ci]
            for cname, style in (("all25", "-o"), ("dipole", "--s")):
                ys = []
                for sg in sig:
                    key = "s3_pk_%d_%s" % (ci, ("%.0e" % sg).replace("-",
                                                                     "m"))
                    pk = out[key]
                    ys.append(np.nanmax(pk[:, S["sel"][cname]],
                                        axis=1).mean())
                ax[0].loglog(sig, ys, style, ms=4,
                             label="%s [%s]" % (label.split(" x ")[0][:22],
                                                cname))
        ax[0].set_xlabel("sigma (per complex observable)")
        ax[0].set_ylabel("mean per-trial max |dT| / band-peak |T|")
        ax[0].set_title("ifreq %d: noise ladder" % ifreq)
        ax[0].grid(True, which="both", alpha=0.3)
        ax[0].legend(fontsize=6)
        names = out["s4_names"]
        for i, nm in enumerate(names):
            rc = out["s4_res_c_%s" % nm]
            ax[1].plot(np.arange(1, len(rc) + 1), np.sort(rc)[::-1], "o-",
                       ms=3.5, label="%s (%d+%d runs)"
                       % (nm, out["s4_n_struct"][i], out["s4_n_empty"][i]))
        ax[1].axhline(RES_C_GATE, color="crimson", ls="--", lw=1)
        ax[1].set_xlabel("complex direction (sorted)")
        ax[1].set_ylabel("res_c at the physical prior")
        ax[1].set_title("ifreq %d: observability vs angle set" % ifreq)
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(RESULTS_DIR, "gate_study_ifreq%02d.png" % ifreq)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
        print("  figure: %s" % p)
    return paths


# ------------------------------------------------------------- markdown

def write_markdown(fm, store, args, path):
    L = []
    A = L.append
    A("# Gate study: recalibrating the synthetic and acceptance gates")
    A("")
    A("Generated by `retrieval/gate_study.py` (synthetic only -- no CST, "
      "no license).  Frequencies: %s; angle set `%s`; direction %+d; "
      "sigma placeholder %.1e; %d noise trials per point; seed %d."
      % (sorted(store), args.angles, args.direction, SIGMA_REF,
         args.trials, args.seed))
    A("")
    A("All numbers below are MEASURED by this script on the reference "
      "`test/single/saw_gold_wl15p0025um.tmat.h5` T-matrix propagated "
      "through the validated Floquet forward model.  Numbers that require "
      "knowing the truth are labelled ORACLE-PRIMED and are NOT achievable "
      "on real data.")
    A("")

    A("## 1. Span ladder")
    A("")
    A("| ifreq | span | mask entries | complex rank | orbits | C4-conf "
      "orbits | identity (rel) | model floor max\\|dS\\| | floor RMS | "
      "floor/sigma |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for ifreq in sorted(store):
        for d in store[ifreq]["spans"]:
            A("| %d | %s | %s | %d | %s | %s | %.1e | %.3e | %.3e | %.2f |"
              % (ifreq, d["name"],
                 d["n_entries"] if d["n_entries"] >= 0 else "n/a",
                 d["rank"],
                 d["n_orbits"] if d["n_orbits"] >= 0 else "n/a",
                 d["n_orbits_c4"] if d["n_orbits_c4"] >= 0 else "n/a",
                 d["ident_rel"], d["floor_max"], d["floor_rms"],
                 d["floor_over_sigma"]))
    A("")
    A("**Structural identity.** For every MASK-DEFINED span the Frobenius "
      "projection of `T_proj` onto the span reproduces `T_proj` EXACTLY on "
      "that span's mask entries (worst relative deviation %.1e; the script "
      "RAISES above 1e-12).  Consequence: every bright-entry recovery "
      "failure reported below is an ESTIMATOR or INFORMATION problem, never "
      "a representability one."
      % max(d["ident_rel"] for ifr in store
            for d in store[ifr]["spans"] if d["n_entries"] >= 0))
    A("")
    A("`obs68` (the SVD-observable restriction of the 68-dim symmetry "
      "basis at the reference linearization, cutoff %.0e) is NOT "
      "mask-defined: its restriction is chosen by data leverage, not entry "
      "support, so an orthogonal projection onto it discards physical "
      "bright content.  Its 'floor' row is therefore a Frobenius-projection "
      "number, not a fit floor -- read its Study-2 `max|dS|` instead."
      % OBS_CUTOFF)
    A("")

    A("## 2. Estimator ladder (noise-free)")
    A("")
    A("Estimators: **E0** unweighted / no prior (the protocol "
      "`synthetic_test.py` used); **E1** weighted `w = 1/sigma^2`; "
      "**E2** unweighted + isotropic Tikhonov `1/tau^2` at the physical "
      "prior `tau = ||t_true||/sqrt(n_par)`; **E3** weighted + isotropic "
      "Tikhonov (the protocol the campaign was going to use); "
      "**E4o** orbit-pure basis + DIAGONAL prior `tik_k = 1/|z_k^true|^2` "
      "(ORACLE-PRIMED); **E4r** the same prior estimated by a %d-round "
      "empirical-Bayes iteration seeded by an isotropic E3 fit "
      "(`tau_k = |z_k|` of the previous round, floored at 1e-3 of its max) "
      "-- REALIZABLE, nothing in it uses the truth." % args.eb_rounds)
    A("")
    A("`seed = truth` rows start the optimizer AT the projection of the "
      "target into the span.  They are not a protocol -- they are the "
      "ACHIEVABILITY BOUND: what the data would give if the optimizer "
      "always found the global minimum.  The gap between the `truth` and "
      "`born` rows of the same (span, estimator) is cause (c) below, "
      "measured.")
    A("")
    A("| ifreq | span | est | seed | data | objective | max\\|dS\\| | "
      "\\|t\\|/\\|t*\\| | passivity | all25 pk max | dipole pk max | "
      "even_m pk max |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ifreq in sorted(store):
        for r in store[ifreq]["rows"]:
            A("| %d | %s | %s | %s | %s | %.3e | %.2e | %.3f | %.3f | "
              "%.3e | %.3e | %.3e |"
              % (ifreq, r["span"], r["est"], r["seed"], r["data"],
                 r["objective"], r["maxdS"], r["excursion"], r["passivity"],
                 r["pk_stats"]["all25"][0], r["pk_stats"]["dipole"][0],
                 r["pk_stats"]["even_m"][0]))
    A("")
    A("**Achievability (truth-seeded) vs Born-seeded, per frequency.**  "
      "The best truth-seeded row and the SAME (span, estimator) started "
      "from the Born seed:")
    A("")
    A("| ifreq | span | est | truth-seeded all25 / dipole | Born-seeded "
      "all25 / dipole | ratio (all25) |")
    A("|---|---|---|---|---|---|")
    for ifreq in sorted(store):
        rows = store[ifreq]["rows"]
        tr_rows = [r for r in rows
                   if r["seed"] == "truth" and r["data"] == "S(T_proj)"]
        if not tr_rows:
            continue
        bt = min(tr_rows, key=lambda r: r["pk_stats"]["all25"][0])
        bo = [r for r in rows if r["seed"] == "born"
              and r["data"] == "S(T_proj)" and r["span"] == bt["span"]
              and r["est"] == bt["est"]]
        b = bo[0] if bo else None
        A("| %d | %s | %s | %.3e / %.3e | %s | %s |"
          % (ifreq, bt["span"], bt["est"], bt["pk_stats"]["all25"][0],
             bt["pk_stats"]["dipole"][0],
             ("%.3e / %.3e" % (b["pk_stats"]["all25"][0],
                               b["pk_stats"]["dipole"][0]))
             if b else "n/a",
             ("%.1f" % (b["pk_stats"]["all25"][0]
                        / max(bt["pk_stats"]["all25"][0], 1e-300)))
             if b else "n/a"))
    A("")
    for ifreq in sorted(store):
        S = store[ifreq]
        dab, scale = S["basis_inv"]
        A("**Basis invariance, ifreq %d.** Same noisy data, same isotropic "
          "penalty, SVD-orthonormal bright basis vs orbit-pure basis: "
          "`max|T_svd - T_orbit| = %.3e` against scale %.3e (relative "
          "%.2e).  The two bases are related by a real orthogonal transform "
          "under which `||t||^2` is invariant, so the regularized minimiser "
          "is mathematically IDENTICAL: a nonzero difference is a Born-seed "
          "BASIN difference, not an estimator improvement."
          % (ifreq, dab, scale, dab / scale))
        A("")
        A("**Shrinkage-bias decomposition, ifreq %d.** For the strongest "
          "E-dipole diagonal %s the linear shrinkage bias of the isotropic-"
          "prior estimator is %.3e = %.2f%% of that entry's band-peak "
          "\\|T\\|.  The measured error of the same entry is far larger, so "
          "the failure is NOT dominated by shrinkage -- it is nonlinearity "
          "/ wrong basin."
          % (ifreq, fitmod.entry_label(fm.modes, *S["bias_entry"]),
             abs(S["bias_total"]),
             100 * abs(S["bias_total"])
             / S["peak"][np.argmax([1 if tuple(e) == S["bias_entry"] else 0
                                    for e in S["entries"]])]))
        A("")

    A("## 2b. Frequency-continuation seeding (E6)")
    A("")
    A("C-clean loop (target lies EXACTLY in the bright-10 span, so the only "
      "possible error is a landscape/basin effect).  Seeding each fit with "
      "the previous frequency's `t_hat` along a contiguous run of %d "
      "frequencies:" % args.cont_span)
    A("")
    A("| ifreq | Born seed all-25 max | continuation up | continuation "
      "down | best |")
    A("|---|---|---|---|---|")
    for ifreq in sorted(store):
        c = store[ifreq]["cont"]
        up = ("%.3e" % c["cont_up_pk"].max()) if "cont_up_pk" in c else "n/a"
        dn = ("%.3e" % c["cont_down_pk"].max()) if "cont_down_pk" in c \
            else "n/a"
        A("| %d | %.3e | %s | %s | %.3e |"
          % (ifreq, c["born_pk"].max(), up, dn, c["best_pk"]))
    A("")

    A("## 3. Noise ladder")
    A("")
    A("**Entry classes.** The bright-25 entry list contains: %s.  Note "
      "that the doc-par.-4 classes `m33_11` and `m33_33` (the "
      "(3,-+3)<->(1,+-1) and (3,+-3)<->(3,+-3) content that is VISIBLE at "
      "theta = 0) contain NO bright-25 entry at all -- they are entirely "
      "sub-threshold, which is exactly why they dominate the model-error "
      "floor in section 1 while being ungateable as \"bright\" entries."
      % ", ".join("%s: %d" % (c, int(store[sorted(store)[0]]["sel"][c].sum()))
                  for c in CLASS_NAMES))
    A("")
    A("### 3a. Protocol screening at sigma = %.0e" % SIGMA_REF)
    A("")
    A("Selection criterion: mean over %d seeded trials of the per-trial "
      "MAX all-25 peak-normalized error.  Only WEIGHTED protocols are "
      "screened so that the linear error-bar theory used by the proposed "
      "gates applies to the winner." % args.screen_trials)
    A("")
    A("Protocols marked `+CONT` use FREQUENCY-CONTINUATION seeding: a "
      "%d-frequency contiguous chain centred on the target frequency is "
      "swept in BOTH directions, each fit seeded from the previous "
      "frequency's `t_hat` (the first fit of each sweep uses the Born "
      "seed), and the smaller-objective result at the target frequency is "
      "kept.  Nothing in that uses the truth -- it is realizable on real "
      "data, and a 49-frequency campaign sweep produces it for free."
      % args.cont_chain)
    A("")
    A("| ifreq | candidate | all25 mean | dipole mean | dipole gain/zero | "
      "mean \\|t\\|/\\|t*\\| | capped | selected |")
    A("|---|---|---|---|---|---|---|---|")
    for ifreq in sorted(store):
        S = store[ifreq]
        z = S["st_zero"]
        A("| %d | _ZERO ESTIMATOR (T_hat = 0)_ | %.3e | %.3e | 1.00 | "
          "0.000 | - | |"
          % (ifreq, z["all25"][0], z["dipole"][0]))
        for row in S["screen"]:
            label, a25, dip, st, si, exc = row[:6]
            cap = row[6] if len(row) > 6 else 0
            A("| %d | %s | %.3e | %.3e | %.2f | %.3f | %d | %s |"
              % (ifreq, label, a25, dip,
                 z["dipole"][0] / max(dip, 1e-300), exc, cap,
                 "**yes**" if S["selected"].endswith(label) else ""))
    A("")
    A("The ZERO ESTIMATOR row is the honest floor: a heavily regularized "
      "protocol can \"win\" a peak-normalized comparison by returning "
      "almost nothing, so the mean excursion column must be read with the "
      "error columns.  **Selection rule:** realizable (no oracle prior), "
      "non-degenerate (mean excursion in [0.3, 3]), must BEAT the zero "
      "estimator on the dipole class, then smallest all-25 error.")
    A("")
    A("### 3b. Noise ladder for the selected / baseline / oracle protocols")
    A("")
    A("Per-class MEAN / MEDIAN / P90 of the per-trial MAX peak-normalized "
      "entry error, noise `n = (sigma/sqrt 2)(g_re + i g_im)` per complex "
      "observable.  The unregularized baseline uses %d trials (its fits "
      "cost ~10x the regularized ones); everything else uses %d."
      % (min(args.trials, args.baseline_trials), args.trials))
    A("")
    A("`gain/zero` = (zero-estimator error) / (protocol error): values <= 1 "
      "mean the protocol is no better than returning nothing for that "
      "class.")
    A("")
    A("| ifreq | protocol | sigma | all25 mean | all25 p90 | all25 "
      "gain/zero | dipole mean | dipole p90 | dipole gain/zero |")
    A("|---|---|---|---|---|---|---|---|---|")
    for ifreq in sorted(store):
        z = store[ifreq]["st_zero"]
        A("| %d | _ZERO ESTIMATOR_ | - | %.3e | %.3e | 1.00 | %.3e | %.3e "
          "| 1.00 |" % (ifreq, z["all25"][0], z["all25"][0],
                        z["dipole"][0], z["dipole"][0]))
        for (label, sg), st in store[ifreq]["s3"].items():
            A("| %d | %s | %.0e | %.3e | %.3e | %.2f | %.3e | %.3e | "
              "%.2f |"
              % (ifreq, label, sg, st["all25"][0], st["all25"][2],
                 z["all25"][0] / max(st["all25"][0], 1e-300),
                 st["dipole"][0], st["dipole"][2],
                 z["dipole"][0] / max(st["dipole"][0], 1e-300)))
    A("")

    A("## 4. Angle-set ladder")
    A("")
    A("CST cost model: one structure solve per distinct (theta, phi) "
      "(theta = 0 counted once), one empty solve per distinct theta.  "
      "`n_obs` = complex directions of the chosen span with res_c >= %.2f "
      "at the physical prior (sigma = %.0e)." % (RES_C_GATE, SIGMA_REF))
    A("")
    A("> **DEPENDENCY -- the angle-set recommendation is CONDITIONAL.**  "
      "The oblique forward map has been cross-checked against treams only "
      "at the angles stored in `results/treams_oblique_check.npz`: "
      "(0,0), (0,45), (20,0), (20,45), (40,0), (40,45).  **No campaign "
      "oblique angle has ever been validated end-to-end against treams** "
      "-- not theta = 15/30/45/60, and no phi = 22.5 row at any theta.  "
      "Every angle set recommended below is built from such angles, at "
      "the theta where the design doc itself warns the Bloch shell sum is "
      "~7.5x harder to converge (par. 2).  This recommendation is "
      "therefore CONDITIONAL on the treams gate being extended to close "
      "at theta = 60 and phi = 22.5.  Do not spend CST hours on it "
      "before that gate closes.")
    A("")
    A("| ifreq | set | angles | structure runs | empty runs | total solves "
      "| n_obs (selected span) | n_obs (bright-10) | sum res_c | all25 "
      "mean err | dipole mean err |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for ifreq in sorted(store):
        a4 = store[ifreq]["s4"]
        for i, nm in enumerate(a4["names"]):
            rc = a4["res_c_%s" % nm]
            A("| %d | %s | %d | %d | %d | %d | %d/%d | %d | %.2f | %.3e | "
              "%.3e |"
              % (ifreq, nm, len(all_angle_sets()[nm]), a4["n_struct"][i],
                 a4["n_empty"][i], a4["n_struct"][i] + a4["n_empty"][i],
                 a4["n_obs"][i], len(rc), a4["n_obs_ref"][i], rc.sum(),
                 a4["errs"][i][0], a4["errs"][i][1]))
    A("")

    A("## 4b. Per-entry information floor and gateability")
    A("")
    A("Selected protocol, angle set `%s`, sigma = %.0e.  `pred_sigma` is "
      "the linear-theory propagated per-entry error bar "
      "`sqrt(|bias|^2 + var)`; `res` is the entry resolution "
      "`H[i,j] / sum_k |B_k[i,j]|^2` at the physical prior."
      % (args.angles, SIGMA_REF))
    A("")
    A("| ifreq | entry | band-peak \\|T\\| | 1% of peak | pred_sigma | "
      "res (selected span) | res (bright-10) | gateable |")
    A("|---|---|---|---|---|---|---|---|")
    for ifreq in sorted(store):
        S = store[ifreq]
        for e in range(len(S["entries"])):
            i, j = S["entries"][e]
            A("| %d | %s | %.3e | %.3e | %.3e | %.3f | %.3f | %s |"
              % (ifreq, fitmod.entry_label(fm.modes, i, j), S["peak"][e],
                 0.01 * S["peak"][e], S["pred"][e], S["res_entry"][e],
                 S["res10"][e], "yes" if S["gateable"][e] else
                 ("**NO (C4-violating)**" if S["c4v"][e] else "**NO**")))
    A("")
    A("%d of the 25 'bright' entries are C4-VIOLATING -- they are the "
      "reference file's own 0.2-0.4%% symmetry-violation noise.  Every "
      "symmetry-constrained span annihilates them exactly, so a retrieval "
      "can only ever return 0 there; they must be REMOVED from any gate "
      "list rather than counted as failures."
      % int(store[sorted(store)[0]]["c4v"].sum()))
    A("")

    A("## 5. Proposed recalibrated gates")
    A("")
    # aggregate numbers for the proposal
    prop = build_proposal(fm, store, args)
    A(prop)
    return "\n".join(L) + "\n", prop


def build_proposal(fm, store, args):
    """Assemble the numeric gate proposal from the measured tables."""
    L = []
    A = L.append
    # collect p90 at sigma = 3e-3 for the best realizable protocol
    per_class = {c: [] for c in CLASS_NAMES}
    per_class_qa = {c: [] for c in CLASS_NAMES}
    label_best = None
    label_qa = None
    beats_zero = {}
    for ifreq in sorted(store):
        z = store[ifreq]["st_zero"]
        for (label, sg), st in store[ifreq]["s3"].items():
            if abs(sg - SIGMA_REF) > 1e-12:
                continue
            if label.startswith("best-realizable"):
                label_best = label
                beats_zero[ifreq] = z["dipole"][0] / max(st["dipole"][0],
                                                         1e-300)
                for c in CLASS_NAMES:
                    per_class[c].append(st[c][2])
            elif label.startswith("QA-gate bound"):
                label_qa = label
                for c in CLASS_NAMES:
                    per_class_qa[c].append(st[c][2])
    Tmax = max(store[i]["Tmax_global"] for i in store)
    A("### 5.1 What is being replaced")
    A("")
    A("The doc's par.-6.2 gate (\"bright-subspace entries recovered to "
      "<= 1% relative\"), par.-6.3 gate (\"dipole block stable to <= 5% "
      "at sigma = 3e-3\") and par.-8 criterion 2 (\"dipole-quadrupole "
      "entries within 5-10% relative\") are all PER-ENTRY RELATIVE.  This "
      "study measures five distinct reasons they fail; they must not be "
      "reported as one \"information limit\":")
    A("")
    A("1. **(a) The per-entry-relative normalization is unattainable in "
      "principle for the weak bright entries.**  Bright-25 band-peak "
      "\\|T\\| spans %.1e .. %.1e while the off-span model error of the "
      "selected span is ~%.1e in T units and the propagated per-entry "
      "error bar at sigma = 3e-3 is %.1e .. %.1e.  For %d of 25 entries "
      "\"1%% of the entry\" is BELOW the estimator's own error bar."
      % (min(store[i]["peak"].min() for i in store),
         max(store[i]["peak"].max() for i in store),
         max(store[i]["model_err_T"] for i in store),
         min(store[i]["pred"].min() for i in store),
         max(store[i]["pred"].max() for i in store),
         max(int((0.01 * store[i]["peak"] < store[i]["pred"]).sum())
             for i in store)))
    A("2. **(b) Isotropic-prior mis-scaling.**  Section 3a compares an "
      "isotropic `||t||^2` penalty with a per-orbit diagonal prior on the "
      "SAME span, data and noise seeds; the difference is the prior, not "
      "the basis (see the basis-invariance measurement).")
    A("3. **(c) Born-seed basin failure -- the dominant cause.**  "
      "Quantified three ways: the truth-seeded vs Born-seeded table in "
      "section 2 (same span, same estimator, same data -- only the seed "
      "differs); the C-clean Born-reachability numbers in section 2b (the "
      "target is exactly in the span there, so any error is purely the "
      "landscape); and the basis-invariance measurement (a mathematically "
      "no-op change of basis moves the answer -> different local "
      "minimum).  Note also that the shrinkage-bias decomposition below "
      "accounts for only a small fraction of the measured error, so the "
      "remainder is nonlinearity / wrong basin, not the prior.")
    A("4. **(d) Genuine structural darkness**: even-m content is dark at "
      "theta = 0 (|dS/dT| ~ 1e-21 vs ~65 at the campaign set, already "
      "verified by `synthetic_test.py` / `observability.visibility_check`) "
      "and is only reachable with oblique + phi != 0/45 rows.")
    A("5. **(e) %d of the 25 'bright' entries are C4-violating file "
      "noise** and are annihilated by every symmetry-constrained span: "
      "they can only be returned as 0, so counting them as gate failures "
      "is a category error."
      % int(store[sorted(store)[0]]["c4v"].sum()))
    A("")
    A("### 5.2 Proposed gate form")
    A("")
    A("Replace per-entry-relative gates with a **two-part, "
      "observability-preconditioned** criterion.  For bright entry `e`:")
    A("")
    A("```")
    A("ACCEPT e   iff   |dT_e|  <=  max( k * pred_sigma_e ,  floor * "
      "|T|max_global )")
    A("GATE e     only if  res_e >= %.2f            (entry resolution at "
      "the chosen angle set and the physical prior)" % RES_C_GATE)
    A("REPORT e   (do not gate) otherwise")
    A("```")
    A("")
    A("with `pred_sigma_e` the estimator's own propagated per-entry error "
      "bar (linear theory: `sqrt(|bias_e|^2 + var_e)` from "
      "`predicted_entry_rms`; validated by `synthetic_test.py`'s step-3 "
      "theory gate at measured/predicted 0.81-1.93) and `|T|max_global` = "
      "%.3e for this reference file." % Tmax)
    A("")
    if label_qa is not None:
        A("**Table A -- QA-gate application** (a candidate tmat.h5 is "
          "supplied, so its own |z_k| set the per-orbit prior).  Measured "
          "90th percentiles at sigma = 3e-3 under `%s`, worst over "
          "frequencies %s.  **These are the actionable thresholds.**"
          % (label_qa, sorted(store)))
        A("")
        A("| class | measured p90 (peak-normalized) | proposed gate = 2x "
          "p90 |")
        A("|---|---|---|")
        for c in CLASS_NAMES:
            v = per_class_qa[c]
            if not v or not np.isfinite(np.nanmax(v)):
                continue
            A("| %s | %.3e | %.3e |" % (c, np.nanmax(v), 2 * np.nanmax(v)))
        A("")
    if label_best is not None:
        bz = min(beats_zero.values()) if beats_zero else float("nan")
        A("**Table B -- blind-retrieval application.**  Best REALIZABLE "
          "protocol (`%s`), p90 at sigma = 3e-3, worst over frequencies "
          "%s.  Its dipole `gain/zero` is %.2f at worst across the "
          "frequencies studied." % (label_best, sorted(store), bz))
        A("")
        A("| class | measured p90 (peak-normalized) | proposed gate = 2x "
          "p90 |")
        A("|---|---|---|")
        for c in CLASS_NAMES:
            v = per_class[c]
            if not v or not np.isfinite(np.nanmax(v)):
                continue
            A("| %s | %.3e | %.3e |" % (c, np.nanmax(v), 2 * np.nanmax(v)))
        A("")
        if np.isfinite(bz) and bz <= 1.0:
            A("> **Table B must not be used as a pass/fail gate.**  The "
              "selected realizable protocol does not beat the "
              "shrink-to-zero estimator on the dipole class (gain/zero "
              "%.2f <= 1), so a \"pass\" against these numbers would also "
              "be earned by a retrieval that returned nothing.  For the "
              "blind-retrieval application the honest deliverable at "
              "campaign-scale angle sets and sigma = 3e-3 is an UPPER "
              "BOUND per entry (|T_e| <= the Table-B value x band-peak), "
              "plus the observability map -- not a recovered value.  "
              "Closing this needs either a lower sigma (the par.-7 complex "
              "closure may deliver it) or a genuine physical per-orbit "
              "prior; frequency continuation alone fixes the BASIN (see "
              "section 2b: 1.104e+01 -> ~1e-12 on the C-clean loop at "
              "ifreq 48) but not the PRIOR." % bz)
            A("")
    A("Recommended constants from these measurements: **k = 3** (3x the "
      "propagated error bar -- a ~99%% one-sided interval for the roughly "
      "Gaussian per-entry error, and comfortably above the measured "
      "measured/predicted spread of 0.81-1.93) and **floor = 1e-2** of the "
      "GLOBAL \\|T\\|max, i.e. an absolute acceptance window of %.2e in T "
      "units.  Entries whose band-peak is below that window are accepted "
      "by the floor term and REPORTED as \"consistent with zero\", which "
      "is the honest statement for them." % (1e-2 * Tmax))
    A("")
    A("### 5.2b Two different applications, two different gates")
    A("")
    A("The oracle-primed E4o prior (`tik_k = 1/|z_k^true|^2`) is NOT "
      "realizable for a BLIND retrieval, but it IS realizable for the "
      "doc-par.-1 use case #1, the **QA gate for an externally supplied "
      "tmat.h5**: there a candidate T is handed over, and its own |z_k| "
      "legitimately supply the per-orbit prior scale.  The measured "
      "difference between E4o and the realizable E4r/E3 rows is therefore "
      "the difference between the two applications, and the design doc "
      "should gate them separately:")
    A("")
    A("* **QA-gate application** (candidate T supplied): use the E4o-style "
      "prior; the section-3b table gives its per-class error, which is the "
      "resolution with which a supplied T can be confirmed or rejected.")
    A("* **Blind-retrieval application**: use the realizable rows.  Where "
      "their `gain/zero` is <= 1 the retrieval carries NO information "
      "about that class beyond \"small\", and the honest deliverable is an "
      "upper bound, not a value.")
    A("")
    A("### 5.3 Passivity")
    A("")
    p_un, p_reg = [], []
    for i in store:
        o = store[i]["out"]
        for e, p in zip(o["s2_est"], o["s2_passivity"]):
            (p_un if e in ("E0", "E1") else p_reg).append(float(p))
    A("Passivity `max SV(I + 2 T)` is reported for every protocol in "
      "section 2.  Worst value over the UNREGULARIZED protocols (E0/E1): "
      "%.3f; worst over the regularized ones (E2/E3/E4): %.3f.  A passive "
      "cell requires <= 1 + eps, so anything materially above 1 is "
      "unphysical.  Recommendation: keep passivity as a POST-CHECK for "
      "regularized protocols, but any protocol without a prior must carry "
      "passivity as an ACTIVE penalty or constraint -- the doc's "
      "post-check alone accepts a superunitary T without complaint."
      % (max(p_un) if p_un else float("nan"),
         max(p_reg) if p_reg else float("nan")))
    A("")
    A("### 5.4 What changed and why")
    A("")
    A("* **Gate normalization**: per-entry relative -> max(propagated "
      "error bar, fixed fraction of global \\|T\\|max).  Reason: cause (a) "
      "above; several bright entries are 1e-4-scale while the information "
      "floor is 1e-5..1e-4, so a 1%-of-entry gate tests the noise, not "
      "the retrieval.")
    A("* **Observability precondition**: only entries with res_e >= %.2f "
      "at the chosen angle set are gated at all; the rest are published as "
      "UNOBSERVABLE with their resolution value.  Reason: cause (d) -- "
      "gating a structurally dark entry is meaningless." % RES_C_GATE)
    A("* **Estimator specification**: the campaign protocol must specify "
      "the PRIOR, not just the weights.  An isotropic `||t||^2` penalty is "
      "basis-invariant (measured, section 2) and therefore cannot be "
      "\"fixed\" by re-choosing an orthonormal basis; the lever is the "
      "per-direction prior scale.")
    A("* **Seeding**: Born seeding alone is not sufficient at every "
      "frequency; frequency continuation is cheap (the C cache covers all "
      "49 frequencies x 17 angles) and must be part of the protocol, with "
      "the C-clean Born-reachability number published per frequency as a "
      "landscape diagnostic.")
    A("* **par.-8 criterion 2** inherits all of the above: replace "
      "\"dipole-quadrupole entries within 5-10% relative\" with the "
      "section-5.2 form evaluated on the observable subset, and publish "
      "the ungateable list alongside.")
    A("")
    A("### 5.5 Explicitly UNGATEABLE bright entries")
    A("")
    for ifreq in sorted(store):
        S = store[ifreq]
        bad = [fitmod.entry_label(fm.modes, *S["entries"][e])
               for e in range(len(S["entries"])) if not S["gateable"][e]]
        A("* ifreq %d (%d of %d): %s"
          % (ifreq, len(bad), len(S["entries"]), ", ".join(bad)))
    A("")
    A("Reasons, per entry, are in the section-4b table: `res` below the "
      "%.2f cutoff (not resolved at this angle set / prior) or "
      "C4-violating (structurally annihilated).  These entries must be "
      "PUBLISHED WITH THEIR RESOLUTION VALUE, not silently passed or "
      "failed." % RES_C_GATE)
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Gate recalibration study (doc par. 6.2/6.3/8)")
    ap.add_argument("--freqs", default="32,48")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--sigmas", default="1e-3,3e-3,1e-2")
    ap.add_argument("--angles", default="campaign")
    ap.add_argument("--direction", type=int, choices=(-1, 1), default=-1)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--quick", action="store_true",
                    help="small trial counts / tighter nfev caps for a "
                         "smoke run")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--cont-span", type=int, default=6,
                    help="length of the frequency-continuation run")
    ap.add_argument("--max-nfev", type=int, default=300)
    ap.add_argument("--trial-nfev", type=int, default=150)
    ap.add_argument("--aux-trials", type=int, default=8,
                    help="trials for the seed-contrast, truth-seeded and "
                         "oracle rows of Study 3b (context, not headline)")
    ap.add_argument("--cont-chain", type=int, default=7,
                    help="frequency-chain length of the continuation "
                         "protocols (centred on the target frequency); 7 "
                         "matches the chain on which Study 2b measures the "
                         "noise-free continuation effect")
    ap.add_argument("--cont-screen-trials", type=int, default=2,
                    help="trials for the continuation half of the Study-3a "
                         "screening (each costs 2*cont_chain fits)")
    ap.add_argument("--win-nfev", type=int, default=500,
                    help="raised max_nfev for the SELECTED protocol in "
                         "Studies 3b/4 (a gate threshold must never rest "
                         "on an unconverged fit); capped rows are flagged")
    ap.add_argument("--sel-trials", type=int, default=12,
                    help="trials for the SELECTED protocol's noise ladder "
                         "(reduced because --win-nfev is raised)")
    ap.add_argument("--screen-trials", type=int, default=4,
                    help="trials used by the Study-3a protocol screening")
    ap.add_argument("--baseline-trials", type=int, default=6,
                    help="trials for the UNREGULARIZED baseline in Study 3 "
                         "(its fits cost ~10x the regularized ones)")
    ap.add_argument("--angle-trials", type=int, default=5,
                    help="trials per angle set in Study 4")
    ap.add_argument("--eb-rounds", type=int, default=3,
                    help="empirical-Bayes rounds of the realizable E4r")
    args = ap.parse_args(argv)
    args.sigma_list = [float(s) for s in args.sigmas.split(",")]
    if args.quick:
        args.trials = min(args.trials, 3)
        args.screen_trials = 2
        args.baseline_trials = 2
        args.angle_trials = 2
        args.aux_trials = 2
        args.cont_screen_trials = 1
        args.cont_chain = 3
        args.win_nfev = 200
        args.sel_trials = 3
        args.max_nfev = 150
        args.trial_nfev = 120
        args.cont_span = min(args.cont_span, 3)
        args.eb_rounds = 2

    t_all = time.time()
    hr("gate_study: adjudication of the par.-6.2 / 6.3 / 8 gates%s"
       % ("   [QUICK]" if args.quick else ""))
    fm = ForwardModel()
    print("cache: %s" % fm.cache_path)
    B68, _ = par.build_c4v_reciprocity_basis(fm.modes, verify_numeric=False)
    assert B68.shape[0] == 68
    if args.freqs.strip().lower() == "all":
        freqs = list(fm.available_freqs)
    else:
        freqs = [int(x) for x in args.freqs.split(",")]
    want = fitmod.resolve_angles(args.angles)
    store = {}
    done, skipped = [], []
    for ifreq in freqs:
        miss = [a for a in want if not fm.have[ifreq, a]]
        if miss:
            print("\n[skip] ifreq %d: angle indices %s not cached"
                  % (ifreq, miss), flush=True)
            skipped.append((ifreq, miss))
            continue
        out = run_frequency(fm, ifreq, B68, args, store)
        p = os.path.join(RESULTS_DIR, "gate_study_ifreq%02d.npz" % ifreq)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez(p, **{k: v for k, v in out.items() if v is not None})
        print("\n  saved: %s" % p, flush=True)
        done.append(ifreq)

    if not done:
        print("\nNo frequency could be processed (nothing cached).")
        return 1

    if not args.no_figs:
        sub("figures")
        try:
            make_figures(store, args)
        except Exception as exc:              # figures are optional
            print("  figure generation failed (non-fatal): %r" % exc)

    md, prop = write_markdown(fm, store, args, None)
    p = os.path.join(RESULTS_DIR, "GATE_STUDY.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(md)
    hr("Study 5: proposed recalibrated gates (also written to %s)" % p)
    print(prop)
    print("\nfrequencies done: %s; skipped: %s; total wall %.1f s"
          % (done, skipped, time.time() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())

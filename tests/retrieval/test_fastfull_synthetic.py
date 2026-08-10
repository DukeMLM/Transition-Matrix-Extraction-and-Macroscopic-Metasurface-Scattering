"""Gates for retrieval/fastfull/synthetic.py (M2 Gate A blind recovery).

Run:  python test_fastfull_synthetic.py   (from retrieval/, env cst_inference)

  (a) noise-free blind recovery is EXACT for a target that lies exactly in
      the 40-coefficient space -- both the wheel branch (seed + continuation
      + LM) and the generic algebraic branch.  This is the control that
      separates recovery error from the reference file's own D4h violation.
  (b) every perturbation model carries the declared per-entry RMS, and the
      structured ones are genuinely structured (low-rank / congruence form)
      rather than full-rank noise wearing a label
  (c) the M1 error bracket is VALIDATED by direct simulation: injecting iid
      noise reproduces the iid prediction, injecting the adversarial
      direction attains the systematic bound
  (d) the recovery basin is unique -- the blind linear seed and randomly
      perturbed seeds converge to the same T
  (e) the adversarial perturbation for a POOLED candidate is built on the
      stacked system, not per block (regression: doing it per block
      understated a pooled candidate's exposure by 9x)
  (f) the recovered T is exactly D4h + reciprocity by construction
"""
import os
import sys


import numpy as np

from tmatrix.retrieval.fastfull import lattice as lt
from tmatrix.retrieval.fastfull import transforms as xf
from tmatrix.retrieval.fastfull import jacobian as jac
from tmatrix.retrieval.fastfull import symmetry as sym
from tmatrix.retrieval.fastfull import synthetic as sy
from tmatrix.retrieval.fastfull import ewald as ew
from tmatrix.retrieval.fastfull import nuisance as nz
from tmatrix.retrieval.fastfull import design as dz

from tmatrix.aggregation.vswf import ModeBasis
from tmatrix.paths import DEMO_TMAT

RESULTS = []
MODES = ModeBasis.standard(3)
LAM = 8.0
K = 2 * np.pi / LAM
SIGMA = 2.8417e-3

WINNER = dz.Design(10.8121, 7.2371, 92.75, 19.68, -0.0098, -0.4930)
SECOND = dz.Design(9.4, 12.7, 84.0, 61.0, 0.317, -0.204)
CONS = dz.Constraints(kz_min_frac=0.2, wood_margin=0.05, n_orders_min=1,
                      n_orders_max=40, area_max_um2=5000.0)


def record(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s\n         %s" % ("PASS" if ok else "FAIL", name, detail),
          flush=True)


def _pieces(design):
    lat = design.lattice()
    o = lt.enumerate_orders(lat, K, f_bloch=(design.f1, design.f2),
                            kz_min_frac=CONS.kz_min_frac,
                            wood_margin=CONS.wood_margin)
    ch = lt.ChannelSet(o)
    A = xf.build_A(K, ch, MODES)
    W = xf.build_W(K, ch, MODES)
    C = ew.lattice_sum_C(lat, K, MODES, lat.bloch(design.f1, design.f2))
    return ch, A, W, C


# --------------------------------------------------------------- (a) control

def test_noise_free_exact(B):
    rng = np.random.default_rng(3)
    T = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.25)[0][0]
    ch, A, W, C = _pieces(WINNER)
    S = xf.scattered_S(W, T, C, A)
    T_hat, info = sy.recover_wheel([(W, A, C, S)], B, SIGMA, n_multistart=2,
                                   rng=rng)
    e = float(np.linalg.norm(T_hat - T) / np.linalg.norm(T))
    record("(a) noise-free wheel recovery is exact on an exactly-D4h target",
           e < 1e-10,
           "relative ||T_hat - T||_F = %.3e over %d channels, ||C T|| = "
           "%.4f (the problem is nonlinear but nearly Born); multistart "
           "spread %.1e" % (e, ch.n,
                            ew.lattice_dressing_strength(C, T)["norm_CT"],
                            info["multistart_spread"]))

    ch2, A2, W2, C2 = _pieces(dz.Design(13.7913, 13.5664, 89.74, 49.16,
                                        -0.5, -0.5))
    if xf.generic_track_metrics(A2, W2)["full_rank"]:
        S2 = xf.scattered_S(W2, T, C2, A2)
        Tg, ginfo = sy.recover_generic(S2, W2, A2, C2)
        eg = float(np.linalg.norm(Tg - T) / np.linalg.norm(T))
        record("(a) noise-free generic algebraic branch is exact",
               eg < 1e-10,
               "relative error %.3e over %d channels; "
               "sigma_min(I + Teff C) = %.4f"
               % (eg, ch2.n, ginfo["deembed"]["sigma_min"]))
    else:
        record("(a) noise-free generic algebraic branch is exact", False,
               "the reference generic cell is not full rank")


# ------------------------------------------------------ (b) perturbations

def test_perturbation_structure():
    ch, A, W, C = _pieces(WINNER)
    rng = np.random.default_rng(11)
    T = np.zeros((30, 30), dtype=complex)
    B, _ = sym.build_d4h_reciprocity_basis(MODES, verify_numeric=False)
    T = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.25)[0][0]
    S = xf.scattered_S(W, T, C, A)
    H = jac.jacobian(W, A, B, T0=T, C=C) / SIGMA
    rows = []
    ok = True
    for model in sy.ERROR_MODELS:
        dS, info = sy.make_perturbation(model, ch, S, rng, SIGMA,
                                        H_whitened=H)
        rms_ok = abs(info["rms"] - SIGMA) <= 1e-6 * SIGMA
        ok &= rms_ok
        rows.append("%s: rms %.4e, rank %d/%d, max %.2e%s"
                    % (model, info["rms"], info["rank"], ch.n,
                       info["max_abs"],
                       "" if info["parameter"] is None
                       else ", parameter %.4g" % info["parameter"]))
    record("(b) every error model carries the declared per-entry RMS "
           "%.4e" % SIGMA, ok, "; ".join(rows))

    # the diagonal-congruence models must really be congruences: the
    # entrywise ratio (S + dS)/S is then exactly the rank-1 outer product
    # d (x) d of per-channel factors.  A calibrated PARAMETER keeps that
    # exact; rescaling dS to hit the RMS would not (it broke this gate at
    # 1.2e-3 before `_calibrate` was introduced).
    for model in ("reference_plane", "angular_smooth"):
        d0 = sy.make_perturbation(model, ch, S, np.random.default_rng(4),
                                  SIGMA)[0]
        sv = np.linalg.svd((S + d0) / S, compute_uv=False)
        record("(b) the %s model is exactly a congruence S -> D S D" % model,
               sv[1] / sv[0] < 1e-10,
               "entrywise ratio (S+dS)/S has sv[1]/sv[0] = %.2e -- rank 1, "
               "an outer product of per-channel factors, i.e. a genuine "
               "one-parameter systematic rather than noise wearing a label"
               % (sv[1] / sv[0]))

    # iid must be the opposite: full rank in the same ratio
    d_iid = sy.make_perturbation("iid", ch, S, np.random.default_rng(4),
                                 SIGMA)[0]
    sv_iid = np.linalg.svd((S + d_iid) / S, compute_uv=False)
    record("(b) iid noise is full rank in the same ratio (the contrast)",
           sv_iid[1] / sv_iid[0] > 1e-3,
           "sv[1]/sv[0] = %.2e for iid against < 1e-10 for the structured "
           "models -- this is the quantitative sense in which the injected "
           "discrepancies differ" % (sv_iid[1] / sv_iid[0]))


# ------------------------------------------------------- (c) bracket check

def test_bracket_validated(B):
    ch, A, W, C = _pieces(WINNER)
    rng = np.random.default_rng(5)
    T = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.25)[0][0]
    S = xf.scattered_S(W, T, C, A)
    n_obs = ch.n ** 2
    sig = jac.sigma_uniform(n_obs, SIGMA)
    H = jac.jacobian(W, A, B, T0=T, C=C) / SIGMA
    pred = jac.recovery_errors(B, MODES, jac.coefficient_covariance(H),
                               n_obs, T, sigma_used=SIGMA)

    errs = {}
    for model in ("iid", "adversarial"):
        vals = []
        for t in range(6 if model == "iid" else 1):
            r = np.random.default_rng(100 + t)
            dS, _ = sy.make_perturbation(model, ch, S, r, SIGMA,
                                         H_whitened=H)
            T_hat, _ = sy.recover_wheel([(W, A, C, S + dS)], B, SIGMA)
            vals.append(float(np.linalg.norm(T_hat - T) / np.linalg.norm(T)))
        errs[model] = float(np.mean(vals))

    iid_ok = abs(errs["iid"] / pred["fro_err_iid"] - 1.0) < 0.25
    adv_ok = abs(errs["adversarial"] / pred["fro_err_sys"] - 1.0) < 0.02
    record("(c) the M1 error bracket is validated by direct simulation",
           iid_ok and adv_ok,
           "injected iid noise gives %.4f%% against the predicted iid "
           "%.4f%% (ratio %.3f, 6 trials); the adversarial direction gives "
           "%.4f%% against the predicted systematic bound %.4f%% (ratio "
           "%.4f)" % (100 * errs["iid"], 100 * pred["fro_err_iid"],
                      errs["iid"] / pred["fro_err_iid"],
                      100 * errs["adversarial"], 100 * pred["fro_err_sys"],
                      errs["adversarial"] / pred["fro_err_sys"]))


# ------------------------------------------------------------- (d) basin

def test_basin_unique(B):
    ch, A, W, C = _pieces(WINNER)
    rng = np.random.default_rng(9)
    T = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.25)[0][0]
    S = xf.scattered_S(W, T, C, A)
    dS, _ = sy.make_perturbation("reference_plane", ch, S, rng, SIGMA)
    T_hat, info = sy.recover_wheel([(W, A, C, S + dS)], B, SIGMA,
                                   n_multistart=5, rng=rng)
    record("(d) the blind recovery basin is unique under 5 perturbed seeds",
           info["multistart_unique"],
           "worst relative spread between multistart solutions = %.2e; the "
           "seed is the blind C = 0 least-squares solution and continuation "
           "walks eta 0 -> 1.  Contrast the specular phase, where no "
           "realizable seed reached the physical basin (RETRIEVAL_LIMIT.md)"
           % info["multistart_spread"])


# ------------------------------------------------------- (e) pooled worst case

def test_pooled_adversarial(B):
    """The stacked worst case must be at least as bad as any per-block one."""
    rng = np.random.default_rng(13)
    T = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.25)[0][0]
    blocks, Hs = [], []
    for d in (WINNER, SECOND):
        ch, A, W, C = _pieces(d)
        S = xf.scattered_S(W, T, C, A)
        blocks.append((W, A, C, S))
        Hs.append(jac.jacobian(W, A, B, T0=T, C=C) / SIGMA)
    H = np.vstack(Hs)
    n_obs = sum(b[3].size for b in blocks)

    U, _, _ = np.linalg.svd(H, full_matrices=False)
    v = sy._scale_to_rms(U[:, -1], SIGMA)
    stacked, off = [], 0
    for (W, A, C, S) in blocks:
        stacked.append((W, A, C, S + v[off:off + S.size].reshape(S.shape)))
        off += S.size
    T_st, _ = sy.recover_wheel(stacked, B, SIGMA)
    e_st = float(np.linalg.norm(T_st - T) / np.linalg.norm(T))

    perblock, r = [], np.random.default_rng(2)
    for (W, A, C, S), Hb in zip(blocks, Hs):
        ch = lt.ChannelSet(lt.enumerate_orders(
            (WINNER if len(perblock) == 0 else SECOND).lattice(), K,
            f_bloch=((WINNER.f1, WINNER.f2) if len(perblock) == 0
                     else (SECOND.f1, SECOND.f2)),
            kz_min_frac=CONS.kz_min_frac, wood_margin=CONS.wood_margin))
        dS, _ = sy.make_perturbation("adversarial", ch, S, r, SIGMA,
                                     H_whitened=Hb)
        perblock.append((W, A, C, S + dS))
    T_pb, _ = sy.recover_wheel(perblock, B, SIGMA)
    e_pb = float(np.linalg.norm(T_pb - T) / np.linalg.norm(T))

    pred = jac.recovery_errors(B, MODES, jac.coefficient_covariance(H),
                               n_obs, T, sigma_used=SIGMA)
    record("(e) the pooled adversarial case is built on the stacked system",
           e_st >= e_pb and abs(e_st / pred["fro_err_sys"] - 1.0) < 0.05,
           "stacked worst case %.3f%% vs per-block construction %.3f%% "
           "(%.1fx); the stacked value matches the predicted bound %.3f%% "
           "to %.1f%%" % (100 * e_st, 100 * e_pb, e_st / max(e_pb, 1e-12),
                          100 * pred["fro_err_sys"],
                          100 * abs(e_st / pred["fro_err_sys"] - 1.0)))


# --------------------------------------------------------- (f) symmetry

def test_seed_determinism():
    """The trial seeds must not depend on Python's process hash randomization.

    The published Gate A run derived trial seeds from `hash(model)`, which is
    randomized per process; two nominally identical probes moved the iid
    error from 3.014% to 2.774%.  This gate pins the replacement seed tree.
    """
    want = [sy.trial_seed(12345, m, 0) for m in sy.ERROR_MODELS]
    again = [sy.trial_seed(12345, m, 0) for m in sy.ERROR_MODELS]
    distinct = len(set(want)) == len(want)
    ordered = sy.trial_seed(1, "iid", 0) != sy.trial_seed(1, "iid", 1)
    record("(g) trial seeds are deterministic and collision free",
           want == again and distinct and ordered,
           "%d models give %d distinct seeds, stable within and (by "
           "construction, SeedSequence) across processes; trial index "
           "changes the seed" % (len(want), len(set(want))))


def test_joint_recovery_matched(B):
    """Fitting the calibration removes the dominant systematic."""
    ch, A, W, C = _pieces(WINNER)
    rng = np.random.default_rng(sy.trial_seed(12345, "reference_plane", 0))
    T = None
    from tmatrix.aggregation.tmat_io import TMatrixData
    data = TMatrixData(str(DEMO_TMAT))
    T = data.T[int(np.argmin(np.abs(data.wavelength_um - LAM)))]
    S = xf.scattered_S(W, T, C, A)
    dS, info = sy.make_perturbation("reference_plane", ch, S, rng, SIGMA)
    T1, _ = sy.recover_wheel([(W, A, C, S + dS)], B, SIGMA)
    e1 = float(np.linalg.norm(T1 - T) / np.linalg.norm(T))
    cal = nz.Calibration(ch, ("ref_plane",))
    T2, eta, _ = sy.recover_joint([(W, A, C, S + dS)], B, SIGMA, [cal])
    e2 = float(np.linalg.norm(T2 - T) / np.linalg.norm(T))
    record("(g) joint T + calibration recovery removes the port-plane alias",
           e2 < 0.05 and e2 < 0.15 * e1,
           "T-only fit gives %.2f%%, joint fit %.2f%% (%.0fx better); the "
           "recovered offset is %.4f um against a true %.4f um.  The alias "
           "is removable by ESTIMATING it, not by a better cell."
           % (100 * e1, 100 * e2, e1 / e2, eta[0], info["parameter"]))


def test_joint_recovery_misspecified(B):
    """...and is catastrophic when the fitted family is wrong.

    This is the gate that stops "fit the calibration" from becoming a free
    win.  Because the nuisance tangents are ~99.98% collinear with T, giving
    them free rein inflates variance enormously when the systematic they
    describe is not actually present.
    """
    ch, A, W, C = _pieces(WINNER)
    from tmatrix.aggregation.tmat_io import TMatrixData
    data = TMatrixData(str(DEMO_TMAT))
    T = data.T[int(np.argmin(np.abs(data.wavelength_um - LAM)))]
    S = xf.scattered_S(W, T, C, A)
    rng = np.random.default_rng(sy.trial_seed(12345, "iid", 0))
    dS, _ = sy.make_perturbation("iid", ch, S, rng, SIGMA)
    T1, _ = sy.recover_wheel([(W, A, C, S + dS)], B, SIGMA)
    e1 = float(np.linalg.norm(T1 - T) / np.linalg.norm(T))
    cal = nz.Calibration(ch, ("ref_plane", "angular_rx", "angular_tx"))
    T2, _, _ = sy.recover_joint([(W, A, C, S + dS)], B, SIGMA, [cal])
    e2 = float(np.linalg.norm(T2 - T) / np.linalg.norm(T))
    record("(g) an unmatched joint fit is much WORSE than not fitting",
           e2 > 3.0 * e1,
           "on iid data (no systematic present) the T-only fit gives "
           "%.2f%% and the 15-parameter joint fit %.2f%% -- %.0fx worse.  "
           "The joint estimator is only admissible with a CALIBRATED prior; "
           "free nuisance parameters are not a free win."
           % (100 * e1, 100 * e2, e2 / e1))


def test_marginalized_matches_joint_loss(B):
    """The marginalized objective predicts the joint fit's variance cost."""
    ch, A, W, C = _pieces(WINNER)
    rng = np.random.default_rng(7)
    T = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.25)[0][0]
    info = nz.marginalized_information(W, A, B, T, C, SIGMA, ch,
                                       classes=("ref_plane",))
    S = xf.scattered_S(W, T, C, A)
    errs = {}
    for tag, joint in (("t_only", False), ("joint", True)):
        vals = []
        for t in range(4):
            r = np.random.default_rng(500 + t)
            dS, _ = sy.make_perturbation("iid", ch, S, r, SIGMA)
            if joint:
                cal = nz.Calibration(ch, ("ref_plane",))
                Th, _, _ = sy.recover_joint([(W, A, C, S + dS)], B, SIGMA,
                                            [cal])
            else:
                Th, _ = sy.recover_wheel([(W, A, C, S + dS)], B, SIGMA)
            vals.append(float(np.linalg.norm(Th - T) / np.linalg.norm(T)))
        errs[tag] = float(np.mean(vals))
    ratio = errs["joint"] / errs["t_only"]
    record("(g) the generalized loss BOUNDS the joint fit's variance cost",
           1.0 <= ratio <= info["generalized_loss"] * 1.05
           and info["generalized_loss"] >= info["sigma_ratio"],
           "admitting one port-plane parameter multiplies the iid T error by "
           "%.2fx (measured, 4 trials).  The bound is the GENERALIZED worst "
           "inflation sqrt(lambda_max(F_free, F_marg)) = %.2fx; the ratio of "
           "the two smallest singular values, %.2fx, is a weaker diagnostic "
           "and is not a bound because the weakest directions before and "
           "after marginalization differ.  Four random realizations cannot "
           "establish a bound -- this gate only checks consistency with one."
           % (ratio, info["generalized_loss"], info["sigma_ratio"]))


def test_ensemble_diversity(B):
    """The design ensemble must cover more than one direction.

    The original `convex` generator S = (1-t) I + t Y_hat gives every draw the
    same dominant -I component: measured identity cosine 0.884-0.906,
    pairwise 0.743-0.830, participation-ratio effective rank **1.44 of 40**,
    against the reference wheel's identity cosine of 0.208.  A design
    optimized on that is optimized on one direction, which is why the
    marginalized winner did not transfer to the wheel.
    """
    rng = np.random.default_rng(20260807)
    old, _ = sym.random_passive_d4h(B, rng, n_draw=6, target_fro=0.25,
                                    method="convex")
    d_old = sym.ensemble_diversity(B, old)
    # exercise the PRODUCTION configuration: the mixed loss grid the search
    # actually uses, not a single favourable bin
    from tmatrix.retrieval.fastfull import opt_marginalized as om
    rng = np.random.default_rng(20260807)
    per = max(1, 6 // len(om.LOSS_GRID))
    new = np.concatenate([
        sym.random_passive_d4h(B, rng, n_draw=per, target_fro=0.25,
                               loss_factor=lf)[0] for lf in om.LOSS_GRID])
    d_new = sym.ensemble_diversity(B, new)
    pas = sym.passivity_max_sv(new)
    res = float(np.abs(sym.symmetry_residual(new, B)).max())
    record("(h) the Cayley ensemble is diverse, passive and exactly D4h",
           d_new["effective_rank"] > 3.0 * d_old["effective_rank"]
           and d_new["identity_cosine_max"] < 0.8
           and pas <= 1.0 + 1e-8 and res < 1e-12,
           "effective rank %.2f vs %.2f (convex), identity cosine %.3f vs "
           "%.3f (wheel 0.208), pairwise max %.3f vs %.3f; max SV(I+2T) = "
           "%.6f, symmetry residual %.1e"
           % (d_new["effective_rank"], d_old["effective_rank"],
              d_new["identity_cosine_max"], d_old["identity_cosine_max"],
              d_new["pairwise_cosine_max"], d_old["pairwise_cosine_max"],
              pas, res))

    # the loss knob must move the identity alignment monotonically
    cos = []
    for lf in (0.05, 0.5, 1.0):
        rng = np.random.default_rng(7)
        e, _ = sym.random_passive_d4h(B, rng, n_draw=4, target_fro=0.25,
                                      loss_factor=lf)
        cos.append(sym.ensemble_diversity(B, e)["identity_cosine_max"])
    record("(h) the loss factor controls how grey the ensemble is",
           cos[0] < cos[1] < cos[2],
           "identity cosine %.3f -> %.3f -> %.3f at loss factor 0.05 -> 0.5 "
           "-> 1.0; a lossless resonant draw sits near the wheel, a fully "
           "absorbing one reproduces the convex generator's collapse"
           % tuple(cos))


def test_prior_and_shared_map(B):
    """The calibrated-prior and shared-parameter paths must actually work."""
    ch, A, W, C = _pieces(WINNER)
    rng = np.random.default_rng(sy.trial_seed(12345, "reference_plane", 0))
    from tmatrix.aggregation.tmat_io import TMatrixData
    data = TMatrixData(str(DEMO_TMAT))
    T = data.T[int(np.argmin(np.abs(data.wavelength_um - LAM)))]
    S = xf.scattered_S(W, T, C, A)
    dS, info = sy.make_perturbation("reference_plane", ch, S, rng, SIGMA)
    dL = info["parameter"]

    cal = nz.Calibration(ch, ("ref_plane",))
    # a correlated prior centred on the TRUE offset must beat a wrong centre
    T_good, e_good, _ = sy.recover_joint([(W, A, C, S + dS)], B, SIGMA, [cal],
                                         prior_precision=np.array([[100.0]]),
                                         prior_mean=np.array([dL]))
    T_bad, e_bad, _ = sy.recover_joint([(W, A, C, S + dS)], B, SIGMA, [cal],
                                       prior_precision=np.array([[100.0]]),
                                       prior_mean=np.array([0.0]))
    g = float(np.linalg.norm(T_good - T) / np.linalg.norm(T))
    b = float(np.linalg.norm(T_bad - T) / np.linalg.norm(T))
    record("(h) a correlated prior with a calibrated mean is used correctly",
           g < 0.05 and b > 3 * g,
           "prior centred on the true offset %.4f um gives %.2f%% T error "
           "(fitted %.4f); the same precision centred on 0 gives %.2f%%.  "
           "The prior MEAN is what carries the calibration, and a wrong one "
           "is worse than none." % (dL, 100 * g, e_good[0], 100 * b))

    # two blocks sharing ONE physical offset
    ch2, A2, W2, C2 = _pieces(SECOND)
    S2 = xf.scattered_S(W2, T, C2, A2)
    d2 = (sy._congruence(S2, np.exp(0.5j * dL * np.asarray(ch2.kz))) - S2)
    blocks = [(W, A, C, S + dS), (W2, A2, C2, S2 + d2)]
    cals = [nz.Calibration(ch, ("ref_plane",)),
            nz.Calibration(ch2, ("ref_plane",))]
    T_tied, e_tied, i_tied = sy.recover_joint(blocks, B, SIGMA, cals,
                                              param_map=[[0], [0]])
    T_free, e_free, i_free = sy.recover_joint(blocks, B, SIGMA, cals)
    record("(h) param_map ties one physical offset across two cells",
           i_tied["n_eta"] == 1 and i_free["n_eta"] == 2
           and abs(e_tied[0] - dL) < 0.05,
           "tied fit has 1 parameter and recovers %.4f um against a true "
           "%.4f; the default disjoint map allocates 2 and fits (%.4f, "
           "%.4f).  Passing the same Calibration object does NOT tie them -- "
           "only param_map does."
           % (e_tied[0], dL, e_free[0], e_free[1]))


def test_cayley_is_exact(B):
    """At zero loss the generator must produce a UNITARY S, not merely
    max SV(S) <= 1.

    Rescaling T after the Cayley map leaves the bounded-real manifold and
    injects absorption that nobody asked for: it gave ||S^H S - I||_2 = 0.043
    and mean apparent absorption 0.0075 at loss_factor 0, against the
    reference wheel's 0.00055.  `max SV(S) <= 1` cannot see that, because the
    attenuation sits in the other singular directions.  The scale of K is now
    root-solved BEFORE the map, so the map itself is exact.
    """
    rows, ok = [], True
    for lf in (0.0, 0.05, 0.5):
        rng = np.random.default_rng(7)
        e, _ = sym.random_passive_d4h(B, rng, n_draw=3, target_fro=0.113992,
                                      loss_factor=lf)
        a = sym.absorption_spectrum(e)
        u = max(x["unitarity_resid"] for x in a)
        ab = float(np.mean([x["mean_absorption"] for x in a]))
        sv = max(x["sv_max"] for x in a)
        nrm = float(np.abs([np.linalg.norm(T) for T in e] - np.array(0.113992)
                           ).max())
        rows.append("lf %.2f: unitarity %.2e, absorption %+.6f, maxSV %.6f"
                    % (lf, u, ab, sv))
        if lf == 0.0:
            ok &= u < 1e-9 and abs(ab) < 1e-9
        ok &= sv <= 1.0 + 1e-9 and nrm < 1e-9
    # and the loss knob must move absorption monotonically
    ab = []
    for lf in (0.0, 0.05, 0.5):
        rng = np.random.default_rng(7)
        e, _ = sym.random_passive_d4h(B, rng, n_draw=3, target_fro=0.113992,
                                      loss_factor=lf)
        ab.append(float(np.mean([x["mean_absorption"]
                                 for x in sym.absorption_spectrum(e)])))
    ok &= ab[0] < ab[1] < ab[2]
    record("(h) the Cayley map is exact: zero loss gives a unitary S",
           ok, "%s; absorption rises %.6f -> %.6f -> %.6f with the loss knob "
               "and ||T||_F hits the target exactly (the wheel sits at "
               "unitarity 4.06e-3, absorption 5.51e-4, i.e. near loss 0.005)"
               % ("; ".join(rows), ab[0], ab[1], ab[2]))


def test_param_map_validation(B):
    """Invalid parameter maps must be rejected, not silently mis-solved."""
    ch, A, W, C = _pieces(WINNER)
    ch2, A2, W2, C2 = _pieces(SECOND)
    rng = np.random.default_rng(3)
    T = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.2)[0][0]
    blocks = [(W, A, C, xf.scattered_S(W, T, C, A)),
              (W2, A2, C2, xf.scattered_S(W2, T, C2, A2))]
    cals = [nz.Calibration(ch, ("ref_plane",)),
            nz.Calibration(ch2, ("ref_plane",))]
    try:
        sy.recover_joint(blocks, B, SIGMA, cals[:1])
        extra = "calib/block mismatch ACCEPTED"
    except ValueError:
        extra = "calib/block mismatch rejected"
    bad = [("negative index", dict(param_map=[[-1], [0]])),
           ("hole", dict(param_map=[[0], [2]])),
           ("wrong block count", dict(param_map=[[0]])),
           ("n_eta too small", dict(param_map=[[0], [1]], n_eta=1)),
           ("fractional index", dict(param_map=[[0.9], [0.1]])),
           ("n_eta too large", dict(param_map=[[0], [0]], n_eta=3)),
           ("bad seed_eta length", dict(param_map=[[0], [1]],
                                        seed_eta=[0.0, 0.0, 0.0]))]
    rows, ok = [], True
    for name, kw in bad:
        try:
            sy.recover_joint(blocks, B, SIGMA, cals, **kw)
            rows.append("%s ACCEPTED" % name)
            ok = False
        except ValueError:
            rows.append("%s rejected" % name)
    ok &= extra.endswith("rejected")
    rows.append(extra)
    record("(h) invalid param_map is rejected", ok,
           "; ".join(rows) + " -- a hole or an oversized n_eta is an "
           "unidentifiable flat direction; a fractional index would floor "
           "into a silent parameter share")


def test_generator_contract(B):
    """The absorption diagnostic must be the operator norm, the loss grid
    must bracket the wheel, and invalid inputs must be rejected."""
    rng = np.random.default_rng(7)
    e, _ = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.25,
                                  loss_factor=0.5)
    S = np.eye(30, dtype=complex) + 2.0 * e[0]
    R = S.conj().T @ S - np.eye(30)
    a = sym.absorption_spectrum(e)[0]
    ok_norm = abs(a["unitarity_resid"] - np.linalg.norm(R, 2)) < 1e-12
    record("(h) absorption_spectrum reports the true operator norm",
           ok_norm and a["unitarity_resid"] > np.abs(R).max(),
           "operator norm %.5f against the max element magnitude %.5f "
           "(%.1f%% higher) -- the docstring promised the former and the "
           "code computed the latter"
           % (a["unitarity_resid"], np.abs(R).max(),
              100 * (a["unitarity_resid"] / np.abs(R).max() - 1)))

    rows, ok = [], True
    for name, kw in (("target_fro <= 0", dict(target_fro=-0.1)),
                     ("target_fro NaN", dict(target_fro=float("nan"))),
                     ("loss_factor < 0", dict(target_fro=0.2,
                                              loss_factor=-1.0))):
        try:
            sym.random_passive_d4h(B, rng, n_draw=1, **kw)
            rows.append("%s ACCEPTED" % name)
            ok = False
        except ValueError:
            rows.append("%s rejected" % name)
    record("(h) the generator rejects invalid norms and loss factors", ok,
           "; ".join(rows))

    # Gate the EXACT PRODUCTION configuration -- same norm, same seed, same
    # construction the optimizer uses -- and report the true mismatch rather
    # than a favourable one measured at the reference norm.  The earlier
    # version of this gate drew at target_fro = 0.113992 while the optimizer
    # draws at ENSEMBLE_FRO = 0.25, so it certified a configuration nothing
    # ran.
    from tmatrix.retrieval.fastfull import opt_marginalized as om
    rng = np.random.default_rng(20260807)
    per = max(1, 6 // len(om.LOSS_GRID))
    prod = np.concatenate([
        sym.random_passive_d4h(B, rng, n_draw=per,
                               target_fro=om.ENSEMBLE_FRO, loss_factor=lf)[0]
        for lf in om.LOSS_GRID])
    ab_prod = [x["mean_absorption"] for x in sym.absorption_spectrum(prod)]
    w = om.WHEEL_ABSORPTION_8UM
    ratio = float(np.mean(ab_prod) / w)
    # at the REFERENCE norm the grid does bracket the wheel; at production
    # norm it does not, and the gate records both rather than hiding one
    ab_ref = []
    for lf in om.LOSS_GRID:
        r = np.random.default_rng(11)
        d, _ = sym.random_passive_d4h(B, r, n_draw=3, target_fro=0.113992,
                                      loss_factor=lf)
        ab_ref.append(float(np.mean([x["mean_absorption"]
                                     for x in sym.absorption_spectrum(d)])))
    record("(h) the loss grid is measured at the PRODUCTION configuration",
           min(ab_ref) < w < max(ab_ref) and om.TARGET_CONDITIONED_PRIOR
           and 1.0 < ratio < 10.0,
           "at the reference norm the grid brackets the wheel (%s vs %.3e); "
           "at the PRODUCTION norm %.2f it gives %.3e..%.3e, mean %.2fx the "
           "wheel -- so the production ensemble is NOT wheel-matched.  The "
           "grid is also flagged TARGET_CONDITIONED_PRIOR because it was "
           "chosen from the reserved reference T, so nothing selected with "
           "it can close Gate A or E."
           % (["%.2e" % x for x in ab_ref], w, om.ENSEMBLE_FRO,
              min(ab_prod), max(ab_prod), ratio))


D_VALID = dz.Design(9.0, 9.0, 90.0, 0.0, 0.1, 0.1)


_FIXTURE_ENS = {}


def _fixture_ensembles(n=2, seed=1, grid=(0.0025, 0.005), stress=0.05,
                       fro=0.25):
    """The fixture's ensembles, built by the PRODUCTION helper and cached.

    The verifier now rebuilds these from the hashed config, so the fixture
    cannot invent side labels: `p0`/`s0` used to satisfy a check that asked
    only for nonempty, unique, disjoint strings.
    """
    key = (n, seed, grid, stress, fro)
    if key not in _FIXTURE_ENS:
        from tmatrix.retrieval.fastfull import opt_marginalized as om
        Bl, _ = sym.build_d4h_reciprocity_basis(MODES, verify_numeric=False)
        ep, es, _ = om.build_paired_ensembles(Bl, seed, n, grid=grid,
                                              stress_loss=stress,
                                              target_fro=fro)
        _FIXTURE_ENS[key] = (ep, es, om.paired_ensemble_ids(Bl, seed, n))
    return _FIXTURE_ENS[key]


def _paired_stub(n=2, seed=1):
    """A REAL paired block, rebuildable from the fixture's hashed config.

    Three weaker versions were caught here.  A stub of `{n_pairs,
    paired_by_latent_id: True}` satisfied a verifier that trusted the flag;
    rows with arbitrary sha-shaped ids satisfied one that checked only their
    syntax; and `production_row="p0"` satisfied one that asked only for
    distinct nonempty strings.  Both ensembles are now regenerated during
    verification and every row hash must match.
    """
    from tmatrix.retrieval.fastfull import opt_marginalized as om
    ep, es, ids = _fixture_ensembles(n, seed)
    pid, sid = om.ensemble_row_ids(ep), om.ensemble_row_ids(es)
    rows_p = [{"sigma_marg": 10.0 + i} for i in range(n)]
    rows_s = [{"sigma_marg": 9.5 + i} for i in range(n)]
    return om.paired_stress_stats(
        rows_p, rows_s, pair_loss=[0.0025, 0.005][:n], pair_ids=ids,
        prod_ids=pid, expect_prod=pid, stress_ids=sid, expect_stress=sid)


def _make_run(runs, om, design, tag, lineage=None, marker=None, manifest=True,
              tamper=None, rec_rid=None, mutate=None, drop=(), rid=None,
              rec_mutate=None, res_mutate=None, sel_mutate=None,
              post=None, no_receipt=False, cfg_mutate=None):
    """Hand-build a VALID run directory, then damage it in one specific way.

    The run id is DERIVED from the snapshot and config bodies exactly as the
    optimizer derives it, so the baseline case is genuinely well-formed and
    every rejection below is attributable to the single change made.  `tag`
    enters the config, so each case derives its own distinct run id instead of
    every case colliding into one directory.
    """
    import json
    lin = lineage or om.LINEAGE_CONDITIONED
    snap = {"fastfull/probe.py": "0" * 64, "__env__": "test"}
    cfg = {"target_conditioned": (lin == om.LINEAGE_CONDITIONED),
           "probe": tag, "seed": 1, "samples": 4, "polish": 0,
           "n_ensemble": 2, "lam_um": 8.0, "sigma": 2.8417e-3,
           "stress_loss": 0.05, "ensemble_fro": 0.25, "generator": "cayley",
           "loss_grid": [0.0025, 0.005], "skip_gate_a": True,
           "q_eta": 0.0, "constraints": {"kz_min_frac": 0.2},
           "gate_a_candidates": [],
           "archive_n": 0, "archive_sha256": om._canon_hash({}),
           "archive_lineages": []}
    if cfg_mutate:
        cfg_mutate(cfg)
    rid = rid or om.derive_run_id(snap, cfg)
    rd = os.path.join(runs, rid)
    os.makedirs(rd, exist_ok=True)
    D = design or D_VALID
    rec = om.make_record(D, "search", rec_rid or rid,
                         om._canon_hash(snap), om._canon_hash(cfg))
    rec["lineage"] = lin
    rec["target_conditioned"] = (lin == om.LINEAGE_CONDITIONED)
    rec["proposal_lineage"] = lin
    sel = om.make_record(
        D, "selected:cand-" + tag, rec_rid or rid,
        om._canon_hash(snap), om._canon_hash(cfg),
        proposal_proof=om._proof(
            "same_run", rec_rid or rid, "cand-" + tag, D, lin,
            parent_record_digest=om.record_digest(rec)))
    sel["lineage"] = lin
    sel["target_conditioned"] = (lin == om.LINEAGE_CONDITIONED)
    sel["proposal_lineage"] = lin
    if design is None:                       # malformed-design counterexample
        rec["design"] = {}
    if rec_mutate:
        rec_mutate(rec)
    if sel_mutate:
        sel_mutate(sel)
    for nm, body in (("result.json",
                      dict(schema_version=om.RESULT_SCHEMA_VERSION,
                           evidence_status="screening-only",
                           run_id=rid,
                           snapshot_sha256=om._canon_hash(snap),
                           config_sha256=om._canon_hash(cfg),
                           snapshot=snap, config=cfg,
                           winner=D.to_dict(),
                           winner_source="cand-" + tag,
                           target_conditioned_prior=(
                               lin == om.LINEAGE_CONDITIONED),
                           lam_um=8.0, sigma=2.8417e-3, seed=1, n_samples=4,
                           polish=0, n_ensemble=2, search_seconds=1.0,
                           ensemble_fro=0.25, loss_grid=[0.0025, 0.005],
                           stress_loss=0.05,
                           leaderboard=[["cand-" + tag, D.to_dict(), 1.0]],
                           comparison={"cand": {"area": 1.0}},
                           audits={"cand": {"sigma": 1.0}},
                           stress_audit={"cand": {"paired": _paired_stub()}},
                           ensemble_diversity={"effective_rank": 2.0},
                           gate_a={}, constraints={"kz_min_frac": 0.2},
                           nuisance_classes=list(nz.DEFAULT_CLASSES),
                           generator="cayley", q_eta=0.0,
                           gate_a_schema_version=om.GATE_A_SCHEMA_VERSION)),
                     ("candidates.json",
                      {lin: {"cand-" + tag: rec, "sel-" + tag: sel}})):
        if nm in drop:
            continue
        if nm == "result.json" and res_mutate:
            res_mutate(body)
        with open(os.path.join(rd, nm), "w") as fh:
            json.dump(body, fh)
    if manifest:
        man = dict(run_id=rid, snapshot_sha256=om._canon_hash(snap),
                   snapshot=snap, config_sha256=om._canon_hash(cfg),
                   config=cfg, lineage=lin, archive_body={},
                   artifacts={n: om._sha_file(os.path.join(rd, n))
                              for n in ("result.json", "candidates.json")
                              if n not in drop})
        if mutate:
            mutate(man)
        with open(os.path.join(rd, "manifest.json"), "w") as fh:
            json.dump(man, fh)
    root = om.output_root(rd, rid)
    with open(os.path.join(rd, "complete"), "w") as fh:
        fh.write(("%s %s" % (rid, root)) if marker is None else marker)
    if root is not None and not no_receipt:
        om.append_receipt(runs, rid, root)
    if tamper:
        with open(os.path.join(rd, tamper), "a") as fh:
            fh.write(" ")          # one byte, after the manifest was written
    if post:
        post(rd, om)
    return rd


def _replace_outputs(rd, om):
    """Rewrite the published candidates and REFRESH the manifest labels.

    This is the probe that used to verify under the identical run id: the id
    hashes inputs, and the artifact digests were mutable labels inside the
    manifest with nothing anchoring the manifest itself.
    """
    import json
    with open(os.path.join(rd, "candidates.json")) as fh:
        book = json.load(fh)
    lin = list(book)[0]
    for r in book[lin].values():
        r["design"] = dz.Design(4.4, 5.5, 87.0, 11.0, 0.2, -0.2).to_dict()
    with open(os.path.join(rd, "candidates.json"), "w") as fh:
        json.dump(book, fh)
    with open(os.path.join(rd, "manifest.json")) as fh:
        man = json.load(fh)
    man["artifacts"]["candidates.json"] = om._sha_file(
        os.path.join(rd, "candidates.json"))
    with open(os.path.join(rd, "manifest.json"), "w") as fh:
        json.dump(man, fh)


def _screen_report(names=("small@8", "winner")):
    """A complete, low-error error-screen report over `names`."""
    m = {mn: dict(fro_err=0.005, fro_err_worst=0.01, block_err_worst=0.02,
                  dS_rank=1, multistart_unique=True,
                  position_in_bracket=0.5) for mn in sy.ERROR_MODELS}
    return {nm: {"models": dict(m)} for nm in names}


def _rewrite_receipt_body(rd, om):
    """Keep the filename, change the body's output_root.

    `read_receipt` returned the body's `run_id` and never looked at its
    stored root, so a receipt could name a completely different root and
    still satisfy verification.
    """
    import json
    runs = os.path.dirname(os.path.normpath(rd))
    rid = os.path.basename(os.path.normpath(rd))
    root = om.output_root(rd, rid)
    p = os.path.join(runs, om.RECEIPTS, "%s.json" % root)
    with open(p, "w") as fh:
        json.dump({"run_id": rid, "output_root": "d" * 64}, fh)


def _steal_receipt(rd, om):
    """Publish a receipt for these bytes that names a DIFFERENT run."""
    import json
    runs = os.path.dirname(os.path.normpath(rd))
    rid = os.path.basename(os.path.normpath(rd))
    root = om.output_root(rd, rid)
    p = os.path.join(runs, om.RECEIPTS, "%s.json" % root)
    with open(p, "w") as fh:
        json.dump({"run_id": "9" * 64, "output_root": root}, fh)


def _forge_receipt(rd, om):
    """Corrupt the receipt that describes these exact bytes."""
    runs = os.path.dirname(os.path.normpath(rd))
    rid = os.path.basename(os.path.normpath(rd))
    p = os.path.join(runs, om.RECEIPTS,
                     "%s.json" % om.output_root(rd, rid))
    with open(p, "w") as fh:
        fh.write("{truncated")


def test_admission_is_manifest_verified():
    """Only a run whose manifest, marker, hashes and record identity all
    agree may contribute a candidate.

    Three loaders have now been attacked here.  The first checked only that
    `complete` and `candidates.json` existed and trusted the record's
    self-declared lineage.  The second iterated `manifest["artifacts"]`, so an
    EMPTY artifact map passed vacuously: a directory with no `result.json`, no
    snapshot or config body, and arbitrary hash strings was admitted.  The
    third must recompute identity from the stored bodies, and must reject --
    not crash on -- JSON of the wrong shape, since one malformed directory
    would otherwise take down all selection.
    """
    import shutil
    import tempfile
    from tmatrix.retrieval.fastfull import opt_marginalized as om
    d = tempfile.mkdtemp()
    D = dz.Design(9.0, 9.0, 90.0, 0.0, 0.1, 0.1)
    try:
        runs = os.path.join(d, "runs")
        good = _make_run(runs, om, D, "valid")

        def wipe(man):                      # the reviewer's vacuous manifest
            man["artifacts"] = {}
            man.pop("snapshot", None)
            man.pop("config", None)
            man["snapshot_sha256"] = "not-a-hash"
            man["config_sha256"] = "also-not-a-hash"

        def shape(field, value):
            return lambda man: man.__setitem__(field, value)

        import json
        mk = lambda tag, **kw: _make_run(runs, om, D, tag, **kw)
        cases = {
            "no manifest": mk("t1", manifest=False),
            "wrong marker text": mk("t2", marker="whatever"),
            "tampered candidates": mk("t3", tamper="candidates.json"),
            "spoofed lineage label": mk(
                "t4", mutate=shape("lineage", om.LINEAGE_INDEPENDENT)),
            "partial marker": mk("t5", marker="deadbeef"),
            "foreign record id": mk("t6", rec_rid="9" * 64),
            "empty artifacts, no bodies": mk("t7", mutate=wipe),
            "no result.json": mk("t8", drop=("result.json",)),
            "config body edited": mk(
                "t9", mutate=shape("config", {"target_conditioned": True,
                                              "probe": "EDITED"})),
            "run id not derivable": mk("t10", rid="7" * 64),
            "artifacts is a list": mk("t11", mutate=shape("artifacts", [])),
            "manifest is a list": mk("t12"),
            # a self-consistent run whose design is EMPTY used to verify, and
            # then killed the next run with KeyError: 'p1_um'
            "empty design": _make_run(runs, om, None, "t16"),
            # a conditioned `search` record RELABELLED as an independent
            # proposal: a fresh proposal cannot claim a lineage the proposing
            # run did not use
            "search relabelled independent": mk(
                "t18", rec_mutate=shape("proposal_lineage",
                                        om.LINEAGE_INDEPENDENT)),
            # an ABSENT proposal lineage was silently inferred from the
            # manifest, promoting unknown provenance to known provenance
            "proposal lineage absent": mk(
                "t19", rec_mutate=lambda r: r.pop("proposal_lineage")),
            # a result body that is not the manifest's body
            "foreign result bodies": mk(
                "t20", res_mutate=shape("config", {"target_conditioned": True,
                                                   "probe": "FOREIGN"})),
            # a result whose winner is not the selected candidate
            "winner is not the selection": mk(
                "t21", res_mutate=shape(
                    "winner", dz.Design(3.3, 4.4, 80.0, 5.0, 0.2,
                                        0.3).to_dict())),
            # a result claiming the opposite conditioning to the hashed config
            "result contradicts lineage": mk(
                "t22", res_mutate=shape("target_conditioned_prior", False)),
            # ---- provenance must RESOLVE, not merely parse ----------------
            # a proof citing a parent run/record that does not exist
            "parent does not exist": mk("t23", sel_mutate=lambda r: r.update(
                proposal_proof=dict(r["proposal_proof"],
                                    parent_run_id="does-not-exist",
                                    parent_record="imaginary"))),
            # a same-lineage selection with NO proof at all used to pass
            "selected without proof": mk(
                "t24", sel_mutate=lambda r: r.pop("proposal_proof")),
            # a proof naming a source outside the closed set
            "unknown proof source": mk("t25", sel_mutate=lambda r: r.update(
                proposal_proof=dict(r["proposal_proof"], source="hearsay"))),
            # an incumbent proof for a geometry that is not that incumbent
            "incumbent proof, wrong geometry": mk(
                "t26", sel_mutate=lambda r: r.update(
                    proposal_proof=dict(r["proposal_proof"],
                                        source="incumbent",
                                        parent_run_id="incumbent-constant",
                                        parent_record="small@8"))),
            # ---- the scientific result, not just its headline ------------
            "winner_source names nothing": mk(
                "t27", res_mutate=shape("winner_source", "does-not-exist")),
            "NaN in the stress audit": mk(
                "t28", res_mutate=shape("stress_audit",
                                        {"fake": float("nan")})),
            "winner absent from its leaderboard": mk(
                "t29", res_mutate=shape("leaderboard", [])),
            "result contradicts the hashed config": mk(
                "t30", res_mutate=shape("seed", 999)),
            "missing scientific fields": mk(
                "t31", res_mutate=lambda b: b.pop("gate_a")),
            "unknown evidence status": mk(
                "t32", res_mutate=shape("evidence_status", "publishable")),
            # ---- the result may not claim a status it did not earn -------
            "claims gate-candidate": mk(
                "t33", res_mutate=shape("evidence_status", "gate-candidate")),
            # bool passes an int check because bool subclasses int
            "schema_version is True": mk(
                "t34", res_mutate=shape("schema_version", True)),
            # extra fields were tolerated
            "extra result field": mk(
                "t35", res_mutate=shape("smuggled", {"x": 1})),
            # config duplicates beyond the six that were compared
            "loss_grid contradicts config": mk(
                "t36", res_mutate=shape("loss_grid", [999.0])),
            "ensemble_fro contradicts config": mk(
                "t37", res_mutate=shape("ensemble_fro", 999.0)),
            "generator contradicts config": mk(
                "t38", res_mutate=shape("generator", "fabricated")),
            # a paired STUB with no rows behind the flag
            "paired stub without rows": mk(
                "t39", res_mutate=shape(
                    "stress_audit",
                    {"cand": {"paired": {"n_pairs": 2,
                                         "paired_by_latent_id": True}}})),
            # empty required reports
            "empty comparison": mk("t40", res_mutate=shape("comparison", {})),
            "empty stress audit": mk("t41",
                                     res_mutate=shape("stress_audit", {})),
            # a winner_source that is on the board but not the best
            # ---- saved summaries must be RECOMPUTABLE ------------------
            "fabricated p10": mk("t43", res_mutate=lambda b: b[
                "stress_audit"]["cand"]["paired"].__setitem__(
                    "p10", "fabricated")),
            "contradictory aggregates": mk("t44", res_mutate=lambda b: b[
                "stress_audit"]["cand"]["paired"].__setitem__(
                    "n_degraded", -3)),
            "rewritten latent ids": mk("t45", res_mutate=lambda b: [
                r.__setitem__("latent_id", "f" * 64)
                for r in b["stress_audit"]["cand"]["paired"]["pairs"]]),
            "loss label off the grid": mk("t46", res_mutate=lambda b: b[
                "stress_audit"]["cand"]["paired"]["pairs"][0].__setitem__(
                    "loss", 0.777)),
            "one T-row hash on both sides": mk(
                "t47", res_mutate=lambda b: b["stress_audit"]["cand"][
                    "paired"]["pairs"][0].__setitem__("stress_row", "p0")),
            "prose where a metric belongs": mk(
                "t48", res_mutate=lambda b: b["audits"]["cand"].__setitem__(
                    "sigma", "looks fine")),
            # ---- the run id names inputs; the RECEIPT names outputs ------
            # rewriting candidates and refreshing the two artifact labels
            # used to verify under the identical run id
            "outputs replaced under one id": mk(
                "t49", post=lambda rd, om_: _replace_outputs(rd, om_)),
            # a receipt that disagrees with the published bytes
            "receipt is corrupt": mk(
                "t50", post=lambda rd, om_: _forge_receipt(rd, om_)),
            # ---- side row labels must be the real ensemble hashes --------
            "invented side row labels": mk(
                "t51", res_mutate=lambda b: [
                    r.update(production_row="a" * 64, stress_row="b" * 64)
                    for r in b["stress_audit"]["cand"]["paired"]["pairs"]]),
            "by_loss deleted": mk("t52", res_mutate=lambda b: b[
                "stress_audit"]["cand"]["paired"].pop("by_loss")),
            "nuisance classes rewritten": mk(
                "t53", res_mutate=shape("nuisance_classes", ["invented"])),
            # ---- the receipt is MANDATORY, not "nothing to contradict" --
            # deleting the journal and re-signing the marker used to let a
            # rewritten run verify under the same input-derived id
            "no receipt at all": mk("t54", no_receipt=True),
            "receipt names another run": mk(
                "t55", post=lambda rd, om_: _steal_receipt(rd, om_)),
            # ---- the error screen must identify its protocol -------------
            "no declared candidate set": mk(
                "t56", cfg_mutate=lambda c: (c.pop("gate_a_candidates"),
                                             c.__setitem__("skip_gate_a",
                                                           False)),
                lineage=om.LINEAGE_INDEPENDENT,
                res_mutate=lambda b: b.update(
                    gate_a=_screen_report(), evidence_status="error-screen-"
                                                             "passed")),
            "screen over an invented candidate": mk(
                "t57", cfg_mutate=lambda c: (
                    c.__setitem__("gate_a_candidates", ["invented-only"]),
                    c.__setitem__("skip_gate_a", False)),
                lineage=om.LINEAGE_INDEPENDENT,
                res_mutate=lambda b: b.update(
                    gate_a=_screen_report(("invented-only",)),
                    evidence_status="error-screen-passed")),
            # ---- byte binding is MANDATORY, not opportunistic -----------
            "proof digest omitted": mk(
                "t58", sel_mutate=lambda r: r["proposal_proof"].pop(
                    "parent_record_digest")),
            "proof digest is wrong": mk(
                "t59", sel_mutate=lambda r: r["proposal_proof"].__setitem__(
                    "parent_record_digest", "c" * 64)),
            # ---- the receipt BODY is parsed, not just its run_id ---------
            "receipt body names another root": mk(
                "t60", post=lambda rd, om_: _rewrite_receipt_body(rd, om_)),
            "winner is not the best": mk(
                "t42", res_mutate=lambda b: b.__setitem__(
                    "leaderboard",
                    [[b["winner_source"], b["winner"], 1.0],
                     ["other", dz.Design(7.1, 6.2, 88.0, 3.0, 0.1,
                                         0.2).to_dict(), 99.0]])),
        }
        # the ORIGINAL label-only mismatch stays covered
        rd = mk("t17")
        with open(os.path.join(rd, "result.json")) as fh:
            res = json.load(fh)
        res["config_sha256"] = "a" * 64
        with open(os.path.join(rd, "result.json"), "w") as fh:
            json.dump(res, fh)
        with open(os.path.join(rd, "manifest.json")) as fh:
            man = json.load(fh)
        man["artifacts"]["result.json"] = om._sha_file(
            os.path.join(rd, "result.json"))
        with open(os.path.join(rd, "manifest.json"), "w") as fh:
            json.dump(man, fh)
        cases["result hash label mismatch"] = rd
        # a crash right after result.json: no candidates, manifest or marker
        crashed = os.path.join(runs, "1" * 64)
        os.makedirs(crashed, exist_ok=True)
        with open(os.path.join(crashed, "result.json"), "w") as fh:
            fh.write('{"run_id": "%s"}' % ("1" * 64))
        cases["crash after result"] = crashed

        # wrong-shape JSON that is still VALID JSON: these used to raise
        # AttributeError/TypeError out of iter_completed and kill selection
        for tag, nm in (("t13", "shelf is a list"),
                        ("t14", "record is null")):
            rd = mk(tag)
            with open(os.path.join(rd, "candidates.json")) as fh:
                book = json.load(fh)
            lin = list(book)[0]
            book[lin] = [] if nm == "shelf is a list" else {"c": None}
            with open(os.path.join(rd, "candidates.json"), "w") as fh:
                json.dump(book, fh)
            cases[nm] = rd
        with open(os.path.join(cases["manifest is a list"],
                               "manifest.json"), "w") as fh:
            fh.write("[1, 2, 3]")
        rd = mk("t15")
        with open(os.path.join(rd, "manifest.json"), "w") as fh:
            fh.write("{not json at all")
        cases["unparseable manifest"] = rd

        verdicts = {nm: om.verify_completed_run(p)[0] is None
                    for nm, p in cases.items()}
        good_ok = om.verify_completed_run(good)[0] is not None
        # every malformed case must be SKIPPED, not raise: one bad directory
        # among many must not be able to take down all selection
        n_runs = len(list(om.iter_completed(runs)))
        admitted = om.load_registry(runs_dir=runs, include_fallbacks=False)
        # a staging directory must be invisible
        os.makedirs(os.path.join(runs, ".staging-1-xyz"), exist_ok=True)
        n_after = len(list(om.iter_completed(runs)))
        index = om.rebuild_index(runs_dir=runs,
                                 out_path=os.path.join(d, "idx.json"))
        n_index = sum(len(v) for k, v in index.items()
                      if not k.startswith("_"))
        # exactly ONE run survives, contributing its search + selected records
        ok = (good_ok and all(verdicts.values()) and n_runs == 1
              and len(admitted) == 2 and n_after == 1 and n_index == 2)
        bad = sorted(k for k, v in verdicts.items() if not v)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    record("(h) candidate admission is manifest-verified, not marker-trusted",
           ok,
           "1 valid run admitted out of %d directories (contributing its "
           "search + selected records); all %d damaged "
           "variants refused%s; selection and index rebuilding both survive "
           "them (a malformed shape is a rejection, not an exception), and a "
           "staging directory is invisible"
           % (len(verdicts) + 1, len(verdicts),
              "" if not bad else " EXCEPT %s" % bad))


def test_archive_is_bound_to_identity():
    """The candidate archive is a SELECTOR INPUT, so it must be frozen at
    entry, taken from the requested namespace, and hashed into the run id.

    Two defects are gated here:

    (a) the archive was loaded *after* the run id was computed and was allowed
        to replace the fresh winner, so one run id could publish different
        winners depending on which other runs finished first;
    (b) `run(out_dir=...)` called a bare `load_registry()`, which reads the
        repository-global RUNS_DIR -- a temporary or independent campaign
        would silently select global candidates while writing elsewhere.
    """
    import shutil
    import tempfile
    from tmatrix.retrieval.fastfull import opt_marginalized as om
    d = tempfile.mkdtemp()
    D = dz.Design(9.0, 9.0, 90.0, 0.0, 0.1, 0.1)
    D2 = dz.Design(9.5, 8.5, 91.0, 5.0, 0.2, -0.1)
    try:
        runs = os.path.join(d, "runs")
        _make_run(runs, om, D, "arch1")
        a1, p1, h1 = om.freeze_archive(runs)
        # the SAME archive must give the same fingerprint...
        h1b = om.freeze_archive(runs)[2]
        # ...and one more exact candidate must change it
        _make_run(runs, om, D2, "arch2")
        a2, p2, h2 = om.freeze_archive(runs)
        stable, moved = (h1 == h1b), (h1 != h2)
        grew = len(a2) == len(a1) + 1
        # provenance is preserved exactly, not rounded
        # provenance is keyed by canonical GEOMETRY, designs by unique label
        exact = all(abs(dz.Design.from_dict(v["design"]).alpha_deg
                        - a2[v["label"]].alpha_deg) < 1e-15
                    for v in p2.values())
        # the SELECTOR MAP must be a bijection with the hashed archive.  It
        # was `{record_name: design}`, so two verified runs carrying distinct
        # geometries under the same record name overwrote one another and
        # `archive_sha256`/`archive_n` certified a candidate the leaderboard
        # never saw (prov=4 against 3 selectable names).
        import json as _json
        for tag, geo in (("clash1", dz.Design(7.7, 6.6, 95.0, 40.0, .4, -.4)),
                         ("clash2", dz.Design(8.8, 5.5, 85.0, 12.0, -.3, .35))):
            rd = _make_run(runs, om, geo, tag)
            # rename both shelves to ONE common record name, then refresh the
            # artifact hash so the run still verifies -- this is the exact
            # state that used to collapse two geometries into one entry
            with open(os.path.join(rd, "candidates.json")) as fh:
                book = _json.load(fh)
            lin = list(book)[0]
            book[lin] = {("collide" if k.startswith("cand") else "collide-sel"):
                         v for k, v in book[lin].items()}
            with open(os.path.join(rd, "candidates.json"), "w") as fh:
                _json.dump(book, fh)
            with open(os.path.join(rd, "manifest.json")) as fh:
                man = _json.load(fh)
            man["artifacts"]["candidates.json"] = om._sha_file(
                os.path.join(rd, "candidates.json"))
            with open(os.path.join(rd, "manifest.json"), "w") as fh:
                _json.dump(man, fh)
        a4, p4, _ = om.freeze_archive(runs)
        bijection = len(a4) == len(p4)
        # and negative zero must not create a representational duplicate
        zpos = om.design_key(dz.Design(9.0, 9.0, 90.0, 0.0, 0.1, 0.1))
        zneg = om.design_key(dz.Design(9.0, 9.0, 90.0, -0.0, 0.1, 0.1))
        canon = (zpos == zneg)
        # namespace isolation: an empty namespace sees ONLY the fallbacks,
        # never the repository-global runs directory
        empty = os.path.join(d, "elsewhere", "runs")
        os.makedirs(empty)
        a3, _, _ = om.freeze_archive(empty)
        isolated = set(a3) == set(om._TRANSCRIBED)
        # and the fingerprint is inside the hashed config, hence the run id
        import inspect
        # strip comments first: the explanatory comment in run() names the old
        # bare call, and searching raw source would match its own retraction
        src = "\n".join(ln.split("#")[0]
                        for ln in inspect.getsource(om.run).splitlines())
        wired = ("archive_sha256=archive_hash" in src
                 and "load_registry()" not in src)
        ok = (stable and moved and grew and exact and isolated
              and wired and bijection and canon)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    record("(h) the candidate archive is frozen, namespaced and hashed into "
           "the run id", ok,
           "the same archive fingerprints identically (%s); adding one exact "
           "candidate moves it (%s -> %s) and grows the archive %d -> %d with "
           "full-precision provenance; an alternate out_dir sees only the "
           "labelled fallbacks, never the global runs directory; and the "
           "fingerprint enters the hashed config, so two archives differing "
           "by one candidate cannot share a run id; %d distinct geometries "
           "sharing one record name stay %d distinct selector entries (the "
           "map used to collapse them, certifying a candidate the leaderboard "
           "never saw); and alpha=-0.0 and +0.0 give one canonical key"
           % ("stable" if stable else "UNSTABLE", h1[:12], h2[:12],
              len(a1), len(a2), len(p4), len(a4)))


def test_proposal_lineage_cannot_be_laundered():
    """A target-conditioned proposal must never become selectable by, or be
    republished as, a target-independent run.

    `freeze_archive` scanned every verified run without a lineage filter, so
    with `TARGET_CONDITIONED_PRIOR=False` a conditioned candidate still
    entered the frozen selector.  If it won, the selected record was stamped
    with the CURRENT (independent) lineage, so the next manifest made a
    conditioned proposal look independent.  Hashing the archive made that
    contamination reproducible; it did not make it eligible.
    """
    import shutil
    import tempfile
    from tmatrix.retrieval.fastfull import opt_marginalized as om
    d = tempfile.mkdtemp()
    Dc = dz.Design(10.5, 7.2, 92.0, 19.0, -0.01, -0.49)
    Di = dz.Design(11.0, 8.0, 90.0, 10.0, 0.10, 0.20)
    try:
        runs = os.path.join(d, "runs")
        _make_run(runs, om, Dc, "cond", lineage=om.LINEAGE_CONDITIONED)
        _make_run(runs, om, Di, "indep", lineage=om.LINEAGE_INDEPENDENT)

        a_i, p_i, _ = om.freeze_archive(runs,
                                        want_lineage=om.LINEAGE_INDEPENDENT)
        a_c, p_c, _ = om.freeze_archive(runs,
                                        want_lineage=om.LINEAGE_CONDITIONED)
        props_i = set(v["proposal_lineage"] for v in p_i.values())
        props_c = set(v["proposal_lineage"] for v in p_c.values())
        # independent selection sees ONLY independent proposals; conditioned
        # development may look at both (it is already gate-disqualified)
        filtered = (props_i == {om.LINEAGE_INDEPENDENT}
                    and props_c == {om.LINEAGE_INDEPENDENT,
                                    om.LINEAGE_CONDITIONED})
        # a conditioned record may not sit on an independent shelf even if
        # every other field is edited to look independent
        laundered = om.make_record(Dc, "search", "r", "s", "c",
                                   proposal_lineage=om.LINEAGE_CONDITIONED)
        laundered["lineage"] = om.LINEAGE_INDEPENDENT
        laundered["target_conditioned"] = False
        rejected = not om._record_ok(laundered, om.LINEAGE_INDEPENDENT)
        # and the selected record preserves the winner's PROPOSAL lineage
        kept = om.make_record(Dc, "selected:x", "r", "s", "c",
                              proposal_lineage=om.LINEAGE_CONDITIONED)
        preserved = (kept["proposal_lineage"] == om.LINEAGE_CONDITIONED
                     and kept["lineage"] == om._lineage())

        # DEDUPLICATION: republishing the same geometry must not move the
        # archive fingerprint (that endless growth is what stopped the
        # completed-identity refusal from ever being reached)
        fp1 = om.freeze_archive(runs, want_lineage=om.LINEAGE_CONDITIONED)[2]
        _make_run(runs, om, Dc, "dup", lineage=om.LINEAGE_CONDITIONED)
        fp2 = om.freeze_archive(runs, want_lineage=om.LINEAGE_CONDITIONED)[2]
        _make_run(runs, om, dz.Design(12.0, 9.0, 88.0, 30.0, 0.3, -0.3),
                  "new", lineage=om.LINEAGE_CONDITIONED)
        fp3 = om.freeze_archive(runs, want_lineage=om.LINEAGE_CONDITIONED)[2]
        dedup = (fp1 == fp2) and (fp2 != fp3)

        # an archive PIN refuses a campaign against an unexpected archive
        pinned = False
        try:
            om.freeze_archive(runs, want_lineage=om.LINEAGE_CONDITIONED,
                              pin=fp1)
        except RuntimeError:
            pinned = True

        # INCUMBENTS are source-code constants and used to bypass the filter
        # entirely, entering the leaderboard with no lineage at all
        inc_i = om.eligible_incumbents(om.LINEAGE_INDEPENDENT)
        inc_c = om.eligible_incumbents(om.LINEAGE_CONDITIONED)
        incumbents_filtered = (not inc_i
                               and set(inc_c) == set(om.INCUMBENT_DESIGNS))

        # A SELECTED record must RESOLVE to an existing parent.  A proof used
        # to be syntactic -- nonempty strings agreeing with the child's own
        # design -- so a record citing `does-not-exist/imaginary` verified and
        # its geometry entered an independent freeze.
        shelf = {"src": om.make_record(Di, "search", "R", "s", "c",
                                       proposal_lineage=om.LINEAGE_INDEPENDENT)}
        shelf["src"]["lineage"] = om.LINEAGE_INDEPENDENT
        body = {om.design_key(Di): dict(name="arch-rec", first_run="PARENT",
                                        proposal_lineage=(
                                            om.LINEAGE_INDEPENDENT),
                                        design=om.canonical_design(Di))}

        def sel_with(proof, design=Di, lineage=om.LINEAGE_INDEPENDENT):
            r = om.make_record(design, "selected:x", "R", "s", "c",
                               proposal_lineage=lineage,
                               proposal_proof=proof)
            r["lineage"] = om.LINEAGE_INDEPENDENT
            return r

        resolves = om.resolve_selected_parent(
            sel_with(om._proof("same_run", "R", "src", Di,
                               om.LINEAGE_INDEPENDENT,
                               parent_record_digest=om.record_digest(
                                   shelf["src"]))),
            shelf, body, "R", runs_dir=runs) is None

        # AN ARCHIVE CITATION MUST REACH A REAL PARENT RUN ON DISK.  The
        # previous resolver looked the key up in the child's OWN persisted
        # archive body: a self-consistent artifact naming a nonexistent
        # parent verified and entered an independent freeze.  Build a genuine
        # parent run and cite it.
        prunes = os.path.join(d, "chain")
        preal = _make_run(prunes, om, Di, "root",
                          lineage=om.LINEAGE_INDEPENDENT)
        pid = os.path.basename(preal)
        real_body = {om.design_key(Di): dict(
            name="cand-root", first_run=pid,
            proposal_lineage=om.LINEAGE_INDEPENDENT,
            design=om.canonical_design(Di))}
        import json as _jj
        _rb = _jj.load(open(os.path.join(preal, "candidates.json")))
        _rrec = _rb[list(_rb)[0]]["cand-root"]
        arch_res = om.resolve_selected_parent(
            sel_with(om._proof("archive", pid, "cand-root", Di,
                               om.LINEAGE_INDEPENDENT,
                               parent_output_root=om.output_root(preal, pid),
                               parent_record_digest=om.record_digest(_rrec))),
            shelf, real_body, "R", runs_dir=prunes) is None
        # ...and the SAME citation with no parent directory must fail
        ghost_body = {om.design_key(Di): dict(
            name="cand-root", first_run="0" * 64,
            proposal_lineage=om.LINEAGE_INDEPENDENT,
            design=om.canonical_design(Di))}
        ghost = om.resolve_selected_parent(
            sel_with(om._proof("archive", "0" * 64, "cand-root", Di,
                               om.LINEAGE_INDEPENDENT,
                               parent_output_root="a" * 64,
                               parent_record_digest="b" * 64)),
            shelf, ghost_body, "R", runs_dir=prunes)
        external_enforced = arch_res and ghost is not None

        def cite(run_dir, rec_name):
            """A VALID outer archive citation to `rec_name` in `run_dir`.

            The legacy fixtures below built their outer proofs WITHOUT the
            now-mandatory digests, so they were refused at the missing-field
            guard and never walked to the ancestor the assertion named.  This
            makes the outer hop valid so the poisoned ancestor is what
            actually decides the outcome.
            """
            import json as _j
            rid_ = os.path.basename(os.path.normpath(run_dir))
            with open(os.path.join(run_dir, "candidates.json")) as fh:
                bk = _j.load(fh)
            rc = bk[list(bk)[0]][rec_name]
            return om._proof("archive", rid_, rec_name, Di,
                             om.LINEAGE_INDEPENDENT,
                             parent_output_root=om.output_root(run_dir, rid_),
                             parent_record_digest=om.record_digest(rc))

        # AN INVALID ANCESTOR MUST SINK THE WHOLE CHAIN.  `_walk_provenance`
        # treated any nested proof whose source string said `incumbent` or
        # `transcribed` as a terminal root without checking the named
        # constant.  So a parent whose own selected record cited an INVALID
        # incumbent was rejected by full verification, while a child citing
        # that same record verified -- `iter_completed` admitted the child,
        # rejected the parent, and the child's geometry entered the freeze.
        poison = os.path.join(d, "poison")
        prd = _make_run(poison, om, Di, "bad", lineage=om.LINEAGE_INDEPENDENT,
                        sel_mutate=lambda r: r.update(
                            proposal_proof=om._proof(
                                "incumbent", "incumbent-constant",
                                "not-an-incumbent", Di,
                                om.LINEAGE_INDEPENDENT)))
        bad_id = os.path.basename(prd)
        parent_refused = om.verify_completed_run(prd)[0] is None
        # the child cites the parent's INVALID selected record
        poison_body = {om.design_key(Di): dict(
            name="sel-bad", first_run=bad_id,
            proposal_lineage=om.LINEAGE_INDEPENDENT,
            design=om.canonical_design(Di))}
        child_why = om.resolve_selected_parent(
            sel_with(cite(prd, "sel-bad")),
            shelf, poison_body, "R", runs_dir=poison)
        # the outer hop is VALID, so the rejection must come from the
        # poisoned terminal itself
        child_refused = (child_why is not None
                         and "incumbent" in child_why)
        # ...and an ancestor that fails a LOCAL invariant must sink the chain
        # too.  `_verify_structural` suppressed `_archive_body_ok` along with
        # the recursion, so a middle run whose archive fingerprint
        # contradicted its own body failed full verification yet still
        # legitimized a child that cited it.
        mid = os.path.join(d, "mid")
        mrd = _make_run(mid, om, Di, "mid", lineage=om.LINEAGE_INDEPENDENT,
                        mutate=lambda m: m.__setitem__(
                            "archive_body",
                            {om.design_key(Di): dict(
                                name="ghost", first_run="z" * 64,
                                proposal_lineage=om.LINEAGE_INDEPENDENT,
                                design=om.canonical_design(Dc))}))
        mid_id = os.path.basename(mrd)
        mid_refused = om.verify_completed_run(mrd)[0] is None
        mid_body = {om.design_key(Di): dict(
            name="sel-mid", first_run=mid_id,
            proposal_lineage=om.LINEAGE_INDEPENDENT,
            design=om.canonical_design(Di))}
        mid_why = om.resolve_selected_parent(
            sel_with(cite(mrd, "sel-mid")),
            shelf, mid_body, "R", runs_dir=mid)
        mid_child_refused = (mid_why is not None
                             and "does not verify" in mid_why)
        # BOUNDED ANCESTRY.  `descend=False` was threaded to the resolver
        # and never consulted, so structural checks re-walked the whole chain
        # suffix and a long chain recursed instead of stopping at
        # MAX_PROVENANCE_DEPTH.
        #
        # NOTE ON CYCLES.  The previous fixture here claimed to build a
        # two-run cycle; it edited `candidates.json` after the manifest was
        # hashed, so it was refused on the artifact hash and the walker never
        # saw a repeated node.  Worse, the assertion could not be repaired:
        # once every hop must cite its parent's OUTPUT ROOT, a genuine
        # published cycle is UNCONSTRUCTIBLE -- run A's root depends on its
        # candidates, which would have to contain B's root, which depends on
        # A's.  Content addressing rules cycles out by construction, so the
        # `seen` guard is defence-in-depth on an unreachable path.  It is
        # therefore tested at unit level, and the reachable bound -- DEPTH --
        # is tested with a real chain.
        import time as _t
        cyc_why = om._walk_provenance(
            "q" * 64, "rec", om.design_key(Di), om.LINEAGE_INDEPENDENT,
            os.path.join(d, "nowhere"), seen={("q" * 64, "rec")})
        cyc_bounded = cyc_why is not None and "cyclic" in cyc_why

        # a REAL chain longer than the hop budget must stop with the depth
        # diagnostic, not recurse
        deep = os.path.join(d, "deep")
        n_hops = om.MAX_PROVENANCE_DEPTH + 3
        prev_dir = _make_run(deep, om, Di, "d0",
                             lineage=om.LINEAGE_INDEPENDENT)
        prev_rec = "cand-d0"
        for i in range(1, n_hops):
            pf = cite(prev_dir, prev_rec)
            bd = {om.design_key(Di): dict(
                name=prev_rec,
                first_run=os.path.basename(os.path.normpath(prev_dir)),
                proposal_lineage=om.LINEAGE_INDEPENDENT,
                design=om.canonical_design(Di))}
            fp = om._canon_hash(
                {k: dict(design=om.canonical_design(v["design"]),
                         proposal_lineage=v["proposal_lineage"])
                 for k, v in bd.items()})
            prev_dir = _make_run(
                deep, om, Di, "d%d" % i, lineage=om.LINEAGE_INDEPENDENT,
                sel_mutate=lambda r, _p=pf: r.__setitem__("proposal_proof",
                                                          _p),
                mutate=lambda m, _b=bd: m.__setitem__("archive_body", _b),
                cfg_mutate=lambda c, _f=fp: c.update(
                    archive_sha256=_f, archive_n=1,
                    archive_lineages=[om.LINEAGE_INDEPENDENT]))
            prev_rec = "sel-d%d" % i
        t0 = _t.time()
        try:
            deep_why = om.resolve_selected_parent(
                sel_with(cite(prev_dir, prev_rec)), shelf,
                {om.design_key(Di): dict(
                    name=prev_rec,
                    first_run=os.path.basename(os.path.normpath(prev_dir)),
                    proposal_lineage=om.LINEAGE_INDEPENDENT,
                    design=om.canonical_design(Di))},
                "R", runs_dir=deep)
            deep_bounded = deep_why is not None and "hops" in deep_why
        except RecursionError:
            deep_bounded = False
        cyc_dt = _t.time() - t0

        # A REAL THREE-RUN CHAIN, with each hop's digest mutated in turn.
        # Digest checking used to stop after the FIRST hop: the walker
        # validated the incoming proof, set it to None, and then loaded the
        # intermediate parent's proof without ever adopting it -- so a middle
        # record could carry a deliberately wrong parent root and record
        # digest and the walk still succeeded.
        def build_chain(root_dir, break_hop=None, mode="wrong"):
            """root <- mid (selected, cites root) <- child cites mid."""
            import json as _j
            r0 = _make_run(root_dir, om, Di, "h0",
                           lineage=om.LINEAGE_INDEPENDENT)
            id0 = os.path.basename(r0)
            with open(os.path.join(r0, "candidates.json")) as fh:
                b0 = _j.load(fh)
            lin0 = list(b0)[0]
            rec0 = b0[lin0]["cand-h0"]
            p_mid = om._proof("archive", id0, "cand-h0", Di,
                              om.LINEAGE_INDEPENDENT,
                              parent_output_root=om.output_root(r0, id0),
                              parent_record_digest=om.record_digest(rec0))
            if break_hop == "mid":
                p_mid["parent_record_digest" if mode == "wrong"
                      else "parent_output_root"] = "9" * 64
                if mode == "omit":
                    p_mid.pop("parent_record_digest")
            # h1's OWN manifest must carry the archive entry it cites, or
            # its local (descend=False) check rejects the citation before the
            # chain is ever walked
            mid_body = {om.design_key(Di): dict(
                name="cand-h0", first_run=id0,
                proposal_lineage=om.LINEAGE_INDEPENDENT,
                design=om.canonical_design(Di))}
            mid_fp = om._canon_hash(
                {k: dict(design=om.canonical_design(v["design"]),
                         proposal_lineage=v["proposal_lineage"])
                 for k, v in mid_body.items()})
            r1 = _make_run(root_dir, om, Di, "h1",
                           lineage=om.LINEAGE_INDEPENDENT,
                           sel_mutate=lambda r: r.__setitem__(
                               "proposal_proof", p_mid),
                           mutate=lambda m: m.__setitem__("archive_body",
                                                          mid_body),
                           cfg_mutate=lambda c: c.update(
                               archive_sha256=mid_fp, archive_n=1,
                               archive_lineages=[om.LINEAGE_INDEPENDENT]))
            id1 = os.path.basename(r1)
            with open(os.path.join(r1, "candidates.json")) as fh:
                b1 = _j.load(fh)
            rec1 = b1[list(b1)[0]]["sel-h1"]
            p_child = om._proof("archive", id1, "sel-h1", Di,
                                om.LINEAGE_INDEPENDENT,
                                parent_output_root=om.output_root(r1, id1),
                                parent_record_digest=om.record_digest(rec1))
            if break_hop == "child":
                p_child["parent_record_digest" if mode == "wrong"
                        else "parent_output_root"] = "9" * 64
                if mode == "omit":
                    p_child.pop("parent_record_digest")
            body = {om.design_key(Di): dict(
                name="sel-h1", first_run=id1,
                proposal_lineage=om.LINEAGE_INDEPENDENT,
                design=om.canonical_design(Di))}
            return sel_with(p_child), body

        chains = {}
        base = os.path.join(d, "chain-ok")
        rec_ok, body_ok = build_chain(base)
        chain_ok = om.resolve_selected_parent(rec_ok, shelf, body_ok, "R",
                                              runs_dir=base) is None
        for hop in ("child", "mid"):
            for mode in ("wrong", "omit"):
                nm2 = "%s hop, digest %s" % (hop, mode)
                dd = os.path.join(d, "chain-%s-%s" % (hop, mode))
                r_, b_ = build_chain(dd, break_hop=hop, mode=mode)
                chains[nm2] = om.resolve_selected_parent(
                    r_, shelf, b_, "R", runs_dir=dd) is not None
        chain_enforced = (parent_refused and child_refused
                          and mid_refused and mid_child_refused
                          and cyc_bounded and deep_bounded and cyc_dt < 60.0
                          and chain_ok and all(chains.values()))
        refused = []
        # a parent that does not exist
        # These five must each reject for their OWN stated reason.  Four of
        # them used to omit the newly mandatory digests and were refused at
        # the missing-field guard instead, so the aggregate overstated
        # coverage.  Each now carries syntactically valid digest fields (real
        # ones where a real parent exists).
        _dig = om.record_digest(shelf["src"])
        reasons = {}
        reasons["nonexistent same-run parent"] = om.resolve_selected_parent(
            sel_with(om._proof("same_run", "R", "imaginary", Di,
                               om.LINEAGE_INDEPENDENT,
                               parent_record_digest=_dig)),
            shelf, body, "R", runs_dir=runs)
        # a geometry that is not in the frozen archive at all (using Di here
        # would hit the body-agreement guard first, not the absence guard)
        reasons["geometry absent from the archive"] =             om.resolve_selected_parent(
                sel_with(om._proof("archive", "e" * 64, "ghost", Dc,
                                   om.LINEAGE_INDEPENDENT,
                                   parent_output_root="a" * 64,
                                   parent_record_digest="b" * 64),
                         design=Dc),
                shelf, body, "R", runs_dir=runs)
        # ...and one whose named parent run disagrees with the body
        reasons["archive body names another parent"] =             om.resolve_selected_parent(
                sel_with(om._proof("archive", "e" * 64, "ghost", Di,
                                   om.LINEAGE_INDEPENDENT,
                                   parent_output_root="a" * 64,
                                   parent_record_digest="b" * 64)),
                shelf, body, "R", runs_dir=runs)
        reasons["no proof at all"] = om.resolve_selected_parent(
            sel_with(None), shelf, body, "R", runs_dir=runs)
        reasons["unrelated geometry"] = om.resolve_selected_parent(
            sel_with(om._proof("same_run", "R", "src", Dc,
                               om.LINEAGE_INDEPENDENT,
                               parent_record_digest=_dig), design=Dc),
            shelf, body, "R", runs_dir=runs)
        _bad = om._proof("same_run", "R", "src", Di,
                         om.LINEAGE_INDEPENDENT, parent_record_digest=_dig)
        _bad["design_key"] = om.design_key(Dc)
        reasons["design_key is not the design carried"] =             om.resolve_selected_parent(sel_with(_bad), shelf, body, "R",
                                       runs_dir=runs)
        want = {
            "nonexistent same-run parent": "is not in this run's shelf",
            "geometry absent from the archive": "absent from the frozen",
            "archive body names another parent": "archive disagrees with",
            "no proof at all": "carries no proposal_proof",
            "unrelated geometry": "different geometry",
            "design_key is not the design carried": "different geometry",
        }
        refused = [reasons[k] for k in reasons]
        exact = {k: (v is not None and want[k] in v)
                 for k, v in reasons.items()}

        proof_enforced = (resolves and external_enforced and chain_enforced
                          and all(exact.values()))
        _conds = dict(filtered=filtered, rejected=rejected,
                      preserved=preserved, dedup=dedup, pinned=pinned,
                      incumbents_filtered=incumbents_filtered,
                      resolves=resolves, external=external_enforced,
                      chain=chain_enforced, chain_ok=chain_ok,
                      refused=all(exact.values()))
        ok = all(_conds.values())
        _bad = sorted(k for k, v in _conds.items() if not v)
        if _bad:
            print("      sub-conditions failed: %s" % _bad, flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    record("(h) proposal lineage is immutable and the archive is "
           "lineage-filtered and deduplicated", ok,
           "an independent run's frozen archive holds %d candidate(s), all "
           "independent; a conditioned development run holds %d and may see "
           "both; a conditioned record edited to look independent is refused "
           "by the schema; a selected record keeps its proposal lineage; "
           "republishing the same geometry leaves the fingerprint at %s while "
           "a new geometry moves it to %s; a stale pin is refused; both "
           "hard-coded incumbents are declared conditioned, so an independent "
           "run inherits %d of them (a conditioned run sees %d); and a "
           "selected record must RESOLVE to an existing parent -- a same-run "
           "record resolves locally and an archive citation resolves against "
           "a REAL parent run directory on disk (a self-consistent body "
           "naming a nonexistent parent is refused), while %d further "
           "unresolvable citations, each carrying VALID digest fields so it "
           "reaches its own guard (%s), are all refused for their stated "
           "reason; and an INVALID ancestor sinks the chain -- a parent "
           "whose selected record cites a bogus incumbent is refused, and so "
           "is a child citing that record (it used to verify while its own "
           "parent did not, and each now rejects for its OWN reason rather "
           "than at the missing-digest guard); a repeated node is refused "
           "with a `cyclic` diagnostic (a genuine published cycle is "
           "unconstructible once every hop cites its parent's output root, "
           "so that guard is unreachable defence-in-depth), and a REAL "
           "%d-hop chain stops on the depth bound in %.2f s instead of "
           "recursing. On a REAL three-run chain the intact citation "
           "resolves while all %d per-hop digest mutations (%s) are refused "
           "-- digest checking used to stop after the first hop"
           % (len(a_i), len(a_c), fp1[:10], fp3[:10], len(inc_i), len(inc_c),
              len(exact), ", ".join(sorted(exact)), n_hops, cyc_dt,
              len(chains),
              ", ".join(sorted(chains))))


def test_receipt_publication_is_crash_recoverable():
    """Receipt publication must recover from every crash boundary AND be
    mutually exclusive under contention.

    The mechanism was rebuilt seven times.  The lease version recovered from
    crashes but was not exclusion: a freshly `O_EXCL`-created lock is briefly
    EMPTY and was stolen as "age infinity"; an ABA interleaving let a
    contender rename a lock already replaced by a live one; nothing fenced the
    original owner, so a paused publisher could resume and overwrite the thief
    with BOTH returning success; the orphan sweep deleted an active
    publisher's temp; an expired owner unlinked its successor's lock; and the
    torn-receipt reclaim ran OUTSIDE the reservation so two recoverers could
    both proceed.  `.stale.*` tombstones were also never deleted, while the
    gate filtered only `.tmp`/`.lock` and reported "no leftovers".
    """
    import json as _j
    import shutil
    import tempfile
    from tmatrix.retrieval.fastfull import opt_marginalized as om
    root, rid, foreign = "a" * 64, "b" * 64, "c" * 64
    lockp = lambda r: os.path.join(r, om.RECEIPTS, "%s.json.lock" % root)
    finalp = lambda r: os.path.join(r, om.RECEIPTS, "%s.json" % root)
    stale = _j.dumps({"token": "t", "owner": 1, "ts": 0.0})

    def fresh():
        d = tempfile.mkdtemp()
        r = os.path.join(d, "runs")
        os.makedirs(os.path.join(r, om.RECEIPTS))
        return d, r

    def leftovers(r):
        # EXACT directory contents, not a `.tmp`/`.lock` filter: the previous
        # gate reported "no leftovers" while `.lock.stale.*` tombstones piled
        # up for every recovered case.
        return sorted(x for x in os.listdir(os.path.join(r, om.RECEIPTS))
                      if x != "%s.json" % root)

    verdicts, dirs = {}, []
    try:
        for name, plant in (
                ("clean", lambda r: None),
                ("crash after reservation",
                 lambda r: open(lockp(r), "w").write(stale)),
                ("crash mid temp write",
                 lambda r: (open(lockp(r), "w").write(stale),
                            open(finalp(r) + ".x.tmp", "w").write("{part"))),
                ("torn final receipt",
                 lambda r: open(finalp(r), "w").close()),
                ("torn final receipt + stale lock",
                 lambda r: (open(finalp(r), "w").close(),
                            open(lockp(r), "w").write(stale))),
                ("crash after replace, lock left",
                 lambda r: (open(finalp(r), "w").write(
                     _j.dumps({"run_id": rid, "output_root": root})),
                     open(lockp(r), "w").write(stale))),
                ("abandoned EMPTY lock",
                 lambda r: (open(lockp(r), "w").close(),
                            os.utime(lockp(r), (0, 0))))):
            d, r = fresh()
            dirs.append(d)
            plant(r)
            om.append_receipt(r, rid, root, _lease=0.0)
            verdicts[name] = (om.read_receipt(r, root) == rid
                              and not leftovers(r))

        # a FRESH empty lock is NOT stale: stealing it raced a live publisher
        d, r = fresh()
        dirs.append(d)
        open(lockp(r), "w").close()
        try:
            om.append_receipt(r, rid, root, _lease=0.0)
            fresh_empty_respected = False
        except RuntimeError:
            fresh_empty_respected = True

        # FENCING: a publisher that loses its reservation mid-flight must not
        # write, must not leave a temp, and must not unlink its successor
        d, r = fresh()
        dirs.append(d)
        def steal(_tok):
            with open(lockp(r), "w") as fh:
                _j.dump({"token": "SUCCESSOR", "owner": 99999, "ts": 9e18},
                        fh)
        om._PUBLISH_HOOK = steal
        try:
            om.append_receipt(r, rid, root, _lease=0.0)
            fenced = False
        except RuntimeError:
            fenced = True
        finally:
            om._PUBLISH_HOOK = None
        fenced = (fenced and om.read_receipt(r, root) is None
                  and _j.load(open(lockp(r)))["token"] == "SUCCESSOR"
                  and not [x for x in os.listdir(os.path.join(r, om.RECEIPTS))
                           if x.endswith(".tmp")])

        # a FOREIGN id may not repair a valid published run's torn receipt --
        # the owner check must work precisely when the receipt is unreadable
        d = tempfile.mkdtemp()
        dirs.append(d)
        runs = os.path.join(d, "runs")
        rdir = _make_run(runs, om, D_VALID, "own")
        own_id = os.path.basename(rdir)
        own_root = om.output_root(rdir, own_id)
        open(os.path.join(runs, om.RECEIPTS, "%s.json" % own_root),
             "w").close()
        owner_seen = om.marker_owner(runs, own_root) == own_id
        try:
            om.append_receipt(runs, foreign, own_root, _lease=0.0)
            foreign_refused = False
        except RuntimeError:
            foreign_refused = True
        om.append_receipt(runs, own_id, own_root, _lease=0.0)
        owner_repairs = om.verify_completed_run(rdir)[0] is not None

        ok = (all(verdicts.values()) and fresh_empty_respected and fenced
              and owner_seen and foreign_refused and owner_repairs)
    finally:
        om._PUBLISH_HOOK = None
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
    record("(h) receipt publication is crash-recoverable AND fenced", ok,
           "%d/%d crash boundaries recover with the receipt directory holding "
           "NOTHING but the receipt (%s); a FRESH empty reservation is %s "
           "rather than stolen as 'age infinity'; a publisher that loses its "
           "reservation mid-flight is %s -- it writes no receipt, leaves no "
           "temp and does not unlink its successor; and with a valid run's "
           "receipt zeroed the owner is still identified (%s), a foreign id "
           "is %s, and the rightful owner repairs it"
           % (sum(verdicts.values()), len(verdicts),
              ", ".join(sorted(verdicts)),
              "respected" if fresh_empty_respected else "STOLEN",
              "fenced out" if fenced else "ALLOWED TO PUBLISH",
              "yes" if owner_seen else "NO",
              "refused" if foreign_refused else "ALLOWED"))


def test_error_screen_is_not_a_passed_gate():
    """The strongest evidence status must name what was actually tested.

    `gate_a_verdict` iterated whatever candidate and model names the result
    supplied and tested two thresholds, so a one-cell report naming an
    invented model with errors -1 and -2 returned `(ran=True, passed=True)`
    and derived a `gate-passed` label.  Negative errors are impossible, an
    arbitrary subset is not the declared experiment, and neither is a passed
    acceptance gate: rank 40, useful-direction SNR, basin stability,
    passivity and frozen trial identities are NOT in the saved report and so
    cannot be checked here.  The strongest name is therefore
    `error-screen-passed`.
    """
    from tmatrix.retrieval.fastfull import opt_marginalized as om

    def rep(models, fro=0.01, blk=0.02, names=("small@8", "winner")):
        m = {mn: dict(fro_err=fro * 0.5, fro_err_worst=fro,
                      block_err_worst=blk, dS_rank=1,
                      multistart_unique=True, position_in_bracket=0.5)
             for mn in models}
        return {nm: {"models": dict(m)} for nm in names}

    declared = list(sy.ERROR_MODELS)
    cfg = {"skip_gate_a": False}
    cands = ["small@8", "winner"]
    good = om.gate_a_verdict(rep(declared), cfg, cands)
    cases = {
        "invented model, negative errors":
            om.gate_a_verdict(rep(["invented"], fro=-1.0, blk=-2.0), cfg,
                              cands),
        "a subset of the declared models":
            om.gate_a_verdict(rep(declared[:1]), cfg, cands),
        "a subset of the declared candidates":
            om.gate_a_verdict(rep(declared, names=("winner",)), cfg, cands),
        "over the error threshold":
            om.gate_a_verdict(rep(declared, fro=0.4), cfg, cands),
        "worst below its own median":
            om.gate_a_verdict({"small@8": {"models": {
                m: dict(fro_err=0.9, fro_err_worst=0.01, block_err_worst=0.01,
                        dS_rank=1, multistart_unique=True,
                        position_in_bracket=0.5) for m in declared}},
                "winner": {"models": {
                    m: dict(fro_err=0.001, fro_err_worst=0.01,
                            block_err_worst=0.01, dS_rank=1,
                            multistart_unique=True, position_in_bracket=0.5)
                    for m in declared}}}, cfg, cands),
        "a missing required field":
            om.gate_a_verdict({nm: {"models": {m: {"fro_err_worst": 0.01}
                                               for m in declared}}
                               for nm in cands}, cfg, cands),
    }
    # a passing screen over ANY other candidate set must not receive the
    # production-protocol status; and an ABSENT declaration is not a pass
    custom = om.derive_evidence_status(
        om.LINEAGE_INDEPENDENT, cfg, rep(declared, names=("invented-only",)),
        ["invented-only"])
    undeclared = om.derive_evidence_status(om.LINEAGE_INDEPENDENT, cfg,
                                           rep(declared), None)
    # presence is not validation: these three were required only to exist
    def broken(**kw):
        r = rep(declared)
        for e in r.values():
            for m in e["models"].values():
                m.update(kw)
        return om.gate_a_verdict(r, cfg, cands)
    cases["dS_rank is null"] = broken(dS_rank=None)
    cases["non-unique multistart basin"] = broken(multistart_unique=False)
    cases["position_in_bracket is null"] = broken(position_in_bracket=None)
    status = om.derive_evidence_status(om.LINEAGE_INDEPENDENT, cfg,
                                       rep(declared), cands)
    fabricated = om.derive_evidence_status(
        om.LINEAGE_INDEPENDENT, cfg, rep(["invented"], fro=-1.0, blk=-2.0),
        cands)
    ok = (good[1] is True and status == "error-screen-passed"
          and fabricated == "error-screen-attempted"
          and custom == "custom-screen-passed"
          and undeclared == "error-screen-attempted"
          and all(v[1] is not True for v in cases.values())
          and "gate-passed" not in om.EVIDENCE_STATUSES
          and len(om.GATE_A_UNVERIFIED) >= 4)
    record("(h) the strongest status is an ERROR SCREEN, not a passed Gate A",
           ok,
           "a report covering the declared %d candidates x %d perturbation "
           "families within 5%% gives %r; %d fabricated or narrowed reports "
           "(%s) all fail; `gate-passed` is not a state the code can emit, "
           "and %d proposal criteria (%s) are recorded as NOT verified by "
           "this screen; a passing screen over a different candidate set gets "
           "the deliberately weaker %r, and an undeclared set is not a pass"
           % (len(cands), len(declared), status, len(cases),
              ", ".join(sorted(cases)), len(om.GATE_A_UNVERIFIED),
              "; ".join(om.GATE_A_UNVERIFIED), custom))


def test_paired_ensemble_is_the_production_one():
    """The stress gate must call the production helper, not reimplement it.

    The previous test grouped sequential RNG draws by loss, whereas production
    cycles loss over pre-drawn latents.  The two constructions gave different
    ensembles (hashes `87246d...` vs `a00a63...`, max entry difference
    6.094e-4), so the test was not measuring the configuration it claimed to.
    """
    import hashlib
    from tmatrix.retrieval.fastfull import opt_marginalized as om
    Bloc, _ = sym.build_d4h_reciprocity_basis(MODES,
                                             verify_numeric=False)
    ens, ens_stress, pair_loss = om.build_paired_ensembles(
        Bloc, 20260807, 6)
    h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    # pairing: row i of each array shares one latent draw
    cos = [abs(np.vdot(ens[i].ravel(), ens_stress[i].ravel()))
           / (np.linalg.norm(ens[i]) * np.linalg.norm(ens_stress[i]))
           for i in range(len(ens))]
    ratio = [sym.absorption_spectrum(ens_stress[i])[0]["unitarity_resid"]
             / sym.absorption_spectrum(ens[i])[0]["unitarity_resid"]
             for i in range(len(ens))]
    nominal = [om.STRESS_LOSS / L for L in pair_loss]
    tracks = max(abs(r / n - 1.0) for r, n in zip(ratio, nominal))
    # adjacent stress rows must be DISTINCT (they were exactly 0.0 apart)
    adj = min(np.abs(ens_stress[i] - ens_stress[i + 1]).max()
              for i in range(len(ens_stress) - 1))
    # a paired statistic must not be the ratio of two independent worst cases
    rows_p = [{"sigma_marg": 10.0}, {"sigma_marg": 8.0}]
    rows_s = [{"sigma_marg": 9.0}, {"sigma_marg": 8.8}]
    st = om.paired_stress_stats(rows_p, rows_s, pair_loss=[0.0025, 0.005],
                                unbound=True)
    # BROKEN pairing must raise, not be silently truncated to the shorter
    # list: two production rows against one stress row were reported as one
    # valid pair, making broken pairing indistinguishable from a small run
    strict, checks = 0, []
    checks.append(lambda: om.paired_stress_stats(rows_p, rows_s[:1],
                                                unbound=True))
    checks.append(lambda: om.paired_stress_stats([], [], unbound=True))
    # asymmetric value validation: production had to be positive while a ZERO
    # or NEGATIVE stress value was accepted and produced ratios 0.0 / -0.5
    checks.append(lambda: om.paired_stress_stats(
        rows_p, [{"sigma_marg": 0.0}, {"sigma_marg": 1.0}], unbound=True))
    checks.append(lambda: om.paired_stress_stats(
        rows_p, [{"sigma_marg": -5.0}, {"sigma_marg": 1.0}], unbound=True))
    checks.append(lambda: om.paired_stress_stats(
        [{"sigma_marg": 0.0}, {"sigma_marg": 1.0}], rows_s, unbound=True))
    # a NaN loss label used to pass straight into stratification
    checks.append(lambda: om.paired_stress_stats(
        rows_p, rows_s, pair_loss=[float("nan"), 0.005], unbound=True))
    # report rows that are not the draws that were passed
    checks.append(lambda: om.paired_stress_stats(
        rows_p, rows_s, prod_ids=["a", "b"], expect_prod=["a", "ZZ"],
        stress_ids=["c", "d"], expect_stress=["c", "d"]))
    checks.append(lambda: om.paired_stress_stats(
        rows_p, rows_s, prod_ids=["a", "b"], expect_prod=["a", "b"],
        stress_ids=["c", "d"], expect_stress=["ZZ", "d"]))
    # ONE-SIDED binding: production ids with no stress ids used to report
    # rows_bound_to_ensemble=True with nothing binding the stress rows
    checks.append(lambda: om.paired_stress_stats(
        rows_p, rows_s, prod_ids=["a", "b"], expect_prod=["a", "b"]))
    # an unlabelled unpaired statistic is refused outright
    checks.append(lambda: om.paired_stress_stats(rows_p, rows_s))
    # duplicate latent ids cannot identify pairs
    H0, H1 = "0" * 64, "1" * 64
    checks.append(lambda: om.paired_stress_stats(
        rows_p, rows_s, pair_ids=[H0, H0], prod_ids=["a", "b"],
        expect_prod=["a", "b"], stress_ids=["c", "d"],
        expect_stress=["c", "d"]))
    # latent ids must be hash-shaped, not arbitrary labels
    checks.append(lambda: om.paired_stress_stats(
        rows_p, rows_s, pair_ids=["L0", "L1"], prod_ids=["a", "b"],
        expect_prod=["a", "b"], stress_ids=["c", "d"],
        expect_stress=["c", "d"]))
    for fn in checks:
        try:
            fn()
        except ValueError:
            strict += 1
    # the per-pair rows carry ids, losses and both absolute values
    labelled = (st["pairs"][1]["pair_id"] == 1
                and st["pairs"][1]["loss"] == 0.005
                and st["pairs"][1]["production"] == 8.0
                and st["pairs"][1]["stress"] == 8.8
                and set(st["by_loss"]) == {"0.0025", "0.005"})
    # the LATENT id is what production and stress genuinely share; the T
    # matrices differ by construction, so comparing T hashes across the two
    # sides could never have established pairing
    lat = om.paired_ensemble_ids(Bloc, 20260807, 6)
    rid_p = om.ensemble_row_ids(ens)
    rid_s = om.ensemble_row_ids(ens_stress)
    identified = (len(set(lat)) == 6 and rid_p != rid_s
                  and om.paired_ensemble_ids(Bloc, 20260807, 6) == lat)
    st2 = om.paired_stress_stats(rows_p, rows_s, pair_loss=[0.0025, 0.005],
                                 pair_ids=[lat[0], lat[1]],
                                 prod_ids=["a", "b"], expect_prod=["a", "b"],
                                 stress_ids=["c", "d"],
                                 expect_stress=["c", "d"])
    bound = (st2["rows_bound_to_ensemble"] and st2["paired_by_latent_id"]
             and st2["pairs"][1]["latent_id"] == lat[1]
             and not st["rows_bound_to_ensemble"])
    # per-pair: 0.90 and 1.10; the unpaired worst/worst scalar reads 8.8/8 = 1.10
    hides = (abs(st["worst"] - 0.9) < 1e-12
             and abs(st["best"] - 1.10) < 1e-12
             and abs(st["worst_unpaired"] - 1.10) < 1e-12)
    ok = (min(cos) > 0.95 and tracks < 0.05 and adj > 1e-3 and hides
          and st["n_degraded"] == 1 and strict == 12 and labelled
          and identified and bound)
    record("(h) the gate uses the production ensemble builder and a PAIRED "
           "statistic", ok,
           "build_paired_ensembles(seed 20260807, 6 pairs) -> %s / %s; "
           "cos(prod,stress) >= %.4f and the absorption ratio tracks the "
           "nominal loss multiplier to %.1f%%, so the pair differs only in "
           "loss; adjacent stress rows differ by >= %.4f (they were exactly "
           "0); on a fixture whose per-pair ratios are 0.90/1.10 the unpaired "
           "worst/worst scalar reads %.2f, hiding the degraded pair; pairs "
           "carry ids, losses and both absolute values with a by-loss "
           "stratification, and the LATENT id (which is what the two sides "
           "actually share -- their T hashes differ by construction) binds "
           "each pair while reported rows are checked against the arrays "
           "passed; and %d/12 broken-pairing inputs raise instead of being "
           "truncated or silently accepted"
           % (h(ens)[:8], h(ens_stress)[:8], min(cos), 100 * tracks, adj,
              st["worst_unpaired"], strict))


def test_execution_bound_snapshot():
    """A snapshot must describe what was EXECUTED, not merely what is on disk
    when the hash is taken.

    Two counterexamples, both run here:

    (a) a module edited AFTER import but before `run()` was hashed in its new
        state while the old module object kept executing, so the run
        certified bytes it never ran;
    (b) `measured_sigma` cached the closure NPZ on first use, so replacing
        that file changed the snapshot hash while the process kept serving
        the OLD sigma.

    (a) cannot be repaired in-process -- the old modules are already imported
    -- so the only sound response is refusal.  (b) can, and is: the cache is
    keyed on the file's content hash.

    NOTE ON METHOD.  An earlier version of this test appended probe bytes to
    the live `fastfull/cost.py` and restored them in a `finally`.  That is not
    safe: during the window it invalidates any concurrent optimizer at its
    end-of-run snapshot check (it did exactly that to a reviewer's run), and
    the restore would clobber a concurrent edit.  No workspace file is touched
    now.  Part (a) hashes a REAL temporary file that is genuinely modified
    between two `snapshot_inputs()` calls, and then drives `run()`'s refusal
    by substituting the recorded import-time baseline -- the same comparison
    on the same code path, with nothing outside the temp directory written.
    """
    import shutil
    import tempfile
    from tmatrix.retrieval.fastfull import opt_marginalized as om
    d = tempfile.mkdtemp()
    armed = om._IMPORT_SNAPSHOT is not None
    probe = os.path.join(d, "probe_input.py")
    with open(probe, "w") as fh:
        fh.write("# stand-in for a snapshotted input\n")
    orig_files = list(om.SNAPSHOT_FILES)
    refused, named, notices_bytes = False, False, False
    try:
        om.SNAPSHOT_FILES = orig_files + [probe]
        before = om.snapshot_inputs()
        with open(probe, "a") as fh:
            fh.write("# changed after hashing\n")
        after = om.snapshot_inputs()
        # snapshot_inputs must actually see the content change
        notices_bytes = (before != after)
        # and run() must refuse when disk no longer matches what was imported
        keep = om._IMPORT_SNAPSHOT
        try:
            om._IMPORT_SNAPSHOT = before
            om.run(lam_um=8.0, samples=4, polish=0, n_ens=2, seed=1,
                   out_dir=d, skip_gate_a=True)
        except RuntimeError as exc:
            refused, named = True, "probe_input.py" in str(exc)
        finally:
            om._IMPORT_SNAPSHOT = keep
    finally:
        om.SNAPSHOT_FILES = orig_files
        shutil.rmtree(d, ignore_errors=True)

    # (b) swap the closure NPZ underneath a warm cache
    s_before = dz.measured_sigma(8.0)
    npz = tempfile.mktemp(suffix=".npz")
    np.savez(npz, f_THz=np.array([20.0, 40.0]),
             sigma=np.array([9.99e-3, 9.99e-3]))
    orig = dz.SIGMA_CLOSURE_NPZ
    try:
        dz.SIGMA_CLOSURE_NPZ = npz
        s_swapped = dz.measured_sigma(8.0)
    finally:
        dz.SIGMA_CLOSURE_NPZ = orig
        os.unlink(npz)
    s_restored = dz.measured_sigma(8.0)
    sees_new = abs(s_swapped - 9.99e-3) < 1e-9
    restores = abs(s_restored - s_before) < 1e-15

    ok = (armed and refused and named and notices_bytes and sees_new
          and restores)
    record("(h) provenance is bound to what executed, not to disk at hash "
           "time", ok,
           "the import-time snapshot covers %d files; a snapshotted input "
           "changed after hashing is seen by snapshot_inputs and makes run() "
           "refuse and name it (%s), with no workspace file touched; the "
           "closure-NPZ cache follows the file's content hash -- swapping it "
           "moves sigma(8um) %.4e -> %.4e and restoring it returns %.4e"
           % (len(om._IMPORT_SNAPSHOT) - 1, "named" if named else "UNNAMED",
              s_before, s_swapped, s_restored))


def test_optimizer_transaction():
    """A run commits atomically, and TWO CONCURRENT runs on one identity
    produce exactly one winner with no mixed artifacts.

    The old assertion here -- that an identical repeat is refused -- is no
    longer the correct expectation and has been REPLACED.  Now that the
    frozen archive is hashed into the run id, committing a run changes the
    archive, so the next call in the same namespace legitimately has
    different inputs and a different identity.  The refusal path still
    exists, but it is reached by a genuine race rather than by a sequential
    repeat, so that is what is tested: two threads that both freeze the same
    archive derive the same id, and exactly one may publish it.
    """
    import json
    import shutil
    import tempfile
    import threading
    from tmatrix.retrieval.fastfull import opt_marginalized as om
    d = tempfile.mkdtemp()
    d2 = tempfile.mkdtemp()
    ARGS = dict(lam_um=8.0, samples=30, polish=1, n_ens=3, seed=4242,
                skip_gate_a=True)
    try:
        rec = om.run(out_dir=d, **ARGS)
        rid = rec["run_id"]
        man, recs = om.verify_completed_run(os.path.join(d, "runs", rid))
        origins = sorted(v["origin"].split(":")[0] for v in recs.values())
        srch = [v for v in recs.values() if v["origin"] == "search"][0]
        sel = [v for v in recs.values()
               if v["origin"].startswith("selected")][0]
        with open(os.path.join(d, "latest.json")) as fh:
            latest = json.load(fh)
        leftovers = [x for x in os.listdir(os.path.join(d, "runs"))
                     if x.startswith(".")]
        # ONE immutable receipt per output root, naming the published run
        _rroot = om.output_root(os.path.join(d, "runs", rid), rid)
        receipt_ok = (om.read_receipt(os.path.join(d, "runs"), _rroot) == rid
                      and os.path.exists(os.path.join(
                          d, "runs", om.RECEIPTS, "%s.json" % _rroot)))

        # a sequential repeat is NOT a repeat: the archive it freezes now
        # contains this run's candidates, so its identity differs
        rec2 = om.run(out_dir=d, **ARGS)
        archive_bound = (rec2["run_id"] != rid)

        # The real collision: two runs that both freeze the SAME archive.
        # Without a barrier the scheduler may let one publish before the other
        # freezes, giving two legitimate identities and a test that passes for
        # the wrong reason.  `_AFTER_FREEZE_HOOK` fires right after freezing,
        # so both workers are provably past that point before either searches.
        out, err = {}, {}
        barrier = threading.Barrier(2, timeout=120)
        om._AFTER_FREEZE_HOOK = barrier.wait

        def go(tag):
            try:
                out[tag] = om.run(out_dir=d2, **ARGS)
            except Exception as exc:                       # noqa: BLE001
                err[tag] = exc

        ts = [threading.Thread(target=go, args=(i,)) for i in (0, 1)]
        try:
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        finally:
            om._AFTER_FREEZE_HOOK = None
        # both workers must have derived the SAME identity for this to be a
        # race at all, not two legitimate runs
        ids = [r["run_id"] for r in out.values()] + [
            "".join(c for c in str(e) if c in "0123456789abcdef")[:12]
            for e in err.values()]
        same_identity = len(set(i[:12] for i in ids)) == 1
        # directories only: the append-only receipts journal also lives in
        # runs/ and is not a published run
        published = sorted(x for x in os.listdir(os.path.join(d2, "runs"))
                           if not x.startswith(".")
                           and os.path.isdir(os.path.join(d2, "runs", x))
                           and x != om.RECEIPTS)
        stray = [x for x in os.listdir(os.path.join(d2, "runs"))
                 if x.startswith(".")]
        one_winner = (len(out) == 1 and len(err) == 1
                      and len(published) == 1)
        winner_ok = (one_winner
                     and om.verify_completed_run(
                         os.path.join(d2, "runs", published[0]))[0] is not None
                     and published[0] == out[list(out)[0]]["run_id"])
        loser = str(list(err.values())[0])[:60] if err else "NONE"

        # DIFFERENT-ID success race.  The same-id race above cannot reach the
        # derived writes at all, because its loser stops at the rename.  Two
        # concurrently SUCCESSFUL runs do reach them, and `<path>.<pid>.tmp`
        # is not unique across threads in one process -- a probe produced
        # FileNotFoundError as one thread replaced the other's temp file.
        d3 = tempfile.mkdtemp()
        derr = {}

        def go2(tag):
            try:
                om.run(out_dir=d3, seed=4242 + tag, lam_um=8.0, samples=30,
                       polish=1, n_ens=3, skip_gate_a=True)
            except Exception as exc:                       # noqa: BLE001
                derr[tag] = exc

        t2 = [threading.Thread(target=go2, args=(i,)) for i in (0, 1)]
        for t in t2:
            t.start()
        for t in t2:
            t.join()
        pub2 = sorted(x for x in os.listdir(os.path.join(d3, "runs"))
                      if not x.startswith(".")
                      and os.path.isdir(os.path.join(d3, "runs", x))
                      and x != om.RECEIPTS)
        tmps = [x for x in os.listdir(d3) if x.endswith(".tmp")] + [
            x for x in os.listdir(os.path.join(d3, "runs"))
            if x.endswith(".tmp")]
        with open(os.path.join(d3, "latest.json")) as fh:
            lat2 = json.load(fh)
        # the DERIVED INDEX must agree with iter_completed, not lag it.  A
        # rebuild that scanned one run, paused while a second completed, then
        # finished, used to atomically overwrite the two-run book with its own
        # stale one-run view.
        with open(os.path.join(d3, "candidate_registry.json")) as fh:
            idx2 = json.load(fh)
        gen = set(idx2.get("_generation", []))
        completed = set(r for r, _, _ in om.iter_completed(
            os.path.join(d3, "runs")))
        index_current = (gen == completed and len(completed) == 2)
        derived_ok = (not derr and len(pub2) == 2 and not tmps
                      and lat2["run_id"] in pub2 and index_current
                      and all(om.verify_completed_run(
                          os.path.join(d3, "runs", r))[0] is not None
                          for r in pub2))
        shutil.rmtree(d3, ignore_errors=True)

        conds = dict(manifest=man is not None,
                     origins=sorted(set(origins)) == ["polish", "search",
                                                      "selected"],
                     search_distinct=srch["design"] != sel["design"],
                     latest=latest["run_id"] == rid,
                     no_leftovers=not leftovers,
                     snapshot=man["snapshot_sha256"] == rec["snapshot_sha256"],
                     archive_bound=archive_bound, winner_ok=winner_ok,
                     no_stray=not stray, same_identity=same_identity,
                     derived_ok=derived_ok, receipt_ok=receipt_ok)
        ok = all(conds.values())
        failed = sorted(k for k, v in conds.items() if not v)
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)
    record("(h) a run commits atomically and concurrent runs on one identity "
           "yield exactly one winner", ok,
           "run %s: verified manifest, origins %s, search distinct from "
           "selected, `latest` points at it, no staging survives. A "
           "sequential repeat now derives a DIFFERENT id (%s) because the "
           "archive it froze contains this run -- identity is archive-bound. "
           "Two concurrent runs synchronized at a post-freeze barrier so both "
           "provably froze the same archive (one derived identity: %s): %d "
           "published, %d refused (%r); the published run verifies and no "
           "staging directory is left to poison a retry. Two concurrently "
           "SUCCESSFUL different-id runs -- which the same-id race cannot "
           "reach, since its loser stops at the rename -- both publish and "
           "verify, `latest` names one of them, no temp file survives, and "
           "the derived index's generation equals iter_completed rather than "
           "lagging it, and the append-only receipt matches the recomputed "
           "output root.%s"
           % (rid[:12], sorted(set(origins)), rec2["run_id"][:12],
              "yes" if same_identity else "NO", len(out), len(err), loser,
              "" if ok else "  FAILED: %s" % failed))


def test_recovered_is_symmetric(B):
    ch, A, W, C = _pieces(WINNER)
    rng = np.random.default_rng(21)
    T = sym.random_passive_d4h(B, rng, n_draw=1, target_fro=0.25)[0][0]
    S = xf.scattered_S(W, T, C, A)
    dS, _ = sy.make_perturbation("angular_smooth", ch, S, rng, SIGMA)
    T_hat, _ = sy.recover_wheel([(W, A, C, S + dS)], B, SIGMA)
    resid = float(sym.symmetry_residual(T_hat[None], B)[0])
    record("(f) the recovered T is exactly D4h + reciprocal by construction",
           resid < 1e-12,
           "relative ||T_hat - P_D4h(T_hat)||_F = %.2e -- the 40-coefficient "
           "parametrization imposes the symmetry rather than fitting it, so "
           "a symmetry residual can never be used as a validity check on "
           "this branch" % resid)


# ------------------------------------------------------------------ driver

def main():
    print("=== fastfull synthetic / Gate A gates ===", flush=True)
    B, _ = sym.build_d4h_reciprocity_basis(MODES)
    test_noise_free_exact(B)
    test_perturbation_structure()
    test_bracket_validated(B)
    test_basin_unique(B)
    test_pooled_adversarial(B)
    test_seed_determinism()
    test_ensemble_diversity(B)
    test_cayley_is_exact(B)
    test_param_map_validation(B)
    test_generator_contract(B)
    test_admission_is_manifest_verified()
    test_archive_is_bound_to_identity()
    test_proposal_lineage_cannot_be_laundered()
    test_receipt_publication_is_crash_recoverable()
    test_error_screen_is_not_a_passed_gate()
    test_paired_ensemble_is_the_production_one()
    test_execution_bound_snapshot()
    test_optimizer_transaction()
    test_prior_and_shared_map(B)
    test_joint_recovery_matched(B)
    test_joint_recovery_misspecified(B)
    test_marginalized_matches_joint_loss(B)
    test_recovered_is_symmetric(B)

    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n=== SUMMARY (test_fastfull_synthetic) ===")
    for name, ok, _ in RESULTS:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("ALL %d TESTS PASSED" % len(RESULTS) if nfail == 0
          else "%d of %d TEST(S) FAILED" % (nfail, len(RESULTS)))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())

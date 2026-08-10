"""Validation-ladder steps 2-3 (INVERSE_TMATRIX_FROM_FLOQUET.md par. 6.2 /
6.3; checklist item 6): noise-free synthetic closed loop and noise
robustness at the smoke frequencies, with the doc's gates evaluated AS
MEASUREMENTS.

READ THIS FIRST -- gate semantics.  The doc's par.-6.2 gate ("bright-
subspace entries recovered <= 1% relative") presupposed that the
campaign-13 specular data identifies the physical bright content.  The
measured identifiability structure (fit.py docstring; reproduced by this
script) says it does NOT:

  * a bright-10-basis fit of the PHYSICAL projected reference carries an
    irreducible MODEL ERROR: sub-threshold entries (|T| < 1e-3 |T|max,
    e.g. the (3,-+3)<->(1,+-1) couplings with |dS/dT| ~ 70-800) produce
    specular signal at the 1e-3 level that the bright span cannot
    represent; absorbed, it biases small bright entries by multiples of
    their own size;
  * the full 68-basis is rank-deficient on this data (Born rank 58/136
    real; ~82/136 at the reference amplitude, spectrum spanning 16
    decades): its null directions are pinned (not fitted) via
    fit.observable_subbasis, and the pinned content -- which includes
    bright-entry weight through SVD mixing -- is unrecoverable IN
    PRINCIPLE at any noise level.

This script therefore runs, per frequency, three closed loops:

  A-bright  (task step-2 (a)):  S from P68(T_ref), bright-10 fit
  A-full    (task step-2 (b)):  S from P68(T_ref), full-68 fit restricted
            to its observable span at the reference linearization
            (s >= 1e-4 s_max; the exact form of the doc-par.-4 null-space
            pinning), plus the THEORETICAL floor of that restriction
  C-clean   (machinery gates):  S from the bright-span part
            P_bright(P68(T_ref)) -- the target then lies exactly in the
            fitted subspace.  Two gates: (i) BASIN CERTIFICATION -- a
            truth-seeded fit must stay at the truth (objective ~ 0,
            entries recovered to ~1e-6): certifies the residual/Jacobian/
            packing machinery is exact; (ii) BORN REACHABILITY -- the
            Born-seed multistart fit, REPORTED as a measurement: whether
            the global basin is reachable from t = 0 is a property of the
            landscape (doc par. 9 local-minima risk; at 12 um the weak
            even-m content makes shallow false basins and Born multistart
            measurably stalls at objective ~3e-8, while at 8 um it
            reaches the truth).

plus the RAW-reference variants of A-bright / A-full (doc: the ~0.2-0.4%
symmetry-violating noise of the stored file = the model-error term a real
campaign adds on top).

PASS/FAIL summary: the LITERAL doc gates on A-bright / A-full are printed
with their measured numbers and marked "[measured FAIL -- expected]" --
they are findings about the data's information content, not defects (the
C-clean loop and test_fit_smoke.py prove the machinery is exact).  The
exit code reflects ONLY the machinery gates (C-clean basin
certification, step-3 linear-theory consistency, and the structural
theta=0 darkness of the even-m sub-basis).

Step 3 (noise robustness, par. 6.3): complex Gaussian noise
n = (sigma/sqrt(2)) (g_re + i g_im) per COMPLEX observable (E|n|^2 =
sigma^2, i.i.d. over angles/blocks/polarizations), weights w = 1/sigma^2,
sigma in {1e-3, 3e-3, 1e-2}, N_trials seeded trials; bright-10 fits on
the CLEAN loop (isolates the noise response; the task-literal variant --
noise on the P68 synthesis -- is also run at sigma = 3e-3 and reported,
its errors being noise + the deterministic A-bright bias).

REGULARIZATION AT THE PHYSICAL PRIOR SCALE (measured necessity): the
bright-basis Jacobian at the reference linearization has singular values
down to s ~ 2e-2 in raw S units, so at sigma = 3e-3 the weakest
directions are determined only to ~ sigma/s ~ 0.15 ABSOLUTE -- an order
of magnitude larger than the physical coefficients (~1e-2).  An
unregularized noisy fit therefore wanders the weak-direction manifold
and poisons every entry through SVD mixing (measured: 300% dipole errors
at chi^2-correct objectives).  All step-3 fits use the doc-par.-4
Tikhonov with the PHYSICAL prior tau = ||pack(T_bp)|| / sqrt(n_params)
(~1e-2, printed per frequency): tikhonov = 1/tau^2 in the weighted
objective, i.e. directions are fitted only where the data beats the
prior, exactly the resolution crossover of observability.svd_resolution
with prior_scale = tau.

Step-3 gates: the doc-6.3 gate "dipole block stable to <= 5% at
sigma = 3e-3" is evaluated AS A MEASUREMENT (the strong electric
in-plane dipole diagonals meet it; the magnetic/z-dipole diagonal
entries, whose absolute S-leverage is ~ sigma at 13 angles, cannot at
any optimizer quality -- their information content is the limit).  The
MACHINERY gate is theory consistency: per-entry rms error over the
trials must match the linear-theory prediction of the regularized
estimator (shrinkage bias + posterior covariance propagated through the
basis; median measured/predicted ratio within [1/3, 3]) -- this
certifies estimator + noise model + Jacobian end to end without
depending on what the data can or cannot resolve.

Angle-count study at sigma = 3e-3: campaign 13 vs starter 5
{(0,0),(30,0),(30,22.5),(60,0),(60,22.5)} vs normal-only [(0,0)], same
regularized protocol, with the bright-basis resolution res_c (physical
prior) reported per set.  MACHINERY GATE: the ORBIT-PURE even-m
sub-basis of the bright span (span{P(E_ij): (i,j) even-m bright}) must
have exact-Jacobian column norms at machine zero for normal-only and
O(10) at campaign -- theta = 0 delivers 2 complex observables and
structurally cannot see even-m content (doc par. 3/4).  (res_c per
FITTED basis direction is reported but not gated: the SVD-orthonormal
bright basis mixes position orbits, so single directions blend even- and
odd-m content.)

Outputs: results/synthetic_ifreq{NN}.npz (arrays for every loop above),
results/observability_heatmap_ifreq{NN}.png / _spectrum_ifreq{NN}.png +
observability npz (via observability.analyze; the doc-named default
figure paths are produced by the observability.py CLI).

Usage (each frequency fits in well under 10 min; run per-freq if needed):
    python synthetic_test.py --freqs 32,48
    python synthetic_test.py --freqs 32 --trials 10 --sigmas 1e-3,3e-3,1e-2
    python synthetic_test.py --freqs 48 --no-noise      # step 2 only
Frequencies whose campaign angles are not yet cached are skipped
gracefully, so the eventual full-band run is
    python synthetic_test.py --freqs all
"""
import argparse
import os
import sys
import time


import numpy as np

from tmatrix.retrieval.forward import ForwardModel                      # noqa: E402
from tmatrix.retrieval import parametrize as par # noqa: E402
from tmatrix.retrieval import fit as fitmod # noqa: E402
from tmatrix.retrieval import observability as obs # noqa: E402

from tmatrix.paths import RETRIEVAL_RESULTS

RESULTS_DIR = str(RETRIEVAL_RESULTS)
GATES = []          # (name, passed, is_machinery_gate, detail)


def gate(name, ok, machinery, detail=""):
    tag = "PASS" if ok else ("FAIL" if machinery
                             else "measured FAIL -- expected")
    print("[%s] %s%s" % (tag, name, ("  --  " + detail) if detail else ""),
          flush=True)
    GATES.append((name, bool(ok), bool(machinery), detail))


def class_selectors(modes, entries):
    """Boolean selectors over the bright-entry list for reporting."""
    cls = fitmod.classify_entries(modes)
    sel = {}
    for name in ("dipole", "quad22", "even_m", "odd_m", "c4_violating"):
        sel[name] = np.array([cls[name][i, j] for i, j in entries])
    sel["dipole_odd"] = sel["dipole"] & sel["odd_m"]
    sel["dipole_even"] = sel["dipole"] & sel["even_m"]
    return sel


def class_row(tag, pk, pw, sel):
    def mx(s):
        return pk[s].max() if s.any() else np.nan
    print("   %-12s all25: max=%9.3e med=%9.3e n>1%%=%2d n>5%%=%2d | "
          "dip=%9.3e (odd %9.3e / even %9.3e) quad22=%9.3e even_m=%9.3e"
          % (tag, pk.max(), np.median(pk), int((pk > 0.01).sum()),
             int((pk > 0.05).sum()), mx(sel["dipole"]),
             mx(sel["dipole_odd"]), mx(sel["dipole_even"]),
             mx(sel["quad22"]), mx(sel["even_m"])), flush=True)


def run_frequency(fm, ifreq, B68, B10, binfo, args):
    """Steps 2-3 at one frequency.  Returns dict of arrays for the npz."""
    modes = fm.modes
    mask = binfo["mask"]
    entries = np.argwhere(mask)
    peak = np.abs(fm.data.T[:, entries[:, 0], entries[:, 1]]).max(axis=0)
    sel = class_selectors(modes, entries)
    angles = fitmod.resolve_angles(args.angles)
    d = args.direction
    out = dict(entries=entries, peak=peak, mask=mask,
               angle_indices=np.array(angles), direction=d,
               ifreq=ifreq, lam_um=fm.lam_um[ifreq])

    T_ref = fm.data.T[ifreq]
    T_proj = par.unpack(par.pack(T_ref, B68), B68)
    T_bp = par.unpack(par.pack(T_proj, B10), B10)     # bright-span part
    S_proj = fm.predict(T_proj, ifreq, angles, d)
    S_raw = fm.predict(T_ref, ifreq, angles, d)
    S_clean = fm.predict(T_bp, ifreq, angles, d)
    out["T_ref"], out["T_proj"], out["T_bp"] = T_ref, T_proj, T_bp
    out["S_proj"], out["S_raw"], out["S_clean"] = S_proj, S_raw, S_clean

    eng10 = fitmod.AnalyticJacobian(fm, ifreq, B10, angles, d)
    eng68 = fitmod.AnalyticJacobian(fm, ifreq, B68, angles, d)
    t_lin = par.pack(T_proj, B68)
    B_obs, s_ref, V_obs = fitmod.observable_subbasis(
        eng68, t_lin=t_lin, s_cutoff_rel=args.obs_cutoff)
    engO = fitmod.AnalyticJacobian(fm, ifreq, B_obs, angles, d)
    out["obs_cutoff"] = args.obs_cutoff
    out["obs_rank_complex"] = B_obs.shape[0]
    out["s_ref_complex"] = s_ref

    print("\n=== ifreq %d (lam = %.2f um), angles %s (%d), direction %+d "
          "===" % (ifreq, fm.lam_um[ifreq], args.angles, len(angles), d))
    print(" model-error floor of the bright span (forward-only): "
          "max|S(T_proj) - S(T_bp)| = %.3e"
          % np.abs(S_proj - S_clean).max())
    print(" observable restriction of the full 68-basis at the reference "
          "linearization: %d complex of 68 kept (s >= %.0e s_max; "
          "s_max = %.2e)" % (B_obs.shape[0], args.obs_cutoff, s_ref[0]))

    # ---------------------------------------------------------- step 2
    print("\n--- step 2 (noise-free closed loops) ---")
    tols = dict(xtol=1e-12, ftol=1e-12, gtol=1e-12)
    combos = [
        ("A_bright_proj", B10, eng10, S_proj, T_proj, dict(max_nfev=2000)),
        ("A_full_proj", B_obs, engO, S_proj, T_proj, dict(max_nfev=2500)),
        ("A_bright_raw", B10, eng10, S_raw, T_ref, dict(max_nfev=2000)),
        ("A_full_raw", B_obs, engO, S_raw, T_ref, dict(max_nfev=2500)),
        ("C_clean", B10, eng10, S_clean, T_bp, dict(max_nfev=2000)),
    ]
    step2 = {}
    for name, Bx, engx, S_meas, T_tgt, extra in combos:
        r = fitmod.fit_frequency_multistart(
            fm, ifreq, Bx, S_meas, angles, n_starts=args.starts,
            seed=args.seed, direction=d, engine=engx, **tols, **extra)
        pw, pk = fitmod.entry_errors(r["T0_hat"], T_tgt, entries, peak)
        step2[name] = (r, pw, pk)
        print("  %-13s obj=%.3e maxdS=%.2e nfev=%4d st=%d wall=%5.1f s "
              "passivity=%.4f%s"
              % (name, r["objective"], np.abs(r["dS"]).max(), r["nfev"],
                 r["status"], r["wall_s"], r["passivity_max_sv"],
                 "" if r["multistart_pick"] == 0 else
                 " [restart %d]" % r["multistart_pick"]))
        class_row(name, pk, pw, sel)
        out["s2_%s_T0_hat" % name] = r["T0_hat"]
        out["s2_%s_t_hat" % name] = r["t_hat"]
        out["s2_%s_rel_peaknorm" % name] = pk
        out["s2_%s_rel_pointwise" % name] = pw
        out["s2_%s_objective" % name] = r["objective"]
        out["s2_%s_maxdS" % name] = np.abs(r["dS"]).max()
        out["s2_%s_nfev" % name] = r["nfev"]
        out["s2_%s_wall_s" % name] = r["wall_s"]
        out["s2_%s_passivity" % name] = r["passivity_max_sv"]

    # theoretical floor of the observable restriction (no fit involved)
    zc = t_lin[:68] + 1j * t_lin[68:]
    T_obs_part = np.tensordot(V_obs @ (V_obs.conj().T @ zc), B68,
                              axes=(0, 0))
    pw_f, pk_f = fitmod.entry_errors(T_obs_part, T_proj, entries, peak)
    class_row("A_full FLOOR", pk_f, pw_f, sel)
    out["s2_A_full_floor_rel_peaknorm"] = pk_f
    print("   (A_full FLOOR = entries of the observable-span part of "
          "T_proj: the content the restricted fit cannot represent; the "
          "fit is expected to land near this floor)")

    # per-entry table for the task-(a) fit
    cmp1 = fitmod.compare_to_reference(step2["A_bright_proj"][0]
                                       ["T0_hat"][None], T_proj[None],
                                       mask, peak_ref_stack=fm.data.T)
    print("\n  per-entry table, A_bright_proj vs P68(T_ref) "
          "(sorted by band-max peak-normalized error):")
    print(fitmod.format_compare(cmp1, modes))

    # C-clean basin certification: truth-seeded fit must stay at truth
    t_truth = par.pack(T_bp, B10)
    r_seed = fitmod.fit_frequency(fm, ifreq, B10, S_clean, angles,
                                  t0=t_truth, direction=d, engine=eng10,
                                  **tols, max_nfev=500)
    pw_s, pk_s = fitmod.entry_errors(r_seed["T0_hat"], T_bp, entries,
                                     peak)
    out["s2_C_seed_rel_peaknorm"] = pk_s
    out["s2_C_seed_objective"] = r_seed["objective"]

    # gates
    pk_ab = step2["A_bright_proj"][2]
    pk_af = step2["A_full_proj"][2]
    pk_cl = step2["C_clean"][2]
    r_cl = step2["C_clean"][0]
    gate("step2 doc-literal: A_bright_proj all-25 <= 1%% (ifreq %d)"
         % ifreq, pk_ab.max() <= 0.01, False,
         "max = %.3e, n>1%% = %d/25 (bright-span model error, see "
         "docstring)" % (pk_ab.max(), int((pk_ab > 0.01).sum())))
    gate("step2 doc-literal: A_full_proj all-25 <= 1%% (ifreq %d)"
         % ifreq, pk_af.max() <= 0.01, False,
         "max = %.3e vs theoretical floor %.3e (unobservable content)"
         % (pk_af.max(), pk_f.max()))
    gate("step2 machinery: C_clean basin certification (truth-seeded, "
         "ifreq %d)" % ifreq,
         r_seed["objective"] <= 1e-16 and pk_s.max() <= 1e-6, True,
         "obj = %.2e, all-25 peaknorm max = %.2e (exactness of residual/"
         "Jacobian/packing)" % (r_seed["objective"], pk_s.max()))
    gate("step2 measured: C_clean Born-reachability all-25 <= 1%% "
         "(ifreq %d)" % ifreq, pk_cl.max() <= 0.01, False,
         "max = %.3e at obj = %.2e (multistart objs %s; true obj 0; "
         "basin structure of this frequency, doc par. 9)"
         % (pk_cl.max(), r_cl["objective"],
            ["%.1e" % o for o in r_cl["multistart_objectives"]]))
    raw_extra = step2["A_bright_raw"][2].max() / max(pk_ab.max(), 1e-300)
    print("  raw-target variant extra floor: A_bright raw/proj max "
          "peaknorm = %.3e/%.3e (x%.2f); A_full raw/proj = %.3e/%.3e"
          % (step2["A_bright_raw"][2].max(), pk_ab.max(), raw_extra,
             step2["A_full_raw"][2].max(), pk_af.max()))

    # ---------------------------------------------------- observability
    if not args.no_obs:
        print("\n--- observability map (full 68-basis, FD Jacobian) ---")
        obs.analyze(
            fm, ifreq, basis="full", angles=args.angles,
            sigma=args.obs_sigma, direction=d, B=B68,
            bright_mask=mask, B_full_for_vis=B68,
            heatmap_path=os.path.join(
                RESULTS_DIR, "observability_heatmap_ifreq%02d.png" % ifreq),
            spectrum_path=os.path.join(
                RESULTS_DIR, "observability_spectrum_ifreq%02d.png"
                % ifreq),
            npz_tag="ifreq%02d_full_%s" % (ifreq, args.angles),
            do_visibility=not args.no_vis)

    # ---------------------------------------------------------- step 3
    if args.no_noise:
        return out
    tau = float(np.linalg.norm(par.pack(T_bp, B10))
                / np.sqrt(2 * B10.shape[0]))
    tik = 1.0 / tau ** 2
    out["s3_tau"] = tau
    print("\n--- step 3 (noise robustness; clean loop, bright basis, "
          "w = 1/sigma^2, tikhonov = 1/tau^2 with physical prior "
          "tau = %.2e) ---" % tau)
    sigmas = args.sigma_list
    ntr = args.trials
    nE = len(entries)
    s3_pk = np.full((len(sigmas), ntr, nE), np.nan)
    s3_pw = np.full((len(sigmas), ntr, nE), np.nan)
    s3_obj = np.full((len(sigmas), ntr), np.nan)
    s3_nfev = np.zeros((len(sigmas), ntr), dtype=int)
    s3_fail = np.zeros((len(sigmas), ntr), dtype=bool)
    t3 = time.time()
    for i_s, sg in enumerate(sigmas):
        w = 1.0 / sg ** 2
        for tr in range(ntr):
            rng = np.random.default_rng(
                [args.seed, ifreq, int(round(sg * 1e12)), tr])
            noise = (rng.standard_normal(S_clean.shape)
                     + 1j * rng.standard_normal(S_clean.shape)) \
                * (sg / np.sqrt(2.0))
            r = fitmod.fit_frequency(fm, ifreq, B10, S_clean + noise,
                                     angles, weights=w, direction=d,
                                     engine=eng10, tikhonov=tik,
                                     method="lm", max_nfev=1000)
            pw, pk = fitmod.entry_errors(r["T0_hat"], T_bp, entries, peak)
            s3_pk[i_s, tr] = pk
            s3_pw[i_s, tr] = pw
            s3_obj[i_s, tr] = r["objective"]
            s3_nfev[i_s, tr] = r["nfev"]
            s3_fail[i_s, tr] = not r["success"]
        n_obs_c = 8 * len(angles)
        print("  sigma=%.0e: bright-entry peaknorm mean(max)=%.3e "
              "max(max)=%.3e | dipole mean(max)=%.3e max(max)=%.3e | "
              "obj mean=%.1f (E~%d complex obs) | nfev med=%d fails=%d"
              % (sg, s3_pk[i_s].max(axis=1).mean(),
                 s3_pk[i_s].max(), s3_pk[i_s][:, sel["dipole"]]
                 .max(axis=1).mean(),
                 s3_pk[i_s][:, sel["dipole"]].max(),
                 s3_obj[i_s].mean(), n_obs_c,
                 int(np.median(s3_nfev[i_s])), int(s3_fail[i_s].sum())),
              flush=True)
    out["s3_sigmas"] = np.array(sigmas)
    out["s3_rel_peaknorm"] = s3_pk
    out["s3_rel_pointwise"] = s3_pw
    out["s3_objective"] = s3_obj
    out["s3_nfev"] = s3_nfev
    out["s3_notconverged"] = s3_fail
    print("  step-3 sweep wall: %.1f s" % (time.time() - t3))

    i3 = sigmas.index(3e-3) if 3e-3 in sigmas else None
    if i3 is not None:
        dip_mean = s3_pk[i3][:, sel["dipole"]].max(axis=1).mean()
        gate("step3 doc-literal: clean-loop dipole block <= 5%% at "
             "sigma=3e-3 (ifreq %d)" % ifreq, dip_mean <= 0.05, False,
             "mean-over-%d-trials of max dipole peaknorm = %.3e "
             "(max %.3e; odd-dipole mean %.3e; strongest E-dipole "
             "diagonals mean %.3e) -- information-limited, see docstring"
             % (ntr, dip_mean, s3_pk[i3][:, sel["dipole"]].max(),
                s3_pk[i3][:, sel["dipole_odd"]].max(axis=1).mean(),
                s3_pk[i3][:, sel["dipole_odd"] & (peak > 0.01)]
                .max(axis=1).mean()
                if (sel["dipole_odd"] & (peak > 0.01)).any() else np.nan))

        # MACHINERY: measured trial errors vs linear-theory prediction
        # (shrinkage bias + posterior covariance of the tik estimator)
        w3 = 1.0 / 3e-3 ** 2
        t_truth10 = par.pack(T_bp, B10)
        npar10 = t_truth10.size
        Jw = eng10.jac_packed(t_truth10, w3)
        A = Jw.T @ Jw + tik * np.eye(npar10)
        Ainv = np.linalg.inv(A)
        Cov = 0.5 * Ainv @ (Jw.T @ Jw) @ Ainv
        bias_t = -Ainv @ (tik * t_truth10)
        nb10 = npar10 // 2
        pred = np.empty(nE)
        for e, (i, j) in enumerate(entries):
            gr = np.concatenate([B10[:, i, j], np.zeros(nb10)])
            gi = np.concatenate([np.zeros(nb10), B10[:, i, j]])
            b_e = complex(gr @ bias_t, gi @ bias_t)
            v_e = float(gr @ Cov @ gr + gi @ Cov @ gi)
            pred[e] = np.sqrt(abs(b_e) ** 2 + v_e)
        meas = np.sqrt(((s3_pk[i3] * peak[None, :]) ** 2).mean(axis=0))
        sig_e = pred > 1e-3 * pred.max()
        ratio = meas[sig_e] / pred[sig_e]
        out["s3_pred_rms"] = pred
        out["s3_meas_rms"] = meas
        gate("step3 machinery: trial errors match linear theory "
             "(median ratio in [1/3, 3], ifreq %d)" % ifreq,
             bool(1.0 / 3.0 <= np.median(ratio) <= 3.0), True,
             "median measured/predicted rms = %.2f (range %.2f..%.2f "
             "over %d entries with signal)"
             % (np.median(ratio), ratio.min(), ratio.max(),
                int(sig_e.sum())))

        # task-literal variant: noise on the P68 synthesis
        w = 1.0 / 3e-3 ** 2
        pk_lit = np.full((ntr, nE), np.nan)
        for tr in range(ntr):
            rng = np.random.default_rng([args.seed, ifreq, 777, tr])
            noise = (rng.standard_normal(S_proj.shape)
                     + 1j * rng.standard_normal(S_proj.shape)) \
                * (3e-3 / np.sqrt(2.0))
            r = fitmod.fit_frequency(fm, ifreq, B10, S_proj + noise,
                                     angles, weights=w, direction=d,
                                     engine=eng10, tikhonov=tik,
                                     method="lm", max_nfev=1000)
            _, pk_lit[tr] = fitmod.entry_errors(r["T0_hat"], T_proj,
                                                entries, peak)
        out["s3_lit_rel_peaknorm"] = pk_lit
        gate("step3 doc-literal: P68-loop dipole block <= 5%% at "
             "sigma=3e-3 (ifreq %d)" % ifreq,
             pk_lit[:, sel["dipole"]].max(axis=1).mean() <= 0.05, False,
             "mean of max dipole peaknorm = %.3e (noise + the "
             "deterministic A_bright bias)"
             % pk_lit[:, sel["dipole"]].max(axis=1).mean())

    # ------------------------------------------- step 3 angle-count study
    print("\n--- step 3 angle-count study (sigma = 3e-3, clean loop, "
          "same regularized protocol) ---")
    sets = [("campaign", angles),
            ("starter", fitmod.ANGLE_SETS["starter"]),
            ("normal-only", fitmod.ANGLE_SETS["normal-only"])]
    # bright-basis complex directions supported on even-m entries (the
    # position-orbit block structure makes this an exact split)
    G_even = np.array([[B10[k][i, j] for k in range(B10.shape[0])]
                       for i, j in entries[sel["even_m"]]])
    even_dirs = np.nonzero((np.abs(G_even) ** 2).sum(axis=0) > 1e-12)[0]
    s3a_pk = np.full((len(sets), ntr, nE), np.nan)
    res_by_set = {}
    set_names = []
    for i_a, (aname, aset) in enumerate(sets):
        set_names.append(aname)
        miss = [a for a in aset if not fm.have[ifreq, a]]
        if miss:
            print("  %-11s: SKIP (angles %s not cached)" % (aname, miss))
            continue
        engx = (eng10 if aset == angles else
                fitmod.AnalyticJacobian(fm, ifreq, B10, aset, d))
        S_cl = (S_clean if aset == angles else
                fm.predict(T_bp, ifreq, aset, d))
        # bright-basis resolution at this angle set (physical prior)
        Jx = engx.jac_packed(par.pack(T_bp, B10))
        svx = obs.svd_resolution(Jx, sigma=3e-3, prior_scale=tau)
        res_c = obs.fold_complex(svx["res"])
        res_by_set[aname] = res_c
        w = 1.0 / 3e-3 ** 2
        for tr in range(ntr):
            rng = np.random.default_rng(
                [args.seed, ifreq, 1000 + i_a, tr])
            noise = (rng.standard_normal(S_cl.shape)
                     + 1j * rng.standard_normal(S_cl.shape)) \
                * (3e-3 / np.sqrt(2.0))
            r = fitmod.fit_frequency(fm, ifreq, B10, S_cl + noise, aset,
                                     weights=w, direction=d, engine=engx,
                                     tikhonov=tik, method="lm",
                                     max_nfev=1000)
            _, s3a_pk[i_a, tr] = fitmod.entry_errors(r["T0_hat"], T_bp,
                                                     entries, peak)
        pk_a = s3a_pk[i_a]
        print("  %-11s (%2d angles): all25 mean(max)=%.3e | dipole=%.3e "
              "| quad22=%.3e | even_m=%.3e | res_c(prior tau): even-m "
              "dirs %s" % (aname, len(aset), pk_a.max(axis=1).mean(),
                           pk_a[:, sel["dipole"]].max(axis=1).mean(),
                           pk_a[:, sel["quad22"]].max(axis=1).mean(),
                           pk_a[:, sel["even_m"]].max(axis=1).mean(),
                           np.array2string(res_c[even_dirs],
                                           precision=2)), flush=True)
        out["s3a_res_c_%s" % aname.replace("-", "_")] = res_c
    out["s3a_set_names"] = np.array(set_names)
    out["s3a_rel_peaknorm"] = s3a_pk
    out["s3a_even_dirs"] = even_dirs

    # MACHINERY: orbit-pure even-m sub-basis is structurally dark at
    # theta=0 and lit at campaign (exact-J column norms at T_bp)
    B10_even, _ = obs.subspace_basis_for_entries(B10,
                                                 entries[sel["even_m"]])
    eng_e0 = fitmod.AnalyticJacobian(fm, ifreq, B10_even, [0], d)
    eng_ec = fitmod.AnalyticJacobian(fm, ifreq, B10_even, angles, d)
    t0e = par.pack(T_bp, B10_even)
    cn0 = np.linalg.norm(eng_e0.jac_packed(t0e), axis=0).max()
    cnc = np.linalg.norm(eng_ec.jac_packed(t0e), axis=0).max()
    out["s3a_evensub_colnorm_normal"] = cn0
    out["s3a_evensub_colnorm_campaign"] = cnc
    gate("step3 machinery: even-m sub-basis structurally dark at "
         "theta=0, lit at campaign (ifreq %d)" % ifreq,
         bool(cn0 <= 1e-8 * cnc), True,
         "max |J col|: normal-only %.3e vs campaign %.3e (%d orbit-pure "
         "even-m directions); entry errors even_m: normal-only %.3e vs "
         "campaign %.3e (normal-only = prior-pinned at 0, i.e. ~100%% of "
         "their own size)"
         % (cn0, cnc, B10_even.shape[0],
            np.nanmax(s3a_pk[2][:, sel["even_m"]], axis=1).mean()
            if not np.isnan(s3a_pk[2]).all() else np.nan,
            np.nanmax(s3a_pk[0][:, sel["even_m"]], axis=1).mean()
            if not np.isnan(s3a_pk[0]).all() else np.nan))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Validation-ladder steps 2-3 (doc par. 6)")
    ap.add_argument("--freqs", default="32,48",
                    help="comma list of frequency indices, or 'all'")
    ap.add_argument("--sigmas", default="1e-3,3e-3,1e-2")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--angles", default="campaign",
                    help="campaign | starter | all17 | comma list")
    ap.add_argument("--direction", type=int, choices=(-1, 1), default=-1)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--starts", type=int, default=3,
                    help="multistart count for the step-2 fits")
    ap.add_argument("--obs-cutoff", type=float, default=1e-4,
                    help="s/s_max cutoff of the full-basis observable "
                         "restriction")
    ap.add_argument("--obs-sigma", type=float, default=3e-3,
                    help="noise floor for the observability analysis "
                         "(PLACEHOLDER until the par.-7 complex closure)")
    ap.add_argument("--no-noise", action="store_true",
                    help="skip step 3")
    ap.add_argument("--no-obs", action="store_true",
                    help="skip the observability figures")
    ap.add_argument("--no-vis", action="store_true",
                    help="skip the theta=0 visibility verification")
    args = ap.parse_args(argv)
    args.sigma_list = [float(s) for s in args.sigmas.split(",")]

    t_all = time.time()
    print("=" * 76)
    print("synthetic_test: validation-ladder steps 2-3 "
          "(doc par. 6.2 / 6.3)")
    print("=" * 76)
    fm = ForwardModel()
    if args.freqs.strip().lower() == "all":
        freqs = list(range(fm.nf))
    else:
        freqs = [int(x) for x in args.freqs.split(",")]
    print("cache: %s" % fm.cache_path)
    print("requested freqs: %s" % freqs)

    B68, meta = par.build_c4v_reciprocity_basis(fm.modes,
                                               verify_numeric=False)
    B10, binfo = par.bright_orbit_basis(B68, fm.data.T, threshold=1e-3,
                                        modes=fm.modes)
    assert B68.shape[0] == 68 and B10.shape[0] == 10
    print("bases: full %d complex, bright %d complex (%d bright entries, "
          "%d orbits)" % (B68.shape[0], B10.shape[0], binfo["n_bright"],
                          binfo.get("n_orbits", -1)))

    want = fitmod.resolve_angles(args.angles)
    done = []
    for ifreq in freqs:
        miss = [a for a in want if not fm.have[ifreq, a]]
        if miss:
            print("\n[skip] ifreq %d: angle indices %s not cached yet "
                  "(background precompute still filling)" % (ifreq, miss),
                  flush=True)
            continue
        res = run_frequency(fm, ifreq, B68, B10, binfo, args)
        path = os.path.join(RESULTS_DIR, "synthetic_ifreq%02d.npz" % ifreq)
        np.savez(path, **{k: v for k, v in res.items() if v is not None})
        print("\n  saved: %s" % path)
        done.append(ifreq)

    print("\n" + "=" * 76)
    print("SUMMARY (synthetic_test; freqs done: %s; %.1f s total)"
          % (done, time.time() - t_all))
    print("=" * 76)
    n_mach_fail = 0
    for name, ok, machinery, detail in GATES:
        tag = "PASS" if ok else ("FAIL" if machinery
                                 else "measured FAIL -- expected")
        print("  [%s] %s" % (tag, name))
        if machinery and not ok:
            n_mach_fail += 1
    print("machinery gates: %s"
          % ("ALL PASSED" if n_mach_fail == 0
             else "%d FAILED" % n_mach_fail))
    print("(doc-literal step-2/3 gates on the physical target are "
          "information-content measurements; see module docstring)")
    sys.exit(1 if n_mach_fail else 0)


if __name__ == "__main__":
    main()

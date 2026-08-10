"""Acceptance criteria of INVERSE_TMATRIX_FROM_FLOQUET.md par. 8 (+ the
par.-7 channel-dictionary hook), as reusable functions plus a CLI that
exercises all of them end to end on SYNTHETIC data today and consumes real
de-embedded CST data unchanged in the final campaign stage
(checklist item 8, second half).

WHAT THIS MODULE DOES AND DOES NOT PROVE
----------------------------------------
It runs five acceptance checks on a fitted T0 (par. 8.1-8.4) and one
pre-campaign gate (par. 7):

  1 heldout_acceptance            par. 8.1  fit on a few angles, PREDICT the
                                            rest, gate max complex |dS|
  2 bright_comparison             par. 8.2  per-entry errors vs the reference
                                            tmat.h5, per entry class
  3 discrepancy_vs_observability  par. 8.2  is the error pattern explained by
                                            the observability map?
  4 physical_checks               par. 8.3  passivity + reciprocity (CHECKS,
                                            never enforced constraints)
  5 fit_observability             par. 8.4  heatmap + SV spectrum published
                                            with the fit
  6 channel_dictionary_acceptance par. 7    the TE/TM label hypothesis must be
                                            uniquely pinned BEFORE the campaign

Each check is a MEASUREMENT of what the data supports; only the plumbing
around it is a correctness gate.  Following synthetic_test.py, gate(name,
ok, machinery, detail) marks every result as either a MACHINERY gate (a
defect if it fails; it drives the exit code) or a measurement (printed with
its number, never fatal).  Concretely:

  * a held-out residual of 1e-4 proves the fitted T0 predicts unseen angles;
    it does NOT prove the T-matrix entries are right (many T0 reproduce the
    specular channel -- see the observability rank deficiency in fit.py);
  * a bright-entry error of x% is a statement about the campaign angle
    set's information content, not about the optimizer;
  * passivity/reciprocity are necessary, never sufficient;
  * the par.-7 channel-dictionary test proves only that ONE label
    hypothesis survives -- it cannot detect an error common to all 8.

CONVENTIONS (locked; see retrieval/HANDOFF.md -- do not re-derive)
-----------------------------------------------------------------
  * direction = -1 (down-going, k_hat = (sin th cos ph, sin th sin ph,
    -cos th)) everywhere in this module: the CST/campaign illumination.
  * Jones index order 0 = TE, 1 = TM; S[angle, block, a, b] with block
    0 = S11 (reflection), 1 = S21 (transmission), row a = receive,
    column b = incident.
  * Angle table = precompute_C.ANGLES_DEG (17 rows, campaign 13 first);
    named subsets in fit.ANGLE_SETS.
  * Symmetry subspace from parametrize.build_c4v_reciprocity_basis (68
    complex dims); bright span from parametrize.bright_orbit_basis (10).
  * Born seed t0 = 0 (fit.py default).
  * Entry errors are PEAK-NORMALIZED exactly as in synthetic_test.py:
    peak_e = max over ALL 49 reference frequencies of |T_ref[f, i, j]|,
    err_e = |T0_hat - T_tgt|_e / peak_e, via fit.entry_errors.

DEFAULT FIT / HOLD-OUT SPLIT (par. 8.1 says "fit on 4, predict the rest")
------------------------------------------------------------------------
FIT_ANGLES_DEFAULT = [0, 5, 10, 11]
    = (theta, phi) = (0, 0), (30, 22.5), (60, 0), (60, 22.5) degrees.
Rationale (each point is a measured or doc-normative fact):
  * it is a strict SUBSET of the doc-par.-7 "starter" campaign subset
    (theta in {0,30,60} x phi in {0,22.5}), so accepting it costs no CST
    runs beyond the ones the doc already budgets first;
  * it spans the full theta range 0/30/60: even-m content (m, m' in
    {0,+-2}, e.g. the (2,+-2) quadrupole) is STRICTLY dark at theta = 0
    (doc par. 4, re-verified in synthetic_test.py), so oblique angles are
    the only source of that information, and 60 deg carries the most;
  * two of the four are phi = 22.5, the only angles delivering all 8
    complex observables per (angle, frequency) -- on the mirror planes
    phi in {0,45} the cross-pol vanishes identically and only 4 survive
    (doc par. 3);
  * (60, 0) adds a mirror-plane oblique angle: 4 more observables and the
    reference-model-free cross-pol check of deembed.check_mirror_plane_crosspol;
  * (0, 0) is kept because it is the phase/closure anchor of the campaign
    (par. 7 orders it first and its complex closure defines sigma), and it
    is the cheapest run.
  Observable budget: 2 + 8 + 4 + 8 = 22 complex = 44 real observations for
  the 20 real parameters of the bright-10 basis -- over-determined.
  Measured (P68 synthetic, campaign hold-out, ifreq 32/48): held-out worst
  |dS| = 2.0e-4 / 6.0e-4, i.e. this split predicts the 9 unseen campaign
  angles ~20x inside the par.-8.1 gate.  Alternatives were measured too:
  [0,4,5,11] gives 4.6e-4/3.3e-4 and [0,5,8,11] degrades to 1.7e-2 at
  ifreq 48 (a false Born basin), which is why the default is multistarted.

HOLDOUT_ANGLES_DEFAULT = campaign 13 minus the fit angles = 9 angles.
Pass holdout_angles="all17" to additionally predict the 4 treams-validation
angles (20/40 deg), which no campaign run would ever have measured.

CHANNEL-DICTIONARY ACCEPTANCE (par. 7): BAND POOLING AND THE STATISTIC
----------------------------------------------------------------------
The doc says "all 8 de-embedded complex numbers must match ... to <= 1e-2".
Two separate decisions are needed to turn that into a test: how to handle
the 49-frequency band, and what statistic to apply.  Both are measured
below, on the RAW reference T at direction -1 over the full band.

(a) BAND: POOLED, always.  Whatever the statistic, the frequency axis is
    pooled (band_mode="pooled", the default; "per_freq" exists only to
    demonstrate that it fails).  Measured with the max statistic at
    (30, 0), tol 1e-2: pooled over all 49 frequencies -> exactly 1 winner;
    pooled over only the two smoke frequencies {32, 48} -> 4 winners;
    per-frequency -> a unique winner at only 5 of 49 frequencies.  Pool
    over everything you measured.

(b) WHY THE CROSS-SIGN HALF IS HARD -- the structure barely cross-
    polarizes ANYWHERE.  A wrong s11_cross/s21_cross hypothesis perturbs
    only the cross-pol entries, so the separation between the true and the
    nearest wrong hypothesis is exactly 2 x |cross-pol|.  Measured
    band-max |cross-pol| with the raw reference T, and the resulting
    max-statistic separation `sep` = min pooled-max residual over the 7
    wrong hypotheses, at all 13 campaign angles:

      idx (theta,phi)   sep        |cross|max     idx (theta,phi)   sep
       0  ( 0, 0  )   1.201e-02   7.230e-03       7  (45, 0  )   1.235e-02
       1  (15, 0  )   1.105e-02   6.884e-03       8  (45,22.5)   1.301e-02
       2  (15,22.5)   1.174e-02   6.075e-03       9  (45,45  )   1.037e-02
       3  (15,45  )   8.193e-03   4.917e-03      10  (60, 0  )   1.645e-02
       4  (30, 0  )   1.101e-02   6.979e-03      11  (60,22.5)   1.710e-02
       5  (30,22.5)   1.170e-02   6.446e-03      12  (60,45  )   1.321e-02
       6  (30,45  )   9.614e-03   4.809e-03

    Mirror-plane (phi in {0,45}) sep median 1.105e-02; phi = 22.5 sep
    median 1.238e-02.  RATIO 1.12 -- there is NO qualitative advantage to
    the phi = 22.5 angles.  (An earlier version of this docstring claimed
    there was, reasoning from the C4v-PROJECTED T, whose cross-pol is
    identically 0 on the mirror planes and reaches 5.4e-3 at (60, 22.5).
    That projected statement is true but operationally irrelevant: the
    acceptance uses the RAW reference T, whose ~0.3 % C4-violating content
    puts |cross-pol| at 4.8e-3..9.7e-3 at EVERY campaign angle, mirror
    plane or not.  Do not choose the acceptance angle for phi.)  What DOES
    help is theta: sep grows monotonically with theta, 1.6x from 15 to 60
    degrees.

(c) STATISTIC: use chi2, not max.  The doc's max over 8 channels x 49
    frequencies = 392 complex numbers is nearly powerless once noise is
    present: the TRUE hypothesis's own pooled max residual is already
    ~ sigma sqrt(ln 392) = 2.44 sigma, i.e. 7.3e-3 at sigma = 3e-3, while
    `sep` at the doc's (30, 0) is 1.10e-2 -- a usable tolerance window only
    3.7e-3 wide, and the measured margin at tol = 1e-2 is 1.10x.  The
    pooled least-squares statistic

        D_h = sum over the 8 channels and the band of |S_h - S_pred|^2

    uses all 392 numbers instead of the largest one.  Its detection
    significance for the WEAKEST wrong hypothesis, z = sqrt(D_min)/(2 sigma):

      angle          z @ 3e-3   z @ 5e-3   z @ 1e-2   z, single frequency
      (30, 0) doc       8.4        5.1        2.5         0.6 (worst f)
      (45,22.5)        10.8        6.5        3.2         1.1
      (60, 0)          13.1        7.9        3.9         1.0
      (60,22.5)        14.4        8.6        4.3         1.4

    So: pooled chi2 over the band is robust (z = 8..14 at the placeholder
    sigma = 3e-3, still 2.5..4.3 even at sigma = 1e-2), the max statistic
    has essentially no margin, and a single-frequency test is hopeless at
    every angle.  channel_dictionary_acceptance therefore takes
    statistic="chi2" (recommended) or "max" (doc-literal, kept for
    comparison and still exercised in --selftest).

    MEASURED success rate at recovering a KNOWN hypothesis under additive
    complex Gaussian noise, 20 seeded trials per cell (--selftest prints
    this table every run; "oracle" = the max statistic handed the midpoint
    of the usable tolerance window, i.e. what a perfectly informed operator
    could do; "window CLOSED" = 2.44 sigma >= sep, so NO tolerance works):

      angle       sigma    chi2    max(doc tol 1e-2)   max(oracle)  window
      (30, 0)     3e-3     100 %        100 %              90 %     open
      (30, 0)     5e-3      35 %          0 %               0 %     CLOSED
      (30, 0)     1e-2       0 %          0 %               0 %     CLOSED
      (45,22.5)   3e-3     100 %        100 %             100 %     open
      (45,22.5)   5e-3     100 %          0 %              55 %     open
      (45,22.5)   1e-2       0 %          0 %               0 %     CLOSED

    NO test ever accepted a WRONG hypothesis in any cell -- every failure
    is a refusal, so the doc's fail-safe design survives intact under both
    statistics.  The decisive cell is (45, 22.5) at sigma = 5e-3: chi2
    decides correctly every time while the doc's fixed tolerance decides
    never.  At sigma = 3e-3 the doc's tol happens to sit inside the
    3.7e-3-wide window and the max test works -- that is luck, not
    robustness, and it evaporates as soon as the measured closure sigma
    moves.  This is the evidence for amending doc par. 7.

    ACCEPTANCE ANGLE -- a recommendation with the numbers, not a decree:
    (60, 22.5) has the best separation (z = 14.4) but theta = 60 is the
    most expensive FD solve and the one closest to the grazing Rayleigh /
    shell-convergence limit (doc par. 2); (45, 22.5) at z = 10.8 keeps a
    2x margin over z_min = 5 even if the measured closure sigma comes out
    at 5e-3, and the table above shows it deciding 100 % of trials there
    where the doc's (30, 0) manages only 35 %.  (45, 22.5) is the
    pragmatic choice.  The doc's own (30, 0) is adequate at z = 8.4 but
    leaves no headroom if sigma exceeds ~5e-3.  Whatever angle is chosen,
    the closure sigma must be measured FIRST -- at sigma = 1e-2 no angle in
    the campaign set can decide the cross signs at all, and the remedy is
    then deembed.check_mirror_plane_crosspol plus extending the hypothesis
    family, not a looser tolerance.

(d) If the acceptance still refuses, deembed.check_mirror_plane_crosspol on
    the phi in {0,45} structure runs constrains the label ORIENTATION with
    no reference model at all.  An error common to EVERY member of the
    hypothesis family is invisible to this test by construction -- which is
    exactly what happened on the real campaign, see (e).

(e) HYPOTHESIS FAMILY: use the EXTENDED 16, not the shipped 8.  The real
    (theta=60, phi=22.5) acceptance run REFUSED 0-of-8, and the diagnosis
    was the HANDOFF.md caveat come true.  All 8 shipped hypotheses are PORT
    GAUGES, S -> D S D with D = diag(s_1, s_2), so they multiply S[a,b] by
    s_a s_b and can NEVER change a CO-POLAR diagonal entry.  The measured
    defect was on one: S11[TM,TM] disagreed with the model by |d| = 2|S|
    -- a sign, not a phase or a scale.  Re-measured here independently from
    cst_runs/struct_th60_ph22p5 + empty_th60 (de-embedded with
    deembed.deembed_spectra, empty co-pol SZmin(a),Zmax(a) per receive mode
    a), against the reference-T forward model at direction -1, with the
    campaign's measured sigma = 2.6333e-3 and n_obs = 392:

      channel      max|d|      max|meas|      channel      max|d|
      S11[TE,TE]  1.645e-03   3.957e-01      S21[TE,TE]  2.490e-03
      S11[TE,TM]  1.070e-02   5.245e-03      S21[TE,TM]  1.009e-02
      S11[TM,TE]  9.364e-03   5.258e-03      S21[TM,TE]  9.909e-03
      S11[TM,TM]  3.231e-01   1.629e-01      S21[TM,TM]  1.294e-03

    Every channel agrees at the ~1e-2 level except S11[TM,TM], where
    |d| = 2|S|.  Negating the TM RECEIVE ROW of the S11 block only:
    reduced chi2 658.93 -> 2.49 (max|d| 3.231e-01 -> 1.101e-02).  Controls:
    negating the S21 TM row instead gives 71585, negating both gives 70929.
    Enumerating the extended 16 at (60, 22.5): winner (swap=False,
    s11_cross=-1, s21_cross=-1, r11_tm=-1) at reduced chi2 1.595 with
    runner-up z = 5.77 -> ACCEPTED at z_min = 5, while the 8-member family
    reaches only chi2 658.58 at z = 0.91 -> REFUSED.  At theta = 0 the same
    winner family gives chi2 1.134 but all eight r11_tm=-1 members tie
    (z = 0.15): normal incidence is structurally degenerate (C4 forces
    S_TE = S_TM and the cross-pol vanishes), which independently confirms
    why par. 7 places the acceptance at an OBLIQUE angle.

    The r11_tm sign is a CONVENTION CONSEQUENCE, not a free parameter: the
    par.-2 basis evaluates e_TM separately for k_hat_r (transversally
    opposite; sparams_oblique.py's docstring derives e_TM^(r) = -x_hat at
    theta -> 0, and forward.py's selftest already encodes
    S11[TM,TM] = -stored S11 and passes at 1e-10 against run_demo.py),
    whereas CST's Floquet port S11 refers both directions to the same
    transverse mode pattern.  It is carried as a hypothesis dimension ONLY
    so the acceptance stays fail-safe -- able to refuse rather than to
    assume.  Note the margin at (60,22.5) is z = 5.77, only 1.15x above
    z_min: it is still the CROSS-sign dimension that is marginal, exactly
    as section (b)/(c) predicted, and the runner-up differs from the winner
    only in s21_cross.

REAL-DATA ENTRY POINT
---------------------
validate_from_deembedded() takes the de-embedded per-angle blocks on the
49-point tmat frequency grid exactly as deembed.deembed_blocks +
deembed.interp_to_grid produce them, applies the accepted label hypothesis,
and runs checks 1-5.  Array shapes are specified in its docstring and the
path is exercised in --selftest with synthetic arrays pushed through that
same function (so the plumbing is proven, not declared).

CLI
---
    python validate_against_reference.py --selftest
    python validate_against_reference.py --selftest --freqs 8,24,32,48
    python validate_against_reference.py --selftest --sigma 3e-3
    python validate_against_reference.py --selftest --tol-bright 0.05
Options: --fit-angles, --holdout-angles, --tol-heldout, --tag, --sigma-obs,
--starts, --seed, --no-figures, --no-channel-dict.  Frequencies whose
requested angles are not cached are skipped gracefully (fm.have) and listed
in the summary.  The selftest fits the BRIGHT-10 basis (the par.-8.2 gate is
about bright entries and 4 angles carry 44 real observations for its 20 real
parameters); every function takes the basis as an argument, so a caller can
pass parametrize.build_c4v_reciprocity_basis's full 68 instead -- but note
that 4 angles cannot support 136 real parameters with method 'lm' and a
tikhonov > 0 / 'trf' fit is then mandatory.
"""
import argparse
import os
import sys
import time


import numpy as np
from scipy.stats import spearmanr

from tmatrix.plotting import plt                    # noqa: E402

from tmatrix.numerics import maxabs

from tmatrix.retrieval import parametrize as par # noqa: E402
from tmatrix.retrieval import fit as fitmod # noqa: E402
from tmatrix.retrieval import observability as obs # noqa: E402
from tmatrix.retrieval import deembed # noqa: E402
from tmatrix.retrieval.fit import DIRECTION_DEFAULT, resolve_angles, entry_label  # noqa: E402

from tmatrix.paths import RETRIEVAL_RESULTS

RESULTS_DIR = str(RETRIEVAL_RESULTS)

# ------------------------------------------------------------- tolerances
# par. 8.1: held-out complex specular S
TOL_HELDOUT = 1e-2
# par. 8.2: bright-entry relative error.  The doc says "5-10 %, tighten
# after the synthetic study"; a parallel study is recalibrating it, so this
# is ONE named constant -- retune it here and every report follows.
TOL_BRIGHT = 0.10
# par. 8.3
TOL_PASSIVITY = 1e-3
TOL_RECIPROCITY = 1e-3
# par. 7, max statistic (doc-literal)
TOL_CHANNEL_DICT = 1e-2
# par. 7, chi2 statistic (recommended; see module docstring section (c)).
# z_min: required detection significance of the runner-up hypothesis,
# z = sqrt(D_second - D_best) / (2 sigma).  5 sigma is the usual "decide,
# do not guess" bar and leaves the measured z = 8.4..14.4 (at the
# placeholder sigma) a 1.7-2.9x margin.
Z_MIN_CHANNEL_DICT = 5.0
# Largest reduced chi2 of the WINNING hypothesis still accepted,
# chi2_red = D_best / (n_obs sigma^2), expectation 1 when sigma is right.
# 4.0 = residual rms <= 2 sigma; above that the de-embedding or sigma is
# wrong and the label verdict should not be trusted even if the margin is
# large.  Only an UPPER bound is gated: chi2_red << 1 merely means sigma
# was overestimated, which does not endanger the label decision.
CHI2_MAX_CHANNEL_DICT = 4.0
# Per-complex-observable noise scale assumed by the chi2 test.  PLACEHOLDER
# 3e-3 until the par.-7 normal-incidence complex closure measures it; the
# report always prints the z that would follow at 2x and 3x this value.
SIGMA_CHANNEL_DICT = 3e-3

# par. 8.2 second half: what counts as "discrepancy pattern consistent with
# the observability map".  Spearman rank correlation between the per-entry
# error and a per-entry ill-determinedness score.  With the 25 bright
# entries the two-sided critical values are rho = 0.40 (p = 0.05) and
# rho = 0.51 (p = 0.01), so 0.5 is "significant at ~1 %" -- strong enough
# to exclude chance, loose enough to admit that observability is only PART
# of the story (the bright-span model error from sub-threshold entries,
# measured at ~1e-3 in S, is a second, independent error source that the
# heatmap does not describe).  A near-unity correlation would be the wrong
# expectation and demanding it would be dishonest.
CONSISTENCY_RHO = 0.5

# Noise floor used for the observability damping; PLACEHOLDER until the
# par.-7 normal-incidence complex closure measures the real one.
SIGMA_OBS = obs.SIGMA_PLACEHOLDER                       # 3e-3

# par. 8.1 default split -- see the module docstring for the full rationale
FIT_ANGLES_DEFAULT = [0, 5, 10, 11]        # (0,0) (30,22.5) (60,0) (60,22.5)
HOLDOUT_ANGLES_DEFAULT = [a for a in fitmod.ANGLE_SETS["campaign"]
                          if a not in FIT_ANGLES_DEFAULT]

GATES = []          # (name, passed, is_machinery_gate, detail)


def gate(name, ok, machinery, detail=""):
    """Record + print one gate (synthetic_test.py idiom).  machinery=True
    means a failure is a DEFECT and drives the exit code; machinery=False
    means the number is a measurement of the data's information content."""
    tag = "PASS" if ok else ("FAIL" if machinery
                             else "measured FAIL -- expected")
    print("[%s] %s%s" % (tag, name, ("  --  " + detail) if detail else ""),
          flush=True)
    GATES.append((name, bool(ok), bool(machinery), detail))
    return bool(ok)



def _angle_name(fm, ia):
    return "(%g,%g)" % (fm.theta_deg[ia], fm.phi_deg[ia])


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# measured-S container
# ===========================================================================

def rows_for_angles(S_freq, angle_indices):
    """Extract the requested angle rows from one frequency's measured S.

    S_freq is either
      * a dict {angle-table index: (2, 2, 2) complex}, or
      * an array (n_angles_table, 2, 2, 2) whose ROW INDEX IS THE ROW INDEX
        OF THE NORMATIVE 17-ANGLE TABLE (precompute_C.ANGLES_DEG); rows for
        angles that were never measured may be NaN and are never touched
        unless requested.
    Returns (len(angle_indices), 2, 2, 2) complex in the requested order.
    """
    angle_indices = [int(a) for a in angle_indices]
    if isinstance(S_freq, dict):
        return np.array([S_freq[a] for a in angle_indices], dtype=complex)
    S = np.asarray(S_freq, dtype=complex)
    if S.ndim != 4 or S.shape[1:] != (2, 2, 2):
        raise ValueError("S_freq must have shape (n_angles_table, 2, 2, 2) "
                         "or be a dict {angle index: (2,2,2)}; got %s"
                         % (S.shape,))
    bad = [a for a in angle_indices if a >= S.shape[0]]
    if bad:
        raise IndexError("angle indices %s exceed the %d rows supplied"
                         % (bad, S.shape[0]))
    return S[angle_indices]


def synth_S_by_freq(fm, T_stack, ifreq_list, angle_indices=None,
                    direction=DIRECTION_DEFAULT, sigma=0.0, seed=0):
    """Synthetic measured-S container: {ifreq: (17, 2, 2, 2)} filled at
    angle_indices (NaN elsewhere), from the per-frequency T in T_stack.

    Optional complex Gaussian noise with E|n|^2 = sigma^2 per COMPLEX
    observable (the synthetic_test.py convention: sigma/sqrt(2) per real
    component), i.i.d. over angle/block/polarization/frequency.
    """
    if angle_indices is None:
        angle_indices = fitmod.ANGLE_SETS["campaign"]
    angle_indices = [int(a) for a in angle_indices]
    out = {}
    for ifreq in ifreq_list:
        ifreq = int(ifreq)
        S = np.full((fm.n_angles, 2, 2, 2), np.nan, dtype=complex)
        S[angle_indices] = fm.predict(T_stack[ifreq], ifreq, angle_indices,
                                      direction)
        if sigma > 0.0:
            rng = np.random.default_rng([int(seed), ifreq])
            shape = (len(angle_indices), 2, 2, 2)
            S[angle_indices] += (rng.standard_normal(shape)
                                 + 1j * rng.standard_normal(shape)) \
                * (float(sigma) / np.sqrt(2.0))
        out[ifreq] = S
    return out


# ===========================================================================
# 1.  Held-out-angle acceptance (par. 8.1)
# ===========================================================================

def heldout_acceptance(fm, ifreq_list, B, S_meas_by_freq,
                       fit_angles=None, holdout_angles=None,
                       direction=DIRECTION_DEFAULT, weights=None,
                       tikhonov=0.0, tol_heldout=TOL_HELDOUT,
                       t0_by_freq=None, n_starts=3, seed=20260806,
                       engines=None, fig_path=None, verbose=True,
                       **lsq_kwargs):
    """par. 8.1: fit on `fit_angles`, PREDICT `holdout_angles`, gate the
    prediction error.

    Parameters
    ----------
    fm : forward.ForwardModel
    ifreq_list : sequence of int          frequency indices (tmat grid)
    B : (nb, n, n)                        fit basis (bright-10 by default)
    S_meas_by_freq : dict {ifreq: (17, 2, 2, 2) or {ia: (2,2,2)}}
        Measured/synthetic Jones blocks; see rows_for_angles for the layout.
        Must contain BOTH the fit and the hold-out angles.
    fit_angles, holdout_angles : ANGLE_SETS name, comma string, or index
        list (None -> FIT_ANGLES_DEFAULT / HOLDOUT_ANGLES_DEFAULT).
    weights, tikhonov : fit.fit_frequency semantics (w_i = 1/sigma_i^2).
    t0_by_freq : dict {ifreq: t0} or None  seed override (None -> Born 0).
    n_starts : int                         multistart count (1 = Born only).
    engines : dict {ifreq: AnalyticJacobian} or None
        Reused across calls; missing entries are built here and returned in
        the result under 'engines' (ONE engine per (basis, angle set,
        frequency) -- rebuilding is the dominant cost of repeated fits).

    Frequencies whose fit or hold-out angles are not cached (fm.have) are
    SKIPPED and listed under 'skipped'.

    Returns dict with (per-frequency dicts keyed by ifreq unless noted):
      fit_angles, holdout_angles, ifreqs (used, sorted), skipped,
      T0 {ifreq: (n,n)}, T0_stack (nf_used, n, n), t_hat {ifreq: (2nb,)},
      fit_results {ifreq: fit.fit_frequency dict}, engines,
      resid_fit, resid_holdout          {ifreq: max complex |dS|}
      resid_fit_per_angle  (nf_used, n_fit), resid_holdout_per_angle
                                        (nf_used, n_hold)
      worst_fit, worst_holdout (scalars), worst_holdout_ifreq,
      worst_holdout_angle, passed (worst_holdout <= tol_heldout),
      tol_heldout, fig_path
    """
    fit_angles = (FIT_ANGLES_DEFAULT if fit_angles is None
                  else resolve_angles(fit_angles))
    holdout_angles = (HOLDOUT_ANGLES_DEFAULT if holdout_angles is None
                      else resolve_angles(holdout_angles))
    overlap = sorted(set(fit_angles) & set(holdout_angles))
    if overlap:
        raise ValueError("fit and hold-out angle sets overlap at %s -- the "
                         "hold-out prediction would not be held out"
                         % overlap)
    B = np.asarray(B)
    engines = {} if engines is None else dict(engines)
    out = dict(fit_angles=list(fit_angles),
               holdout_angles=list(holdout_angles),
               tol_heldout=float(tol_heldout), direction=int(direction))
    T0, t_hat, fits = {}, {}, {}
    r_fit, r_hold = {}, {}
    rf_ang, rh_ang = [], []
    used, skipped = [], []

    for ifreq in ifreq_list:
        ifreq = int(ifreq)
        miss = [a for a in list(fit_angles) + list(holdout_angles)
                if not fm.have[ifreq, a]]
        if miss:
            skipped.append((ifreq, "angles %s not cached" % miss))
            if verbose:
                print("  [skip] ifreq %2d (lam %6.2f um): angles %s not "
                      "cached" % (ifreq, fm.lam_um[ifreq], miss), flush=True)
            continue
        if ifreq not in S_meas_by_freq:
            skipped.append((ifreq, "no measured S supplied"))
            continue
        S_fit = rows_for_angles(S_meas_by_freq[ifreq], fit_angles)
        S_hold = rows_for_angles(S_meas_by_freq[ifreq], holdout_angles)
        if not np.isfinite(S_fit).all() or not np.isfinite(S_hold).all():
            skipped.append((ifreq, "measured S has non-finite entries"))
            continue

        if ifreq not in engines:
            engines[ifreq] = fitmod.AnalyticJacobian(fm, ifreq, B,
                                                     fit_angles, direction)
        eng = engines[ifreq]
        kw = dict(weights=weights, direction=direction, tikhonov=tikhonov,
                  engine=eng)
        kw.update(lsq_kwargs)
        t0 = None if t0_by_freq is None else t0_by_freq.get(ifreq)
        if n_starts and n_starts > 1 and t0 is None:
            r = fitmod.fit_frequency_multistart(
                fm, ifreq, B, S_fit, fit_angles, n_starts=n_starts,
                seed=seed, **kw)
        else:
            r = fitmod.fit_frequency(fm, ifreq, B, S_fit, fit_angles,
                                     t0=t0, **kw)
        S_hold_pred = fm.predict(r["T0_hat"], ifreq, holdout_angles,
                                 direction)
        d_fit = np.abs(r["dS"])
        d_hold = np.abs(S_hold_pred - S_hold)

        T0[ifreq] = r["T0_hat"]
        t_hat[ifreq] = r["t_hat"]
        fits[ifreq] = r
        r_fit[ifreq] = float(d_fit.max())
        r_hold[ifreq] = float(d_hold.max())
        rf_ang.append(d_fit.reshape(len(fit_angles), -1).max(axis=1))
        rh_ang.append(d_hold.reshape(len(holdout_angles), -1).max(axis=1))
        used.append(ifreq)
        if verbose:
            print("  ifreq %2d (lam %6.2f um): fit-angle resid %.3e | "
                  "HELD-OUT resid %.3e  [obj %.2e, nfev %4d, %.2f s]"
                  % (ifreq, fm.lam_um[ifreq], r_fit[ifreq], r_hold[ifreq],
                     r["objective"], r["nfev"], r["wall_s"]), flush=True)

    out.update(ifreqs=used, skipped=skipped, T0=T0, t_hat=t_hat,
               fit_results=fits, engines=engines,
               resid_fit=r_fit, resid_holdout=r_hold,
               resid_fit_per_angle=np.array(rf_ang) if rf_ang
               else np.zeros((0, len(fit_angles))),
               resid_holdout_per_angle=np.array(rh_ang) if rh_ang
               else np.zeros((0, len(holdout_angles))))
    out["T0_stack"] = (np.array([T0[i] for i in used]) if used
                       else np.zeros((0, fm.modes.n, fm.modes.n),
                                     dtype=complex))
    out["worst_fit"] = max(r_fit.values()) if r_fit else np.nan
    out["worst_holdout"] = max(r_hold.values()) if r_hold else np.nan
    if used:
        wi = max(used, key=lambda i: r_hold[i])
        wa = holdout_angles[int(out["resid_holdout_per_angle"]
                                [used.index(wi)].argmax())]
        out["worst_holdout_ifreq"] = int(wi)
        out["worst_holdout_angle"] = int(wa)
    else:
        out["worst_holdout_ifreq"] = out["worst_holdout_angle"] = -1
    out["passed"] = bool(used) and out["worst_holdout"] <= tol_heldout

    if fig_path and used:
        _plot_heldout_spectrum(fm, out, fig_path)
        out["fig_path"] = fig_path
    else:
        out["fig_path"] = None

    if verbose and used:
        print("  worst over the band: fit-angle %.3e | HELD-OUT %.3e "
              "(ifreq %d, angle %s); gate %.1e"
              % (out["worst_fit"], out["worst_holdout"],
                 out["worst_holdout_ifreq"],
                 _angle_name(fm, out["worst_holdout_angle"]), tol_heldout))
        print("  per-HELD-OUT-angle worst over the band:")
        for j, ia in enumerate(holdout_angles):
            print("    angle %2d %-12s  max|dS| = %.3e"
                  % (ia, _angle_name(fm, ia),
                     out["resid_holdout_per_angle"][:, j].max()))
    return out


def _plot_heldout_spectrum(fm, res, path):
    used = res["ifreqs"]
    lam = np.array([fm.lam_um[i] for i in used])
    o = np.argsort(lam)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.semilogy(lam[o], np.array([res["resid_fit"][i] for i in used])[o],
                "o-", ms=4, label="fitted angles (%d)"
                % len(res["fit_angles"]))
    ax.semilogy(lam[o], np.array([res["resid_holdout"][i] for i in used])[o],
                "s-", ms=4, label="HELD-OUT angles (%d)"
                % len(res["holdout_angles"]))
    ax.axhline(res["tol_heldout"], color="crimson", ls="--",
               label="par. 8.1 gate %.0e" % res["tol_heldout"])
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("max complex |dS| over the angle set")
    ax.set_title("par. 8.1 held-out-angle acceptance\nfit %s -> predict %s"
                 % (res["fit_angles"], res["holdout_angles"]), fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _ensure_results_dir()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ===========================================================================
# 2.  Bright-entry comparison vs the reference tmat.h5 (par. 8.2)
# ===========================================================================

def bright_comparison(fm, T0_stack, ifreq_list, mask, B_proj=None,
                      tol_bright=TOL_BRIGHT, top=None, verbose=True):
    """par. 8.2: per-entry relative errors of the fitted T0 vs the
    reference tmat.h5, pointwise and peak-normalized, per entry class.

    T0_stack : (nf_used, n, n) frequency-aligned with ifreq_list.
    mask     : (n, n) bool bright-entry mask (parametrize.bright_mask).
    B_proj   : optional basis; when given, the comparison is ALSO made
               against the projected target P_B(T_ref) (the in-subspace
               truth: the part of the reference the fit could represent at
               all), reported alongside the literal par.-8.2 comparison
               against the RAW reference.
    tol_bright : gate tolerance actually used, PRINTED in the report.  It
               is module constant TOL_BRIGHT by default -- one place to
               retune when the parallel calibration study lands.

    Returns dict: cmp_raw / cmp_proj (fit.compare_to_reference outputs),
    per-class tables, gate numbers (max_rel_peaknorm, max_rel_at_peak,
    frac_within_tol), passed, tol_bright, table (formatted string).
    """
    T0_stack = np.asarray(T0_stack)
    ifreq_list = [int(i) for i in ifreq_list]
    T_raw = fm.data.T[ifreq_list]
    cmp_raw = fitmod.compare_to_reference(T0_stack, T_raw, mask,
                                          peak_ref_stack=fm.data.T)
    cmp_proj = None
    if B_proj is not None:
        T_proj = np.array([par.unpack(par.pack(fm.data.T[i], B_proj), B_proj)
                           for i in ifreq_list])
        cmp_proj = fitmod.compare_to_reference(T0_stack, T_proj, mask,
                                               peak_ref_stack=fm.data.T)

    entries = cmp_raw["entries"]
    cls = fitmod.classify_entries(fm.modes)
    classes = {}
    for name in ("dipole", "quad22", "even_m", "odd_m", "c4_violating"):
        selm = np.array([cls[name][i, j] for i, j in entries])
        if not selm.any():
            continue
        v = cmp_raw["band_max_peaknorm"][selm]
        classes[name] = dict(n=int(selm.sum()), max=float(v.max()),
                             median=float(np.median(v)),
                             n_over_tol=int((v > tol_bright).sum()))
    pk = cmp_raw["band_max_peaknorm"]
    frac_ok = float((pk <= tol_bright).mean())
    passed = bool(pk.max() <= tol_bright)
    table = fitmod.format_compare(cmp_raw, fm.modes, top=top)

    if verbose:
        print("  bright entries: %d (mask), frequencies: %s"
              % (len(entries), ifreq_list))
        print("  TOLERANCE IN USE: tol_bright = %.4g  (module constant "
              "TOL_BRIGHT; doc par. 8.2 says 5-10 %%)" % tol_bright)
        print("  vs RAW reference tmat.h5: band-max peak-normalized error "
              "max = %.3e, median = %.3e, %d/%d entries within tol"
              % (pk.max(), np.median(pk), int((pk <= tol_bright).sum()),
                 len(pk)))
        if cmp_proj is not None:
            pkp = cmp_proj["band_max_peaknorm"]
            print("  vs PROJECTED target P(T_ref) (in-subspace truth): "
                  "max = %.3e, median = %.3e" % (pkp.max(), np.median(pkp)))
        print("  per entry class (band-max peak-normalized):")
        for name, d in classes.items():
            print("    %-13s n=%2d  max=%.3e  median=%.3e  over tol: %d"
                  % (name, d["n"], d["max"], d["median"], d["n_over_tol"]))
        print("  per-entry table (sorted by band-max peak-normalized "
              "error):")
        print(table)
    return dict(cmp_raw=cmp_raw, cmp_proj=cmp_proj, classes=classes,
                entries=entries, tol_bright=float(tol_bright),
                max_rel_peaknorm=float(pk.max()),
                max_rel_at_peak=float(cmp_raw["max_rel_at_peak"]),
                frac_within_tol=frac_ok, passed=passed, table=table,
                ifreqs=ifreq_list)


# ===========================================================================
# 3 + 5.  Observability of the fitted angle set, and the discrepancy test
# ===========================================================================

def fit_observability(fm, ifreq_list, B, angle_indices, engines=None,
                      T_truth_stack=None, sigma=SIGMA_OBS,
                      direction=DIRECTION_DEFAULT, mask=None, tag="run",
                      make_figures=True, verbose=True):
    """par. 8.4: the observability map OF THE ACTUAL FITTED angle set and
    basis, published with the fit.

    Uses the EXACT analytic Jacobian (fit.AnalyticJacobian.jac_packed,
    reusing the fit's engines) instead of observability.jacobian's central
    differences -- same object, no FD truncation, and no extra forward
    evaluations.  Resolution via observability.svd_resolution with the
    PHYSICAL prior

        prior_scale = tau = ||pack(T_truth_in_span)|| / sqrt(n_params)

    (the synthetic_test.py convention; ~1e-2 physically), then
    observability.fold_complex and observability.entry_heatmap.
    T_truth_stack (default: the reference T stack) supplies the
    linearization point and tau: t_lin = pack(P_B(T_truth[ifreq]), B).  For
    real data with no truth available pass T_truth_stack = the fitted T0
    stack.

    Figures per frequency (make_figures): results/validate_{tag}_obs_
    heatmap_ifreq{NN}.png and ..._obs_spectrum_ifreq{NN}.png.

    Returns {ifreq: dict(J, s, lam, res, res_c, H, tau, n_above,
    heatmap_path, spectrum_path)}.
    """
    B = np.asarray(B)
    angle_indices = resolve_angles(angle_indices)
    engines = {} if engines is None else dict(engines)
    if T_truth_stack is None:
        T_truth_stack = fm.data.T
    npar = 2 * B.shape[0]
    ang_name = ",".join(str(a) for a in angle_indices)
    out = {}
    for ifreq in ifreq_list:
        ifreq = int(ifreq)
        miss = [a for a in angle_indices if not fm.have[ifreq, a]]
        if miss:
            if verbose:
                print("  [skip] observability ifreq %d: angles %s not "
                      "cached" % (ifreq, miss))
            continue
        if ifreq not in engines:
            engines[ifreq] = fitmod.AnalyticJacobian(fm, ifreq, B,
                                                     angle_indices,
                                                     direction)
        eng = engines[ifreq]
        T_t = np.asarray(T_truth_stack)[ifreq] \
            if np.asarray(T_truth_stack).ndim == 3 else T_truth_stack
        t_lin = par.pack(T_t, B)
        tau = float(np.linalg.norm(t_lin) / np.sqrt(npar))
        J = eng.jac_packed(t_lin)
        sv = obs.svd_resolution(J, sigma=sigma, prior_scale=max(tau, 1e-30))
        res_c = obs.fold_complex(sv["res"])
        H = obs.entry_heatmap(res_c, B)
        # complementary "unobservability" weight and the raw basis support:
        #   support[mu,nu] = sum_k |B_k[mu,nu]|^2   (how much of the entry
        #                    the basis can represent at all)
        #   G[mu,nu]       = sum_k (1 - res_k) |B_k[mu,nu]|^2 = support - H
        # G * tau^2 is the leading-order POSTERIOR VARIANCE of entry
        # (mu, nu) under the very Tikhonov model that defines res_k, so G
        # -- not 1/H -- is the quantity the per-entry error should track.
        # See discrepancy_vs_observability.
        G = obs.entry_heatmap(1.0 - res_c, B)
        support = obs.entry_heatmap(np.ones_like(res_c), B)
        d = dict(J=J, s=sv["s"], lam=sv["lam"], res=sv["res"], res_c=res_c,
                 H=H, G=G, support=support, tau=tau, n_above=sv["n_above"],
                 sigma=float(sigma), angle_indices=list(angle_indices),
                 ifreq=ifreq, heatmap_path=None, spectrum_path=None)
        if make_figures:
            _ensure_results_dir()
            hp = os.path.join(RESULTS_DIR,
                              "validate_%s_obs_heatmap_ifreq%02d.png"
                              % (tag, ifreq))
            sp = os.path.join(RESULTS_DIR,
                              "validate_%s_obs_spectrum_ifreq%02d.png"
                              % (tag, ifreq))
            obs.plot_heatmap(
                H, fm.modes, mask, hp,
                "Observability of the FITTED set: ifreq %d (lam %.2f um), "
                "%d complex basis dirs, angles [%s], sigma %.1e, physical "
                "prior tau %.2e" % (ifreq, fm.lam_um[ifreq], B.shape[0],
                                    ang_name, sigma, tau))
            obs.plot_spectrum(
                sv["s"], sv["lam"], sp,
                "SV spectrum of the FITTED Jacobian: ifreq %d, angles [%s]"
                % (ifreq, ang_name), sv["n_above"])
            d["heatmap_path"], d["spectrum_path"] = hp, sp
        out[ifreq] = d
        if verbose:
            print("  ifreq %2d (lam %6.2f um): tau = %.3e, lam_damp = %.3e, "
                  "SV %.3e..%.3e, %d/%d above lam, res_c in [%.3f, %.3f]"
                  % (ifreq, fm.lam_um[ifreq], tau, sv["lam"], sv["s"][0],
                     sv["s"][-1], sv["n_above"], len(sv["s"]),
                     res_c.min(), res_c.max()), flush=True)
    return out


def discrepancy_vs_observability(fm, ifreq_list, T0_stack, mask, obs_by_freq,
                                 T_tgt_stack=None, rho_min=CONSISTENCY_RHO,
                                 fig_path=None, support_tol=1e-12,
                                 verbose=True):
    """par. 8.2 second half: is the per-entry discrepancy pattern CONSISTENT
    with the observability map?

    TWO rank correlations are computed and BOTH are reported, because they
    answer the question differently and the difference is itself a result:

      rho_invH = Spearman(per-entry error, 1/H[mu,nu])
          the literal "error vs 1/observability" pairing.  H is the doc's
          heatmap sum_k res_k |B_k[mu,nu]|^2, bounded by the basis support
          (<= 1), so 1/H is a compressive score that saturates at 1 for
          every well-resolved entry and diverges for entries the basis
          cannot represent AT ALL (support 0), whose error has nothing to
          do with the estimator.  Reported as required, but it is NOT the
          quantity linear theory predicts.

      rho_G    = Spearman(per-entry error, G[mu,nu]),
          G = sum_k (1 - res_k) |B_k[mu,nu]|^2 = support - H.
          G * tau^2 IS the leading-order posterior variance of the entry
          under the same Tikhonov/prior model that defines res_k: it goes
          to 0 both for perfectly resolved entries AND for entries with no
          basis support (which the fit never moves), which is exactly the
          correct behaviour.  This is the PRIMARY consistency number and
          the one `passed` is based on.

    Both are computed for the ABSOLUTE error |dT|_e (the scale G predicts)
    and for the peak-normalized error |dT|_e / peak_e (the locked reporting
    convention), and over all bright entries as well as over the entries
    with basis support > support_tol.  Consistency means large errors sit
    on ill-determined entries: POSITIVE rho.

    What counts as consistent: rho_G(absolute error, all entries) >=
    CONSISTENCY_RHO (0.5) -- for 25 entries that is p ~ 0.01 two-sided (see
    the module docstring).  Reported, never enforced.

    H and G are pooled over the frequencies as the MEAN (the fit is
    per-frequency but the entry errors are band-maxima); per-frequency
    correlations are reported as well.

    Returns dict: rho (= rho_G, primary), rho_invH, rho_table (all four
    combinations x {all, supported}), pvalue, rho_per_freq, err_abs,
    err_peaknorm, H_pooled, G_pooled, support, entries, labels, passed,
    rho_min, fig_path.
    """
    ifreq_list = [int(i) for i in ifreq_list]
    if T_tgt_stack is None:
        T_tgt_stack = fm.data.T[ifreq_list]
    cmp_ = fitmod.compare_to_reference(np.asarray(T0_stack),
                                       np.asarray(T_tgt_stack), mask,
                                       peak_ref_stack=fm.data.T)
    entries = cmp_["entries"]
    ii, jj = entries[:, 0], entries[:, 1]
    err_pk = cmp_["band_max_peaknorm"]
    err_abs = err_pk * cmp_["peak"]

    have = [i for i in ifreq_list if i in obs_by_freq]
    if not have:
        raise ValueError("no observability results for the requested "
                         "frequencies")
    H_f = np.array([obs_by_freq[i]["H"][ii, jj] for i in have])
    G_f = np.array([obs_by_freq[i]["G"][ii, jj] for i in have])
    sup = obs_by_freq[have[0]]["support"][ii, jj]
    H_pool, G_pool = H_f.mean(axis=0), G_f.mean(axis=0)
    floor = max(H_pool[H_pool > 0].min() * 1e-3, 1e-300) \
        if (H_pool > 0).any() else 1e-300
    inv_obs = 1.0 / np.maximum(H_pool, floor)
    supported = sup > support_tol

    def _rho(y, x, m):
        ok = m & np.isfinite(y) & np.isfinite(x)
        if ok.sum() < 3:
            return np.nan, np.nan
        r = spearmanr(y[ok], x[ok])
        return float(r.statistic), float(r.pvalue)

    allm = np.ones(len(entries), dtype=bool)
    rho_table = {}
    for ename, e in (("abs", err_abs), ("peaknorm", err_pk)):
        for sname, x in (("1/H", inv_obs), ("G", G_pool)):
            for mname, m in (("all", allm), ("supported", supported)):
                rho_table["%s|%s|%s" % (ename, sname, mname)] = _rho(e, x, m)
    rho, pval = rho_table["abs|G|all"]
    rho_invH, p_invH = rho_table["abs|1/H|all"]

    rho_per_freq = {}
    for pos, i in enumerate(have):
        e_f = cmp_["rel_peaknorm"][ifreq_list.index(i)] * cmp_["peak"]
        rho_per_freq[i] = _rho(e_f, G_f[pos], allm)[0]

    passed = bool(np.isfinite(rho) and rho >= rho_min)
    labels = [entry_label(fm.modes, i, j) for i, j in entries]

    if fig_path:
        _plot_discrepancy_scatter(H_pool, G_pool, err_abs, labels, rho,
                                  rho_invH, rho_min, have, fig_path)

    if verbose:
        print("  entries: %d bright (%d with basis support > %.0e); "
              "H, G pooled over ifreqs %s (mean)"
              % (len(entries), int(supported.sum()), support_tol, have))
        print("  PRIMARY  Spearman rho(|dT|, G) = %+.3f  (p = %.2g)  "
              "[G = sum_k (1-res_k)|B_k|^2 ~ posterior variance / tau^2]; "
              "consistency threshold %.2f" % (rho, pval, rho_min))
        print("  LITERAL  Spearman rho(|dT|, 1/H) = %+.3f  (p = %.2g)  "
              "[H = the doc's heatmap]" % (rho_invH, p_invH))
        print("  full table (error metric | score | entry set -> rho, p):")
        for key in sorted(rho_table):
            r, p = rho_table[key]
            print("    %-28s %+0.3f  (p = %.2g)" % (key, r, p))
        print("  per-frequency rho(|dT|, G): %s"
              % ", ".join("ifreq %d: %+.3f" % (i, r)
                          for i, r in rho_per_freq.items()))
        o = np.argsort(err_abs)[::-1]
        rank_G = np.argsort(np.argsort(G_pool))    # 0 = best determined
        print("  worst-5 entries by ABSOLUTE error (|dT|, peaknorm, H, G, "
              "G-rank of %d):" % len(entries))
        for e in o[:5]:
            print("    %-12s |dT| %.3e  pk %.3e  H %.3e  G %.3e  "
                  "G-rank %2d" % (labels[e], err_abs[e], err_pk[e],
                                  H_pool[e], G_pool[e], rank_G[e] + 1))
    return dict(rho=float(rho), pvalue=float(pval), rho_invH=float(rho_invH),
                pvalue_invH=float(p_invH), rho_table=rho_table,
                rho_per_freq=rho_per_freq, err=err_abs, err_abs=err_abs,
                err_peaknorm=err_pk, H_pooled=H_pool, G_pooled=G_pool,
                support=sup, inv_obs=inv_obs, entries=entries,
                labels=labels, passed=passed, rho_min=float(rho_min),
                fig_path=fig_path, ifreqs=have)


def _cluster_labels(x, y, labels, tol=0.04):
    """Group points that coincide on the log-log plane (the symmetry orbits
    make many entries numerically identical) so each cluster is annotated
    once, as 'label (xN)'.  Returns [(x, y, text), ...]."""
    lx, ly = np.log10(x), np.log10(y)
    used = np.zeros(len(x), dtype=bool)
    out = []
    for i in np.argsort(-y):
        if used[i]:
            continue
        near = (~used) & (np.abs(lx - lx[i]) < tol) & (np.abs(ly - ly[i])
                                                      < tol)
        used |= near
        k = int(near.sum())
        out.append((x[i], y[i],
                    labels[i] + ("" if k == 1 else "  (x%d)" % k)))
    return out


def _plot_discrepancy_scatter(H, G, err_abs, labels, rho_G, rho_invH,
                              rho_min, ifreqs, path, decades=8.0):
    """Two log-log panels: the doc's H (left) and the posterior-variance
    weight G (right), both against the absolute per-entry error.

    Entries with NO basis support have H = G = 0 (numerically ~1e-33) and
    would stretch the x-axis over 30+ decades; they are clipped to the left
    edge and drawn as open triangles, since their error is set by the truth
    and not by the estimator at all.
    """
    y = np.asarray(err_abs, dtype=float)
    ypos = y[y > 0]
    yfloor = (ypos.min() * 0.5) if ypos.size else 1e-30
    y = np.maximum(y, yfloor)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax, x_raw, xlab, ttl in (
            (axes[0], H, "observability H[mu,nu] = sum_k res_k "
             "|B_k[mu,nu]|^2", "literal: rho(|dT|, 1/H) = %+.3f" % rho_invH),
            (axes[1], G, "G[mu,nu] = sum_k (1 - res_k) |B_k[mu,nu]|^2  "
             "(~ posterior variance / tau^2)",
             "primary: rho(|dT|, G) = %+.3f  (consistent if >= %.2f)"
             % (rho_G, rho_min))):
        x = np.asarray(x_raw, dtype=float)
        xmax = x.max() if x.max() > 0 else 1.0
        xf = xmax * 10.0 ** (-decades)
        clipped = x < xf
        xp = np.maximum(x, xf)
        ax.loglog(xp[~clipped], y[~clipped], "o", ms=6, color="tab:blue")
        if clipped.any():
            ax.loglog(xp[clipped], y[clipped], "^", ms=8, mfc="none",
                      mec="crimson", mew=1.4,
                      label="no basis support (H = G = 0), clipped")
            ax.legend(loc="lower right", fontsize=7)
        for xi, yi, txt in _cluster_labels(xp, y, labels):
            ax.annotate(txt, (xi, yi), fontsize=6.5, xytext=(5, 3),
                        textcoords="offset points")
        ax.set_xlim(xf / 3.0, xmax * 3.0)
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel("band-max ABSOLUTE entry error |dT|")
        ax.set_title(ttl, fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("par. 8.2 consistency: discrepancy vs observability "
                 "(bright entries, ifreqs %s)" % ifreqs, fontsize=10)
    fig.tight_layout()
    _ensure_results_dir()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ===========================================================================
# 4.  Passivity and reciprocity (par. 8.3) -- CHECKS, not constraints
# ===========================================================================

def physical_checks(T0_stack, modes=None, perm=None, sign=None,
                    tol_passivity=TOL_PASSIVITY,
                    tol_reciprocity=TOL_RECIPROCITY, verbose=True):
    """par. 8.3: max SV(I + 2 T0) <= 1 + tol_passivity and
    max|Rec(T0) - T0| <= tol_reciprocity, reported as CHECKS.

    Neither is enforced anywhere in the fit: reciprocity holds by
    construction of the subspace basis (parametrize.build_c4v_reciprocity_
    basis intersects with the reciprocity-symmetric subspace and
    fit.fit_frequency asserts it at 1e-10), so a reciprocity failure here
    would be a MACHINERY defect; passivity is a genuine physical
    post-check of the fitted amplitudes.

    T0_stack : (nf, n, n).  modes (or perm/sign from
    parametrize.reciprocity_perm_sign) is needed for the reciprocity half;
    without it reciprocity is reported as None.

    Returns dict: passivity_max_sv (scalar), passivity_per_freq (nf,),
    passivity_margin (max SV - 1), passivity_ok, reciprocity_max,
    reciprocity_per_freq, reciprocity_ok, tolerances.
    """
    T0_stack = np.asarray(T0_stack)
    if T0_stack.ndim == 2:
        T0_stack = T0_stack[None]
    n = T0_stack.shape[-1]
    I = np.eye(n)
    sv = np.array([np.linalg.svd(I + 2.0 * T, compute_uv=False).max()
                   for T in T0_stack])
    pas_max = float(sv.max()) if sv.size else np.nan
    out = dict(passivity_per_freq=sv, passivity_max_sv=pas_max,
               passivity_margin=pas_max - 1.0,
               passivity_ok=bool(pas_max <= 1.0 + tol_passivity),
               tol_passivity=float(tol_passivity),
               tol_reciprocity=float(tol_reciprocity))
    if perm is None and modes is not None:
        perm, sign = par.reciprocity_perm_sign(modes)
    if perm is None:
        out.update(reciprocity_per_freq=None, reciprocity_max=None,
                   reciprocity_ok=None)
    else:
        rec = np.array([maxabs(par.apply_rec(T, perm, sign) - T)
                        for T in T0_stack])
        out.update(reciprocity_per_freq=rec,
                   reciprocity_max=float(rec.max()) if rec.size else np.nan,
                   reciprocity_ok=bool(rec.max() <= tol_reciprocity))
    if verbose:
        print("  passivity  : max SV(I + 2 T0) = %.6f  (gate 1 + %.0e = "
              "%.6f) -> %s" % (pas_max, tol_passivity, 1.0 + tol_passivity,
                               "OK" if out["passivity_ok"] else "VIOLATED"))
        if out["reciprocity_max"] is not None:
            print("  reciprocity: max|Rec(T0) - T0| = %.3e  (gate %.0e) "
                  "-> %s  [exact by subspace construction]"
                  % (out["reciprocity_max"], tol_reciprocity,
                     "OK" if out["reciprocity_ok"] else "VIOLATED"))
    return out


# ===========================================================================
# 6.  Channel-dictionary acceptance (par. 7) -- BEFORE the CST campaign
# ===========================================================================

def reference_blocks(fm, band, theta_deg, phi_deg,
                     direction=DIRECTION_DEFAULT, T_ref=None):
    """(S11_pred, S21_pred) of shape (2, 2, len(band)) from the forward
    model at one angle with the RAW reference T (par.-2 Jones order).

    Split out so Monte-Carlo studies can compute it once and hand it to
    channel_dictionary_acceptance via `pred=` instead of re-running 49
    forward evaluations per trial.
    """
    ia = fm.angle_index(float(theta_deg), float(phi_deg))
    miss = [i for i in band if not fm.have[i, ia]]
    if miss:
        raise KeyError("C not cached for angle %d at ifreqs %s" % (ia, miss))
    if T_ref is None:
        T_ref = fm.data.T
    S11 = np.empty((2, 2, len(band)), dtype=complex)
    S21 = np.empty((2, 2, len(band)), dtype=complex)
    for p, i in enumerate(band):
        S = fm.predict(T_ref[i], i, [ia], direction)[0]
        S11[..., p], S21[..., p] = S[0], S[1]
    return S11, S21


def channel_dictionary_acceptance(fm, ifreq_or_band, S11_meas_cst_order,
                                  S21_meas_cst_order, theta_deg=30.0,
                                  phi_deg=0.0, tol=TOL_CHANNEL_DICT,
                                  direction=DIRECTION_DEFAULT, T_ref=None,
                                  band_mode="pooled", statistic="chi2",
                                  sigma=SIGMA_CHANNEL_DICT,
                                  z_min=Z_MIN_CHANNEL_DICT,
                                  chi2_max=CHI2_MAX_CHANNEL_DICT, pred=None,
                                  family="extended", verbose=True):
    """par. 7 acceptance, required BEFORE the campaign proceeds: the 8
    de-embedded complex numbers at one oblique angle must identify EXACTLY
    ONE TE/TM label hypothesis against the forward model evaluated with the
    REFERENCE T.

    BOTH statistics are always computed and reported; `statistic` only
    decides which one sets `passed`/`hypothesis`.

      "chi2" (DEFAULT, recommended -- module docstring section (c))
          D_h = sum over the 8 channels and the whole band of
          |S_h - S_pred|^2, i.e. all 8 x nband complex numbers rather than
          the largest one.  It is a likelihood-ratio test, not a fixed
          tolerance:
            winner        = argmin_h D_h
            chi2_reduced  = D_best / (n_obs sigma^2)      (expectation 1)
            z_margin      = sqrt(D_second - D_best) / (2 sigma)
          ACCEPT iff z_margin >= z_min AND chi2_reduced <= chi2_max.
          Only an UPPER bound on chi2_reduced is gated: a value far below 1
          means sigma was overestimated, which does not endanger the label
          decision, while a value far above 1 means the de-embedding or
          sigma is wrong and no label verdict should be trusted.
          `tol` is unused in this mode.

      "max" (doc-literal)
          max over the 8 channels and the band of |S_h - S_pred|; accept
          iff EXACTLY ONE hypothesis is within `tol`.  This is the
          statistic deembed.select_hypothesis implements, and a
          DeembedError from it (0 or >1 winners) is CAUGHT and surfaced as
          passed=False with the full residual table -- the doc's fail-safe
          design (refuse rather than mis-pick).  Measured to have almost no
          margin at the placeholder sigma (module docstring (c)); kept for
          comparison, not recommended for the campaign.

    Parameters
    ----------
    ifreq_or_band : int or sequence of int
        Frequency index/indices on the tmat 49-point grid.
    S11_meas_cst_order, S21_meas_cst_order : complex arrays
        De-embedded Jones blocks in CST MODE ORDER (the deembed.
        deembed_blocks layout), shape (2, 2) for a single frequency or
        (2, 2, nband) with the last axis aligned with ifreq_or_band.
        Index [a, b] = [receive mode a+1, incident mode b+1].
    theta_deg, phi_deg : the scan angle; must be a row of the normative
        17-angle table (fm.angle_index raises otherwise).
    sigma : per-COMPLEX-observable noise scale for the chi2 test.  DEFAULT
        SIGMA_CHANNEL_DICT = 3e-3 is a PLACEHOLDER until the par.-7
        normal-incidence complex closure measures it; the report always
        also prints z_margin at 2x and 3x sigma so the campaign can see how
        the verdict degrades if the closure comes out worse.
    band_mode : "pooled" (DEFAULT) pools the frequency axis; "per_freq"
        requires a unique, agreeing winner at EVERY frequency -- strictly
        harder and measured to fail on this structure with either statistic
        (module docstring (a)/(c)); it exists to demonstrate that.
    family : "extended" (DEFAULT -- the 16 hypotheses including the S11
        TM-row sign r11_tm) or "base" (the 8 port gauges, i.e. the shipped
        deembed default, kept for comparison).  The real campaign needed
        the extended family: no port gauge can flip a CO-POLAR diagonal,
        so the 8 could not express the measured S11[TM,TM] sign and the
        acceptance refused 0-of-8.  See the module docstring section (e).
    pred : optional (S11_pred, S21_pred) from reference_blocks(), to skip
        the forward evaluations (Monte-Carlo studies).

    Returns dict: passed, hypothesis, residual (winner's statistic value),
    statistic, sigma, z_min, chi2_max, n_obs,
    table [(hyp, max residual)], D_table [(hyp, D)], z_table [(hyp, z)],
    best_residual, second_residual, margin (= second_residual/tol),
    D_best, D_second, chi2_reduced, z_margin, z_margin_2sigma,
    z_margin_3sigma, n_winners, winners, error, angle_index, band,
    band_mode, tol, S11_pred, S21_pred, per_freq (when band_mode="per_freq").
    """
    band = ([int(ifreq_or_band)] if np.isscalar(ifreq_or_band)
            else [int(i) for i in ifreq_or_band])
    ia = fm.angle_index(float(theta_deg), float(phi_deg))
    if statistic not in ("chi2", "max"):
        raise ValueError("statistic must be 'chi2' or 'max'")
    if band_mode not in ("pooled", "per_freq"):
        raise ValueError("band_mode must be 'pooled' or 'per_freq'")
    if family not in ("extended", "base"):
        raise ValueError("family must be 'extended' or 'base'")
    ext = (family == "extended")
    if pred is None:
        S11_pred, S21_pred = reference_blocks(fm, band, theta_deg, phi_deg,
                                              direction, T_ref)
    else:
        S11_pred, S21_pred = (np.asarray(pred[0], dtype=complex),
                              np.asarray(pred[1], dtype=complex))

    S11_m = np.asarray(S11_meas_cst_order, dtype=complex)
    S21_m = np.asarray(S21_meas_cst_order, dtype=complex)
    if S11_m.ndim == 2:
        S11_m = S11_m[..., None]
    if S21_m.ndim == 2:
        S21_m = S21_m[..., None]
    if S11_m.shape != S11_pred.shape or S21_m.shape != S21_pred.shape:
        raise ValueError("measured blocks %s / %s do not match the band "
                         "shape %s (2, 2, %d)"
                         % (S11_m.shape, S21_m.shape, S11_pred.shape,
                            len(band)))

    hyps = deembed.label_hypotheses(extended=ext)
    table, D_table, D_per_freq = [], [], []
    for hyp in hyps:
        h11, h21 = deembed.apply_hypothesis(S11_m, S21_m, hyp)
        d11, d21 = np.abs(h11 - S11_pred), np.abs(h21 - S21_pred)
        table.append((hyp, max(float(d11.max()), float(d21.max()))))
        D_table.append((hyp, float((d11 ** 2).sum() + (d21 ** 2).sum())))
        D_per_freq.append((d11 ** 2).sum(axis=(0, 1))
                          + (d21 ** 2).sum(axis=(0, 1)))
    resids = np.array([r for _, r in table])
    D = np.array([d for _, d in D_table])
    D_per_freq = np.array(D_per_freq)                 # (8, nband)
    n_obs = 8 * len(band)
    sigma = float(sigma)

    z_of = (lambda dd, s: float(np.sqrt(max(dd, 0.0)) / (2.0 * s)))
    o_max = np.argsort(resids)
    o_chi = np.argsort(D)
    D_best, D_second = float(D[o_chi[0]]), float(D[o_chi[1]])
    chi2_red = (D_best / (n_obs * sigma ** 2)) if sigma > 0 else np.inf
    z_margin = z_of(D_second - D_best, sigma)
    z_table = [(hyps[j], z_of(D[j] - D_best, sigma)) for j in range(len(hyps))]
    winners = [h for (h, r) in table if r <= tol]

    out = dict(angle_index=int(ia), theta_deg=float(theta_deg),
               phi_deg=float(phi_deg), band=band, tol=float(tol),
               band_mode=band_mode, statistic=statistic, sigma=sigma,
               family=family, n_hypotheses=len(hyps),
               z_min=float(z_min), chi2_max=float(chi2_max), n_obs=n_obs,
               table=table, D_table=D_table, z_table=z_table,
               n_winners=len(winners), winners=winners,
               S11_pred=S11_pred, S21_pred=S21_pred,
               best_residual=float(resids[o_max[0]]),
               second_residual=float(resids[o_max[1]]),
               margin=(float(resids[o_max[1]] / tol) if tol > 0 else np.inf),
               D_best=D_best, D_second=D_second, chi2_reduced=chi2_red,
               z_margin=z_margin,
               z_margin_2sigma=z_of(D_second - D_best, 2.0 * sigma),
               z_margin_3sigma=z_of(D_second - D_best, 3.0 * sigma),
               error=None, hypothesis=None, residual=None,
               direction=int(direction))

    # ---------------------------------------------------------- verdict
    if statistic == "chi2":
        chi2_ok = chi2_red <= chi2_max
        marg_ok = z_margin >= z_min
        out["passed"] = bool(chi2_ok and marg_ok)
        if out["passed"]:
            out["hypothesis"] = hyps[o_chi[0]]
            out["residual"] = D_best
        else:
            why = []
            if not marg_ok:
                why.append("runner-up margin z = %.2f < z_min = %.2f "
                           "(sigma = %.1e): the data cannot separate %s "
                           "from %s" % (z_margin, z_min, sigma,
                                        hyps[o_chi[0]], hyps[o_chi[1]]))
            if not chi2_ok:
                why.append("winner's reduced chi2 = %.2f > %.2f: residual "
                           "rms is %.1f sigma, so sigma or the de-embedding "
                           "is wrong and no label verdict is trustworthy"
                           % (chi2_red, chi2_max, np.sqrt(chi2_red)))
            out["error"] = ("chi2 acceptance REFUSED: " + "; ".join(why))
    else:
        try:
            hyp, r = deembed.select_hypothesis(S11_m, S21_m, S11_pred,
                                               S21_pred, tol, extended=ext)
            out.update(hypothesis=hyp, residual=float(r), passed=True)
        except deembed.DeembedError as e:
            out.update(passed=False, error=str(e))

    if band_mode == "per_freq":
        per = []
        for p, i in enumerate(band):
            if statistic == "chi2":
                dp = D_per_freq[:, p]
                op = np.argsort(dp)
                zp = z_of(dp[op[1]] - dp[op[0]], sigma)
                c2 = dp[op[0]] / (8 * sigma ** 2) if sigma > 0 else np.inf
                ok_p = zp >= z_min and c2 <= chi2_max
                per.append((i, hyps[op[0]] if ok_p else None,
                            float(dp[op[0]]) if ok_p else None,
                            None if ok_p else "z = %.2f, chi2red = %.2f"
                            % (zp, c2)))
            else:
                try:
                    h, r = deembed.select_hypothesis(
                        S11_m[..., p:p + 1], S21_m[..., p:p + 1],
                        S11_pred[..., p:p + 1], S21_pred[..., p:p + 1],
                        tol, extended=ext)
                    per.append((i, h, float(r), None))
                except deembed.DeembedError as e:
                    per.append((i, None, None, str(e).splitlines()[0]))
        out["per_freq"] = per
        uniq = {tuple(sorted(h.items())) for _, h, _, _ in per
                if h is not None}
        ok = (len(per) > 0 and all(h is not None for _, h, _, _ in per)
              and len(uniq) == 1)
        out["passed"] = bool(ok)
        if ok:
            out["hypothesis"] = per[0][1]
            out["residual"] = max(r for _, _, r, _ in per)
        else:
            out["hypothesis"] = None
            out["error"] = ("per_freq mode (%s): %d/%d frequencies gave a "
                            "unique winner, %d distinct winners"
                            % (statistic,
                               sum(1 for _, h, _, _ in per if h is not None),
                               len(per), len(uniq)))

    if verbose:
        print("  angle %d %s, band %s (%d freqs = %d complex observables), "
              "statistic = %s, band_mode = %s"
              % (ia, _angle_name(fm, ia),
                 band[:3] + (["..."] if len(band) > 3 else []), len(band),
                 n_obs, statistic, band_mode))
        if statistic == "chi2":
            print("    sigma = %.2e (PLACEHOLDER unless measured), "
                  "z_min = %.1f, chi2_max = %.1f"
                  % (sigma, z_min, chi2_max))
        else:
            print("    tol = %.2e" % tol)
        print("    family = %s (%d hypotheses)" % (family, len(hyps)))
        print("    hypothesis table (D = pooled sum |dS|^2; z = "
              "sqrt(D - D_best)/(2 sigma); r = pooled max |dS|):")
        for j in np.argsort(D):
            h = hyps[j]
            print("      swap=%-5s s11=%+d s21=%+d%s : D %.4e  z %6.2f  "
                  "r %.4e%s"
                  % (h["swap"], h["s11_cross"], h["s21_cross"],
                     (" r11tm=%+d" % h["r11_tm"]) if "r11_tm" in h else "",
                     D[j], z_table[j][1], resids[j],
                     "   <-- best" if j == o_chi[0] else
                     ("   <-- within tol" if resids[j] <= tol else "")))
        print("    chi2_reduced(best) = %.3f (expect ~1); z_margin = %.2f "
              "(at 2x sigma %.2f, at 3x sigma %.2f); max-statistic "
              "next-best = %.3e = %.2fx tol"
              % (chi2_red, z_margin, out["z_margin_2sigma"],
                 out["z_margin_3sigma"], out["second_residual"],
                 out["margin"]))
        if out["passed"]:
            print("    ACCEPTED: %s" % (out["hypothesis"],))
        else:
            print("    REFUSED (fail-safe): %s"
                  % (out["error"] or "").splitlines()[0])
    return out


# ===========================================================================
# 7.  Real-data entry point
# ===========================================================================

def blocks_to_S_by_freq(fm, S11_by_angle, S21_by_angle, meas_angles,
                        hypothesis=None):
    """De-embedded per-angle blocks -> the {ifreq: (17, 2, 2, 2)} container
    that heldout_acceptance consumes.

    S11_by_angle, S21_by_angle : complex arrays of shape
        (n_meas_angles, 2, 2, nf) with
          axis 0 = measured angle slot, aligned with meas_angles,
          axis 1 = receive index a, axis 2 = incident index b,
          axis 3 = frequency on the tmat 49-point grid (fm.nf).
        This is exactly deembed.deembed_blocks' (2, 2, nf) output per angle,
        stacked over angles after deembed.interp_to_grid has put it on the
        tmat grid.  If `hypothesis` is given the a/b indices are CST MODE
        numbers (0 -> Zmax mode 1) and the hypothesis is applied here; if
        it is None they must ALREADY be the par.-2 Jones order (0 = TE,
        1 = TM).
    meas_angles : sequence of row indices into the normative 17-angle table
        (precompute_C.ANGLES_DEG), len = n_meas_angles.

    Returns {ifreq: (fm.n_angles, 2, 2, 2)} with NaN in every angle row not
    supplied, block index 0 = S11, 1 = S21.
    """
    S11 = np.asarray(S11_by_angle, dtype=complex)
    S21 = np.asarray(S21_by_angle, dtype=complex)
    meas_angles = [int(a) for a in meas_angles]
    if S11.shape != S21.shape:
        raise ValueError("S11/S21 shapes differ: %s vs %s"
                         % (S11.shape, S21.shape))
    if S11.ndim != 4 or S11.shape[:3] != (len(meas_angles), 2, 2):
        raise ValueError("expected shape (n_meas_angles=%d, 2, 2, nf); got %s"
                         % (len(meas_angles), S11.shape))
    nf = S11.shape[3]
    if nf != fm.nf:
        raise ValueError("frequency axis has %d points, the tmat grid has "
                         "%d -- interpolate with deembed.interp_to_grid "
                         "first" % (nf, fm.nf))
    if hypothesis is not None:
        S11, S21 = zip(*[deembed.apply_hypothesis(S11[j], S21[j], hypothesis)
                         for j in range(len(meas_angles))])
        S11, S21 = np.array(S11), np.array(S21)
    out = {}
    for i in range(nf):
        S = np.full((fm.n_angles, 2, 2, 2), np.nan, dtype=complex)
        for j, ia in enumerate(meas_angles):
            S[ia, 0] = S11[j, :, :, i]
            S[ia, 1] = S21[j, :, :, i]
        out[i] = S
    return out


def validate_from_deembedded(fm, S11_by_angle, S21_by_angle, meas_angles,
                             hypothesis=None, ifreq_list=None, B=None,
                             mask=None, fit_angles=None, holdout_angles=None,
                             direction=DIRECTION_DEFAULT, weights=None,
                             tikhonov=0.0, tol_heldout=TOL_HELDOUT,
                             tol_bright=TOL_BRIGHT, sigma_obs=SIGMA_OBS,
                             n_starts=3, tag="real", make_figures=True,
                             verbose=True, **lsq_kwargs):
    """REAL-DATA ENTRY POINT: run acceptance checks 1-5 on de-embedded CST
    blocks.  No CST or license is needed to CALL this -- it only consumes
    arrays.

    Input shapes (precisely; see blocks_to_S_by_freq):
      S11_by_angle, S21_by_angle : complex (n_meas_angles, 2, 2, 49)
          [angle slot, receive index, incident index, tmat frequency index].
          The (2, 2, nf) per-angle slabs are exactly what
          deembed.deembed_blocks returns; put them on the 49-point grid with
          deembed.interp_to_grid before stacking.
      meas_angles : list of row indices into precompute_C.ANGLES_DEG,
          len = n_meas_angles, aligned with axis 0.
      hypothesis : the ACCEPTED deembed label hypothesis (dict with keys
          swap / s11_cross / s21_cross, from channel_dictionary_acceptance)
          -- applied here, so pass the blocks in raw CST mode order.  Pass
          None if they are already TE/TM.

    Other parameters mirror the individual checks; **lsq_kwargs go straight
    to scipy least_squares through fit.fit_frequency (xtol/ftol/gtol/
    max_nfev...).  B defaults to the bright-10 basis and mask to the 1e-3
    bright mask, both built from the reference file.  fit_angles must be a
    subset of meas_angles; holdout_angles defaults to meas_angles minus
    fit_angles.

    Returns dict: heldout, bright, observability, discrepancy, physical,
    S_meas_by_freq, fit_angles, holdout_angles, ifreqs, npz_path.
    """
    meas_angles = [int(a) for a in meas_angles]
    if B is None or mask is None:
        B68, _, B10, binfo = obs.build_bases(fm)
        B = B10 if B is None else B
        mask = binfo["mask"] if mask is None else mask
        B68_ref = B68
    else:
        B68_ref = None
    fit_angles = (FIT_ANGLES_DEFAULT if fit_angles is None
                  else resolve_angles(fit_angles))
    bad = [a for a in fit_angles if a not in meas_angles]
    if bad:
        raise ValueError("fit angles %s are not among the measured angles %s"
                         % (bad, meas_angles))
    if holdout_angles is None:
        holdout_angles = [a for a in meas_angles if a not in fit_angles]
    else:
        holdout_angles = resolve_angles(holdout_angles)
    if ifreq_list is None:
        ifreq_list = list(range(fm.nf))

    S_by_freq = blocks_to_S_by_freq(fm, S11_by_angle, S21_by_angle,
                                    meas_angles, hypothesis)

    if verbose:
        print("\n--- (1) par. 8.1 held-out-angle acceptance ---")
    hres = heldout_acceptance(
        fm, ifreq_list, B, S_by_freq, fit_angles, holdout_angles,
        direction=direction, weights=weights, tikhonov=tikhonov,
        tol_heldout=tol_heldout, n_starts=n_starts, verbose=verbose,
        fig_path=(os.path.join(RESULTS_DIR,
                               "validate_%s_heldout_spectrum.png" % tag)
                  if make_figures else None), **lsq_kwargs)
    used = hres["ifreqs"]
    if not used:
        return dict(heldout=hres, bright=None, observability=None,
                    discrepancy=None, physical=None,
                    S_meas_by_freq=S_by_freq, fit_angles=fit_angles,
                    holdout_angles=holdout_angles, ifreqs=[], npz_path=None)

    if verbose:
        print("\n--- (2) par. 8.2 bright-entry comparison vs the reference "
              "tmat.h5 ---")
    bres = bright_comparison(fm, hres["T0_stack"], used, mask,
                             B_proj=B68_ref, tol_bright=tol_bright,
                             verbose=verbose)

    if verbose:
        print("\n--- (5) par. 8.4 observability map of the FITTED angle "
              "set ---")
    ores = fit_observability(fm, used, B, fit_angles,
                             engines=hres["engines"], sigma=sigma_obs,
                             direction=direction, mask=mask, tag=tag,
                             make_figures=make_figures, verbose=verbose)

    if verbose:
        print("\n--- (3) par. 8.2 discrepancy-vs-observability consistency "
              "---")
    dres = discrepancy_vs_observability(
        fm, used, hres["T0_stack"], mask, ores, verbose=verbose,
        fig_path=(os.path.join(RESULTS_DIR,
                               "validate_%s_discrepancy_scatter.png" % tag)
                  if make_figures else None))

    if verbose:
        print("\n--- (4) par. 8.3 passivity and reciprocity checks ---")
    pres = physical_checks(hres["T0_stack"], modes=fm.modes, verbose=verbose)

    npz_path = save_results(fm, tag, hres, bres, ores, dres, pres)
    return dict(heldout=hres, bright=bres, observability=ores,
                discrepancy=dres, physical=pres, S_meas_by_freq=S_by_freq,
                fit_angles=fit_angles, holdout_angles=holdout_angles,
                ifreqs=used, npz_path=npz_path)


# ===========================================================================
# persistence
# ===========================================================================

def save_results(fm, tag, hres, bres, ores, dres, pres, extra=None):
    """results/validate_reference_{tag}.npz with every gate array."""
    _ensure_results_dir()
    used = hres["ifreqs"]
    d = dict(tag=tag, ifreqs=np.array(used),
             lam_um=np.array([fm.lam_um[i] for i in used]),
             fit_angles=np.array(hres["fit_angles"]),
             holdout_angles=np.array(hres["holdout_angles"]),
             direction=hres["direction"],
             tol_heldout=hres["tol_heldout"],
             T0_stack=hres["T0_stack"],
             resid_fit=np.array([hres["resid_fit"][i] for i in used]),
             resid_holdout=np.array([hres["resid_holdout"][i]
                                     for i in used]),
             resid_fit_per_angle=hres["resid_fit_per_angle"],
             resid_holdout_per_angle=hres["resid_holdout_per_angle"],
             worst_fit=hres["worst_fit"], worst_holdout=hres["worst_holdout"],
             heldout_passed=hres["passed"])
    if bres is not None:
        c = bres["cmp_raw"]
        d.update(bright_entries=c["entries"], bright_peak=c["peak"],
                 bright_rel_peaknorm=c["rel_peaknorm"],
                 bright_rel_pointwise=c["rel_pointwise"],
                 bright_band_max_peaknorm=c["band_max_peaknorm"],
                 bright_rel_at_peak=c["rel_at_peak"],
                 bright_frob_rel=c["frob_rel"],
                 tol_bright=bres["tol_bright"],
                 bright_passed=bres["passed"])
        if bres["cmp_proj"] is not None:
            d["bright_band_max_peaknorm_proj"] = \
                bres["cmp_proj"]["band_max_peaknorm"]
    if ores:
        ks = sorted(ores)
        d.update(obs_ifreqs=np.array(ks),
                 obs_H=np.array([ores[i]["H"] for i in ks]),
                 obs_res_c=np.array([ores[i]["res_c"] for i in ks]),
                 obs_s=np.array([ores[i]["s"] for i in ks]),
                 obs_tau=np.array([ores[i]["tau"] for i in ks]),
                 obs_lam=np.array([ores[i]["lam"] for i in ks]),
                 obs_sigma=ores[ks[0]]["sigma"])
    if dres is not None:
        d.update(disc_rho=dres["rho"], disc_pvalue=dres["pvalue"],
                 disc_rho_invH=dres["rho_invH"],
                 disc_err_abs=dres["err_abs"],
                 disc_err_peaknorm=dres["err_peaknorm"],
                 disc_H_pooled=dres["H_pooled"],
                 disc_G_pooled=dres["G_pooled"], disc_support=dres["support"],
                 disc_labels=np.array(dres["labels"]),
                 disc_rho_min=dres["rho_min"], disc_passed=dres["passed"])
    if pres is not None:
        d.update(passivity_per_freq=pres["passivity_per_freq"],
                 passivity_max_sv=pres["passivity_max_sv"],
                 passivity_ok=pres["passivity_ok"])
        if pres["reciprocity_per_freq"] is not None:
            d.update(reciprocity_per_freq=pres["reciprocity_per_freq"],
                     reciprocity_max=pres["reciprocity_max"],
                     reciprocity_ok=pres["reciprocity_ok"])
    if extra:
        d.update(extra)
    path = os.path.join(RESULTS_DIR, "validate_reference_%s.npz" % tag)
    np.savez(path, **{k: v for k, v in d.items() if v is not None})
    return path


# ===========================================================================
# 8.  Self-test (synthetic, no CST, no license)
# ===========================================================================

MAX_STAT_FACTOR = np.sqrt(np.log(392.0))    # ~2.44: E[max of n_obs Rayleigh]


def _cd_noise_study(fm, band, angle, hyp, sigmas, n_trials, seed=4242,
                    tol=TOL_CHANNEL_DICT, family="extended"):
    """Monte-Carlo success rate of three tests at one angle.

    Synthesizes CST-mode-order blocks from the reference-T prediction under
    a KNOWN hypothesis, adds i.i.d. complex Gaussian noise of scale sigma
    (E|n|^2 = sigma^2 per complex observable), and counts how often each
    test returns EXACTLY that hypothesis:

      chi2       given the trial's TRUE sigma (the campaign gets it from
                 the par.-7 normal-incidence closure);
      max        given the DOC's fixed tol -- the realistic case, since the
                 doc fixes 1e-2 with no knowledge of the noise floor;
      max-oracle given the tolerance a PERFECTLY INFORMED operator would
                 pick -- the midpoint of the usable window,
                 tol* = (f sigma + sep) / 2 with f = sqrt(ln n_obs) ~ 2.44
                 the expected pooled max of the TRUE hypothesis's own
                 noise.  When f sigma >= sep the window is CLOSED and NO
                 tolerance whatsoever can work (reported as
                 window_open = False); that is the decisive statement,
                 because it is independent of how the tolerance is chosen.
                 The midpoint is indicative, not optimal -- measured, it
                 can score slightly BELOW the doc's fixed tol in a cell
                 where the doc's value happens to sit better in the window
                 (90 % vs 100 % at (30, 0), sigma = 3e-3).  Its purpose is
                 to keep the chi2-vs-max comparison from being an artefact
                 of tol = 1e-2.

    Returns {sigma: dict(...rates, counts, chi2_z_mean, tol_oracle,
    window_open, sep)}.
    """
    th, ph = angle
    S11p, S21p = reference_blocks(fm, band, th, ph)
    S11c, S21c = deembed.apply_hypothesis(S11p, S21p, hyp, inverse=True)
    sep = np.inf
    for h in deembed.label_hypotheses(extended=(family == "extended")):
        if all(h[k] == v for k, v in hyp.items()):
            continue                      # skip the TRUE hypothesis itself
        a, b = deembed.apply_hypothesis(S11p, S21p, h)
        sep = min(sep, max(maxabs(a - S11p), maxabs(b - S21p)))
    out = {}
    for sg in sigmas:
        tol_or = 0.5 * (MAX_STAT_FACTOR * sg + sep)
        window_open = bool(MAX_STAT_FACTOR * sg < sep)
        tests = (("chi2", "chi2", tol), ("max", "max", tol),
                 ("max_oracle", "max", tol_or))
        n_ok = {k: 0 for k, _, _ in tests}
        n_wrong = dict(n_ok)
        n_ref = dict(n_ok)
        zs = []
        for tr in range(n_trials):
            rng = np.random.default_rng([seed, int(round(sg * 1e9)), tr])

            def noise(shape):
                return (rng.standard_normal(shape)
                        + 1j * rng.standard_normal(shape)) * (sg / np.sqrt(2))

            m11, m21 = S11c + noise(S11c.shape), S21c + noise(S21c.shape)
            for key, stat, tl in tests:
                r = channel_dictionary_acceptance(
                    fm, band, m11, m21, th, ph, tol=tl, statistic=stat,
                    sigma=sg, pred=(S11p, S21p), family=family,
                    verbose=False)
                if r["passed"] and r["hypothesis"] == hyp:
                    n_ok[key] += 1
                elif r["passed"]:
                    n_wrong[key] += 1        # accepted the WRONG hypothesis
                else:
                    n_ref[key] += 1          # fail-safe refused
                if key == "chi2":
                    zs.append(r["z_margin"])
        out[sg] = dict(
            chi2_rate=n_ok["chi2"] / n_trials,
            max_rate=n_ok["max"] / n_trials,
            max_oracle_rate=n_ok["max_oracle"] / n_trials,
            chi2_wrong=n_wrong["chi2"], max_wrong=n_wrong["max"],
            max_oracle_wrong=n_wrong["max_oracle"],
            chi2_refused=n_ref["chi2"], max_refused=n_ref["max"],
            max_oracle_refused=n_ref["max_oracle"],
            chi2_z_mean=float(np.mean(zs)), tol_oracle=float(tol_or),
            window_open=window_open, sep=float(sep), n_trials=n_trials)
    return out


def _channel_dict_selftest(fm, band, tol_strict=2e-3, n_trials=20,
                           verbose=True):
    """par.-7 hook exercises with KNOWN label hypotheses.

    Synthesizes 'measured' CST-mode-order blocks from the reference-T
    forward prediction by applying a known hypothesis (apply_hypothesis is
    an INVOLUTION -- asserted here -- so the same call is its own inverse
    map), perturbs them, and requires the acceptance to return EXACTLY that
    hypothesis; then requires the fail-safe to REFUSE when the perturbation
    is far too large.  Run for BOTH statistics:

      * "max": three known hypotheses at tol_strict = 2e-3 (not the doc's
        1e-2 -- this structure's band-max cross-pol is only 4.8e-3..9.7e-3,
        so at 1e-2 the max statistic has almost no margin), plus the
        doc-literal (30, 0)/1e-2 case reported as a measurement;
      * "chi2": the same recovery under REAL additive complex Gaussian
        noise at sigma = 3e-3 and sigma = 1e-2, over n_trials seeded
        trials, with the max statistic run on the identical trials so the
        two success rates can be compared directly.

    `band` is the FULL 49-frequency tmat grid: pooling over the band is
    what makes either statistic work (module docstring (a)).
    """
    hyps = deembed.label_hypotheses()
    ext_hyps = deembed.label_hypotheses(extended=True)
    # round-trip checks.  The BASE 8 are involutions (unchanged property);
    # with r11_tm = -1 AND swap = True the forward map is NOT (the row sign
    # lives in Jones indices and does not commute with the swap), so the
    # "inverse-mapped" step uses apply_hypothesis(..., inverse=True).
    rng = np.random.default_rng(7)
    A = rng.normal(size=(2, 2, 3)) + 1j * rng.normal(size=(2, 2, 3))
    Bb = rng.normal(size=(2, 2, 3)) + 1j * rng.normal(size=(2, 2, 3))
    inv_err = 0.0
    for h in hyps:
        a1, b1 = deembed.apply_hypothesis(A, Bb, h)
        a2, b2 = deembed.apply_hypothesis(a1, b1, h)
        inv_err = max(inv_err, maxabs(a2 - A), maxabs(b2 - Bb))
    gate("par.7 machinery: apply_hypothesis is still an involution on the "
         "BASE 8 (no regression)", inv_err == 0.0, True,
         "max round-trip error = %.1e" % inv_err)
    rt_err, n_noninv = 0.0, 0
    for h in ext_hyps:
        a1, b1 = deembed.apply_hypothesis(A, Bb, h)
        a2, b2 = deembed.apply_hypothesis(a1, b1, h, inverse=True)
        rt_err = max(rt_err, maxabs(a2 - A), maxabs(b2 - Bb))
        a3, b3 = deembed.apply_hypothesis(a1, b1, h)
        if max(maxabs(a3 - A), maxabs(b3 - Bb)) > 0:
            n_noninv += 1
    gate("par.7 machinery: apply_hypothesis(inverse=True) inverts all 16 "
         "extended hypotheses exactly", rt_err == 0.0, True,
         "max round-trip error = %.1e; %d of 16 members are NOT involutions "
         "(exactly the swap=True & r11_tm=-1 ones), which is why the "
         "inverse flag exists" % (rt_err, n_noninv))

    cases = [
        ("H0 (swap=F,+,+,r11tm=+) at (30,0) [doc angle]",
         ext_hyps[0], 30.0, 0.0),
        ("H5 (swap=T,+,-,r11tm=+) at (60,22.5)", ext_hyps[5], 60.0, 22.5),
        ("H3+row (swap=F,-,-,r11tm=-) at (30,22.5) [the campaign's "
         "winner shape]", ext_hyps[8 + 3], 30.0, 22.5),
        ("H5+row (swap=T,+,-,r11tm=-) at (60,22.5) [non-involutive]",
         ext_hyps[8 + 5], 60.0, 22.5),
    ]
    results = []
    for name, hyp, th, ph in cases:
        ia = fm.angle_index(th, ph)
        S11p, S21p = reference_blocks(fm, band, th, ph)
        # inverse-map the known hypothesis -> "CST mode order" measurement
        S11c, S21c = deembed.apply_hypothesis(S11p, S21p, hyp, inverse=True)
        pred = (S11p, S21p)
        rng = np.random.default_rng([2026, ia, ext_hyps.index(hyp)])
        eps = 0.1 * tol_strict
        pert = (rng.standard_normal(S11c.shape)
                + 1j * rng.standard_normal(S11c.shape)) * (eps / np.sqrt(2))
        pert2 = (rng.standard_normal(S21c.shape)
                 + 1j * rng.standard_normal(S21c.shape)) * (eps / np.sqrt(2))
        r = channel_dictionary_acceptance(
            fm, band, S11c + pert, S21c + pert2, th, ph, tol=tol_strict,
            statistic="max", pred=pred, verbose=False)
        ok = (r["passed"] and r["hypothesis"] == hyp)
        gate("par.7 machinery: 'max' statistic recovers %s (tol %.0e, "
             "pooled band)" % (name, tol_strict), ok, True,
             "winners = %d, picked %s, residual %.2e, next-best %.2e "
             "(%.1fx tol)" % (r["n_winners"], r["hypothesis"],
                              r["residual"] or np.nan,
                              r["second_residual"], r["margin"]))
        rc = channel_dictionary_acceptance(
            fm, band, S11c + pert, S21c + pert2, th, ph, statistic="chi2",
            sigma=eps, pred=pred, verbose=False)
        gate("par.7 machinery: 'chi2' statistic recovers %s (sigma = %.0e)"
             % (name, eps), rc["passed"] and rc["hypothesis"] == hyp, True,
             "picked %s, z_margin = %.1f (z_min %.1f), chi2_reduced = %.2f "
             "(expect ~1)" % (rc["hypothesis"], rc["z_margin"], rc["z_min"],
                              rc["chi2_reduced"]))
        results.append((name, r, rc))

        # fail-safe: perturbation far above tol -> 0 winners -> DeembedError
        big = 50.0 * tol_strict
        rb = (rng.standard_normal(S11c.shape)
              + 1j * rng.standard_normal(S11c.shape)) * (big / np.sqrt(2))
        rb2 = (rng.standard_normal(S21c.shape)
               + 1j * rng.standard_normal(S21c.shape)) * (big / np.sqrt(2))
        rf = channel_dictionary_acceptance(
            fm, band, S11c + rb, S21c + rb2, th, ph, tol=tol_strict,
            statistic="max", pred=pred, verbose=False)
        raised = False
        try:
            deembed.select_hypothesis(S11c + rb, S21c + rb2,
                                      rf["S11_pred"], rf["S21_pred"],
                                      tol_strict)
        except deembed.DeembedError:
            raised = True
        gate("par.7 machinery: 'max' fail-safe refuses a %.0e perturbation "
             "(%s)" % (big, name.split(" at ")[0]),
             raised and (not rf["passed"]) and rf["n_winners"] == 0, True,
             "winners = %d, best residual %.2e > tol %.0e; DeembedError "
             "raised = %s" % (rf["n_winners"], rf["best_residual"],
                              tol_strict, raised))
        # chi2 fail-safe, both trip conditions:
        #  (i) sigma so large that the margin collapses below z_min
        #  (ii) a perturbation so large that the winner's chi2 blows up
        rc_i = channel_dictionary_acceptance(
            fm, band, S11c + pert, S21c + pert2, th, ph, statistic="chi2",
            sigma=3e-2, pred=pred, verbose=False)
        rc_ii = channel_dictionary_acceptance(
            fm, band, S11c + rb, S21c + rb2, th, ph, statistic="chi2",
            sigma=SIGMA_CHANNEL_DICT, pred=pred, verbose=False)
        gate("par.7 machinery: 'chi2' fail-safe refuses on margin AND on "
             "chi2 blow-up (%s)" % name.split(" at ")[0],
             (not rc_i["passed"]) and (not rc_ii["passed"]), True,
             "(i) sigma = 3e-2 -> z_margin = %.2f < z_min %.1f, REFUSED; "
             "(ii) %.0e perturbation at sigma = %.0e -> chi2_reduced = "
             "%.3g > %.1f, REFUSED"
             % (rc_i["z_margin"], rc_i["z_min"], big, SIGMA_CHANNEL_DICT,
                rc_ii["chi2_reduced"], rc_ii["chi2_max"]))

    # --- hypothesis separation across the whole campaign angle set -------
    print("\n  hypothesis separation at every campaign angle (raw reference "
          "T, direction %+d, pooled over %d frequencies):"
          % (DIRECTION_DEFAULT, len(band)))
    print("    idx  (theta,phi)    sep(max stat)  |cross|max   z@3e-3  "
          "z@5e-3  z@1e-2")
    sep_rows = []
    for ia in fitmod.ANGLE_SETS["campaign"]:
        s11, s21 = reference_blocks(fm, band, fm.theta_deg[ia],
                                    fm.phi_deg[ia])
        xp = max(maxabs(s11[[0, 1], [1, 0]]), maxabs(s21[[0, 1], [1, 0]]))
        sep, dmin = np.inf, np.inf
        for h in ext_hyps:
            if h == ext_hyps[0]:
                continue
            a, b = deembed.apply_hypothesis(s11, s21, h)
            sep = min(sep, max(maxabs(a - s11), maxabs(b - s21)))
            dmin = min(dmin, float((np.abs(a - s11) ** 2).sum()
                                   + (np.abs(b - s21) ** 2).sum()))
        zz = [np.sqrt(dmin) / (2 * s) for s in (3e-3, 5e-3, 1e-2)]
        sep_rows.append((ia, fm.phi_deg[ia], sep, xp, dmin, zz))
        print("    %2d   (%2.0f,%5.1f)     %.3e     %.3e   %6.1f  %6.1f  "
              "%6.1f" % (ia, fm.theta_deg[ia], fm.phi_deg[ia], sep, xp,
                         zz[0], zz[1], zz[2]))
    mirror = [s for _, p, s, _, _, _ in sep_rows if p % 45 == 0]
    off = [s for _, p, s, _, _, _ in sep_rows if p % 45 != 0]
    ratio = float(np.median(off) / np.median(mirror))
    gate("par.7 measured: phi = 22.5 angles separate the cross-sign "
         "hypotheses MUCH better than the mirror planes", ratio >= 2.0,
         False,
         "mirror-plane sep median %.3e vs phi=22.5 median %.3e -- RATIO "
         "%.2f, i.e. NO qualitative advantage.  This structure barely "
         "cross-polarizes anywhere (|cross|max 4.8e-3..9.7e-3 at every "
         "campaign angle with the raw reference T), which is the real "
         "reason the cross-sign half is hard; theta, not phi, is what "
         "helps (sep grows 1.6x from theta 15 to 60)"
         % (np.median(mirror), np.median(off), ratio))

    # --- the doc's literal acceptance, as a MEASUREMENT ------------------
    print("\n  doc-literal par.-7 acceptance (theta=30, phi=0, tol=1e-2, "
          "'max' statistic, reference T, NO label distortion, NO noise), "
          "FULL %d-frequency band:" % len(band))
    S11p, S21p = reference_blocks(fm, band, 30.0, 0.0)
    pred30 = (S11p, S21p)
    r_pool = channel_dictionary_acceptance(
        fm, band, S11p, S21p, 30.0, 0.0, tol=TOL_CHANNEL_DICT,
        statistic="max", band_mode="pooled", family="base", pred=pred30,
        verbose=True)
    gate("par.7 doc-literal: (30,0) tol 1e-2 'max' POOLED over the full "
         "band gives exactly 1 winner",
         r_pool["passed"] and r_pool["n_winners"] == 1, False,
         "winners = %d; next-best residual %.3e = %.2fx tol.  NOISE-FREE "
         "only: the true hypothesis's own pooled max is ~2.44 sigma "
         "(7.3e-3 at sigma = 3e-3), leaving a usable window of just 3.7e-3 "
         "-- see the noise study below"
         % (r_pool["n_winners"], r_pool["second_residual"],
            r_pool["margin"]))
    r_pf = channel_dictionary_acceptance(
        fm, band, S11p, S21p, 30.0, 0.0, tol=TOL_CHANNEL_DICT,
        statistic="max", band_mode="per_freq", family="base", pred=pred30,
        verbose=False)
    nuniq = sum(1 for _, h, _, _ in r_pf["per_freq"] if h is not None)
    gate("par.7 doc-literal: (30,0) tol 1e-2 PER-FREQUENCY gives a unique "
         "winner at every frequency", r_pf["passed"], False,
         "unique winner at %d/%d frequencies -- the band-POOLED reading is "
         "the only one under which the doc's acceptance can pass at all"
         % (nuniq, len(band)))
    r_chi30 = channel_dictionary_acceptance(
        fm, band, S11p, S21p, 30.0, 0.0, statistic="chi2",
        sigma=SIGMA_CHANNEL_DICT, pred=pred30, verbose=False)
    print("  same data, 'chi2' statistic at the placeholder sigma = %.0e: "
          "z_margin = %.1f (z_min %.1f), chi2_reduced = %.3g -> %s"
          % (SIGMA_CHANNEL_DICT, r_chi30["z_margin"], r_chi30["z_min"],
             r_chi30["chi2_reduced"],
             "ACCEPT" if r_chi30["passed"] else "REFUSE"))

    # --- NOISE STUDY: success rate of both statistics --------------------
    sigmas = (3e-3, 5e-3, 1e-2)
    print("\n  NOISE STUDY -- success rate over %d seeded trials per cell "
          "(known hypothesis H0, additive complex Gaussian noise).  chi2 "
          "gets the trial's true sigma; 'max(doc)' gets the doc's fixed "
          "tol = %.0e; 'max(oracle)' gets the MIDPOINT of the usable "
          "tolerance window, i.e. what a perfectly informed operator would "
          "pick (indicative, not optimal); 'window CLOSED' means 2.44 sigma "
          ">= sep, so NO tolerance can work at all:"
          % (n_trials, TOL_CHANNEL_DICT))
    print("    angle        sigma    chi2 (z_mean)   max(doc)   max(oracle) "
          "  oracle tol   window   wrong picks")
    noise_study = {}
    for th, ph in ((30.0, 0.0), (45.0, 22.5)):
        st = _cd_noise_study(fm, band, (th, ph), ext_hyps[0], sigmas,
                             n_trials)
        noise_study[(th, ph)] = st
        for sg in sigmas:
            d = st[sg]
            print("    (%2.0f,%5.1f)   %.0e   %4.0f%% (%5.1f)   %5.0f%%     "
                  "%6.0f%%      %.2e   %-6s   %d"
                  % (th, ph, sg, 100 * d["chi2_rate"], d["chi2_z_mean"],
                     100 * d["max_rate"], 100 * d["max_oracle_rate"],
                     d["tol_oracle"], "open" if d["window_open"] else
                     "CLOSED",
                     d["chi2_wrong"] + d["max_wrong"]
                     + d["max_oracle_wrong"]), flush=True)
    d30 = noise_study[(30.0, 0.0)]
    d45 = noise_study[(45.0, 22.5)]
    gate("par.7 machinery: 'chi2' recovers the known hypothesis in EVERY "
         "noise trial at sigma = 3e-3 (both angles)",
         all(st[3e-3]["chi2_rate"] == 1.0 for st in noise_study.values()),
         True,
         "(30,0): %.0f %% over %d trials, mean z_margin %.1f; (45,22.5): "
         "%.0f %%, mean z %.1f"
         % (100 * d30[3e-3]["chi2_rate"], n_trials,
            d30[3e-3]["chi2_z_mean"], 100 * d45[3e-3]["chi2_rate"],
            d45[3e-3]["chi2_z_mean"]))
    gate("par.7 machinery: NO test ever accepts a WRONG hypothesis "
         "(fail-safe holds under noise)",
         all(d["chi2_wrong"] == 0 and d["max_wrong"] == 0
             and d["max_oracle_wrong"] == 0
             for st in noise_study.values() for d in st.values()), True,
         "%d wrong acceptances in %d trials x %d sigmas x 2 angles x 3 "
         "tests; every other failure is a REFUSAL"
         % (sum(d["chi2_wrong"] + d["max_wrong"] + d["max_oracle_wrong"]
                for st in noise_study.values() for d in st.values()),
            n_trials, len(sigmas)))
    max_as_good = all(st[s]["max_rate"] >= st[s]["chi2_rate"]
                      for st in noise_study.values() for s in sigmas)
    gate("par.7 measured: the doc's 'max' statistic at tol = 1e-2 survives "
         "noise as well as chi2 does", max_as_good, False,
         "chi2 / max(doc) / max(oracle) success rates -- (30,0): "
         "sigma 3e-3 %.0f/%.0f/%.0f %%, 5e-3 %.0f/%.0f/%.0f %%, "
         "1e-2 %.0f/%.0f/%.0f %%; (45,22.5): 3e-3 %.0f/%.0f/%.0f %%, "
         "5e-3 %.0f/%.0f/%.0f %%, 1e-2 %.0f/%.0f/%.0f %%.  At sigma = 3e-3 "
         "the doc's fixed tol happens to sit inside the 3.7e-3-wide window "
         "and max works; by 5e-3 the window has moved and the doc's tol "
         "fails while chi2 still decides; at 1e-2 the window is CLOSED "
         "(2.44 sigma > sep) so NO tolerance can work at all, while chi2 "
         "merely loses significance (z = %.1f / %.1f) and refuses "
         "honestly.  That is the argument for amending doc par. 7 to a "
         "pooled chi2 test."
         % tuple([100 * d30[s][k] for s in sigmas
                  for k in ("chi2_rate", "max_rate", "max_oracle_rate")]
                 + [100 * d45[s][k] for s in sigmas
                    for k in ("chi2_rate", "max_rate", "max_oracle_rate")]
                 + [d30[1e-2]["chi2_z_mean"], d45[1e-2]["chi2_z_mean"]]))
    return results, r_pool, r_pf, sep_rows, noise_study


def selftest(argv=None):
    ap = argparse.ArgumentParser(
        description="par. 8 acceptance criteria, synthetic end-to-end "
                    "(no CST, no license)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--freqs", default="32,48",
                    help="comma list of frequency indices (default the "
                         "smoke frequencies 32, 48)")
    ap.add_argument("--fit-angles", default=None,
                    help="ANGLE_SETS name or comma list (default "
                         "%s)" % FIT_ANGLES_DEFAULT)
    ap.add_argument("--holdout-angles", default=None,
                    help="default: campaign 13 minus the fit angles")
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="complex Gaussian noise on the synthetic S "
                         "(E|n|^2 = sigma^2 per complex observable)")
    ap.add_argument("--sigma-obs", type=float, default=SIGMA_OBS,
                    help="noise floor for the observability damping "
                         "(PLACEHOLDER 3e-3 until the par.-7 closure)")
    ap.add_argument("--tol-bright", type=float, default=TOL_BRIGHT)
    ap.add_argument("--tol-heldout", type=float, default=TOL_HELDOUT)
    ap.add_argument("--starts", type=int, default=3,
                    help="multistart count for the fits")
    ap.add_argument("--tag", default="selftest")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-channel-dict", action="store_true")
    args = ap.parse_args(argv)

    t_all = time.time()
    from tmatrix.retrieval.forward import ForwardModel
    print("=" * 76)
    print("validate_against_reference --selftest: par. 8 acceptance "
          "criteria on SYNTHETIC data")
    print("=" * 76)
    fm = ForwardModel()
    print("cache: %s  (%d frequencies with cached angles)"
          % (fm.cache_path, len(fm.available_freqs)))
    B68, meta, B10, binfo = obs.build_bases(fm)
    mask = binfo["mask"]
    print("bases: full %d complex dirs, bright %d complex dirs "
          "(%d bright entries, %d orbits)"
          % (B68.shape[0], B10.shape[0], binfo["n_bright"],
             binfo.get("n_orbits", -1)))

    fit_angles = (FIT_ANGLES_DEFAULT if args.fit_angles is None
                  else resolve_angles(args.fit_angles))
    holdout = ([a for a in fitmod.ANGLE_SETS["campaign"]
                if a not in fit_angles] if args.holdout_angles is None
               else resolve_angles(args.holdout_angles))
    all_ang = sorted(set(fit_angles) | set(holdout))
    freqs = [int(x) for x in args.freqs.split(",")]
    print("fit angles   %s = %s" % (fit_angles,
                                    [_angle_name(fm, a) for a in fit_angles]))
    print("hold-out     %s = %s" % (holdout,
                                    [_angle_name(fm, a) for a in holdout]))
    print("frequencies  %s (lam = %s um); direction %+d; noise sigma = %g"
          % (freqs, ["%.2f" % fm.lam_um[i] for i in freqs],
             DIRECTION_DEFAULT, args.sigma))
    print("tolerances   heldout %.1e | bright %.4g (TOL_BRIGHT) | "
          "passivity 1+%.0e | reciprocity %.0e | consistency rho %.2f"
          % (args.tol_heldout, args.tol_bright, TOL_PASSIVITY,
             TOL_RECIPROCITY, CONSISTENCY_RHO))

    figs = not args.no_figures
    weights = (1.0 / args.sigma ** 2) if args.sigma > 0 else None

    # ------------------------------------------------------------------
    # synthetic truths
    # ------------------------------------------------------------------
    T_proj = np.array([par.unpack(par.pack(fm.data.T[i], B68), B68)
                       for i in range(fm.nf)])
    T_bp = np.array([par.unpack(par.pack(T_proj[i], B10), B10)
                     for i in range(fm.nf)])
    S_proj = synth_S_by_freq(fm, T_proj, freqs, all_ang,
                             sigma=args.sigma, seed=args.seed)
    S_span = synth_S_by_freq(fm, T_bp, freqs, all_ang, sigma=0.0)

    tik = 0.0
    if args.sigma > 0:
        taus = [np.linalg.norm(par.pack(T_bp[i], B10))
                / np.sqrt(2 * B10.shape[0]) for i in freqs]
        tik = 1.0 / float(np.mean(taus)) ** 2
        print("noise mode: weights = 1/sigma^2 = %.3e, tikhonov = 1/tau^2 "
              "= %.3e (physical prior tau = %.3e; the synthetic_test.py "
              "protocol)" % (weights, tik, float(np.mean(taus))))

    # ==================================================================
    print("\n" + "=" * 76)
    print("(0) MACHINERY control: in-span truth, truth-seeded fit, held-out "
          "prediction")
    print("=" * 76)
    t0_seed = {i: par.pack(T_bp[i], B10) for i in freqs}
    ctrl = heldout_acceptance(fm, freqs, B10, S_span, fit_angles, holdout,
                              t0_by_freq=t0_seed, n_starts=1,
                              tol_heldout=args.tol_heldout,
                              xtol=1e-14, ftol=1e-14, gtol=1e-14,
                              max_nfev=500, verbose=True)
    engines = ctrl["engines"]
    dT_ctrl = max(maxabs(ctrl["T0"][i] - T_bp[i]) for i in ctrl["ifreqs"]) \
        if ctrl["ifreqs"] else np.inf
    gate("par.8.1 machinery: in-span truth predicts held-out angles exactly",
         bool(ctrl["ifreqs"]) and ctrl["worst_holdout"] <= 1e-8, True,
         "worst held-out |dS| = %.3e (gate 1e-8), worst |T0_hat - T_truth| "
         "= %.3e -- certifies the fit -> predict -> compare plumbing"
         % (ctrl["worst_holdout"], dT_ctrl))
    born = heldout_acceptance(fm, freqs, B10, S_span, fit_angles, holdout,
                              n_starts=args.starts, seed=args.seed,
                              tol_heldout=args.tol_heldout,
                              xtol=1e-14, ftol=1e-14, gtol=1e-14,
                              max_nfev=2000, engines=engines, verbose=False)
    gate("par.8.1 measured: Born-seeded multistart reaches the in-span "
         "truth from t = 0", born["worst_holdout"] <= 1e-8, False,
         "worst held-out |dS| = %.3e (basin reachability of this angle set, "
         "doc par. 9 -- the truth-seeded control above is the machinery "
         "gate)" % born["worst_holdout"])

    # ==================================================================
    print("\n" + "=" * 76)
    print("(1) par. 8.1 HELD-OUT-ANGLE ACCEPTANCE on the physical target "
          "P68(T_ref)")
    print("=" * 76)
    hres = heldout_acceptance(
        fm, freqs, B10, S_proj, fit_angles, holdout, weights=weights,
        tikhonov=tik, tol_heldout=args.tol_heldout, n_starts=args.starts,
        seed=args.seed, engines=engines, xtol=1e-14, ftol=1e-14, gtol=1e-14,
        max_nfev=2000,
        fig_path=(os.path.join(RESULTS_DIR, "validate_%s_heldout_spectrum"
                               ".png" % args.tag) if figs else None))
    used = hres["ifreqs"]
    if not used:
        print("\nNo frequency had all requested angles cached -- nothing to "
              "validate.")
        return 1
    gate("par.8.1: held-out complex |dS| <= %.0e across the band"
         % args.tol_heldout, hres["passed"], False,
         "worst held-out %.3e at ifreq %d angle %s; worst FITTED-angle "
         "residual %.3e (contrast). Fit on %d angles, predict %d."
         % (hres["worst_holdout"], hres["worst_holdout_ifreq"],
            _angle_name(fm, hres["worst_holdout_angle"]), hres["worst_fit"],
            len(fit_angles), len(holdout)))

    # ==================================================================
    print("\n" + "=" * 76)
    print("(2) par. 8.2 BRIGHT-ENTRY COMPARISON vs the reference tmat.h5")
    print("=" * 76)
    bres = bright_comparison(fm, hres["T0_stack"], used, mask, B_proj=B68,
                             tol_bright=args.tol_bright)
    gate("par.8.2: bright entries within %.4g of the reference (peak-"
         "normalized)" % args.tol_bright, bres["passed"], False,
         "max band-max peaknorm = %.3e over %d entries (%.0f%% within tol); "
         "dipole class max = %.3e -- information content of this angle set, "
         "not an optimizer defect (see synthetic_test.py)"
         % (bres["max_rel_peaknorm"], len(bres["entries"]),
            100 * bres["frac_within_tol"],
            bres["classes"].get("dipole", {}).get("max", np.nan)))

    # ==================================================================
    print("\n" + "=" * 76)
    print("(5) par. 8.4 OBSERVABILITY MAP published with the fit")
    print("=" * 76)
    ores = fit_observability(fm, used, B10, fit_angles, engines=engines,
                             T_truth_stack=T_proj, sigma=args.sigma_obs,
                             mask=mask, tag=args.tag, make_figures=figs)
    fig_ok = True
    if figs:
        for i in used:
            for key in ("heatmap_path", "spectrum_path"):
                p = ores[i][key]
                fig_ok &= bool(p) and os.path.isfile(p) \
                    and os.path.getsize(p) > 0
            print("  ifreq %2d figures: %s ; %s"
                  % (i, os.path.basename(ores[i]["heatmap_path"]),
                     os.path.basename(ores[i]["spectrum_path"])))
    gate("par.8.4 machinery: observability heatmap + SV spectrum written "
         "for the fitted angle set", fig_ok or not figs, True,
         "%d frequencies x 2 figures under results/ (tag '%s')"
         % (len(used), args.tag) if figs else "figures disabled")

    # ==================================================================
    print("\n" + "=" * 76)
    print("(3) par. 8.2 DISCREPANCY-vs-OBSERVABILITY CONSISTENCY")
    print("=" * 76)
    dres = discrepancy_vs_observability(
        fm, used, hres["T0_stack"], mask, ores,
        fig_path=(os.path.join(RESULTS_DIR, "validate_%s_discrepancy_"
                               "scatter.png" % args.tag) if figs else None))
    gate("par.8.2: discrepancy pattern consistent with the observability "
         "map (Spearman rho >= %.2f)" % CONSISTENCY_RHO, dres["passed"],
         False,
         "PRIMARY rho(|dT|, G) = %+.3f (p = %.2g, n = %d entries), "
         "G = sum_k (1-res_k)|B_k|^2 ~ posterior variance / tau^2; the "
         "doc-LITERAL rho(|dT|, 1/H) = %+.3f (p = %.2g) does NOT pass -- "
         "1/H saturates at ~1 for resolved entries and diverges where the "
         "basis has no support, so it is the wrong pairing (see "
         "discrepancy_vs_observability docstring); threshold %.2f ~ p 0.01 "
         "two-sided at n = 25"
         % (dres["rho"], dres["pvalue"], len(dres["entries"]),
            dres["rho_invH"], dres["pvalue_invH"], CONSISTENCY_RHO))

    # ==================================================================
    print("\n" + "=" * 76)
    print("(4) par. 8.3 PASSIVITY and RECIPROCITY checks")
    print("=" * 76)
    pres = physical_checks(hres["T0_stack"], modes=fm.modes)
    ref_pas = par.passivity_max_sv(fm.data.T[used])
    gate("par.8.3 machinery: reciprocity of the fitted T0 <= %.0e"
         % TOL_RECIPROCITY, bool(pres["reciprocity_ok"]), True,
         "max|Rec(T0) - T0| = %.3e (exact by subspace construction; a "
         "failure would be a basis/packing defect)"
         % pres["reciprocity_max"])
    gate("par.8.3: passivity max SV(I + 2 T0) <= 1 + %.0e" % TOL_PASSIVITY,
         pres["passivity_ok"], False,
         "fitted %.6f vs reference tmat.h5 %.6f over the same frequencies "
         "(a physical property of the amplitudes, never enforced)"
         % (pres["passivity_max_sv"], ref_pas))

    # ==================================================================
    print("\n" + "=" * 76)
    print("(6) par. 7 CHANNEL-DICTIONARY ACCEPTANCE hook")
    print("=" * 76)
    cd = None
    if not args.no_channel_dict:
        cd_band = [i for i in range(fm.nf)
                   if fm.have[i, fm.angle_index(30.0, 0.0)]
                   and fm.have[i, fm.angle_index(60.0, 22.5)]
                   and fm.have[i, fm.angle_index(30.0, 22.5)]]
        print("  band for the channel-dictionary hook: %d of %d tmat "
              "frequencies with the required angles cached"
              % (len(cd_band), fm.nf))
        cd = _channel_dict_selftest(fm, cd_band)

    # ==================================================================
    print("\n" + "=" * 76)
    print("(7) REAL-DATA PATH: the same checks through "
          "validate_from_deembedded()")
    print("=" * 76)
    # build de-embedded-style blocks in CST mode order from the SAME
    # synthetic S, using a known hypothesis, then push them through the
    # real-data entry point and require identical results.
    # swap=True, s11_cross=+1, s21_cross=-1, r11_tm=-1: deliberately the
    # NON-INVOLUTIVE member, so the real-data plumbing is proven on the
    # hardest case rather than the easiest.
    hyp = deembed.label_hypotheses(extended=True)[8 + 5]
    n_ang = len(all_ang)
    S11_ang = np.full((n_ang, 2, 2, fm.nf), np.nan, dtype=complex)
    S21_ang = np.full((n_ang, 2, 2, fm.nf), np.nan, dtype=complex)
    for j, ia in enumerate(all_ang):
        for i in freqs:
            S11_ang[j, :, :, i] = S_proj[i][ia, 0]
            S21_ang[j, :, :, i] = S_proj[i][ia, 1]
        # inverse-map into "CST mode order"
        S11_ang[j], S21_ang[j] = deembed.apply_hypothesis(
            S11_ang[j], S21_ang[j], hyp, inverse=True)
    print("  synthetic de-embedded blocks: S11/S21 shape %s "
          "(angle slot, receive, incident, tmat freq), CST mode order, "
          "hypothesis %s" % (S11_ang.shape, hyp))
    # (a) the ARRAY plumbing must be bit-exact
    S_round = blocks_to_S_by_freq(fm, S11_ang, S21_ang, all_ang, hyp)
    dS_plumb = max(maxabs(rows_for_angles(S_round[i], all_ang)
                           - rows_for_angles(S_proj[i], all_ang))
                   for i in freqs)
    gate("real-data path machinery: blocks_to_S_by_freq + apply_hypothesis "
         "round trip is bit-exact", dS_plumb == 0.0, True,
         "max|dS| = %.1e over %d angles x %d frequencies (CST mode order "
         "-> hypothesis -> par.-2 Jones order)"
         % (dS_plumb, len(all_ang), len(freqs)))
    # (b) measured optimizer repeatability floor (same inputs, same
    #     process, same engine): MINPACK/threaded-BLAS reduction order is
    #     not bit-reproducible and the 4-angle bright optimum has near-flat
    #     directions, so T0 agreement is checked at a relative tolerance,
    #     not at bit level.
    i_rep = used[-1]
    S_rep = rows_for_angles(S_proj[i_rep], fit_angles)
    kw_rep = dict(direction=DIRECTION_DEFAULT, engine=engines[i_rep],
                  xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
    T_r1 = fitmod.fit_frequency(fm, i_rep, B10, S_rep, fit_angles,
                                **kw_rep)["T0_hat"]
    T_r2 = fitmod.fit_frequency(fm, i_rep, B10, S_rep, fit_angles,
                                **kw_rep)["T0_hat"]
    rep_floor = maxabs(T_r1 - T_r2)
    real = validate_from_deembedded(
        fm, S11_ang, S21_ang, all_ang, hypothesis=hyp, ifreq_list=freqs,
        B=B10, mask=mask, fit_angles=fit_angles, holdout_angles=holdout,
        weights=weights, tikhonov=tik, tol_heldout=args.tol_heldout,
        tol_bright=args.tol_bright, sigma_obs=args.sigma_obs,
        n_starts=args.starts, tag=args.tag + "_realpath",
        make_figures=figs, verbose=False,
        xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
    dT = (max(maxabs(real["heldout"]["T0"][i] - hres["T0"][i])
              for i in used) if real["ifreqs"] == used else np.inf)
    dR = abs(real["heldout"]["worst_holdout"] - hres["worst_holdout"])
    scale = max(maxabs(hres["T0_stack"]), 1e-30)
    tol_path = 1e-6 * scale
    gate("real-data path machinery: validate_from_deembedded reproduces the "
         "direct-path fit", dT <= tol_path and dR <= 1e-6, True,
         "max|T0_realpath - T0_direct| = %.2e (tol 1e-6 x max|T0| = %.2e; "
         "MEASURED same-input optimizer repeatability floor = %.2e -- "
         "MINPACK/threaded-BLAS reduction order is not bit-reproducible and "
         "the 4-angle bright optimum has near-flat directions), "
         "|d worst-holdout| = %.2e; ifreqs %s"
         % (dT, tol_path, rep_floor, dR, real["ifreqs"]))
    print("  real-data path outputs: %s" % real["npz_path"])

    # ==================================================================
    extra = {}
    if cd is not None:
        _, r_pool, r_pf, sep_rows, noise_study = cd
        extra.update(
            cd_angle_index=r_pool["angle_index"],
            cd_tol=r_pool["tol"],
            cd_residuals=np.array([r for _, r in r_pool["table"]]),
            cd_D=np.array([d for _, d in r_pool["D_table"]]),
            cd_n_winners=r_pool["n_winners"],
            cd_second_residual=r_pool["second_residual"],
            cd_perfreq_unique=int(sum(1 for _, h, _, _ in r_pf["per_freq"]
                                      if h is not None)),
            cd_perfreq_n=len(r_pf["per_freq"]),
            cd_sep_angle_idx=np.array([r[0] for r in sep_rows]),
            cd_sep_max=np.array([r[2] for r in sep_rows]),
            cd_sep_xpol=np.array([r[3] for r in sep_rows]),
            cd_sep_D=np.array([r[4] for r in sep_rows]),
            cd_sep_z=np.array([r[5] for r in sep_rows]),
            cd_noise_angles=np.array([list(k) for k in noise_study]),
            cd_noise_sigmas=np.array(sorted(
                next(iter(noise_study.values())))),
            cd_noise_chi2_rate=np.array(
                [[noise_study[a][s]["chi2_rate"]
                  for s in sorted(noise_study[a])] for a in noise_study]),
            cd_noise_max_rate=np.array(
                [[noise_study[a][s]["max_rate"]
                  for s in sorted(noise_study[a])] for a in noise_study]),
            cd_noise_max_oracle_rate=np.array(
                [[noise_study[a][s]["max_oracle_rate"]
                  for s in sorted(noise_study[a])] for a in noise_study]),
            cd_noise_window_open=np.array(
                [[noise_study[a][s]["window_open"]
                  for s in sorted(noise_study[a])] for a in noise_study]),
            cd_z_min=Z_MIN_CHANNEL_DICT,
            cd_sigma_placeholder=SIGMA_CHANNEL_DICT)
    npz = save_results(fm, args.tag, hres, bres, ores, dres, pres, extra)
    print("\narrays: %s" % npz)

    print("\n" + "=" * 76)
    print("SUMMARY (validate_against_reference --selftest; %.1f s; freqs "
          "%s; skipped %s)" % (time.time() - t_all, used,
                               hres["skipped"] or "none"))
    print("=" * 76)
    n_mach_fail = 0
    for name, ok, machinery, _ in GATES:
        tag = "PASS" if ok else ("FAIL" if machinery
                                 else "measured FAIL -- expected")
        print("  [%-26s] %s" % (tag, name))
        if machinery and not ok:
            n_mach_fail += 1
    print("machinery gates: %s"
          % ("ALL PASSED" if n_mach_fail == 0 else "%d FAILED" % n_mach_fail))
    print("(par.-8 acceptance numbers on the physical target are "
          "information-content MEASUREMENTS; tolerance in force: "
          "TOL_BRIGHT = %.4g)" % args.tol_bright)
    return 1 if n_mach_fail else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--selftest" in argv:
        sys.exit(selftest(argv))
    print(__doc__)
    print("Run:  python validate_against_reference.py --selftest")


if __name__ == "__main__":
    main()

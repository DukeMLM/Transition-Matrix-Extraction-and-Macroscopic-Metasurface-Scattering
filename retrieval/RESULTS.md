# Multi-angle Floquet → isolated-cell T-matrix: results

Campaign executed and verified 2026-08-06/07. Every number below was either
produced by the verifier re-running the code, or reproduced by an
independently written script that re-derives its own references. Companion
documents: `HANDOFF.md` (state + conventions), `VERIFICATION_ITEMS_5_6.md`
(machinery verification), `RETRIEVAL_LIMIT.md` (what limits the retrieval),
`results/REAL_RETRIEVAL.md` (the full §8 report),
`results/GATE_STUDY.md` (the estimator/angle-set study).

---

## 1. The two questions, separated

The design doc's §1 gives two distinct motivations. They have opposite
answers, and keeping them apart is the main result.

### Q1 — "Can a supplied tmat.h5 be validated against cheap periodic runs?" **YES.**

The reference `saw_gold_wl15p0025um.tmat.h5`, pushed through the multiple-
scattering forward map with no fitting at all, reproduces the de-embedded CST
data at **all 13 campaign angles** to **2.74e-3 … 3.58e-3**, i.e. **≤ 1.36 σ
at every angle**, against a measured noise floor σ = 2.633e-3. (Against the
raw reference T the residual is 4.8e-3…9.7e-3; the 2.7× gap is that file's own
~0.3 % C4v-violation noise, not a model error.)

This is the QA gate of §1 reason 1, and it works — a collaborator's T-matrix
can be accepted or rejected from periodic solves alone, at ~1 minute per
angle, with no isolated-cell rerun.

### Q2 — "Can T0 be *retrieved* from Floquet data?" **NO — and we can now show it on real data.**

A T0 fitted to 4 angles predicts the other 9 to **1.87e-3** (§8.1 gate 1e-2,
5.4× margin, 0.67 σ) — it reproduces unmeasured angles *inside the noise* —
while being **wrong**:

| §8 criterion | measured | verdict |
|---|---|---|
| §8.1 held-out angles ≤ 1e-2 | **1.8657e-3** | **PASS** (5.4×) |
| §8.2 bright entries ≤ 10 % | worst 61.6, median 12.0, **0 of 25** within tol | FAIL (~600×) |
| §8.2 discrepancy vs observability, ρ ≥ 0.50 | ρ_G = +0.311 (p 0.13) | FAIL |
| §8.3 passivity ≤ 1 + 1e-3 | **1.1461** (reference 1.00007) | **VIOLATED** |
| §8.3 reciprocity ≤ 1e-3 | 2.29e-17 | PASS (exact by construction) |
| §8.4 observability heatmap published | 49 × 2 figures at measured σ | PASS |

**A T-matrix that predicts every held-out angle to within noise is 600 % wrong
in its entries and 14.6 % super-unitary.** That is a direct, real-data
demonstration of the non-uniqueness that §5 predicted from the literature —
far stronger evidence than the synthetic argument, and the single most
citable outcome of this work.

The mechanism is decomposed in `RETRIEVAL_LIMIT.md`: the rich span *has* the
information (a truth-seeded noise-free fit reaches 0.41 % dipole error at
12 µm), but no realizable seed reaches that basin, and across 23 candidate
protocols none beats the trivial `T̂ = 0` estimator on the dipole class.
Passivity violation is the cleanest single symptom.

---

## 2. The campaign

19 FD solves, all §7 gates passed. **Solver time ≈ 13 minutes total** —
structure runs 50–78 s, empty runs 16–23 s. The doc's "tens of minutes per
solve / hours for the campaign" was wrong by ~30×, which retroactively makes
the angle-set economising moot.

| §7 gate | result |
|---|---|
| Normal-incidence complex closure ≤ 5e-3 | **3.617e-3 PASS** |
| **σ (the fit noise floor), measured not assumed** | **2.6333e-3** (per-freq 2.31e-3…3.62e-3, median 2.68e-3), S11-phase-dominated |
| Pinned domain, all 19 runs | L = 11.714687 µm, **diff +0.00 nm** |
| Empty-cell phase advance, every θ | rel. err ≤ 6.9e-5, ‖S21‖−1 ≤ 3.4e-4, \|S11_empty\| ≤ 7.1e-5 |
| Channel-dictionary acceptance at (60°,22.5°) | **PASS**, χ²_red 1.595, z 5.77 |
| Model-free mirror-plane cross-pol (no reference model) | **4.34e-5** vs 5e-3 tol |

Two independent routes agree on the label map: the χ² acceptance against the
reference-T forward model, and the reference-free mirror-plane cross-pol check.

**Noise-floor calibration (§7's two prescribed routes):** empty vs analytic
`e^{+i k_z L}` gives 9.52e-4 = 0.36 σ; the perturbed mesh
(`AccuracyTet` 1e-4 → 3e-4) gives 1.04e-4 = 0.039 σ. Both are far below σ,
so **the fit floor is model error, not solver noise** — exactly as §7
predicted, and the reason σ is a systematic scale rather than a random one
(every χ² z-value here is therefore indicative, not a calibrated confidence).

---

## 3. Corrections this work forced on the design document

Each is amended inline in `INVERSE_TMATRIX_FROM_FLOQUET.md` with the original
text preserved.

1. **§7's acceptance test was calibrated against an artifact.** Cross-sign
   hypothesis separation at mirror-plane angles came entirely from the
   reference file's C4v-violation noise; on a physical C4v cell it is 4e-16,
   and real CST data confirms it (cross-pol 9.8e-5 at (0,0)). Replaced by a
   pooled χ² discriminant at (60°,22.5°). A fixed-tolerance max test is also
   powerless once σ ≥ 1e-2 (its usable window closes entirely).
2. **The 8-hypothesis label family could not contain the truth.** The live
   acceptance refused with χ²_red 658. Cause: the S11 block carries a −1 on
   its TM *receive* row (§2's own `ê_TM^(r)` convention), and **no port-mode
   sign can express it** — a mode sign multiplies `S11[a,b]` by `s_a·s_b`,
   leaving the co-pol diagonal invariant. Extended to 16 members; χ²_red
   dropped to 1.595. The fail-safe worked exactly as designed.
3. **§4's Born seed is not sufficient**, and §4's deferred "frequency
   smoothness as a constraint" is the identified next lever.
4. **§3's "φ=22.5° gives 8 observables not 4"** is true as a count but weak as
   information for this cell (extra observables 8.7e-5…5.4e-3, at or below σ).
5. **§6.2/§6.3's gates** fail for four separable reasons, only one of which is
   an information limit.

## 4. Residual risk and what was not measured

- **The label map's `s21_cross` rests on a thin margin.** Pooling χ² over all
  four φ=22.5° angles raises z from 5.77 to **6.95–7.36** (χ²_red 1.03–1.16
  over 1568 observables, winner unchanged at every pooling stage) — but it
  **still refuses at 2σ**. Pooling does not rescue it; the mirror-plane
  argument does (a cross-sign error flips entries that vanish identically at
  φ ∈ {0°,45°}, so it is harmless exactly where it is unmeasurable).
- Only Zmax modes were excited, so cross-port transmission reciprocity is
  untestable with this campaign.
- Every held-out angle lies inside the same 13-angle grid; the θ=20°/40°
  treams-validation angles were never solved in CST.
- σ(f) varies 2.31e-3…3.62e-3 but the fit uses the single band RMS.
- The comparison target is the same reference tmat.h5 for both routes, so any
  error shared by both — the lmax=3 truncation above all — is invisible here.
## 5. Full-band synthetic study (all 49 frequencies)

`python -m tmatrix.retrieval.synthetic_test --freqs all` completed 49/49.
The doc-literal §6.2/§6.3 gates fail band-wide, as the two smoke frequencies
predicted and for the reasons `RETRIEVAL_LIMIT.md` decomposes.

**Machinery gates: 195 of 196 pass.** The single failure is worth stating
precisely rather than rounding away:

> `step3 machinery: trial errors match linear theory` FAILS at **ifreq 39
> only** (λ = 10.25 µm): median measured/predicted rms **3.39** against a
> [1/3, 3] window, range 0.33 … 49.87 over 23 entries.

It is **not** a code defect, and two independent observations rule out the
obvious alternatives:

- *Not basin hopping across trials.* Re-running 30 seeded trials at ifreq 39
  gives a unimodal objective distribution, 94.5 … 125.6 with median **106.5**
  against the expected E ≈ 104 — i.e. χ² is exactly right — and a max/median
  entry-error ratio of only 1.6 (neighbours ifreq 38/40: 1.1 and 1.4). The
  trials agree with each other.
- *The gate's own premise fails.* The check linearizes the regularized
  estimator **at the truth**. At ifreq 39 the fit sits 21.8 band-peaks away
  from the truth, far outside any linear neighbourhood, so a linearization
  there cannot describe the deviation. The failure direction — measured
  **larger** than predicted — is what nonlinearity produces.

So the honest reading is that this gate certifies the estimator/noise-model/
Jacobian chain *only while the estimator stays near the truth*, and at one
frequency in 49 it does not. That is one more manifestation of the landscape
limit, not a new problem.

---

Every number above rests on the full 49-frequency real campaign and, for the
synthetic studies, on all 49 frequencies except where a smoke-frequency pair
(ifreq 32 / 48) is named explicitly.

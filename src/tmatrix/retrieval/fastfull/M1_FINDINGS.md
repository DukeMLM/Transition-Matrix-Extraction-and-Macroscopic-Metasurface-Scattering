# M1 findings — what the rank/SNR/cost designer says before any CST run

Companion to the auto-generated numbers in
`retrieval/results/fastfull/M1_DESIGN_STUDY.md` (regenerate with
`python -m tmatrix.retrieval.fastfull.m1_study --samples 700`). Milestone M1 of
`FAST_FULL_TMATRIX_WHEEL_PROPOSAL.md`.

**Revision 2 (2026-08-07), after the external review logged in
`review.md`.** Three of the four findings below changed as a result; §2 and
§5 reversed. See §8 for what was withdrawn and why.

All numbers are at the **frequency-matched measured** complex-S discrepancy
from `retrieval/results/fit_sigma_from_closure.npz` (2.8417e-3 at 8 µm,
3.1751e-3 at 20 µm), with the C = 0 screening Jacobian, and the reference
wheel used only for benchmarking — never for design.

---

## 0. The error model, and why every number is a bracket

The whitening level σ is the campaign's normal-incidence **closure**
residual. `retrieval/results/REAL_RETRIEVAL.md` §4.3 establishes that this
residual is dominated by *model* error, not CST numerical noise: "because the
dominant error is systematic rather than i.i.d. Gaussian, every χ²
significance
quoted … is *indicative*."

That matters because a coded cell has hundreds of modal observables
(576 for the 6-order winner, 28224 for the proposal's seed cell). An iid
model lets the discrepancy average down like 1/√n_obs — a factor of 6.8 for
the winner, 168 for the seed. A systematic discrepancy delivers none of it.
Every recovery figure is therefore reported as a **bracket**:

| model | assumption | formula |
|---|---|---|
| iid (optimistic) | errors independent across modal entries | `E‖δT‖²_F = tr(Cov)` |
| systematic (conservative) | one deterministic error vector of per-entry RMS σ, worst-case direction | `‖δT‖_F ≤ √(n_obs · λ_max(Cov))` |

Neither end is the answer: the iid figure assumes an averaging gain that is
not available, and the systematic figure assumes the discrepancy aligns with
the single worst-conditioned direction of `H⁺`, which nothing suggests. The
true value depends on the *structure* of the discrepancy, which is unmeasured
until M3's per-channel covariance study. **M1 makes no Gate E claim.**

## 1. The proposal's par. 6 seed cell is beaten decisively by a much smaller one

| | par. 6 seed | `small@8` |
|---|---:|---:|
| cell | 26.0 × 33.8 µm (878.8 µm²) | 10.81 × 7.24 µm, γ 92.75°, α 19.68° (78.2 µm²) |
| Bloch (fractional) | (0.090, −0.460) | (−0.0098, −0.4930) |
| propagating orders @8 µm | 42 | 6 |
| channels | 168 | 24 |
| wheel rank | 40/40 | 40/40 |
| σ₄₀ (whitened) | 305.5 | **523.8** |
| scattered signal / σ | 2.1 | **20.3** |
| predicted global δT (sys / iid) | 482 % / 10.9 % | **40 % / 5.9 %** |
| cost proxy | 5471 min, 172 RHS | **78 min, 28 RHS** |

A 6-order cell 11× smaller in area gives 1.7× better worst-direction
conditioning, 10× more signal, and 70× less compute. The gap widened under
the corrected error model: the seed cell's 168 channels are a *liability*
once the discrepancy stops averaging down.

The one model-independent number here is **signal / σ = 20.3**: the winner's
scattered modal amplitudes stand 20× above the measured discrepancy. The
physics is present; what M1 cannot yet settle is how that discrepancy
propagates into T.

## 2. Gate E is NOT cleared — at either wavelength, under either model

| | global δT (sys / iid) | worst dominant block (sys / iid) | target |
|---|---:|---:|---:|
| `small@8` | 40.2 % / 5.9 % | 97.6 % / 4.9 % | 5 % / 2 % |
| `small@20` | 800.6 % / 117.0 % | 1579.9 % / 79.3 % | 5 % / 2 % |

Even the optimistic end fails at 8 µm (5.88 % > 5 %), and the conservative
end fails by 8×. At 20 µm every configuration fails by a wide margin under
both models — max|T| falls 18× (0.07807 → 0.00431) while the discrepancy
floor does not move.

The dominant blocks are the invariant multipole blocks **E1←E1** and
**M1←M1** (electric and magnetic dipole), identified by their share of
‖T‖_F, not by coordinates of the symmetry basis.

## 3. The generic algebraic track is dominated for this wheel

`generic@8` needs 48 channels and 341 min to reach σ₄₀ = 416 — worse than
`small@8`'s 524 at 78 min, and its larger observable count costs it further
under the systematic model (101 % vs 40 % global). The wheel track needs only
24 channels; at 24 channels the generic branch is *impossible*
(rank(A) = 24 < 30), which is the proposal's par. 3 point that the generic
gate is the stronger one. Inequality (D1), σ₄₀(H) ≥ σ₃₀(W)·σ₃₀(A), is gated
in `tests/retrieval/test_fastfull_design.py`.

The generic track keeps its value as an object-independent correctness
reference — branch G of par. 9.3 runs exactly, noise-free, in gate (d) — not
as the faster route.

## 4. Two structural facts the proposal does not state

**(a) The Wood-margin rule caps the usable cell area.** Par. 7.2 requires
every order — propagating *or* evanescent — to stay a fixed margin from
|q| = k. The expected number of orders inside that forbidden annulus is

```
N_ann ≈ margin · k² · A_cell / (2π)
```

so the acceptance probability of a random Bloch vector decays like
exp(−N_ann). At λ = 8 µm with margin 0.05 that is ~1 forbidden order at
A_cell ≈ 200 µm² and ~5 at 1000 µm². Measured hit rates for uniformly
sampled Bloch vectors were 45/200, 6/200 and 1/200 for the 2–6, 6–20 and
8–24 order configurations, matching the estimate. Consequences:

* the design search cannot sample (f1, f2) blindly; `design.best_bloch`
  optimizes the Bloch vector per cell, which raises the generic-config hit
  rate from 0.5 % to 38 %;
* **the par. 6 seed cell violates the proposal's own par. 7.2 constraint at
  8 µm**: its Wood margin there is 0.008, 6× inside the declared 0.05. It
  satisfies the rule at 20 µm, where it was designed.

**(b) No single fixed cell serves both ends of the band.** The 8 µm winners
have *no* propagating orders at 20 µm (they are subwavelength there), and the
20 µm winners are the exact 2.5× rescalings of the 8 µm ones — the whitened
operator is λ-scale invariant, gated in
`tests/retrieval/test_fastfull_design.py`. Par. 11's overlapping-subband
strategy is mandatory, not a refinement.

## 5. Pooling encodings does not help

Under the systematic model, pooling *hurts*: global δT goes 40.2 % (1 cell)
→ 46.4 % (2) → 52.0 % (3), while cost rises 78 → 105 → 146 min. Extra
encodings raise σ₄₀ more slowly than they raise √n_obs, so the worst-case
bound degrades. Only under the iid model does pooling look mildly favourable
(5.9 % → 6.0 % → 6.1 %, i.e. flat).

This reverses revision 1, which used the iid model and concluded pooling gave
a 1.5× conditioning gain. The proposal's par. 8.5 hypothesis — "8–12 channels
pooled across two or three encodings" — is not supported by M1 for this
wheel; a single 24-channel cell is better on every axis measured here.

## 6. The speed claim is not yet supported

Under the cost proxy the conventional isolated-particle benchmark
(46 illuminations × 64 s ≈ 49 min) is *cheaper* than the best coded cell
(78 min). Two reasons not to conclude anything from that yet:

* the 64 s per conventional run is borrowed from the campaign's 2 × 2 µm
  periodic solve; a genuine isolated-particle project needs open boundaries,
  a larger domain and near-field monitors, and is certainly slower;
* the proxy's exponents are anchored on one measured point and extrapolated
  ~100× in degrees of freedom.

Calibrating both sides is M3/M5 work. The honest M1 statement is that the
coded cell is now in the same order of magnitude as the benchmark, which the
par. 6 seed cell (5471 min) was not.

## 7. What M1 does and does not establish

**Does.**
* The analytic measurement operator is *consistent with the validated
  single-order model*: flux-normalized `W T_eff A + S_empty` reproduces
  `sparams_oblique`'s Jones blocks to 3e-16 in both illumination directions,
  and the CST TE/TM gauge reproduces that module's θ→0 mapping table entry
  for entry.
* The 40-coefficient D4h ⊗ reciprocity basis is right: ranks 58 and 40
  matching an independent character calculation, sector table matching
  par. 3.1, σ_h derived numerically and in closed form.
* A 6-order 78 µm² cell reaches rank 40/40 and stands 20× above the measured
  discrepancy in raw signal, at 1/70 of the seed cell's cost.
* The par. 7 design problem is solvable, and its solutions are far better
  than the published seed dimensions.

**Does not.**
* **No CST convention is validated.** The gates compare the new transforms
  against the repository's existing *analytic* single-order model. They say
  nothing about CST multimode port fields, per-mode gauge and labels,
  reference-plane phases, or a diffractive cell. Those belong to Gate B at
  M3 and are untouched.
* **No Gate A or Gate E verdict.** The error bracket is too wide to certify
  either, and the C = 0 Jacobian is a screening statistic (see
  `jacobian.py`'s scope note).
* **Information, not recoverability.** The previous project phase found that
  the information was present and no realizable seed reached it
  (`RETRIEVAL_LIMIT.md`); nothing here contradicts or repeats that. Two
  reasons for cautious optimism, both to be tested at M2: at C = 0 the
  wheel-track problem is complex-**linear** in the 40 coefficients, so there
  is no basin to miss (gate (e) recovers all 40 exactly from one
  least-squares solve); and the winning cell's neighbours sit 7–11 µm away
  versus the campaign's 2 µm, so the lattice dressing that makes the problem
  nonlinear is far weaker.

## 8. Changed in revision 2

| revision 1 claim | status |
|---|---|
| "clears Gate E's initial targets at 8 µm (5 % global / 1 % dominant)" | **withdrawn.** The 5 % rested on an iid averaging gain of 6.8× that a systematic discrepancy does not deliver; the unrounded iid value is 5.88 %, itself a fail. Bracket is now 5.9–40 %. |
| "dominant-block error 1 %" over coefficients within a decade of the largest | **withdrawn.** That statistic lived in the arbitrary eigenbasis of a degenerate projector and moved by ~4× under a rotation of the basis. Replaced by invariant multipole blocks; the honest figure for E1←E1 / M1←M1 is 4.9 % (iid) / 97.6 % (systematic). |
| "pooling 3 encodings improves the global figure to 4 %" | **reversed.** Under the systematic model pooling degrades it (40 → 52 %). |
| "the flux-normalized measurement operator is correct" | **narrowed** to consistency with the existing single-order analytic model; CST multimode correctness is Gate B at M3. |
| band-RMS σ = 2.6333e-3 everywhere | **replaced** by the frequency-matched spectrum (2.8417e-3 at 8 µm, 3.1751e-3 at 20 µm). |

Two permanent gates were added so none of these can silently return:
`tests/retrieval/test_fastfull_design.py` (k) checks that every reported
recovery number is invariant under a random orthogonal rotation of the
40-dimensional basis
(holds to 1.4e-15), and (k) checks that the systematic bound is the exact
worst case and brackets the iid figure.

## 9. M2 progress — the C = 0 caveat is discharged

`tmatrix.retrieval.fastfull.ewald` now supplies a converged lattice coupling
for exactly the cells `coupling.py` refuses (14 gates in
`tests/retrieval/test_fastfull_ewald.py`).

**The lattice-sum operator agrees between two implementations** — but Gate D is NOT closed; see the README header for why the separate
`tmatrix.aggregation` repo-vs-treams discrepancy has to be resolved in a common
basis first. `treams.sw.translate_periodic` reproduces the
repository's convention with no transpose, conjugation, polarization flip or
Bloch-sign change, and agrees with the *campaign's own* tapered C — the one
built with the normative per-θ taper scaling — to 2.9e-8 … 1.2e-6 relative
over six (frequency, angle) pairs. Driven through the forward map the two
implementations differ by 5.3e-7 in complex S, i.e. **5000× below the
measured CST-vs-model discrepancy**. Ewald also runs in ~10 ms against the
tapered sum's ~7 s, is eta-independent to 2.7e-14 over a safe bracket, and
refuses (rather than silently degrading) outside it.

**And the C = 0 screening verdict survives.** With the real coupling, over a
passive D4h ensemble:

| design | ‖C T‖ | σ_min(I + T_eff C) | σ₄₀ ratio | global δT ratio | rank |
|---|---:|---:|---:|---:|---:|
| `small@8` | 0.167 | 0.905 | 0.983 | 1.0014 | 40 → 40 |
| `medium@8` | 0.128 | 0.923 | 0.948 | 1.0035 | 40 → 40 |
| `small@20` | 0.010 | 0.994 | 0.999 | 1.0001 | 40 → 40 |

Nothing in §1–§6 moves by more than 5 % in σ₄₀ or 0.4 % in predicted error.
The reason is physical: a coding cell is large, so its lattice coupling is
weak — ‖C T‖ = 0.167 against **5.1** for the campaign's 2 µm cell. The wheel
problem on a coding cell is therefore close to Born and close to *linear*,
which is a materially different regime from the one in which the previous
project phase's optimization landscape defeated blind retrieval
(`RETRIEVAL_LIMIT.md`). Gate D's error-amplification worry also does not
bite: σ_min(I + T_eff C) ≥ 0.90, cond ≤ 1.17.

## 10. Gate A — blind noisy recovery, and the bracket closed empirically

`tmatrix.retrieval.fastfull.synthetic` + `gate_a_run.py` (10 gates in
`tests/retrieval/test_fastfull_synthetic.py`; numbers in
`retrieval/results/fastfull/GATE_A_STUDY.md`). A known T0 is synthesized
through the real Ewald C, perturbed, and recovered **blind** — C = 0 linear
seed,
continuation in the coupling, Levenberg-Marquardt with the analytic
Jacobian, no mask and no oracle.

**The recovery machinery is exact and the basin is unique.**

| control | value |
|---|---|
| noise-free wheel recovery, exactly-D4h target | 1.8e-16 |
| noise-free generic algebraic branch | 1.8e-15 |
| multistart spread over 5 perturbed seeds | ≤ 2e-16 |

The 3.9e-3 residual seen when the *reference file* is the target is that
file's own D4h violation (5.96e-3), not recovery error — which is why the
control uses an exactly-symmetric draw. **Basin uniqueness is the sharpest
contrast with the specular phase**, where no realizable seed reached the
physical basin (`RETRIEVAL_LIMIT.md`); here every seed lands in the same
place, because a coding cell is near-linear (§9).

**Where structured discrepancies land in the §0 bracket** (0 = iid end,
1 = adversarial end), global δT at σ = 2.8417e-3:

| error model | `small@8` | `generic@8` | `pool-3@8` | bracket position |
|---|---:|---:|---:|---:|
| iid | 5.1 % | 7.1 % | 6.7 % | −0.02 … +0.01 |
| TE/TM mode mixing | 4.3 % | 11.2 % | 6.3 % | −0.05 … +0.05 |
| smooth angular (mesh) | 19.9 % | 47.1 % | 27.3 % | +0.41 … +0.44 |
| port reference plane | **24.0 %** | 53.3 % | 26.6 % | +0.42 … +0.53 |
| adversarial | 40.6 % | 95.7 % | 49.9 % | +0.91 … +1.01 |

Three things follow, and none of them was visible from M1 alone.

1. **The bracket is real but not symmetric in risk.** The iid and adversarial
   ends are both attained exactly (validating the M1 predictions), but
   realistic errors do not sit in the middle uniformly: gauge/label mixing is
   essentially harmless, while a **port-plane offset and smooth angular
   error land at 0.4–0.5** and dominate. The effective error for the best
   design is ~24 %, roughly 4× the iid estimate, not 7×.
2. **That names what M3 must control.** The limiting classes are the
   reference plane and slow angular (mesh) variation — exactly the two the
   reviewer's recommendation 5 proposes measuring by running the empty cell
   at multiple mesh levels and two reference-plane locations. Label/gauge
   hypotheses, which cost the specular campaign the most effort, are the
   *least* damaging class here.
3. **Gate A does not pass at the measured discrepancy.** 24 % against Gate
   E's 5 % target needs ~5× better σ *in the reference-plane class
   specifically* — a calibration problem, not an identifiability one.

**Candidate ranking is unchanged and now rests on recovery, not prediction:**
`small@8` is best on every error model, `generic@8` is ~2× worse (its 2304
observables inflate systematic exposure), `pool-3@8` is worse than the single
cell on every structured model. Pooling remains unsupported.

**Cell independence.** `small@8`'s blindly recovered T0 predicts a
geometrically distinct held-out encoding (9.4 × 12.7 µm, γ 84°, α 61°, a
different Bloch point, never used in any fit) to max |δS| = 4.68e-3 = 1.65 σ
on a signal of 1.96e-2. `generic@8`'s predicts it to 3.68 σ.

## 11. Nuisance-marginalized design, and why the estimator had to change too

Following reviewer recommendation 6 (`review.md`, 2026-08-07 15:45).
`tmatrix.retrieval.fastfull.nuisance` builds the joint Fisher information of
T and the declared calibration parameters and reports its Schur complement
`F_T = J_c^T (I − P_η) J_c`; `tmatrix.retrieval.fastfull.opt_marginalized`
optimizes against it. Four results, in the order they were forced on me.

**(a) The tangent audit reproduces independently.** For `small@8`: the
port-plane tangent has 99.979 % of its norm inside col(H) (reviewer: 99.981 %),
leaving 2.05 % distinguishable, and induces 23.92 % apparent T error
(reviewer/Gate A: 23.9–24.0 %). Extending it to classes Gate A could not
represent:

| class | params | max projection | smallest principal angle | worst δT | (leading-vector) |
|---|---:|---:|---:|---:|---:|
| phase_tx / rx | 24 each | 99.9887 % | 0.86° | **24.4 %** | 6.7 % |
| angular_tx / rx | 7 each | 99.9885 % | 0.87° | **24.2 %** | 21.2 % |
| ref_plane | 1 | 99.9790 % | 1.17° | **23.9 %** | 23.9 % |
| tm_row | 1 | 74.289 % | 42.0° | 14.7 % | 14.7 % |
| mix_tx / rx | 12 each | 34.781 % | 69.6° | 8.9 % | 2.3 % |

**Corrected after review (2026-08-08).** The first version of this table took
the leading left singular vector of each family, which maximizes output norm
per unit parameter and says nothing about collinearity with T. The
constrained worst member at fixed output RMS is what matters, and it changes
the conclusion: the per-channel **phase** families are as damaging as the
port plane (24.4 % vs 23.9 %), not 3.6× less. Five families sit at ~24 %
with principal angles under 1.2°. The reviewer's independently computed
numbers (24.37/24.38 %, 8.85/8.86 %, 99.989 %) are reproduced exactly.

**This retracts a §10 claim.** I wrote that label/gauge error is the least
damaging class. That rested on `mode_mixing`, a congruence `S → M S Mᵀ`,
which — as the reviewer pointed out — cannot express the campaign's actual
fault, a *receive-only* TM-row defect. Represented properly as `tm_row`, the
gauge class costs 14.7 %, third largest. The congruence-like `mix_*` members
really are benign (2.3 %); the receive-only member is not.

**(b) Optimizing the marginalized objective without a stability constraint is
gamed.** The first search returned a 3.5 µm-pitch cell with the best ratio
(loss 2.1× vs 33.3×) and a *catastrophic* Gate A recovery, 284 % against the
incumbent's 5.7 %. It sat at ‖C T‖ = 0.89–0.95 — a collective resonance,
where the Jacobian is huge for the ensemble draws and collapses 100× at a
different T. The constraints recommendation 4 specified — ‖C T‖ ≤ 0.5,
σ_min(I + T_eff C) ≥ 0.5, signal/σ ≥ 3 — are now `Constraints` fields and the
rejection is gated.

**They reject that one pathology; the ensemble itself was a second one, now
fixed.** The original passive generator, `S = (1−t)I + t·Ŷ`, gives every
draw the same dominant `−I` component. Measured over six draws at
‖T‖_F = 0.25: identity cosine 0.884–0.906, pairwise 0.743–0.830,
participation-ratio effective rank **1.44 of 40**, against the reference
wheel's identity cosine of 0.208. The "ensemble" was a one-parameter family
wearing 40 coordinates.

`symmetry.random_passive_d4h_cayley` replaces it, using the bounded-real
correspondence `S = (I−K)(I+K)⁻¹`, `T = −K(I+K)⁻¹`, which is exactly passive
whenever `Herm(K) ⪰ 0` and stays in the subspace because V is closed under
conjugate transpose and under analytic functions of a *single* element. A
`loss_factor` knob scales the absorptive part of K, so a low-loss resonant
draw — the wheel's own regime — has a small identity component. Measured at
loss factor 0.05: identity cosine 0.251, pairwise 0.019–0.305, effective
rank **5.51** (of a maximum 6 for six draws), still exactly passive and
exactly D4h.

**The first version of that generator was still not exact.** It applied the
Cayley map and *then* rescaled T to the requested Frobenius norm, which
leaves the bounded-real manifold and injects absorption nobody asked for: at
`loss_factor = 0`, where S must be unitary, it gave ‖SᴴS − I‖₂ = 0.043 and
mean absorption 0.0075 against the wheel's 0.00055 — invisible to a
`max SV(S) ≤ 1` check because the attenuation sits in the other singular
directions. The scale of K is now root-solved *before* the map. At zero loss
the draw is now unitary to 1e-16 with absorption −0.0e0, and absorption rises
monotonically with the knob (0 → 0.0069 → 0.049 at 0 / 0.05 / 0.5). The
wheel sits near `loss_factor ≈ 0.005`. `symmetry.absorption_spectrum`
reports the full singular spectrum so this cannot hide again.

**(c) The apparent gain was mostly the degenerate ensemble, and the optimum
still does not transfer.** Three successive runs, each correcting the
previous one's defect:

| ensemble | `small@8` σ_marg | winner σ_marg | gain |
|---|---:|---:|---:|
| norm from ‖T_ref‖, convex generator | 13.50 | 17.77 | 31.6 % |
| declared norm 0.25, convex generator | 27.12 | 34.87 | 28.6 % |
| **declared norm, Cayley loss grid (eff. rank 5.08)** | **33.74** | **34.38** | **1.9 %** |

**The 28.6 % gain is withdrawn.** Once the ensemble spans more than one
direction, the marginalized objective finds essentially nothing better than
the incumbent, and it selects a *different* cell each time the ensemble
changes — the optimum was tracking the generator, not the physics.

The third run's winner (10.86 × 7.10 µm, 20 channels, 77 µm², 70 min) is
still worse on the reference wheel in blind recovery for four of five error
models: iid 9.60 vs 5.72, mode mixing 12.40 vs 4.48, angular 18.74 vs 15.05,
adversarial 70.08 vs 40.59 %. It is better only on the class it was
implicitly tuned for (reference plane 21.88 vs 23.99 %).

**And a stochastic restart is not a selector.** `search` resamples and never
reconsiders earlier points, so a run can label a cell "winner" while a
previously proposed cell scores higher on the *same* ensemble. It did: the
04:58 run reported a cell at penalized objective 5.637 while the archived
04:24 cell scored 5.924. `opt_marginalized` now carries an explicit
`ARCHIVE`, re-evaluates every past candidate under the current ensemble, and
reports a leaderboard. On the next run the fresh search scored **4.174**
against the archived candidate's **5.805** — 39 % worse than a point already
known — and the archive correctly overrode it.

The leaderboard on that ensemble: `cand-0458` 5.805, `cand-0424` 5.488,
`small@8` 5.386, fresh search 4.174, `generic@8` 0.418. So an archived cell
does beat `small@8` by 7.8 % on the training objective — and still loses on
the wheel in four of five recovery classes. **The training objective and
blind recovery disagree, consistently, in every run so far.**

**That leaderboard was itself corrupted by hand-transcription.** The archive
entries were typed from printed, rounded log lines; the true 04:24 candidate
had α = 63.4185055507°, not the 56.22° transcribed. On the exact-Cayley
ensemble the true point scores 5.828 against the malformed 5.488 — the
transcription suppressed a known point by 5.84 % and reversed the ordering.
The exact coordinates are **not recoverable** from this repository, because
the artifact that held them was overwritten by the next run (see below).
`opt_marginalized` now records every candidate at full precision inside its
own run directory; the transcribed entries survive only under names ending
`(rounded)` so they can never again be mistaken for the real points.

*(Superseded three times since first written. The append-only
`candidate_registry.json` described here was replaced first by scanning
completed run directories — an append-only file has a read-modify-write race
and can lose candidates — then by manifest-verified admission, because a
`complete` marker turned out to be no evidence at all (a directory with no
manifest, an arbitrary marker string and a foreign run id was accepted), and
then again because that manifest check iterated the manifest's own artifact
list, so an **empty** artifact map passed vacuously: a directory with no
`result.json`, no snapshot or config body and arbitrary hash strings was still
admitted. `candidate_registry.json` is now derived, non-selecting metadata
rebuilt from verified runs, and `append_registry` raises. Selection reads run
directories through `verify_completed_run`, which **recomputes** every hash,
**rederives** the run id from the stored snapshot and config bodies, requires
exactly `{result.json, candidates.json}`, and **derives lineage from the
hashed config** rather than from any label — so relabelling a run cannot move
it between lineages, and changing the fact changes its identity.)*

**The archive is a selector input, and for several rounds it was not part of
the run's identity.** An archived cell is allowed to beat the fresh search and
become the selected winner, but the archive was loaded *after* the run id was
computed — so one run id could publish different winners depending on which
other runs happened to finish first. Worse, `run(out_dir=...)` called a bare
`load_registry()`, reading the repository-global runs directory: a temporary
or independent campaign would silently select global candidates while writing
elsewhere. The archive is now frozen once at entry from the requested
namespace and its fingerprint is hashed into the run id.

**Hashing the archive made a scientific violation reproducible without making
it eligible.** The frozen archive initially had no lineage filter, so a
target-conditioned candidate entered the selector even under
`TARGET_CONDITIONED_PRIOR=False`; and because the selected record was stamped
with the *current* lineage, a conditioned proposal that won would have been
republished as independent — laundering it across the evidence boundary the
proposal (par. 7.3) draws. Records now carry a `proposal_lineage` distinct
from the evaluation `lineage`; independent runs admit only independent
proposals; and an attempted promotion raises rather than being relabelled.

**Calling that lineage "immutable" was itself premature, twice.** For one
round it was a self-declared label: mandatory nowhere, silently inferred from
the manifest when absent, and accepted as written when present — so a
conditioned `search` record relabelled `target_independent` verified and was
exposed to an independent selector. The fix for *that* was also overclaimed: a
`proposal_proof` whose `design_key` merely agreed with the record's own design
is a self-consistency check the record can always satisfy, nothing resolved
the named parent, and same-lineage selections needed no proof at all — so a
selected record could carry a geometry unrelated to its own run's search, its
polish shortlist, or the archive. **Every** selected record now resolves
through `resolve_selected_parent` against one of four sources — a same-run
`search`/`polish` record, an archive entry, a declared incumbent, or a
labelled transcription — matching on canonical geometry and proposal lineage.
A run refuses to publish a selection whose source cannot be shown.

**That was still a third overclaim, because the archive branch resolved
against the child's own config.** A hash over a claim makes the claim
immutable, not true: a self-consistent artifact whose persisted archive body
named `first_run=does-not-exist`, with a ghost record and no parent directory,
verified and contributed its geometry to an independent freeze. Archive
citations are now walked to a **fresh search/polish root** — each hop opens
the named run directory, verifies its manifest and artifact hashes, finds the
named record and requires identical canonical geometry and proposal lineage,
with cycles and runaway depth refused.

**The strongest evidence label is `error-screen-passed`, not `gate-passed`,
and the latter is not a value the code can emit.** A verdict function that
iterated whatever candidate and model names a result happened to supply, and
tested two error thresholds, returned "passed" for a one-cell report naming an
invented model with errors −1 and −2. The screen now requires the *declared*
candidate set (pinned in the hashed config) and the *declared* perturbation
families, with every field present, finite and non-negative. But it still does
not establish proposal Gate A: `GATE_A_UNVERIFIED` records in code that
rank-40 identifiability, useful-direction SNR above 10, noise-free and basin
stability, passivity, and frozen trial/holdout identities are **not** checked,
because the saved report does not carry them.

**A run id names inputs; an output root names what was published.** The id
hashes the snapshot and config, and the artifact digests were mutable labels
inside a manifest that nothing anchored — so rewriting every candidate
geometry and refreshing the two labels verified under the identical id.
`output_root` is recomputed from the artifact bytes plus the manifest with its
own labels removed, written into the completion marker and appended to
`runs/receipts.jsonl`, and all three must agree. This is **tamper-evidence,
not tamper-proofing**: anyone who can rewrite the artifacts can rewrite the
marker and the receipt too. What it buys is that a partial edit cannot pass
and that a child citing a parent binds the parent's exact bytes.

**The first receipt design failed open and could poison its own winner.** A
shared append-only JSONL keyed by run id was optional at verification, so
deleting it and re-signing the marker let a rewritten run verify under the
same input-derived id; it was an unlocked append, so one truncated fragment
made every run in the namespace unreadable; and because `result.json` carries
wall-clock `search_seconds`, two same-identity workers wrote two contradictory
rows and the rename winner stopped verifying. Receipts are now **one immutable
file per output root** (`runs/receipts/<root>.json`, write-once), mandatory
for any published run, and written before the rename — where they can no
longer conflict. Parent proofs additionally carry `parent_output_root` and
`parent_record_digest`, so a citation names the parent's exact bytes rather
than only its id and a record name.

**Both of those claims were premature when first written, in the same way.**
The digest fields existed and were populated by the production caller, but
`_proof` omitted them when a caller did not supply them and both resolvers
compared them *only when present* — so a citation could downgrade silently to
run-id/record-name semantics. And digest checking stopped after the FIRST hop:
the walker validated the incoming proof, cleared it, and never adopted the
intermediate parent's own proof, so a middle record could carry a wrong parent
root and record digest and still pass. Both are now mandatory per source
(`archive` needs both fields, `same_run` needs the record digest) and every
run-backed hop adopts and checks the next proof.

**A side effect worth recording: content addressing makes a published
provenance CYCLE unconstructible.** Once every hop must cite its parent's
output root, run A's root depends on its candidates, which would have to
contain B's root, which depends on A's. The `seen` guard in the walker is
therefore defence-in-depth on an unreachable path — it is checked at unit
level, while the *reachable* bound (depth) is gated with a real 19-hop chain
of published runs that stops on the hop budget in 0.28 s.

The receipt write was also described in code and in the response log as
`O_CREAT|O_EXCL` while actually being a check-then-`os.rename` that returned
success for any pre-existing path — so a stale or contradictory receipt for a
root was silently preserved and treated as ours. It is now an exclusive per-root
reservation, a temp body that is written and fsynced, and an atomic rename
into place; an existing file is parsed and must already match, or the publish
raises and the stage is quarantined.

The exclusive-create-then-write version was itself not crash-atomic: a crash
between the create and a complete close left a zero-byte receipt at the final
path, which read as unreadable and made every retry raise while preserving it
— a permanent wedge for that root. Unreadable residue is now reclaimed
automatically, but **only after proving that no VALID published run depends on
the root** — the marker check runs the same structural verification the
selector uses, with only the receipt exempted, so a forged partial directory
can no longer veto recovery indefinitely.

**Fixing the torn receipt moved the wedge rather than removing it.** A crash
between creating the exclusive reservation and unlinking it left the lock
forever, and acquisition retried without waiting, liveness checking or
reclamation. The reservation became a lease — and a
timestamp lease turned out not to be mutual exclusion. A freshly `O_EXCL`-
created lock is briefly EMPTY and was stolen as "age infinity"; read-then-
rename had an ABA race; nothing fenced the dispossessed owner, so a paused
publisher could resume and overwrite the thief with *both* returning success;
the orphan sweep deleted an active publisher's temp; an expired owner unlinked
its successor's lock; torn-receipt reclamation ran outside the reservation; and
`.stale.*` tombstones were never deleted while the gate reported "no
leftovers".

Acquisition now returns a **fencing token**, re-checked immediately before the
atomic replace and again on release, and *everything* — reading and repairing a
torn receipt, sweeping orphan temps — happens inside the reservation. Seven
crash boundaries are fault-injected and gated, each recovering with the receipt
directory holding nothing but the receipt; a fresh empty reservation is
respected while an abandoned one recovers past a grace; and a publisher that
loses its reservation mid-flight writes nothing, leaves no temp, and does not
unlink its successor. **Not claimed:** cross-process liveness. A lock naming
another live process past the lease will be stolen — fencing is what makes that
safe, not the lease being right.

Ownership is also receipt-independent. `verify_completed_run` takes
`receipt_mode` in `require` / `allow_missing` / `skip`, because two states were
not enough: the marker-authority check previously used `allow_missing`, which
exempts only an *absent* receipt, so in the one state it existed for — a
present but unreadable receipt — it returned False and a **foreign run id could
install itself over a valid published run's root**. `read_receipt` parses an exact
`{run_id, output_root}` schema and requires the stored root to equal both the
requested root and the filename.

**The walk itself then had to be fixed three times.** It stopped at any nested proof
whose `source` string read `incumbent` or `transcribed` and called that a
root without checking the named constant — so a parent whose own selected
record cited an invalid incumbent was rejected by full verification while a
*child* citing that record verified, and the child's geometry reached the
freeze while its parent was refused. The terminal checks are now single
functions used at every level, an intermediate archive hop must agree with
that parent's own archive body, and one cycle/depth budget spans the whole
resolved chain. The third fix: structural parent verification suppressed the
archive-invariant check along with the recursion, so an ancestor whose
fingerprint contradicted its own body failed full verification yet still
legitimised a child citing it — only the *descent* is suppressed now, and the
proof's own key and lineage are checked before any terminal name lookup.
**That descent suppression was itself inert for a round**: the flag was
threaded through three call layers and never consulted, so structural checks
re-walked the whole chain suffix and a two-run archive cycle recursed to
`RecursionError` instead of stopping at `MAX_PROVENANCE_DEPTH`. Exactly one
outer walker now owns the `seen` set and hop budget.

**The strongest status also had to be split.** `error-screen-passed` now
requires the version-1 protocol candidate set (`small@8`, `winner`); a screen
that passes over any other set receives `custom-screen-passed`, which carries
no production-protocol implication, and a config declaring no candidate set at
all is not a pass. The archive invariants
(`archive_sha256`, `archive_n`, `archive_lineages`, and each entry's design
against its own key) are recomputed rather than read off labels.

**Run identity carries the geometry epoch, not provenance.** Hashing the full
archive body reintroduced the churn deduplication was meant to close: adding
the same geometry under a lexicographically earlier run left the geometry
fingerprint unchanged but moved the retained name and `first_run`, so the next
invocation derived a new id and repeated the same deterministic search. The
provenance body now lives in the manifest — auditable, recomputed against the
fingerprint, outside the run id. The selector also stores canonicalized
`Design` objects rather than raw ones, and Bloch fractions are folded into the
half-open zone `[-0.5, 0.5)` alongside signed zero and alpha mod 360.

**The two incumbents are declared TARGET-CONDITIONED, and an independent run
inherits neither.** They previously entered the leaderboard as lineage-free
source constants, exempt from every filter, and a win would have stamped them
with whatever lineage the current run used. They were chosen by the M1 design
study with the reserved reference wheel in the loop, and no hash-bound proof
exists that their selection was independent of it — so `INCUMBENT_LINEAGE`
declares them conditioned. This is the conservative direction: it costs an
independent run two starting points rather than smuggling target-derived
geometry across the boundary. **Promoting either requires a rerun under a
declared-independent prior, not an edit to a constant.**

**The first version of archive-bound identity also never terminated.** Keying
the archive on record *name* meant every run republished the same
deterministic search geometry under a new run prefix, moving the fingerprint,
giving the next run a fresh id, and rerunning the identical search forever
while archive cost grew linearly. The archive is now keyed by canonical
full-precision **geometry** (`design_key`) and its fingerprint covers the
unique geometry set, not the runs that republished it: a run that discovers no
new geometry derives its predecessor's id and is refused. `archive_pin=`
pins an epoch explicitly. Still open: content-addressing the deterministic
search artifact so a new archive candidate does not rerun the optimization at
all.

**The training prior was also one to two orders too absorptive.** At the
reference norm ‖T‖_F = 0.113992 the 8 µm wheel has mean absorption
5.515e-4; the production `LOSS_GRID = (0.05, 0.15, 0.5)` produced 6.9e-3 to
4.9e-2 — 13× to 89× the wheel — so every completed search measured
performance on deliberately over-lossy random matrices. The grid is now
`(0.0025, 0.005, 0.01)`, which brackets the wheel at the reference norm
(3.5e-4 / 7.0e-4 / 1.4e-3). **No search has yet been run on it**, so every
candidate ranking above remains provisional.

**Two things about that recalibration must be said plainly.**

*It is target-conditioned, and therefore development-only.* The grid was
chosen so the draws bracket the **reference wheel's** measured absorption.
Par. 7.3 reserves that T for benchmarking and forbids using it to choose
priors, so this branch consumes part of the Gate-E ground truth and **cannot
close Gate A or Gate E**. The constant is flagged `TARGET_CONDITIONED_PRIOR`
in code. A genuinely independent anchor — from declared Au dispersion,
geometry-scale bounds and causal sector weights rather than from the answer —
is required before any candidate is selected on this basis, and a different
untouched T must then be reserved for validation. That work is not done.

*And even so the production ensemble is not wheel-matched.* The grid was
calibrated at ‖T‖_F = 0.113992 while the optimizer draws at
`ENSEMBLE_FRO = 0.25`, where the same grid gives 5.8e-4 … 3.0e-3, mean
**3.06×** the wheel. The gate now measures the exact production
configuration and reports that factor, rather than certifying a
configuration nothing runs.

**Independent fixed-cell evidence, on the corrected grid, favours the
incumbent.** Across the training seed plus three disjoint seeds, worst raw
information is 35.18 (`small@8`) / 30.22 (true 04:24) / 28.44
(`cand-0458`), and worst penalized objective 5.460 / 4.696 / 4.663. Moving
toward the physical-loss regime therefore *reverses* the old ranking and
makes `small@8` the robust fixed-cell incumbent — while still supplying no
useful-direction SNR, no finite-recovery improvement and no Gate-A pass.

**`small@8` therefore remains the operational incumbent**, now on stronger
evidence than before: it survives an ensemble correction that removed almost
all of its challenger's advantage. `generic@8` is 4.3× the cost with weaker
sheet-signal SNR and no recovery evidence.

That is not a failure of the objective; it is a mismatch between the
objective and the estimator. The marginalized information is the information
about T that survives *if the fit also estimates the calibration*. A T-only
fit does not, so it cannot collect it.

**(d) So the estimator was extended — and that is where the gain is.**
`synthetic.recover_joint` fits T and the calibration parameters together
(`nuisance.Calibration` gives the finite, not tangent, form). On the
reference-plane class:

| cell | T-only | joint | recovered δL vs true |
|---|---:|---:|---|
| `small@8` | 23.99 % | **1.16 %** | 0.4666 vs 0.4889 µm |
| marginalized winner | 24.36 % | **0.46 %** | 0.4945 vs 0.4908 µm |

and on the smooth-angular class 20.5 % → 1.8 % for `small@8`. **The dominant
error class is removable by estimating it, not by redesigning the cell** —
and here the marginalized winner does pay off, 2.5× better than the incumbent
and with a 10× more accurate offset estimate.

**But a free joint fit is not a free win.** The misspecification matrix
(injected class × fitted family, `small@8`, global δT):

| injected \ fitted | none | ref_plane | ref+ang | ref+ang+tm |
|---|---:|---:|---:|---:|
| iid | 5.40 % | 68.99 % | 130.64 % | 131.48 % |
| reference_plane | 23.99 % | **1.16 %** | 2.19 % | 2.75 % |
| angular_smooth | 20.54 % | 29.00 % | **2.19 %** | 3.16 % |
| mode_mixing | 4.66 % | 5.88 % | 100.00 % | 100.00 % |

Because the nuisance tangents are ~99.98 % collinear with T, giving them free
rein inflates variance enormously whenever the systematic they describe is
*not* present. And no single prior strength fixes it — the true offset is
δL ≈ 0.49 µm, so a prior tight enough to protect the iid case (sd ≤ 0.1 µm)
shrinks the correction away and leaves 23.7 %:

| injected \ prior sd | free | 1.0 | 0.3 | 0.1 | 0.03 |
|---|---:|---:|---:|---:|---:|
| iid | 130.6 % | 54.5 % | 13.1 % | 5.8 % | 5.4 % |
| reference_plane | 2.2 % | 15.5 % | 22.3 % | 23.7 % | 24.0 % |
| angular_smooth | 2.2 % | 3.2 % | 12.6 % | 21.6 % | 22.5 % |
| mode_mixing | 100.0 % | 182.2 % | 225.4 % | 4.6 % | 4.7 % |

**Conclusion — a model-matched conditional proof of concept, NOT a Gate E
result.** The ~0.5–2 % figures above are oracle-family fits: the same forward
model generated and fitted the data, the correct nuisance family was enabled
by hand, no nuisance distribution was measured, no calibrated positive
`Q_eta` produced them, and no locked cell or hardware data were involved.
**Gate E stays open.** What the experiment does establish is narrower and
still useful: *if* the dominant discrepancy is a port-plane offset *and* that
is known, the 24 % bias is removable — and *if* it is not, the same freedom
costs 24×. Calibration-model uncertainty currently dominates the error
budget; encoding performance remains conditional on the calibrated nuisance
distribution (the same experiment shows a 2.5× difference between cells, so
the encoding is not irrelevant).

Three further limits, all recorded rather than worked around:

* **Objective and estimator are not the same algorithm.** The objective
  removes 88 nuisance columns; `nuisance.Calibration` implements finite forms
  for only 16 of them (`ref_plane`, `tm_row`, angular rx/tx). The 72
  per-channel phase and per-order mixing columns are optimized against but
  not fittable — and the corrected audit puts the phase families at the same
  ~24 % level as the port plane, so this gap is material, not cosmetic.
* **Sharing a calibration parameter across cells is now explicit.** An
  earlier docstring claimed that passing one `Calibration` object to two
  blocks tied their parameters; it did not. `recover_joint` takes a
  `param_map`.
* **The marginalized score still assumes an iid remainder.** Dividing or not
  dividing by √n_obs presumes that projecting the declared tangents leaves
  something that averages. No calibration measurement demonstrates that yet,
  so the raw score is an optimistic likelihood screen, not an acceptance
  metric. Both `sigma_marg` and `sigma_marg_per_obs` are now emitted in the
  artifact so neither is privileged.
* **The reported loss is now the generalized worst inflation.** σ_free/σ_marg
  — the ratio of the two *smallest* singular values — is not a bound, because
  the weakest direction before and after marginalization need not coincide.
  The bound is sqrt of the largest eigenvalue of the pencil (F_free, F_marg);
  for `small@8` with all classes it is 46.4× where the ratio reads 149→ the
  ratio is kept only as `sigma_ratio`, a diagnostic.

## 12. Next actions, in order

1. **M3's calibration measurement, with a specific brief.** §11 shows the
   accuracy is set by how well the nuisance *distribution* is known.
   The corrected worst-direction table makes the per-channel phase, angular
   and port-plane families **similarly damaging** (24.4 / 24.2 / 23.9 %), so
   this ranking must not set the experiment order — a measured class
   covariance must. Measure all of: reference-plane shifts, a mesh ladder,
   and empty-cell/port-field repeats (which are what resolve the phase and
   receive-only gauge families at all). Report means AND covariances, as a
   **precision** `Q = Σ⁻¹` when handed to the code — passing a covariance
   inverts the regularization.
2. **Re-run the marginalized optimization with the measured Q_eta.**
   `nuisance.marginalized_information` already takes `q_eta`; with a measured
   prior the objective becomes well posed and the incumbent-vs-winner
   comparison becomes decidable. Until then the two are not separable on
   evidence.
3. **Deterministic, locked-holdout Gate A re-run.** The seed tree is now
   fixed and gated, but the reviewer is right that the current holdout is
   consumed development data and that the perturbation amplitudes are
   normalized hypotheticals rather than measured physics. Both need M3
   numbers before Gate A can be declared closed.

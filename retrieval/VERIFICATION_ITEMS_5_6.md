# Independent verification of retrieval items 5–6 (`tmatrix.retrieval.fit`, `tmatrix.retrieval.observability`, `tmatrix.retrieval.synthetic_test`)

Verifier pass, 2026-08-06. Every number below was produced by the verifier
re-running the shipped code and by an **independently written** check script
that re-derives its own references (it does not import the delivered test
files). Conventions used throughout: campaign 13 angles, `direction = -1`,
smoke frequencies ifreq 32 (λ = 12.00 µm) and ifreq 48 (λ = 8.00 µm).

---

## 1. Machinery — all independent checks PASS

| Check | Result |
|---|---|
| Cached `C` is bit-for-bit unchanged by `predict` / `fit` / `AnalyticJacobian` (17 angles) | PASS |
| `pack_S` applies `sqrt(w)` to **both** Re and Im; `objective == Σ wᵢ\|dSᵢ\|² == ‖weighted residual‖²` | PASS (agreement to 1e-12 relative) |
| `tmatrix.retrieval.fit` defaults: `direction = -1`, `t0 = None` (Born seed 0) | PASS |
| Analytic Jacobian vs an **O(h⁴) Richardson-extrapolated** central FD (independent of the shipped G2 test) | PASS — max column relative error **1.08e-11** (ifreq 32) / **5.41e-11** (ifreq 48) |
| Truth-seeded closed loop (`C_clean`) stays at the truth | PASS — objective **8.7e-31** / **2.6e-30**, entry error **3.1e-12** / **3.8e-12** |
| Shipped `tests/retrieval/test_fit_smoke.py` | 15/15 PASS |
| Shipped `python -m tmatrix.retrieval.synthetic_test --freqs 32,48` | all 6 machinery gates PASS, 574 s |
| Shipped `python -m tmatrix.retrieval.observability --ifreq 32 --basis bright --angles campaign` | reproduces: 20/20 SVs above λ, cond 2.23e4, `s_min` 2.11e-2 |

The analytic-Jacobian result is **five orders of magnitude tighter** than the
gate `tests/retrieval/test_fit_smoke.py` claims, because Richardson
extrapolation removes the FD truncation term that limited the shipped
comparison.

### A suspicion that was checked and found harmless

`observability.jacobian` uses central FD at `step_scale = 1e-6`. Measured
`‖J_FD − J_exact‖_F = 1.375` at ifreq 32 against `s_max = 4811`, so singular
values below ≈1.4 are FD artefact — three decades **above** the resolution
damping `λ = 2.12e-3`. This looked like it would inflate `res_c`. It does
not: recomputing the whole analysis with the exact analytic Jacobian gives
**identical** results — `n_above` 76 vs 76, `Σ res_c` 37.93 vs 37.93, dark
directions 8 vs 8, heatmap agreeing to 3e-4. The contaminated singular
values sit far below λ and are filtered out either way. No change required;
switching to the exact Jacobian would only be a speed/cleanliness win.

## 2. The doc §6.2 / §6.3 gate failures reproduce — and split into **two distinct** mechanisms

The previous session reported these as a single "information limit". That is
incomplete. The independent search separates them:

### (a) Model-error bias — the global minimum is **not** at the truth

A 16-start independent search (Born seed + truth seed + 14 random seeds at
the physical scale) on the noise-free `A_bright` problem finds:

| | ifreq 32 | ifreq 48 |
|---|---|---|
| best objective found | 3.313e-07 | 4.430e-07 |
| objective **at the truth** `T_bp` | 1.277e-05 | 2.759e-05 |
| ratio best/truth | **0.026** | **0.016** |
| bright-entry peak-normalized error at the best point | 8.58 | 23.3 |

The optimizer is not failing: it finds a point that fits the data **38×
(ifreq 32) / 62× (ifreq 48) better than the truth does**. The bright-span
model is genuinely biased, so no optimizer can pass the doc's gate on this
data. The shipped 3-start multistart already reaches the same objective
(3.313e-07), so the shipped protocol was not under-searching.

**Why:** the bright span omits sub-threshold entries whose S-leverage is
large. Off-bright residual magnitude in T is 1.13e-05 (ifreq 32) /
7.71e-05 (ifreq 48), which produces
`max|S(T_proj) − S(T_bp)| = 1.02e-03 / 1.17e-03`.

**Structural fact (new, verified to 3e-18):** the bright-span projection
`T_bp` agrees with `T_proj = P68(T_ref)` **exactly on every bright-mask
entry**. So the bright basis *can* represent the bright content perfectly —
the failure is estimation, never representability. This holds for any
threshold and should be asserted for any span used downstream.

### (b) A separate Born-seed landscape trap

On the `C_clean` loop the target lies exactly in the fitted span and the
truth **is** the global minimum (truth-seeded objective 1e-30). Yet a
16-start Born-blind search reaches only:

| | ifreq 32 | ifreq 48 |
|---|---|---|
| best objective (truth = 0) | 3.26e-08 | 9.79e-07 |
| bright-entry peak-normalized error | 1.45 | 49.7 |

So even with *zero* model error the Born basin does not contain the truth at
these frequencies. This is an independent failure mode from (a) and needs a
seeding fix (frequency continuation is the obvious candidate — the reference
T is smooth in frequency and the full 49×17 `C` cache exists).

### (c) The per-entry-relative gate is unattainable in principle for the weakest bright entries

Several bright entries have band-peak `|T| ~ 1e-4` (e.g. `1-1Mx2-1E`
9.10e-05, `2+1Ex1+1M` 1.01e-04, `2+0Ex2+0E` 1.35e-04). The doc's "≤ 1 %
relative" then demands ≈1e-6 absolute accuracy, while the bright-span model
error alone is 1.1e-5 in T and the noise-propagated resolution at σ = 3e-3 is
of the same order. No estimator can pass. Any recalibrated gate must be
normalized to a global scale (e.g. `|T|max`) and/or paired with the
estimator's own propagated error bar.

### (d) Structural darkness — confirmed exactly as the doc claims

`even-m` content at θ = 0: `|dS/dT| = 5.3e-21` (ifreq 32) / 3.3e-21 (ifreq
48) → 65.5 / 16.2 with oblique angles. The orbit-pure even-m sub-basis has
exact-Jacobian column norms 1.7e-14 (normal-only) vs 1.24e+02 (campaign).
The doc's θ = 0 visibility claims (i) and (ii) both PASS at both
frequencies.

## 3. The regularization protocol is itself a major error source

`tmatrix.retrieval.synthetic_test`'s step 3 uses an isotropic Tikhonov prior
`tik = 1/τ²`, `τ = ‖pack(T_bp)‖/√n_par`. Measured consequences:

- **The noise ladder is inverted** at ifreq 32: dipole error *falls* as σ
  rises (0.75 at σ = 1e-3, 0.64 at 3e-3, 0.31 at 1e-2). That is the
  signature of a **bias-dominated**, not noise-dominated, estimator.
- At σ = 3e-3 the strongest E-dipole diagonals come out at ~31 %
  peak-normalized error, while the noise-propagated resolution for that
  entry is ~0.3 % — a factor ~100 that is *not* information-limited.
- The linear shrinkage bias explains only part of it: decomposing the bias
  of `Re T[1-1Ex1-1E]` over the weighted-Jacobian SVD gives a total bias of
  5.28e-4 (0.68 % of that entry's band-peak) at ifreq 32 — the remainder is
  nonlinearity / wrong basin.

### Orbit-pure basis: a mathematical no-op (measured)

The 10 C4-conforming bright position orbits give an **exactly orthonormal**
basis (max|Gram − I| = 5.3e-17) with **zero pairwise support overlap**,
spanning the same real space as the SVD bright basis (residual 2.5e-15).
Because both are real-orthonormal bases of the same span, the isotropic
`‖t‖²` penalty is invariant, so the regularized estimate must be identical —
confirmed at ifreq 48 (`max|T_svd − T_orbit| = 5.7e-09` vs scale 2.7e-02).
At ifreq 32 the two runs differ (1.36e-02 vs scale 6.8e-03) **only because
they land in different local minima**, which is failure mode (b) again, not
a property of the basis. *Switching to an orbit-pure basis alone fixes
nothing.*

### What does help: an orbit-**scaled** prior

Diagonal prior `tik_k = 1/|z_k|²` per orbit direction (oracle-primed here;
a two-pass data-driven surrogate is the realizable version). Mean over 12
trials of the per-trial max peak-normalized entry error, σ = 3e-3, clean
loop:

| protocol | ifreq 32 all-25 / dipole / strong-E-dip | ifreq 48 all-25 / dipole / strong-E-dip |
|---|---|---|
| unregularized | 2.22e+01 / 2.55e+00 / 4.64e-01 | 8.17e+01 / 9.45e+00 / 2.60e+00 |
| isotropic (SVD basis) | 1.39e+01 / 7.08e-01 / 5.34e-01 | 4.83e+01 / 2.46e+00 / 1.73e+00 |
| isotropic (orbit basis) | 7.87e+00 / 5.78e-01 / 3.18e-01 | 4.70e+01 / 2.70e+00 / 1.80e+00 |
| **orbit-scaled** | **1.54e-01 / 3.94e-02 / 3.79e-02** | 7.33e+00 / 1.93e+00 / 1.93e+00 |

At ifreq 32 the orbit-scaled prior is a 90× improvement on all-25 and brings
the dipole to 3.9 %, i.e. it **meets the doc's §6.3 "dipole ≤ 5 %" gate**.
At ifreq 48 it helps much less — and ifreq 48 is also where the Born basin
fails worst, consistent with (b) being the binding constraint there.

## 4. Verdict

- Items 5–6 are **correct as engineering**: every machinery gate passes under
  independent re-derivation, including a Jacobian check five orders tighter
  than the shipped one.
- The reported §6.2/§6.3 failures are **real and reproduce exactly**, but the
  previous session's single-cause framing ("campaign-13 information limit")
  is **not supported**. The failures decompose into (a) bright-span model
  error, (b) a Born-seed landscape trap, (c) a gate normalization that is
  unattainable by construction for the weakest entries, and (d) genuine
  structural darkness. Only (d) — and part of (a) — is an information limit.
- Consequently the recalibration must change the **estimator and the gate
  normalization**, not only the gate numbers; and the angle-set decision for
  the CST campaign must be re-measured under the corrected protocol before
  any solver time is spent.

Follow-up measurements (span ladder, estimator ladder including
frequency-continuation seeding, angle-set ladder, recalibrated gate proposal)
are in `results/GATE_STUDY.md`.

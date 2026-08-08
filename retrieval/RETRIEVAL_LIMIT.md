# What limits the Floquet → T0 retrieval: a verified decomposition

Verifier pass, 2026-08-06/07. Every number here was produced by the verifier
re-running the code, or by an independently written script that re-derives its
own references. Campaign 13 angles, `direction = -1`, smoke frequencies
ifreq 32 (λ = 12.00 µm) and ifreq 48 (λ = 8.00 µm) unless stated.

This supersedes the previous session's single-cause framing ("the doc's
§6.2/§6.3 gates fail as an information limit of the campaign-13 angle set").
That framing is **not supported**. The failure decomposes, and the pieces have
different remedies — some already applied.

---

## The decomposition

### 1. Bright-span model error — real, and it caps the small span

The bright-10 span (threshold 1e-3, 25 entries) **represents the bright
entries exactly**: its projection of `P68(T_ref)` agrees with `P68(T_ref)` on
every bright-mask entry to 3e-18. Recovery failures are never
representability failures. The same identity was asserted and passes at every
threshold tested (1e-3, 3e-4, 1e-4, 3e-5; worst 9.9e-16).

But the span omits sub-threshold entries with large S-leverage. The
off-bright residual is 1.1e-5 (ifreq 32) / 7.7e-5 (ifreq 48) in T units and
produces `max|S(T_proj) − S(T_bp)| = 1.02e-3 / 1.17e-3`. Consequence: the
global minimum of the bright-10 fit is **not at the truth**. A 16-start
independent search finds a point fitting the data **38× / 62× better** than
the truth does. Noise-free, perfectly optimized, the bright-10 dipole error
at 8 µm is **269 %**.

Lowering the threshold fixes this: at 3e-5 (473 entries, rank 53) the model
floor drops 8×, and a truth-seeded fit recovers the dipole to **0.41 %
(12 µm) / 2.3 % (8 µm)** — meeting the doc §6.3 5 % target. **The information
is in the data.**

### 2. The optimization landscape — the binding constraint, and it is not fixed

The information being present does not make it reachable.

| span | seed | dipole error @ ifreq 48, noise-free |
|---|---|---|
| bright1e-3 (rank 10) | truth | 2.69 |
| bright1e-3 (rank 10) | continuation | 2.69 |
| bright1e-3 (rank 10) | Born | 3.50 |
| bright3e-5 (rank 53) | **truth** | **0.023** |
| bright3e-5 (rank 53) | continuation | 5.75 |
| bright3e-5 (rank 53) | Born | 7.20 |

Frequency-continuation seeding (chain 42→48, seeding each fit from the
previous frequency) **completely solves the basin problem when the target
lies in the fitted span**: on the C-clean loop ifreq 48 goes from 11.04
(Born) to **6e-12**, objective 3.17e-6 → 1.7e-30, every step of the chain at
~1e-30. That is a real, realizable win and it is why continuation is now a
first-class protocol dimension in `gate_study.py`.

**But on the physical target it does not transfer to the rich span.** With
model error present, each frequency's minimum is displaced from the truth and
the chain propagates a displaced solution; at ifreq 42 the rich-span Born fit
is already in a bad basin (all-25 43.6) where on the clean loop it was exact.
So: continuation fixes the **basin**, not the **model error**, and the rich
span — the only one with the information — remains unreachable from any
realizable seed.

### 3. Prior mis-scaling — real, fixable only with oracle knowledge

An isotropic Tikhonov prior at the physical scale inverts the noise ladder
(dipole error *falls* as σ rises: 0.75 → 0.64 → 0.31 for σ = 1e-3 → 1e-2),
the signature of a bias-dominated estimator. An **orbit-scaled** diagonal
prior `tik_k = 1/|z_k|²` is 90× better on all-25 and brings the dipole to
3.9 % at 12 µm.

Two things must be said about it. First, an **orbit-pure basis alone is a
mathematical no-op** under an isotropic prior — both bases are
real-orthonormal for the same span, so `‖t‖²` is invariant and the estimate is
identical (confirmed to 3.5e-7 relative at ifreq 48; where they differ, at
ifreq 32, it is two local minima, i.e. §2 again). Second, the orbit-scaled
prior needs `|z_k|` of the truth. A realizable two-step empirical-Bayes
version was implemented and **fails** (`gain/zero ≈ 1.00`): the first
isotropic pass shrinks, so the estimated `tau_k` is off by 2.7–15× median and
up to 650×.

### 4. Gate normalization — unattainable by construction for weak entries

Several bright entries have band-peak `|T| ~ 1e-4`. "≤ 1 % relative" demands
≈1e-6 absolute while the bright-span model error alone is 1.1e-5. 21–23 of
the 25 bright entries are ungateable in per-entry-relative terms at any noise
level. Gates must be normalized to a global scale and paired with the
estimator's own propagated error bar — which is validated (measured/predicted
rms ratio 0.81–1.93).

### 5. Structural darkness — the one genuine information limit, exactly as the doc says

Even-m content at θ=0: `|dS/dT| = 5.3e-21` → 65.5 with oblique angles. The
orbit-pure even-m sub-basis has exact-Jacobian column norms 1.7e-14
(normal-only) vs 1.24e+02 (campaign). Doc §4's visibility claims (i) and (ii)
both hold at both frequencies. Two of the 25 "bright" entries are C4-violating
file noise and are structurally returnable only as 0.

---

## Bottom line for the campaign

**Blind retrieval of T0 from Floquet data alone is limited by §2, not by the
angle set.** Measured over 23 candidate protocols at σ = 3e-3, no *realizable*
protocol beats the trivial `T̂ = 0` estimator on the dipole class at either
frequency (best realizable gain/zero 0.97 at 12 µm, 0.86 at 8 µm). Adding
angles does not address this: the limit is the landscape, not the data.

**The QA-gate application — the doc §1 reason this project exists — does
work.** There, the supplied tmat.h5's own `|z_k|` legitimately set the prior,
so the oracle-primed protocol is realizable by construction: dipole gain 8.17×
over zero, fully converged (0 of 112 fits capped), with actionable per-class
thresholds. And the §8.1 held-out-angle criterion **passes with margin**: fit
on 4 angles, predict the other 9, worst complex |dS| = 6.0e-4 against the
1e-2 gate (9.1e-3 at σ = 3e-3). That is the acceptance test the QA gate needs.

**Identified next lever, deliberately not built here** (doc §4 defers it:
"frequency smoothness is a diagnostic, not a constraint, in v1"): use
smoothness as a *constraint* rather than a seed — a joint fit over the band
penalizing `‖t(f) − t(f−1)‖²`. Continuation as a seed adds no information;
smoothness as a constraint does, it is realizable without oracle knowledge,
and it is physically justified because T is analytic in frequency. Doc §5's
hybrid completion (periodic data + a few isolated-cell runs for the dark
residual) is the other route. Both are out of the current scope.

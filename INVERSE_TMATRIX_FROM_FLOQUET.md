# Retrieving the isolated-cell T-matrix from multi-angle Floquet S-parameters

**Status: IMPLEMENTED and independently verified (2026-08-07).** The whole
§10 checklist exists in `src/tmatrix/retrieval/`; every module has been read
line-by-line against this document and its gates re-run by a verifier. (The
companion `.cst` is a 48 KB history-only archive with **no result data** — it
provides geometry provenance only; all reference CST data comes from fresh
periodic solves, cf. `aggregation/cst_direct/run_v3/s_params_complex.csv`.)

**Read these three before trusting any claim below:**
- `retrieval/HANDOFF.md` — state, locked conventions, next steps.
- `retrieval/VERIFICATION_ITEMS_5_6.md` — the independent verification record.
- `retrieval/RETRIEVAL_LIMIT.md` — **what actually limits the retrieval.**

**Where this document has been overtaken by measurement** (each amendment is
marked inline, with the original text preserved for provenance):
- §3's "φ=22.5° delivers 8 observables instead of 4" is true as a count but
  weak as an information argument for this cell — the extra observables run
  8.7e-5…5.4e-3, at or below σ across most of the band.
- §6.2/§6.3's synthetic gates fail as written, but **not** for the reason a
  first pass suggested. They decompose into span model error, an
  optimization-landscape limit, prior mis-scaling, an unattainable gate
  normalization, and genuine structural darkness. Only the last (and part of
  the first) is an information limit. See `RETRIEVAL_LIMIT.md`.
- §7's channel-dictionary acceptance is amended to a pooled χ² discriminant at
  (θ=60°, φ=22.5°); the original fixed-tolerance max test at (30°,0°) was
  calibrated against an artifact of the reference file.
- §4's "Born seed `T0 = 0` (sufficient for v1)" is **not** sufficient: the
  Born basin does not contain the truth at the smoke frequencies. §4's
  "frequency smoothness is a diagnostic, not a constraint, in v1" is the
  identified next lever.

Every numerical claim originally in this document was either measured directly
on the repo's data or adversarially re-verified (2026-08-06); items marked
**NEW** were built in `src/tmatrix/retrieval/` and are listed in the §10
checklist.

---

## 1. Purpose and positioning

Stages 1–2 (isolated-cell extraction: plane-wave sweep with open boundaries →
near-field VSWF projection) are owned by a collaborator's external tool
(`cst_tmatrix`; see the tmat.h5 `/computation` attributes). This document
specifies an **independent, complementary route** that never touches that code:

> Fit the isolated-cell T-matrix `T0` directly from **periodic (Floquet
> unit-cell) CST simulations at multiple incidence angles**, by inverting the
> multiple-scattering forward map that Stage 3 already implements and has
> validated against treams (≤ 3.4e-4 complex) and direct CST
> (≤ 3e-3 — **in |S| only**; see §7 on the complex-S gate).

Three reasons this is worth building:

1. **QA gate for externally supplied T-matrices.** Whatever tmat.h5 a
   collaborator delivers can be accepted/rejected by checking that it
   reproduces cheap periodic runs at held-out angles — no isolated-cell rerun
   needed.
2. **Robustness.** Periodic Floquet runs are CST's native, best-converged
   workflow (no open-boundary truncation error, no near-field export).
3. **Novelty.** Published inverse retrievals stop at **dipole order**
   (Scher & Kuester 2009; Karamanos et al. 2014/2018); nothing recovering an
   lmax=3 T-matrix from Floquet data was found through Aug 2026. Slots into
   `aggregation/NOVELTY.md` Lane 1 and feeds Lane 3. Honest framing:
   *constrained retrieval with an explicit observability map*, not
   "full T from S-parameters" (provably ill-posed; §5).

## 2. The forward map

For an infinite square lattice (pitch `p`, cell area `A = p²`) illuminated by a
plane wave with direction `k̂` (in-plane Bloch vector
`k∥ = k(sinθ cosφ, sinθ sinφ)`) and polarization `ê_b`:

```
a_inc,b = plane_wave_coeffs(k̂, ê_b, modes)                 # src/tmatrix/aggregation/vswf.py:184
C       = lattice_sum_C_bloch(k, p, modes, k∥, r0, quad)    # NEW (extends tmatrix.aggregation.translate)
f_b     = (I − T0·C)⁻¹ · T0 · a_inc,b                       # = aggregate.solve_periodic
F_b(r̂)  = far_field_amplitude(k, f_b, modes, r̂)            # src/tmatrix/aggregation/vswf.py:203

S21_ab = δ_ab + (2πi/(k A cosθ)) · ê_a^(t)* · F_b(k̂_t)     # transmitted specular order
S11_ab =        (2πi/(k A cosθ)) · ê_a^(r)* · F_b(k̂_r)     # reflected specular order
```

`k̂_t = k̂`; `k̂_r = k̂ − 2(k̂·ẑ)ẑ` (same k∥, flipped k_z). **Row a = receive
polarization, column b = incident polarization, in both blocks**; `F_b` is the
far field under incidence `ê_b`. Polarization basis (normative):
`ê_TE(k̂) = (ẑ×k̂)/|ẑ×k̂|`, `ê_TM(k̂) = ê_TE×k̂`, evaluated **separately** for
`k̂_t` and `k̂_r` (this fixes the reflected-TM sign convention); the θ→0 limit
is taken by continuity along fixed φ, so at φ=0 the basis reduces to (x̂, ŷ)
and the `sparams_normal` comparison is exact.

The bracket `(I − T0·C)⁻¹T0` equals `solve_periodic`'s `T(I − CT)⁻¹` by the
push-through identity, and is the "effective T-matrix" `Teff` of Rahimzadegan
et al., Adv. Opt. Mater. 10, 2102059 (2022), whose closed forms go to exactly
octupole order = our lmax=3.

**Bloch-sum sign (verified against `build_finite_system`'s conventions):** with
the `e^{+ik·r}` spatial convention, site `R` sees `a_inc·e^{+i k∥·R}`, giving
`C(k, k∥) = Σ_{R≠0} A(R) e^{+i k∥·R}`; the `k∥ → 0` limit must reproduce
`lattice_sum_C` to machine precision.

**Bloch-sum convergence warning:** Bloch phases *slow* the propagating-channel
shell-sum convergence — the controlling parameter becomes `k(1 − sinθ)Rc`
(distance to the grazing Rayleigh anomaly), ≈ 7.5× worse at θ = 60° than at
normal incidence. Scale taper lengths as `kRc ≳ 10/(1 − sinθ)` at oblique
angles, re-run `tests/aggregation/test_translate.py`-style
Richardson-stability checks at the largest k∥ (θ=60°, λ=20 µm), and fall back
to Ewald (treams) near grazing if tapered sums degrade.

Conventions throughout: tmat.h5 / Stage 3 (`e^{−iωt}`, `h_l^{(1)}` outgoing,
Condon–Shortley, `f = T a`, mode order from `/modes`). CST S-parameters arrive
in `e^{+jωt}` → conjugate before use (§7 specifies the verification).

### What is new vs. what exists

| Piece | Status | Where |
|---|---|---|
| `plane_wave_coeffs` at arbitrary `k̂` | exists | `src/tmatrix/aggregation/vswf.py:184` |
| `solve_periodic`, `effective_array_T` | exist | `src/tmatrix/aggregation/aggregate.py:87,94` |
| Lattice sum at `k∥ = 0` | exists, validated | `src/tmatrix/aggregation/translate.py:150` (tapered shells + Richardson) |
| **Bloch-phased lattice sum** | **NEW** | extend `tmatrix.aggregation.translate`; per-site phases in `assemble_shell_sum` (sites enumerated by `square_lattice_shells`) |
| **Oblique specular Jones blocks** (`1/cosθ`) | **NEW** (small) | generalize `src/tmatrix/aggregation/sparams.py:26` per the formulas above |
| **Complex-S de-embedding vs CST** | **NEW** — nothing complex-valued exists (see §7) | new module |
| **Fit driver + constraints + observability** | **NEW** | this project |
| Oblique treams cross-check | **NEW work** — `tmatrix.aggregation.treams_reference` is k∥=0-only (see §6) | extend it |
| Oblique CST campaign | **NEW runs** | `src/tmatrix/aggregation/cst_direct/` + `cma_infinite/cst` templates, with required edits (§7) |

## 3. Empirical structure of the target (measured on the reference file)

Measured on `test/single/saw_gold_wl15p0025um.tmat.h5` (49 freqs, 30×30) and
independently re-verified:

- **Sparsity: 25 of 900 entries exceed 1e-3 of the global |T|max at any
  frequency** (exact count). Practical unknowns: tens, not 900.
- **C4 selection rule** (4-spoke wheel): entries violating
  `(m − m') mod 4 = 0` are ≤ 1.5e-3 relative (noise level). The rule allows
  228 of 900 entries; 23 of the 25 bright entries obey it.
- **Reciprocity** `T_{lm,l'm'} = (−1)^{m+m'} T_{l'(−m'),l(−m)}` (the repo's
  validated convention, cf. `src/tmatrix/aggregation/run_demo.py:45` and
  IMPLEMENTATION_GUIDE §7.4):
  mid-band 3.7e-5 absolute vs |T|max 1.1e-2; band-worst 1.5e-4 vs 7.8e-2
  (both at λ = 8 µm) — **0.2–0.4 % relative at every frequency**, consistent
  with the file's stored diagnostic.
- **Unknown count (computed with the exact group representation):** the
  C4v-commutant ∩ reciprocity subspace at lmax=3 has **68 complex dimensions**
  (C4-only commutant: 228 → C4v: 114 → ∩ reciprocity: 68). Restricted to the
  bright entries: **25 entries collapse into 11 symmetry orbits ≈ 11–15
  complex unknowns per frequency.**
- **Observables per (angle, frequency):** 8 complex (2×2 Jones × {R, T}) at
  generic φ (e.g. 22.5°); **only 4** on the mirror planes φ ∈ {0°, 45°}
  (cross-pol vanishes identically); **only 2** at θ = 0 (C4 forces
  S_TE = S_TM). Budget angles with these corrected counts — a handful of
  angles still over-determines the bright subspace.
  **Amended 2026-08-06 (measured):** the selection rule is confirmed exactly
  (projected-T cross-pol is 3e-17…7e-16 at φ ∈ {0°,45°} and non-zero at
  φ=22.5°), but the four *extra* observables at φ=22.5° are **weak for this
  cell**: band-max |cross-pol| = 8.7e-5 (θ=30, λ=20 µm) rising to 5.4e-3
  (θ=60, λ=8 µm). Against the σ = 3e-3 placeholder they only clear SNR 1 at
  the short-λ end and only at θ=60. So "φ=22.5° delivers 8 observables
  instead of 4" is true as a count and misleading as an information
  argument — most of those extra observables are at or below the noise
  across most of the band. Weight the angle budget by measured leverage,
  not by observable count.

## 4. Inverse problem specification

**Per frequency, independently** (49 small fits; frequency smoothness is a
diagnostic, not a constraint, in v1):

```
min over t   Σ_i  w_i ‖ S_meas(θ_i, φ_i) − S_pred(T0(t); θ_i, φ_i) ‖²_F
subject to   T0(t) ∈ 𝒮   (C4v + reciprocity subspace, 68-dim; bright-restricted ~11–15)
             max SV(I + 2·T0) ≤ 1 + ε      (passivity, post-check or penalty)
```

with `w_i = 1/σ_i²`, σ_i from the calibrated per-angle floor (§7).

**Parametrization `𝒮` — build the group action carefully:**

- **C4 rotations about z** act by conjugation with the *diagonal*
  `D_φ = diag(e^{i m_ν φ})`, φ ∈ {0, π/2, π, 3π/2} — the same phases
  `rotate_inplane` (`src/tmatrix/aggregation/translate.py:94`) applies. No
  Wigner-D machinery needed for z-rotations.
- **The four C4v mirrors are VERTICAL planes (σ_v: xz, yz; σ_d: diagonals) —
  `tmatrix.aggregation.mirror`'s `mirror_parity_signs` is NOT one of them.**
  That function encodes the *horizontal* z-mirror `diag(1,1,−1)` (PEC image
  theory, an element of C4h/D4h, diagonal in (l,m)); a vertical mirror maps
  `m → −m` (a signed permutation, block-antidiagonal — it would trip
  `mirror_parity_signs`' own diagonality assertion). Using σ_h builds the
  **wrong group** (C4h), and — trap — the reference T is *also* invariant
  under that wrong projector because the flat wheel is D4h-symmetric, so
  projector-invariance of the reference T does **not** validate the mirror
  choice.
- The correct σ_v(xz) action in this basis (derived numerically, verified:
  the resulting C4v+reciprocity projector leaves the reference T invariant to
  3.6e-5 = its noise floor):
  `(l,m,E) → (−1)^m (l,−m,E)`, `(l,m,M) → −(−1)^m (l,−m,M)`.
  Derive it in code with the `mirror_parity_signs` *technique* generalized:
  sample outgoing VSWF fields via `vswf_fields`, apply
  `E'(r) = M·E(M·r)` with `M = diag(1,−1,1)` (H picks up an extra −1 as a
  pseudovector), and solve for the full representation matrix. Diagonal
  mirrors = σ_v conjugated by C4 rotations. Validate `D(σ_v)² = I`, closure of
  all 8 elements, and check the projected reference T against the σ_v
  selection rules **explicitly** (not just projector invariance — see trap
  above). Then `T ↦ (1/8) Σ_g D(g) T D(g)⁻¹`, intersect with reciprocity,
  orthonormal basis {B_k}, unknowns `T0(t) = Σ t_k B_k`.

**Optimizer:** Levenberg–Marquardt (`scipy.optimize.least_squares`, real/imag
stacked), **Born seed `T0 = 0`** (sufficient for v1; the Rahimzadegan-2022
closed-form dipole inversion of normal-incidence S is an optional cross-check
seed). The problem is mildly nonlinear in-band — but via the **spectral
radius**, not the norm: measured with the repo's own `lattice_sum_C`,
`ρ(T0·C) = 0.16–0.25` across 8–20 µm and `‖(I − C·T0)⁻¹‖₂ ≤ 27`, while
`‖T0·C‖₂ = 5–25` (`‖C‖₂` reaches ~1e6 from quasi-static l=3 coupling). **Never
gate on `‖T0·C‖ < 1`** — it fails at every frequency; use ρ.

**Precompute the lattice sums.** `C(k, k∥)` is independent of `T0`:
compute all 49 freqs × n_angles Bloch sums **once**
(`tmatrix.retrieval.precompute_C` → `results/C_bloch.npz`; ~1–2 h at
`tmatrix.aggregation.run_demo` rates), and have
`tmatrix.retrieval.forward`/`fit`/`observability` load the cache.
Forward evaluations are then microseconds and finite-difference Jacobians are
cheap; without the cache the fit is computationally infeasible.

**Observability map (first-class deliverable):** Jacobian
`J = ∂vec(S_all-angles)/∂t` at the reference T (synthetic study) and at the
solution; SVD → per-basis-vector resolution `res_k`; entrywise heatmap
`H[μ,ν] = Σ_k res_k |B_k[μ,ν]|²`. Null-space directions are reported
*unobservable*, not fitted (Tikhonov toward 0 keeps them pinned).

**Visibility structure at θ=0 (corrected — measured, not assumed):** the
incident wave carries only m = ±1 and the specular projection observes only
m = ±1 — **but the lattice sum couples Δm ≡ 0 (mod 4), so the m = ∓3 blocks
re-enter the m = ±1 channel and are strongly visible at normal incidence**
(measured sensitivities |dS/dT| ≈ 70–790 per unit entry for (3,∓3)↔(1,±1) and
(3,±3)↔(3,±3), vs ≈ 7 for dipole entries, at 14 µm). What is *strictly* dark
at θ=0 is the **even-m content (m ∈ {0, ±2})** — including the (±2,±2)
quadrupole entries — reachable only via oblique angles. Combinations dark in
the specular direction at every sampled angle (anapole-type, Grahn–Shevchenko
degeneracies) remain unrecoverable in principle. Rayleigh anomalies are a
non-issue here: worst-case onset `λ = p(1 + sinθ) = 3.73 µm` at θ=60°
(lattice-axis G; the φ=45° diagonal onsets even lower) — the 8–20 µm band is
**specular-only at every planned angle** with wide margin.

## 5. Feasibility verdict (from the Aug 2026 literature study)

- Full 30-mode recovery from Floquet data alone: **ill-posed** — do not claim it.
- Bright-subspace recovery with an explicit observability map: **feasible and
  open** — nothing above dipole order is published.
- Hybrid completion (periodic data + a few isolated-cell runs for the dark
  residual): natural follow-up.
- Allayarov/Evlyukhin/Calà Lesina (2026) caveat: for *connected* geometries
  the single-cell T is representation-dependent and this route breaks down.
  Our cell is an isolated island — well-posed — state this in any write-up.

## 6. Validation ladder (synthetic before CST; steps 1–3 need no license)

1. **Oblique forward map vs treams.** Implement `lattice_sum_C_bloch` +
   `sparams_oblique`; compare complex specular S at θ ∈ {0°, 20°, 40°},
   φ ∈ {0°, 45°}, both polarizations, 49 freqs. Gate: ≤ 1e-3 complex; θ=0 must
   reproduce `tmatrix.aggregation.run_demo` exactly.
   **Scope warning:** `tmatrix.aggregation.treams_reference` wraps treams at
   k∥ = 0 *only*, and two steps are normal-incidence-specific. Generalizing
   is part of this
   step's work (~a day of convention-wrangling): pass
   `kpar = k(sinθcosφ, sinθsinφ)` to `latticeinteraction.solve` and
   `PlaneWaveBasisByComp.diffr_orders`; build TE/TM incident coefficients by
   projecting the desired oblique `ê` via `treams.efield` (the `[1,0,0]`
   lstsq trick at line ~139 has *no solution* off-normal); re-verify
   `diffr_orders` mode ordering at k∥ ≠ 0; keep the Windows/numpy2
   gufunc-cast shims; confirm the k∥ sign by matching our forward map at
   small θ.
2. **Noise-free synthetic closed loop.** Synthetic `S_meas` from the reference
   T at the planned angle set; fit; gate: bright-subspace entries ≤ 1%
   relative; observability heatmap produced.
3. **Noise robustness.** Inject complex Gaussian noise at the σ established by
   the §7 normal-incidence complex closure (do **not** assume 3e-3 — that
   number is magnitude-only; the complex floor is unknown until measured).
   Gate: dipole block stable to ≤ 5%; report degradation vs angle count.
4. **Real CST campaign** (§7), acceptance in §8.

## 7. CST campaign specification

**Geometry:** free-standing gold wheel array, no spacer, no ground plane —
matching what `T_iso` describes. Base:
`src/tmatrix/aggregation/cst_direct/build_saw_unitcell.py`
(pitch trap documented in REPORT §4b: set `UnitCellFitToBoundingBox "False"` +
explicit `UnitCellDs1/Ds2`).

**Required edits to the existing script (verified against the source):**

- **Pin the domain with an explicit vacuum cellpad brick** (the
  `cma_infinite/cst/build_run_paper.py` `vba_cellpad` pattern; e.g. p×p ×
  ±several µm in z) in **both** structure and empty projects. As-is, the
  z-extent derives from the metal's bounding box + auto open-space — removing
  the metal changes the domain, destroying the phase reference. This edit
  changes the mesh of the validated normal-incidence run → **re-run the
  normal-incidence pair first and re-establish closure before any oblique
  solve.**
- **Excite both Floquet modes.** The script uses `.Stimulation "List","List"`
  and effectively drives one Zmax mode; the 2×2 Jones blocks need both modes
  of the excitation port driven (switch to `"All","All"` or list both; note
  this doubles excitation count). Keep the script's frequency sampling
  (`AddSampleInterval`) otherwise.
- **Dependency chain (for a fresh machine/session):**
  `tmatrix.aggregation.cst_direct.build_saw_unitcell`
  imports `nir.cst_helpers` from `D:/Claude/auto_cst` (hard-coded sys.path);
  also needs `E:/cst/AMD64/python_cst_libraries` and the
  [cma_infinite](https://github.com/DaryLu0v0/cma_infinite) clone for the
  Floquet/scan-angle VBA templates. Pick ONE automation stack — recommend
  `cst.interface` as in the validated saw script — and port `vba_floquet`'s
  angle block into it, parametrizing **both θ and φ** (the cma template
  hard-codes φ=0).

**Channel dictionary (normative — do not improvise):**

- Excite Zmax modes 1,2; incident wave travels −z:
  `k̂ = (sinθ cosφ, sinθ sinφ, −cosθ)` in CST coordinates. The model is
  evaluated at this k̂ directly, `k∥ = k(sinθ cosφ, sinθ sinφ)`.
- `SetPeriodicBoundaryAnglesDirection`: the two in-repo templates disagree
  ("inward" in `build_run_paper.py` vs "outward" in `cst_common.py`);
  **use "inward" and verify** via the empty-cell S21 phase slope vs
  `k_z = k cosθ` — this also pins the k∥ sign.
- S-tree entries: reflection block = `SZmax(a),Zmax(b)`, transmission block =
  `SZmin(a),Zmax(b)`, a,b ∈ {1,2}; map CST's TE/TM labels to the §2 Jones
  basis per angle using the empty cell (labels' orientation depends on φ —
  spot-check one φ≠0 case against the analytic empty-cell answer).
- **Acceptance before the campaign proceeds** (AMENDED 2026-08-06 — measured;
  the original spec is kept below for provenance): the de-embedded complex
  numbers at one oblique angle must identify exactly one of the 8 TE/TM label
  hypotheses, using a **pooled χ² discriminant over the whole 49-frequency
  band**, not a fixed-tolerance max.

  *Original spec:* "at one oblique angle (θ=30°, φ=0°), all 8 de-embedded
  complex numbers must match the forward model evaluated with the *reference*
  T under exactly one TE/TM label hypothesis to ≤ 1e-2."

  *Why it changed.* The 8 hypotheses split into two halves that behave
  completely differently, and the split is what the original spec missed.

  **The mode-swap half is trivial to decide at any angle.** Measured
  `sep_swap` ≥ 0.546 at every campaign angle — five decades above any
  plausible σ. Nothing needs to change for it.

  **The cross-sign half is decidable only at φ=22.5°.** The discriminating
  signal is `sep = 2·|cross-pol of the measured data|`, and on a genuinely
  C4v structure the cross-pol **vanishes identically on the mirror planes
  φ ∈ {0°,45°}**. Measured with the C4v-projected reference T:

  | angle | `sep_cross` | z @ σ=3e-3 |
  |---|---|---|
  | any φ ∈ {0°,45°} | 3.5e-16 … 7.2e-16 | **0.00** |
  | (15°,22.5°) | 9.3e-4 | 0.53 |
  | (30°,22.5°) | 3.5e-3 | 1.99 |
  | (45°,22.5°) | 7.1e-3 | 3.99 |
  | (60°,22.5°) | 1.06e-2 | **5.92** |

  A first pass using the **raw** reference T appeared to give `sep_cross` =
  8.2e-3…1.71e-2 at *every* angle, mirror planes included, and that number
  is an artifact: it is entirely the reference file's own ~0.3 % C4v-violation
  noise (the documented 3.1e-3), which a real C4v CST cell will not
  reproduce. Any acceptance designed against the raw-T numbers would be
  relying on a property of the reference file rather than of the physics.

  **This is self-consistent rather than a problem.** A cross-sign error
  flips entries that are identically zero at φ ∈ {0°,45°}, so on the mirror
  planes the ambiguity is *harmless*: it exists exactly where it has no
  consequence. It becomes consequential at φ=22.5°, and that is precisely
  where it is measurable. So the acceptance must be run at a φ=22.5° angle —
  not for extra observables, but because it is the only place the cross-sign
  question is both askable and worth asking.

  The original test takes a **max over 8 channels × 49 frequencies = 392**
  complex numbers. Under noise of scale σ that max is ≈ σ·√(ln 392) = 2.44σ
  for the *correct* hypothesis, so the usable tolerance window is
  [2.44σ, sep] — only 3.7e-3 wide at (30°,0°) and σ = 3e-3, and **closed
  entirely (2.44σ ≥ sep) at every angle once σ ≥ 1e-2**, where no tolerance
  can work at all.

  *The amended test.* With `D_h = Σ_{channels,freqs} |S_h − S_pred|²`
  (reference T, direction −1), accept hypothesis `h*` iff

  ```
  z_margin = sqrt(D_second − D_best) / (2σ)  ≥  Z_MIN (= 5)
  chi2_reduced = D_best / (n_obs σ²)         ≤  CHI2_MAX (= 4)
  ```

  and refuse otherwise (the fail-safe is retained: refuse, never mis-pick).
  **Single-frequency acceptance is hopeless at every angle** (z ≈ 0.6–1.4
  even on the optimistic raw-T numbers) — the band must be pooled. Pooling
  over only the two smoke frequencies already drops the raw-T residual to
  6.7e-3 and returns 4 winners. (`z = sqrt(D)/(2σ)` as defined here is
  conservative by √2 relative to a strict two-hypothesis ML analysis, so
  `Z_MIN = 5` here is ≈7 in the ML convention.)

  On raw-T data the χ² test also strictly dominates the original max test
  under noise (20 seeded trials, known hypothesis; **no test in any cell ever
  accepted a wrong hypothesis** — every failure is a refusal):

  | angle | σ | χ² | max @ tol 1e-2 |
  |---|---|---|---|
  | (30°,0°) | 3e-3 | 100 % | 100 % |
  | (30°,0°) | 5e-3 | 35 % | **0 %** |
  | (45°,22.5°) | 5e-3 | **100 %** | **0 %** |
  | either | 1e-2 | 0 % (refuses) | 0 % (window closed: 2.44σ ≥ sep) |

  *Recommended acceptance angle:* **(θ=60°, φ=22.5°)** — on the physical
  (C4v-projected) numbers it is the **only** campaign angle whose cross-sign
  margin clears `Z_MIN = 5` (z = 5.92 at σ = 3e-3); (45°,22.5°) reaches only
  3.99 and every mirror-plane angle is exactly 0. Decide the mode-swap half
  wherever convenient — it is unambiguous everywhere.

  **Standing risk, to be resolved by the closure and not before:** the
  cross-sign margin scales as 1/σ, so if the §7 normal-incidence complex
  closure measures σ much above 3e-3 the cross-sign half becomes
  undecidable even at (60°,22.5°) (z = 1.78 at σ = 1e-2). The fail-safe then
  correctly refuses. Mitigations in preference order: (i) pool the χ²
  statistic over *several* φ=22.5° angles rather than one — the signal adds
  in quadrature; (ii) accept the swap determination alone and pin the
  cross-signs by the mirror-plane argument above (they are inconsequential
  wherever they are unmeasurable); (iii) extend
  `deembed.label_hypotheses` per the HANDOFF caveat. Do **not** fall back to
  raw-T separation numbers — they are a property of the reference file.

  Implemented in `tmatrix.retrieval.validate_against_reference`
  (`channel_dictionary_acceptance(..., statistic="chi2")`); the doc-literal
  `statistic="max"` path is retained for comparison.

  **The hypothesis family was incomplete — corrected 2026-08-07 by the real
  campaign.** The first live acceptance run at (60°,22.5°) refused with
  0 of 8 hypotheses passing and a winner's reduced χ² of **658** (residual rms
  25.7σ). Diagnosis: every channel matched the reference-T forward model to
  ~2e-3 *except* the S11 TM co-pol entry, where `|d| = 2|S|` — a sign.
  Applying a −1 to the **TM receive row of the S11 block only** gives
  χ²_red **2.49** at (60°,22.5°) and **1.14** at θ=0, i.e. the residual lands
  exactly on the independently measured σ.

  This is a consequence of §2's own convention, not a free parameter: the
  polarization basis is evaluated separately for `k̂_t` and `k̂_r`, so
  `ê_TM^(r)` is transversally opposite to the CST Floquet port's fixed mode
  pattern, and the reflection block alone carries a receive-side TM-row sign.
  **No port-mode sign can express it** — a mode sign `s_a` multiplies
  `S11[a,b]` by `s_a·s_b`, leaving the co-pol diagonal invariant — which is
  precisely why the shipped 8-member family could not contain the truth. The
  fail-safe behaved as designed: it refused rather than mis-picking.

  `deembed.label_hypotheses(extended=True)` now returns 16 members (the base
  8 with `r11_tm = ±1`; the first 8 are the base 8 unchanged). Note
  `apply_hypothesis` is **not an involution** for the 4 members with
  `swap=True ∧ r11_tm=−1` — use its `inverse=True` flag.

  Live result at (60°,22.5°): winner
  `(swap=False, s11_cross=−1, s21_cross=−1, r11_tm=−1)`, χ²_red 1.595,
  runner-up **z = 5.77** (passes `Z_MIN = 5`, but only by 1.15×; at 2σ it
  would refuse, and the marginal dimension is still `s21_cross`). At θ=0 all
  eight `r11_tm=−1` members tie at χ²_red ≈ 1.13 while every `r11_tm=+1`
  member sits above 2763 — so **normal incidence determines `r11_tm`
  decisively and nothing else**, which is the cleanest available statement of
  why the acceptance must run off the mirror plane.

**Complex de-embedding (NEW — nothing complex-valued exists in the repo;
`tmatrix.aggregation.plot_cst_comparison` and REPORT §4b validated magnitudes
only):**

- With the z-symmetric pinned domain: `S21_deemb = S21_raw / S21_empty`,
  `S11_deemb = S11_raw / S21_empty` (symmetric domain ⇒ reflected path length
  = port-to-port length). Conjugate CST data first; verify the conjugation
  direction by checking that `arg(S21_empty)` advances as `e^{+i k_z L}` in
  the `e^{−iωt}` convention.
- **Gate (new deliverable):** de-embedded complex S at normal incidence vs
  `results/periodic_results.npz` (complex, both polarizations) to ≤ 5e-3.
  Whatever residual this closure yields **is the true σ for the fit weights**
  — the 3e-3 on record is magnitude-only and cannot be assumed for phase.

**Noise-floor calibration (replaces the naive "independent repeat"):** CST's
FD solver is deterministic — an identical re-run measures nothing. Calibrate
two ways: (i) deviation of each empty-cell run from its *analytic* S
(|S21| = 1, phase = k_z·L, S11 = 0); (ii) one perturbed re-run (e.g.
`AccuracyTet` 1e-4 → 3e-4, or mesh refinement change) for discretization
scatter. The dominant w_i term should be the model-error floor from the
normal-incidence complex closure.

**Angle set and budget (C4v ⇒ φ ∈ [0°, 45°]):**
θ ∈ {0°, 15°, 30°, 45°, 60°} × φ ∈ {0°, 22.5°, 45°} = 15 pairs − 2 duplicate
normal-incidence entries = **13 structure runs**. The empty cell is
φ-independent (S depends only on k_z): **5 empty runs (one per θ) + 1
perturbed repeat = 6**. Total ≈ 19 FD solves (hours at run_v3 scale). Include
φ = 22.5° angles deliberately — they are the only ones delivering all 8
observables (§3). Starter subset: θ ∈ {0, 30, 60} × φ ∈ {0, 22.5} minus
duplicate = 5 structure runs; let the *synthetic* observability map (computed
before any CST time is spent) decide additions.

**Frequency grid:** the file's 49 frequencies; interpolate CST's dense sweep
onto it via separate Re/Im `np.interp` (pattern already in
`src/tmatrix/aggregation/cst_direct/build_saw_unitcell.py` ~line 317).

## 8. Acceptance criteria (real-data test)

1. Fitted `T0` reproduces **held-out angles** (fit on 4, predict the rest) to
   ≤ 1e-2 complex specular S across 8–20 µm.
2. Bright-subspace comparison vs the reference tmat.h5: dipole–quadrupole
   entries within 5–10% relative (tighten after the synthetic study);
   discrepancy pattern consistent with the observability map.
3. Passivity `max SV(I + 2T0) ≤ 1 + 1e-3` and reciprocity ≤ 1e-3 as *checks*
   (not enforced beyond the subspace constraint).
4. Observability heatmap published with the fit (determined entries +
   singular-value spectrum at the chosen angle set).

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Bloch lattice-sum bug (sign/phase) | k∥→0 must equal `lattice_sum_C` to 1e-12; treams cross-check (§6 step 1) before fitting |
| Shell-sum degradation near grazing | `kRc ≳ 10/(1−sinθ)` scaling; per-angle Richardson checks; Ewald fallback |
| Wrong symmetry group (C4h vs C4v trap) | explicit σ_v selection-rule check on projected reference T (§4); `D(σ_v)²=I` + group closure tests |
| Local minima in LM fit | ρ(T0·C) ≤ 0.25 in-band ⇒ near-linear; Born seed + multi-start; synthetic study proves the basin |
| Phase-reference error masquerading as T error | normal-incidence complex closure gate before oblique; channel-dictionary acceptance test at θ=30° |
| Model-error floor misestimated | σ from measured complex closure, never the magnitude-only 3e-3; propagate through J to per-entry error bars |
| CST bounding-box period trap / domain drift | explicit `UnitCellDs1/Ds2` + pinned cellpad; empty-cell analytic check per angle |
| Angle set under-determines even-m blocks | synthetic observability map computed before the campaign; even-m content needs oblique + φ=22.5° runs (§4) |
| lmax=3 truncation (Wiscombe suggests 5) | same truncation as reference & Stage 3 ⇒ consistent comparison; flag as shared systematic |

## 10. Implementation checklist (new session; suggested `src/tmatrix/retrieval/` package)

Numerical parameters (normative, from the validated
`tmatrix.aggregation.run_demo` setup):
`modes = TMatrixData('test/single/saw_gold_wl15p0025um.tmat.h5').modes`
(30-mode `ModeBasis`, order from `/modes`); pitch 2.0 µm; A_cell 4.0 µm²;
`r0 = 0.8`; `quad = make_quad(16, 32)`; `kRc = (10, 14, 20)` at θ=0, scaled
per §2 at oblique; frequency grid = the file's 49 points.

1. `src/tmatrix/retrieval/bloch_lattice.py` — `lattice_sum_C_bloch(...)`
   reusing `translation_shells`/`assemble_shell_sum`; tests: k∥=0 ≡
   `lattice_sum_C` (1e-12); Richardson stability at θ=60°, λ=20 µm.
2. `src/tmatrix/retrieval/sparams_oblique.py` — Jones blocks per §2 incl. the
   normative TE/TM basis and its θ→0 continuity limit; test: θ=0 ≡
   `sparams_normal`.
3. `src/tmatrix/retrieval/forward.py` + `src/tmatrix/retrieval/precompute_C.py`
   — `predict_S` reading the `C_bloch.npz` cache; treams validation script
   (§6 step 1, incl. the oblique treams generalization).
4. `src/tmatrix/retrieval/parametrize.py` — C4v×reciprocity basis per §4
   (numerical σ_v derivation, group-closure tests, explicit σ_v selection-rule
   validation).
5. `src/tmatrix/retrieval/fit.py` + `src/tmatrix/retrieval/observability.py` —
   LM driver (Born seed, `w_i = 1/σ_i²`), Jacobian from cache, SVD resolution
   + heatmap `H[μ,ν] = Σ_k res_k |B_k[μ,ν]|²`.
6. `src/tmatrix/retrieval/synthetic_test.py` — ladder steps 2–3.
7. `src/tmatrix/retrieval/cst_campaign.py` — §7 edits to
   `tmatrix.aggregation.cst_direct.build_saw_unitcell`
   (cellpad, both-mode excitation, θ/φ parametrization), empty references,
   channel-dictionary acceptance run. Dependencies: `D:/Claude/auto_cst`
   (`nir.cst_helpers`), `E:/cst/AMD64/python_cst_libraries`, cma_infinite
   clone.
8. `src/tmatrix/retrieval/deembed.py` +
   `src/tmatrix/retrieval/validate_against_reference.py` —
   complex de-embedding + closure gate (§7), acceptance criteria (§8),
   figures.

Environment: conda env `cst_inference` (numpy/scipy/h5py, treams importable
per `tmatrix.aggregation.treams_reference`).

---

## Appendix A — note for the owners of the isolated-cell extraction code

Independent of this direction, two audit findings (Aug 2026) should reach
whoever maintains `cst_tmatrix` / the shipped scripts:

- `src/tmatrix/extraction/compute_t_matrix_projection.py` (the in-repo
  template, *not* the production tool) has a numerically confirmed sign error
  (Green-identity
  integrand needs `E×curl Ψ* − Ψ*×curl E`; the shipped `+` leaks the incident
  field), a missing Wronskian normalization, and a pole clamp that zeroes
  |m|=1 coefficients for polar illumination nodes.
- `README.md` cites the tmat.h5 standard as arXiv:2404.10399 (an unrelated
  robotics paper); correct: Asadova et al., arXiv:2408.10727 (JQSRT 333,
  109310, 2025).

## Appendix B — key references

- Rahimzadegan et al., Adv. Opt. Mater. 10, 2102059 (2022); arXiv:2108.12364 —
  `Teff = (I − T0·Cs)⁻¹T0` to octupole order; symmetry-zero tables.
- Beutel, Fernandez-Corbaton, Rockstuhl, treams, Comput. Phys. Commun. 297,
  109076 (2024) — oblique Ewald reference.
- Scher & Kuester, Metamaterials 3, 44 (2009); Karamanos et al., IET MAP 8,
  1398 (2014) / PIER B (2018) — dipole-order inverse retrievals (prior art).
- Grahn, Shevchenko, Kaivola, New J. Phys. 14, 093033 (2012) — multipole
  degeneracy / non-uniqueness.
- Allayarov, Evlyukhin, Calà Lesina, Laser Photonics Rev. (2026);
  arXiv:2510.11864 — representation dependence for connected cells.
- Schab et al., IEEE TAP 71(12) (2023) — empty-cell calibration discipline.
- Asadova et al., JQSRT 333, 109310 (2025); arXiv:2408.10727 — tmat.h5
  conventions.

# Session handoff — multi-angle Floquet → T-matrix retrieval implementation

> **Path note (later than this handoff).** The code was reorganized into an
> installable package: every `retrieval/*.py` referenced below now lives at
> `src/tmatrix/retrieval/*.py`, `aggregation/*.py` at
> `src/tmatrix/aggregation/*.py`, and the test suites under `tests/`. The
> directories named here still exist and still hold this study's *data*
> (`retrieval/cst_runs/`, `retrieval/results/`). Conventions below are
> unchanged and remain normative.

**Written 2026-08-06 ~21:55 by the previous (Fable) session at its usage limit.
You are the successor session. Read this file and
`../INVERSE_TMATRIX_FROM_FLOQUET.md` (the normative design doc) before doing
anything.**

## Your role (per the user's standing instruction)

Act as **verifier-orchestrator**: delegate implementation work to subagents
(Agent tool, general-purpose) and verify their output yourself — read every
line of delivered code against the design doc and the validated baseline
modules, then re-run the cheap correctness gates before accepting. Do not
implement large items inline; do not accept subagent self-reports without
re-running gates. That protocol caught real bugs in every wave so far.

## Environment (all verified working)

- Repo: `D:\Claude\T matrix` (space in path — always quote). Branch
  `Fast-T-Matrix-Trail`, do NOT commit unless the user asks. Do NOT modify
  `src/tmatrix/aggregation/` or any validated file — it is shared with the
  `dary` branch, which owns the aggregation study; new work goes in
  `src/tmatrix/retrieval/`.
- Python: conda env `cst_inference`. Bash:
  `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate cst_inference && cd "/d/Claude/T matrix" && python -m tmatrix.retrieval.<module> ...`
- Bash calls cap at 10 min; long jobs → `run_in_background` or checkpointed
  scripts.
- Reference data: `../test/single/saw_gold_wl15p0025um.tmat.h5`
  (49 freqs, 30 modes, lmax 3; `tmat_io.TMatrixData`).
- External deps present: `D:/Claude/auto_cst`, `D:/Claude/cma_infinite`
  (shallow clone, made this session), `E:/cst/AMD64/python_cst_libraries`
  (cst.results works offline; cst.interface needs CST).

## State of the doc §10 checklist

| # | Deliverable | Status |
|---|---|---|
| 1 | `bloch_lattice.py` | **DONE + VERIFIED** (brute-force per-site gate 1.3e-15; k∥=0 ≡ `lattice_sum_C`; θ=60° converges at kRc×7.464 — Ewald fallback NOT needed) |
| 2 | `sparams_oblique.py` | **DONE + VERIFIED** (θ→0 ≡ `sparams_normal` exactly, incl. the S11 TM-row sign flip; docstring mapping table is normative) |
| 3 | `precompute_C.py`, `forward.py`, `treams_oblique_validate.py` | **DONE + VERIFIED**. Full cache built: `results/C_bloch.npz` (49×17, 12 MB) + per-freq checkpoints `results/C_bloch/freq_XX.npz` (all 49 at 17/17 angles). **Treams gate (§6 step 1) CLOSED full-band: worst complex \|ΔS\| = 3.66e-7 vs 1e-3 gate** (`results/treams_oblique_check.npz`) |
| 4 | `parametrize.py` | **DONE + VERIFIED** (40/40; ranks 228/114/68 exact; bright basis = **10**, not the doc's 11 — see caveats) |
| 5–6 | `fit.py`, `observability.py`, `synthetic_test.py`, `test_fit_smoke.py` | **DONE + INDEPENDENTLY VERIFIED** (2026-08-06/07). All machinery gates re-derived by a check script that does not import the delivered tests; analytic Jacobian confirmed against an O(h⁴) Richardson FD to **1.1e-11** (5 decades tighter than the shipped 1e-6 gate). Full record: `VERIFICATION_ITEMS_5_6.md`. The §6.2/§6.3 gate failures reproduce but the single-cause "information limit" framing is **WRONG** — see `RETRIEVAL_LIMIT.md` |
| 7 | `cst_campaign.py` | **DONE + VERIFIED** (19-run manifest in `cst_runs/`; `--build` creates projects only). 8 starter projects **BUILT** 2026-08-07. Solve driver is now `cst_solve.py` (separate module) |
| 8 | `deembed.py` (de-embed half) | **DONE + VERIFIED** (10/10 synthetic tests) |
| 8b | `validate_against_reference.py` | **DONE + VERIFIED** — 20 machinery gates PASS on re-run, exit 0. §8.1 held-out acceptance **PASSES**: fit 4 angles, predict 9, worst \|dS\| **6.0e-4** vs the 1e-2 gate (9.1e-3 at σ=3e-3). Carries the new χ² channel-dictionary discriminant |
| 9 | `cst_solve.py` | **DONE + VERIFIED** — sequential, checkpointed, resumable; §7 ordering enforced *in the execution loop* (`enforce_order`, cst_solve.py:1384), not merely documented. **Exactly one solver call site: `cst_solve.py:418`** (`m3d.FDSolver.Start()`), byte-identical to `build_saw_unitcell.py:269`. Use `--timeout-min 0` (the watchdog path is untested) |
| 10 | `gate_study.py` | **DONE** — span/estimator/angle-set/threshold ladders; `results/GATE_STUDY.md` |

Every "VERIFIED" row: I read the full source and re-ran the gates myself;
all numbers reproduce.

## Normative conventions (verified — do not re-derive, do not let a subagent "fix")

- Bloch sign **positive**: C(k,k∥) = Σ_{R≠0} A(R) e^{+i k∥·R}.
- Jones index order **0 = TE, 1 = TM**; rows = receive, cols = incident;
  blocks S11 (reflection), S21 (transmission); prefactor 2πi/(kA cosθ),
  cosθ = |k̂_z|.
- θ→0 at φ=0: every S21 entry and the S11 TE row map **+**, the **S11 TM row
  maps with −1** vs `sparams_normal` (ê_TM^(r) → −x̂). Table in
  `sparams_oblique.py` docstring.
- Campaign/CST/treams illumination: **direction = −1** (down-going,
  k̂ = (sinθcosφ, sinθsinφ, −cosθ)); the C cache is direction-independent.
- σ_v(xz): (l,m,E) → +(−1)^m (l,−m,E), (l,m,M) → −(−1)^m (l,−m,M).
  Reciprocity: `Rec(T) = (−1)^{m+m'} · T[ix(perm,perm)].T` (transpose, NO
  conjugation), perm: (l,m,p)→(l,−m,p).
- Taper scaling s(θ) = 1/(1−sinθ), kRc(θ) = (10,14,20)·s(θ),
  r_max = 3.5·max(kRc)/k. 17-angle table order = campaign 13 (idx 0–12:
  (0,0),(15,0),(15,22.5),(15,45),(30,·),(45,·),(60,·)) then validation 4
  (idx 13–16: (20,0),(20,45),(40,0),(40,45)).
- treams kpar sign **+1** (empirical, two-angle confirmed;
  `results/treams_kpar_sign.npz`).
- CST data is e^{+jωt}: **conjugate on load** (`deembed.conj_cst`).
  Empty-cell arg(S21) must ADVANCE with f (sign = hard gate).
- Pinned domain: Z_PAD = 3 µm ⇒ L_expected = **11.714687 µm**
  (= 6 + 2×λ_center/4; auto-space rule verified on run_v3 to 10 digits).
- Noise σ for fit weights: **placeholder 3e-3** until the campaign's
  normal-incidence complex closure measures it (legacy preview: 3.1e-3,
  S11-phase-dominated).

## CST CAMPAIGN EXECUTED AND COMPLETE — 2026-08-07

All 19 runs solved, all §7 gates PASSED. Checkpoints in `cst_runs/<runid>/`
(`solve_result.npz` + `solve_status.json`), gate records in
`cst_runs/gate_{closure,acceptance}.json`.

- **Timing: the doc's "tens of minutes per solve" was wrong by ~30×.**
  Structure runs 50–78 s, empty runs 16–23 s; the whole campaign is ~13 min
  of solver time. Do not budget hours.
- **Closure gate PASSED**: worst complex residual 3.617e-3 vs the 5e-3 gate.
  **σ is now MEASURED: 2.6333e-3** (RMS over channels+band; per-frequency
  2.31e-3…3.62e-3, median 2.68e-3), S11-phase-dominated. The 3e-3 placeholder
  was right. `results/fit_sigma_from_closure.npz`. **Use this everywhere the
  placeholder appeared** (fit weights, observability λ, gates).
- **All 19 runs report the pinned domain at L = 11.714687 µm, diff +0.00 nm.**
  Every empty cell at every θ passes the phase-ADVANCE gate (rel_err ≤ 7e-5,
  ||S21|−1| ≤ 3.4e-4, |S11_empty| ≤ 7.1e-5). Domain, phase reference,
  conjugation direction and scan-angle convention ("inward") are all pinned by
  measurement.
- **Acceptance gate PASSED at (60°,22.5°)** after the hypothesis family had to
  be extended — see the §7 amendment in the design doc. Winner
  `{swap: False, s11_cross: -1, s21_cross: -1, r11_tm: -1}`, χ²_red 1.595,
  z 5.77. **CAUTION: only 1.15× above z_min; the marginal dimension is
  `s21_cross` and at 2σ it would refuse.** Mitigation (pool χ² over the four
  φ=22.5° angles, all now solved) is in progress.
- **Model-free confirmation of the label map**: mirror-plane cross-pol on
  `struct_th60_ph00` measures **4.34e-5** (tol 5e-3) — no reference model
  involved. Two independent routes agree.
- **Real cross-pol is tiny**: 9.8e-5 at (0,0), 4.3e-5 at (60,0). The reference
  file's ~7e-3 at mirror planes is its own C4v-violation noise. Never
  calibrate an acceptance against raw-reference-T separation numbers.

## VERIFIED FINDINGS 2026-08-06/07 — these supersede "Wave 3 outcome" below

Read `RETRIEVAL_LIMIT.md` first; it is the adjudicated decomposition. Headlines:

- **The bright span represents the bright entries EXACTLY** (3e-18 on every
  bright-mask entry, at every threshold tested). Recovery failures are never
  representability failures. Assert this identity for any new span.
- **Blind retrieval is limited by the OPTIMIZATION LANDSCAPE, not the angle
  set.** The rich span (threshold 3e-5, rank 53) *has* the information — a
  truth-seeded noise-free fit recovers the dipole to 0.41 % (12 µm) / 2.3 %
  (8 µm), meeting the doc §6.3 target — but no realizable seed reaches it.
  Adding CST angles cannot fix this.
- **Frequency continuation fixes the basin only when the target lies in the
  span.** C-clean loop ifreq 48: Born 11.04 → continuation **6e-12**. On the
  physical target it does NOT transfer to the rich span (5.75 vs truth 0.023).
- **An orbit-pure basis is a mathematical NO-OP** under an isotropic prior
  (both real-orthonormal for the same span ⇒ ‖t‖² invariant; confirmed to
  3.5e-7). Only an orbit-*scaled* prior helps, and it needs the truth —
  realizable for the §1 QA-gate use case, not for blind retrieval.
- **No realizable protocol beats `T̂ = 0`** on the dipole class at σ=3e-3
  (23 candidates). The QA-gate application does work (gain 8.17×).
- **The doc §7 acceptance was measured against an artifact.** Cross-sign
  hypothesis separation at mirror-plane angles is ENTIRELY the reference
  file's C4v-violation noise; on a physical C4v cell it is 4e-16. Acceptance
  must run at **(θ=60°, φ=22.5°)** with a **pooled χ² discriminant**. Doc §7
  amended. Never use raw-T separation numbers.
- **treams gate extended**: now closed at (15,22.5), (30,0), (30,22.5),
  (30,45), (45,22.5), (60,0), (60,22.5), (60,45) — worst 5.5e-7 vs 1e-3.
  Growing the taper 1.4× moves our S toward treams (4.3× reduction), proving
  the residual is shell truncation and that **Ewald fallback is NOT needed at
  θ=60**. Still unvalidated: indices 1 (15,0), 3 (15,45), 7 (45,0), 9 (45,45).
- **Recommended angle set `ext_phi`** = θ∈{0,30,60} × φ∈{0,22.5,45}
  (indices [0,4,5,6,10,11,12]) = 7 structure + 3 empty = 10 solves, vs the
  doc's 19. Rests on protocol-independent observability counts (26/26
  directions at 12 µm, 27/30 at 8 µm), NOT on fit-error columns.
- **Identified next lever, NOT built** (out of scope): frequency smoothness as
  a *constraint* (joint band fit penalizing ‖t(f) − t(f−1)‖²), not as a seed.
  Continuation-as-seed adds no information; smoothness-as-constraint does, is
  realizable without oracle knowledge, and is justified because T is analytic
  in frequency. Doc §5's hybrid completion is the other route.

## Wave 3 outcome (SUPERSEDED — kept for provenance; its single-cause framing is wrong)

**Machinery is proven sound** by gates the predecessor re-ran or that are
triple-redundant in the report: truth-seeded closed loops recover exactly
(obj ≤1e-16, entries ≤1e-6); the analytic Jacobian matches an independent
central-FD Jacobian to 5e-8 (and is the default because MINPACK's internal
FD 'lm' thrashed — >21,900 evals without converging); noise-trial errors
match linear posterior theory (measured/predicted rms ~0.8–1.9; chi²
correct).

**The doc's synthetic gates FAIL as measured — reported as an
information-content finding, not a code defect:**
- §6.2 (noise-free bright recovery ≤1%): FAIL at both smoke freqs under
  every protocol. Bright-span model-error floor is ~1e-3 in S (sub-threshold
  entries carry |dS/dT| ~ 70–800 and shadow the bright span); dominant
  E-dipole diagonals recover to ~1–6%, but 1e-4-peak orbit entries sit at up
  to 60× their own size. Born seeding also lands in false basins at physical
  targets (Born-reach 1.45/11.0).
- §6.3 (dipole ≤5% at σ=3e-3): FAIL (E-dipole 0.31–0.81 peak-normalized at
  ifreq 32/48 class stats) — weak M/z-dipole leverage at 13 angles.
- §4 visibility claims CONFIRMED numerically: at θ=0, (3,∓3)↔(1,±1)
  sensitivity 37, (3,±3)↔(3,±3) 294, dipole 4.6; even-m 5e-21 at θ=0 →
  65.5 with obliques. Born rank 58/136 real params; 76–78/136 SVs above the
  σ=3e-3-referenced λ.
- Highest-value refinements identified by the agent: orbit-pure bright
  basis (current SVD basis mixes orbits, shrinkage pollutes strong
  entries); extended threshold-3e-4 basis (25 complex dirs, theoretical
  floors 3.6–4.5 vs 8.6–23); frequency-continuation seeding;
  replace σ placeholder with the measured closure floor.

## ALL SIX STEPS BELOW ARE COMPLETE (2026-08-07)

Items 1–6 of the list that follows were executed and verified; the list is
kept for provenance. Read `RESULTS.md` for the outcome. What a successor
should consider next, in value order:

1. **Frequency smoothness as a CONSTRAINT** (joint band fit penalizing
   ‖t(f) − t(f−1)‖²) — the identified lever for blind retrieval, deliberately
   not built (doc §4 defers it to v2). Continuation-as-a-seed is a different
   thing and adds no information.
2. **Doc §5's hybrid completion** — periodic data + a few isolated-cell runs
   for the dark residual.
3. **Harden `s21_cross`** if a blind campaign ever depends on it: pooling all
   four φ=22.5° angles reaches z = 6.95–7.36 but still refuses at 2σ.
4. Excite Zmin modes too, if cross-port transmission reciprocity is wanted as
   a check (this campaign drove Zmax only).
5. Per-frequency fit weights: σ(f) was measured across 2.31e-3…3.62e-3 but the
   fits use the single band RMS.

## Original next steps, in order (ALL DONE — kept for provenance)

1. **Fully verify items 5–6.** Read `fit.py`, `observability.py`,
   `synthetic_test.py` completely against doc §4/§6 and the conventions
   above; re-run `python synthetic_test.py` at ifreq 32, 48 and
   `python observability.py --ifreq 32 --basis bright --angles campaign`;
   check weights semantics match `ForwardModel.pack_S` (√w on both Re/Im),
   Born seed t=0, direction=−1, no mutation of cached C. Scrutinize
   especially the claims above: the gate-failure finding is
   conclusion-changing, so confirm the truth-seeded loops and the
   model-error floor measurement are what the report says they are
   (`test_fit_smoke.py` already re-ran PASS; `results/synthetic_ifreq*.npz`
   hold the arrays).
2. **Adjudicate the §6.2/§6.3 gate failures** after verification. If
   confirmed: this is a real result about the campaign-13 information
   content — amend the design doc's gates (per-entry-class gates weighted
   by sensitivity, or adopt the agent's orbit-pure / 3e-4-threshold basis
   refinements and re-measure), and record the decision in the doc. The §8
   real-data acceptance criteria inherit whatever recalibration you make.
3. **Full-band synthetic study** (`python synthetic_test.py --freqs all`,
   ~5 min/freq incl. figures; discovers cached freqs via `fm.have`). Run
   AFTER step 2's adjudication so the gates it reports are the recalibrated
   ones.
4. **Write `validate_against_reference.py`** (item 8, second half;
   subagent): held-out-angle acceptance (§8: fit on 4 angles, predict rest
   ≤1e-2), bright-entry comparison vs reference tmat.h5 (5–10% per §8, as
   recalibrated by step 2), discrepancy-vs-observability consistency, the
   (θ=30,φ=0) channel-dictionary acceptance hook
   (`deembed.select_hypothesis` needs the forward model with reference T),
   figures. Verify it. This must exist BEFORE stage 6's acceptance test.
5. **Wrap-up**: results summary doc; update
   `INVERSE_TMATRIX_FROM_FLOQUET.md` status header (design → implemented,
   measured numbers vs its estimates — see caveats + Wave 3 outcome).
6. **CST solve campaign + real-data retrieval — the FINAL stage, and you
   DO execute it** (user instruction 2026-08-06: "cst solve should be the
   last stage by the agent"). Do it only after steps 1–5 are done, because
   step 2's adjudication decides the angle set worth paying CST hours for
   (Wave 3 measured campaign-13 as information-limited — decide
   starter-5 vs full-13 vs an extended set BEFORE solving). Execution
   notes:
   - `cst_campaign.py --build` creates the projects (needs cst.interface,
     i.e. CST on this machine at `E:\cst`); `--solve` intentionally
     refuses — implementing the solve driver (extend `cst_campaign.py` or
     a new `retrieval/cst_solve.py`, via subagent + your verification) is
     part of this stage. Run solves SEQUENTIALLY (license), each as a
     checkpointed background job with logs; each FD solve is tens of
     minutes at run_v3 scale, budget hours for the campaign.
   - §7 order is mandatory: (i) `struct_th00_ph00` + `empty_th00` →
     re-establish `deembed.closure_normal` ≤5e-3 — its residual spectrum
     IS the real fit σ (replaces the 3e-3 placeholder everywhere,
     including the observability λ and step-3 gates); (ii)
     `struct_th30_ph00` + `empty_th30` → channel-dictionary acceptance
     via `deembed.select_hypothesis` + `validate_against_reference`'s
     forward model (exactly ONE of 8 hypotheses ≤1e-2; if 0/8, extend
     the hypothesis family per the caveat below); (iii) remaining runs
     (perturbed empty feeds noise-floor calibration only).
   - Then: de-embed all runs → per-frequency real-data fit (fit on the
     chosen angle subset, hold out the rest) → §8 acceptance criteria
     (held-out ≤1e-2 complex; bright entries vs reference at whatever
     tolerance step 2 recalibrated; passivity/reciprocity checks;
     observability heatmap published with the fit) → final results doc.

## Caveats / deviations already adjudicated (don't re-litigate)

- **Bright basis = 10, not the doc's "11–15"**: the 11th position-orbit is
  purely the two C4-violating noise entries; the C4 average annihilates it.
  Gated on rank == number of C4-conforming orbits.
- **Doc's "~1–2 h" precompute estimate was optimistic**: actual 5.8 h serial
  / 96 min ×4 workers (per-θ taper scaling superset). Done; cache exists.
- **Label-hypothesis family (8) excludes a possible S11 co-pol port gauge**
  (same mode oriented oppositely at Zmax vs Zmin). Fail-safe: acceptance
  refuses (0/8) rather than mis-picking; remedy = extend
  `deembed.label_hypotheses` with per-block co-pol signs.
- **Legacy run_empty never solved** (zero solids, "Could not read mesh") —
  the magnitude-only 3e-3 record used no empty division. Pinned cellpad is
  what makes the empty solvable at all. Legacy complex closure (analytic
  empty): 3.1e-3, S11-phase-dominated (`results/legacy_v3_*.npz`).
- treams TE-diagonal vs stored x-pol reference differs by 3.1e-3 — that is
  the reference T's own C4-violation noise, not a pipeline error.
- `mirror.py`'s `mirror_parity_signs` is the HORIZONTAL mirror (C4h trap) —
  never use it for the C4v projector; `parametrize.py` guards this at
  runtime.

## Where the previous session's in-flight agent report might be

The items-5–6 subagent's transcript lives under
`C:\Users\93107\AppData\Local\Temp\claude\D--Claude-T-matrix\<session-id>\tasks\`
(JSONL, large). Do NOT read whole files from there into context; if you must,
extract only the final entry (`tail -c` a few KB). The on-disk code + re-run
tests supersede it.

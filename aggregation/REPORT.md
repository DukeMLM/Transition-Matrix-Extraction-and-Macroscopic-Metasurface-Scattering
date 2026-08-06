# T-Matrix Array Aggregation → S-Parameters: Implementation & Validation Report

**Task**: Implement Stage 3 of the T-matrix metasurface pipeline (ref: `ref/main.tex`,
Padilla, *Operational Manual: Transition Matrix Extraction and Macroscopic Metasurface
Scattering*): (1) aggregate per-unit-cell T-matrices into one array system, (2) convert
the aggregated response into S-parameters, run on the demo file
`test/single/saw_gold_wl15p0025um.tmat.h5`.

---

## 1. The algorithm (understanding + validation)

The manual's Stage 3 prescribes:

1. **Foldy–Lax multiple scattering.** Each cell *i* has `f^i = T^i a^i` (outgoing ←
   regular VSWF coefficients). Cells couple through translation-addition operators
   `A^{ij}` that re-expand cell *j*'s outgoing waves as regular waves at cell *i*:
   `a^i = a_inc^i + Σ_{j≠i} A^{ij} f^j`. This closes into one global block system
   `(I − A·diag(T)) a = a_inc`.
2. **S-parameters** from the plane-wave projection of the total scattered far field:
   `S21 = 1 + (2πi/(kA cosθ)) ê*·F(k_fwd)`, `S11 = (2πi/(kA cosθ)) ê*·F(k_spec)`.

**Verdict: the algorithm is sound.** An independent audit of every equation in the
manual (Wiscombe bounds, N = 2L(L+2), VSWF construction, Foldy–Lax forms,
T_coupled = (I − T_iso R)⁻¹ T_iso, S-parameter formulas) confirmed them against the
standard literature, with these findings:

- **(minor, documentation)** The manual's S21/S11 equations use `+j` (Eq. for S21),
  which is correct in the *physics* convention e^{−iωt} — but the manual's own
  `(−j)^n` prefactors and `Im(k_z) ≤ 0` branch choice imply the *engineering*
  e^{+jωt} convention, where the sign must be `−j`. The tmat.h5 file (and this
  implementation) use e^{−iωt}, where `+i` is correct.
- **(minor, notation)** The dagger on `D_ν†` in the reflection-matrix integrand reads
  as complex conjugation, which would break analyticity over the evanescent spectrum;
  it should denote a transposed projection operator.
- **(caveat)** The demo file truncates at lmax = 3, while Wiscombe at the
  circumscribing radius a = 0.721 µm and f_max (λ = 8 µm; x = ka ≈ 0.57) suggests
  L = 5 (70 modes vs 30). Wiscombe is conservative at small x and the extraction
  diagnostics look healthy, but an L-convergence check is advisable for production.
- **(caveat)** The area A in the S-parameter formulas is the unit-cell area for the
  periodic case and the *total* array area (asymptotically) for finite arrays.
- **(caveat)** pitch/(2·r_circ) = 2.0/1.44 = 1.39: the nearest-neighbor translation
  re-expansion is formally valid (2.0 > 1.44) but slowly convergent — another reason
  the lmax-truncation caveat matters for tightly packed cells.

## 2. Implementation

All in `aggregation/` (pure numpy/scipy; conventions locked to the tmat.h5 standard —
e^{−iωt}, h_l^{(1)}, Jackson-normalized VSWFs with Condon–Shortley, parity basis):

| file | contents |
|---|---|
| `vswf.py` | VSWF evaluation (M/N, regular/outgoing), plane-wave expansion, far-field amplitude, sphere-projection of (E,H) → regular coefficients (GEMM-batched) |
| `translate.py` | Translation operators A(d) by numerical field projection (convention-safe by construction); in-plane rotation identity `A(Rot_φ d) = e^{i(m_ν−m_μ)φ}A(d)`; shell-grouped lattice sum with Gaussian taper + **Richardson extrapolation in 1/Rc²** (the taper's g=0 channel error is an even series in 1/(kRc) — Dawson asymptotics) |
| `aggregate.py` | **Deliverable 1**: finite-array Foldy–Lax global block system; periodic unit-cell solve `a = (I − CT)⁻¹a_inc`; effective array T-matrix `T_eff = T(I − CT)⁻¹` |
| `sparams.py` | **Deliverable 2**: S11/S21 (co/cross-pol), energy balance, single-scatterer cross sections (optical theorem) |
| `mirror.py` | Stage-2-lite bonus: PEC ground plane via exact image theory (image parity signs, image-lattice sum, `f = (I − T R_m)⁻¹ T a_inc` — the manual's T_coupled with R = image operator) |
| `tmat_io.py` | tmat.h5 reader |
| `run_demo.py`, `run_mirror_demo.py`, `plot_results.py` | drivers |

## 3. Validation (all tests in `test_*.py`, all passing)

**Layer 0 — VSWF core** (`test_vswf.py`)
- Maxwell curl consistency (FD): 2e-9 · plane-wave reconstruction (E and H): 3e-8
- far field vs direct at kr=1e5: 5e-5 · projection roundtrip: 1e-13
- **Mie sphere: optical theorem vs |F|² integral vs classic Mie series: 3e-15**

**Layer 1 — translation/lattice** (`test_translate.py`)
- re-expansion field check: 2e-7 · operator r0-invariance: 3e-7
- rotation identity: 4e-10 · lattice-sum taper/quadrature stability: ≤2e-6

**Layer 2 — end-to-end**
- T-matrix diagnostics: passivity max SV(I+2T) = 1.00007; reciprocity residual
  0.006–0.012 — **matches the file's own stored metric exactly** (conventions aligned)
- energy balance A ∈ [0,1] at all 49 frequencies; reciprocity S21 = S12 to 2e-4
- finite arrays (5×5 → 9×9 → 13×13) converge monotonically to the lattice-sum result
- **External cross-check (treams, independent Ewald-based code): max complex-amplitude
  deviation 2.7e-4 over all 49 frequencies** (fig4)
- **Feature fidelity** (`test_feature_fidelity.py`): a synthetic 15-µm Lorentzian
  dipole T pushed through the same pipeline produces a deep collective resonance
  (min |S21| = 0.265 at 18.0 µm; the 15→18 µm shift is the dense-lattice coupling,
  **reproduced by treams to 3 decimals in depth and the same wavelength**)
- **Mirror unitarity** (`test_mirror.py`): lossless Mie array over PEC gives
  R = 1.000000 (machine-exact energy conservation through the full image machinery)

## 4. Demo results (`results/`)

- `fig1_cross_sections.png` — isolated resonator σ_ext/σ_sca/σ_abs
- `fig2_sparams_periodic.png` — infinite array |S11|, |S21|, R/T/A
- `fig3_finite_vs_periodic.png` — finite-array convergence
- `fig4_treams_crosscheck.png` — overlay with treams (indistinguishable)
- `fig5_feature_fidelity.png` — synthetic-resonance transfer
- `fig6_mirror_absorber.png` — ground-plane (image-theory) absorber response
- `sparams_periodic.csv` — S11/S21 spectra (complex + magnitudes + R/T/A)

**Key physical finding.** The free-standing array's S-parameters are smooth and
featureless (|S21| ≈ 0.99→0.94 from 20→8 µm, |S11| ≈ 0.12→0.32): *the isolated
resonator has no resonance inside the 8–20 µm band* (its σ_ext rises monotonically
toward 8 µm — the tail of a resonance below 8 µm). This is not a pipeline artifact
(the feature-fidelity test proves features transfer faithfully; treams reproduces
the same flat curves). The designed λc = 15 µm response of this structure is an
**MIM ground-plane resonance** — the demo file is `T_iso` (Stage 1: substrate/ground
omitted by design). Once a ground plane is added via image theory (Stage-2-lite,
`run_mirror_demo.py`), an absorption resonance emerges at ~13 µm (peak A = 0.093)
for the design's mirror distance h = 350 nm — and vanishes at h = 550 nm —
qualitatively recovering the absorber behavior near the design λc = 15 µm.
The residual position/amplitude discrepancy is expected: PEC mirror + vacuum
spacer replace the real dielectric spacer (its permittivity is not in the file),
and the image distance 2h = 0.70 µm sits at the Rayleigh-hypothesis boundary of
the r_circ = 0.72 µm cell, where the lmax = 3 truncation underestimates the
near-field image coupling.

**Noise note.** The digitized-field-monitor noise in T (reciprocity residual ~1%,
passivity overshoot 7e-5) propagates to ≲1e-3 in |S| — invisible at the "overall
shape" tolerance requested.

## 4b. Direct CST validation ("the real S-parameters")

A direct CST frequency-domain periodic simulation of the same free-standing
array (`cst_direct/build_saw_unitcell.py`: unit-cell Floquet boundaries,
pitch 2 µm, lossy gold σ = 4.561e7 S/m, geometry from the tmat.h5 attributes,
verified visually against the design sketch) agrees with the T-matrix
aggregation to

    max | |S21|_CST − |S21|_aggregation | = 0.0011
    max | |S11|_CST − |S11|_aggregation | = 0.0029

across the whole 8–20 µm band (`results/fig7_cst_direct_comparison.png`).
This closes the loop end-to-end: CST near-field extraction → T-matrix →
Foldy–Lax aggregation → S-parameters reproduces a direct CST periodic
solve to ~0.3% — which also bounds the total error contributed by the
digitized-monitor extraction itself.

**Pitfall discovered on the way** (cost two wrong runs, documented for future
CST automation): with `Boundary "unit cell"`, CST sizes the transverse unit
cell to the *geometry bounding box* unless told otherwise. The spoke-wheel's
bounding box is its 1.44 µm diameter, so the first runs simulated a lattice of
*touching* rings — a connected inductive mesh reflecting ~97% (the solver even
warned "edges ... treated as infinitely thin PEC wires" at the fused contacts).
Diagnosed by control tests (a 1 µm patch behaved identically to a continuous
sheet); fixed with `Boundary.UnitCellFitToBoundingBox "False"` +
`UnitCellDs1/Ds2 = pitch`. The archived extraction project itself
(`saw_gold_wl15p0025um.cst`, 48 KB) contains no result data — the reference
here is a fresh periodic solve.

## 5. How to reproduce

```
conda activate cst_inference
cd "D:/Claude/T matrix/aggregation"
python test_vswf.py && python test_translate.py && python test_mirror.py
python run_demo.py          # ~8 min: 49 freqs, periodic + finite arrays
python test_feature_fidelity.py
python run_mirror_demo.py   # ~15 min: ground-plane variant
python plot_results.py
```

For a new multi-atom problem: load per-atom T-matrices, call
`aggregate.build_finite_system` + `solve_finite` (arbitrary in-plane positions,
mixed T's supported), then `sparams.sparams_normal` with A = N·A_cell; for periodic
arrays use `translate.lattice_sum_C` + `aggregate.solve_periodic`.

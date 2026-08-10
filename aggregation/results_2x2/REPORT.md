# `test/2x2` — T-matrix → S-parameters

Stage-3 reconstruction of `test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5`.

```bash
python -m tmatrix.aggregation.run_case test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5 --pitch 8.0 --r0 3.0 --lmax auto --finite 2 3 5 9 --out results_2x2
python -m tmatrix.aggregation.treams_case test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5 --pitch 8.0 --out results_2x2/treams_reference.npz --lmax auto
python -m tmatrix.aggregation.cst_packed_reference test/2x2/SAW_gold_noSub_packed.cst --list
python -m tmatrix.aggregation.cst_packed_reference test/2x2/SAW_gold_noSub_packed.cst --run 6 --out results_2x2/cst_direct_reference.csv
python -m tmatrix.aggregation.plot_case results_2x2
```

## Input

| | |
|---|---|
| T-matrix | 25 frequencies, 10–34 THz (λ 8.82–29.98 µm), lmax 5 → 70 modes, vacuum embedding |
| provenance | `cst_tmatrix` CST driver, merged from a 10–18 THz and a 19–34 THz extraction |
| lattice | square, **pitch 8.0 µm**, free-standing, normal incidence, x-polarized |
| projection radius | r₀ = 3.0 µm |

The pitch is not in the h5. It was read from the CST project sitting next to it
(`p = 8` in every row of its parametric run table, and the unit cell is the
vacuum brick `Xrange -p/2 … p/2`) and then confirmed independently from that
run's Floquet port data: the first higher-order mode cuts off at
**37.474 THz = c / 8 µm**.

λ_min/pitch = 1.102, so only the 0th diffraction order propagates across the
whole band, but the Rayleigh onset at λ = 8 µm is close to the top of it.

## Result

`periodic_results.csv` / `.npz`, `fig1_sparams.png`, `fig2_balance_truncation.png`.

* Transmission minimum (parabolic fit in frequency): **13.10 µm** — 13.097 µm
  from this code, 13.075 µm from treams, 13.126 µm from the direct CST run.
  At the dip |S21| = 0.06, R = 0.89, A = 0.11.
* Cross-polarized |S| ≤ 5×10⁻¹⁴ (C4-symmetric cell, as expected).
* A = 1 − R − T ∈ [0.010, 0.114] — non-negative everywhere.

Finite N×N Foldy–Lax arrays converge toward the periodic result from both
sides (`finite_results.npz`). At 5 sampled wavelengths the 2×2 array differs
from the infinite lattice by |Δ|S21|| = 0.005–0.15 (largest near the
resonance, at 12 µm), 3×3 by 0.01–0.08, 9×9 by 0.005–0.03.

## Multipole truncation is required — `--lmax auto`

Running with all 70 modes gives A < 0 at six frequencies, |S11| = 1.107 at
24 THz, and a 15 % spike at that point. The cause is conditioning, not the
aggregation: the translation operators grow like
h_l⁽¹⁾(k·pitch) ~ (2l−1)!!/(k·pitch)^(l+1), so on a deep-subwavelength lattice
the high-l block of C is enormous and multiplies whatever extraction noise sits
in the high-l rows of T.

| λ (µm) | cond(I − C T), lmax 5 | lmax picked | cond after |
|---|---|---|---|
| 29.98 | 3.5×10⁵ | 3 | 18 |
| 27.25 | 5.5×10⁵ | 3 | 17 |
| 24.98 | 1.8×10⁵ | 3 | 15 |
| 16.66 | 4.5×10³ | 4 | 47 |
| 12.49 | 7.4×10² | 4 | 11 |
| 8.82 | 14 | 5 | 14 |

`--lmax auto` keeps, per frequency, the largest truncation with
cond(I − C T) ≤ 100: lmax 3 below 17 THz, 4 up to 25 THz, 5 above. This is
physically the right budget — the T-matrix is 99 % electric dipole at 10 THz
and only develops l = 2 weight above ~28 THz — and it restores A ≥ 0
everywhere without changing the well-conditioned part of the band.

## Accuracy

| check | result |
|---|---|
| r₀ (×0.8) and quadrature (20×40 → 28×56) invariance | ≤ 1.3×10⁻¹⁰ |
| lattice-sum taper length (kRc/1.4) | ≤ 4.3×10⁻³ |
| **treams** (independent Ewald code), same truncation, complex S | max 0.070, mean 0.019 |
| **direct CST periodic run** (run 6, same project), complex S | max 0.078, mean 0.022 (S21); max 0.077, mean 0.031 (S11) |
| resonance position vs CST direct | 13.097 µm vs 13.126 µm (0.2 %) |
| absorption vs CST direct | A ∈ [0.010, 0.114] vs [0.008, 0.106], max ΔA 0.040 |
| reciprocity of the array S-params (S21 vs S12) | 1.9×10⁻³ … 1.6×10⁻² |
| energy balance A = 1 − R − T | ≥ 0.010 |

The floor is set by the input file, not by the aggregation. The lattice sum
itself is converged to ~10⁻¹⁰, but the T-matrix violates passivity by up to
2.8 % (max SV(I+2T) = 1.028) and reciprocity by up to 11 % — against 0.6–1.2 %
for the `test/single` demo. The 10–18 THz sub-band is the worse of the two
merged extractions (its own `residual` 0.014–0.024, `reciprocity` 0.07–0.11,
vs 0.002–0.015 and 0.012–0.065 above 19 THz). The largest error against both
references is at 16.66 µm (18 THz) — the seam between the two merged bands.
Expect ~2–8 % on complex S here, not the 3×10⁻⁴ obtained for `test/single`.

## Picking the right run out of the packed CST project

`SAW_gold_noSub_packed.cst` holds a **10-point parametric sweep over `scale`**,
not a single run. Signals are versioned by `choice` = the 3D Run ID in CST's
Result Navigator; taking the newest one gives run 10 (`scale` 3.25), whose
resonance is at 10.93 µm and which has nothing to do with this h5.

`python -m tmatrix.aggregation.cst_packed_reference --list` prints the table
(`p = 8` in all ten). Matching the reconstructed resonance selects **run 6**:
`scale` 4.0, ring outer radius r = 2.877 µm, dip at 13.126 µm. The T-matrix's
own low-frequency dipole polarizability independently implies a scatterer
~1.25× larger than run 10's r = 2.338 µm, i.e. r ≈ 2.9 µm — the same run.

| 3D Run ID | 4 | 5 | 9 | 1 | 10 | 8 | **6** | 7 | 2 | 3 |
|---|---|---|---|---|---|---|---|---|---|---|
| scale | 2.0 | 2.5 | 2.75 | 3.0 | 3.25 | 3.5 | **4.0** | 4.5 | 5.0 | 5.5 |
| dip (µm) | >35 THz | 9.18 | 9.69 | 10.29 | 10.93 | 11.59 | **13.13** | 14.90 | 17.34 | 23.51 |

The extracted reference is de-embedded from the Floquet port planes to z = 0
(L = 168.5 µm for run 6, fitted from the thin-sheet gauge S21 − S11 = 1, which
holds to 0.016 across the band) and conjugated from CST's e^{+jωt} to this
repo's e^{−iωt}. The h5 carries no `/scatterer/geometry` group (unlike
`test/single`), so this identification comes from the physics, not from
metadata.

## Files

| file | contents |
|---|---|
| `periodic_results.csv` | λ, f, lmax, cond, complex S11/S21, R/T/A, uncoupled sheet, σ_ext/σ_sca/σ_abs |
| `periodic_results.npz` | same + the full lattice-sum stack C(f) and input diagnostics |
| `finite_results.npz` | 2×2, 3×3, 5×5, 9×9 Foldy–Lax arrays at 5 wavelengths |
| `treams_reference.npz` | independent Ewald cross-check |
| `cst_direct_reference.csv` | direct CST periodic run 6 of the packed project, de-embedded to z = 0 |
| `fig1_sparams.png` | \|S11\|, \|S21\|: this work vs treams vs direct CST |
| `fig2_balance_truncation.png` | R/T/A and the adaptive truncation / conditioning |

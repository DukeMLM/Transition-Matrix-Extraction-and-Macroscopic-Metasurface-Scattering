# Transition-Matrix Extraction and Macroscopic Metasurface Scattering

A multi-scale simulation pipeline for metasurfaces: extract the **transition
matrix (T-matrix)** of a single unit cell from one full-wave CST simulation,
then predict the S-parameters of arbitrarily large arrays with fast linear
algebra — no full-wave simulation of the array required.

Validated end-to-end on the `dary` branch: the aggregated S-parameters agree
with a direct CST periodic simulation to **|ΔS| ≤ 0.003** and with the
independent [treams](https://github.com/tfp-photonics/treams) code to
**~3×10⁻⁴** across the full 8–20 µm band.

<p align="center">
<img src="aggregation/results/fig7_cst_direct_comparison.png" width="85%">
</p>

---

## Pipeline overview

| Stage | What it does | Code | Status |
|---|---|---|---|
| **1. Near-field extraction** | Illuminates the *isolated* unit cell with a set of plane waves in CST (frequency-domain solver, open boundaries) and exports complex E/H on a spherical monitor | `extract_cst_near_fields_2.py` | main |
| **2. T-matrix projection** | Projects the recorded fields onto vector spherical wave functions (VSWFs) and solves F = T·A in least squares → `*.tmat.h5` | `compute_t_matrix_projection_2.py` | main |
| **3. Array aggregation → S-parameters** | Couples the per-cell T-matrices through the Foldy–Lax multiple-scattering equations (finite arrays or infinite lattices) and projects the collective response onto plane waves → S11/S21 | `aggregation/` | **this branch** |

Theory reference: `ref/main.tex` (operational manual). The physics in one
breath: the scattering of one cell is the linear map `f = T a` between
incoming (regular) and outgoing VSWF coefficients; an array couples cells via
translation operators `A(d)`, giving the self-consistent system
`a^i = a_inc^i + Σ_j A(R_j−R_i) T a^j`; the summed outgoing far fields of a
subwavelength lattice collapse into the 0th diffraction order, giving
`S21 = 1 + (2πi/kA)·ê*·F(+ẑ)` and `S11 = (2πi/kA)·ê*·F(−ẑ)`.

## Repository layout

```
ref/                          theory manual (LaTeX + PDF)
test/single/                  demo unit cell: spoke-and-wheel gold resonator
  saw_gold_wl15p0025um.tmat.h5   49 freqs (8-20 um), lmax 3, tmat.h5 format
aggregation/                  Stage 3 implementation (this branch)
  vswf.py                     VSWF engine: fields, plane waves, far field, projections
  translate.py                translation operators + lattice sums (Richardson-extrapolated)
  aggregate.py                Foldy-Lax finite-array + periodic solves
  sparams.py                  S11/S21, energy balance, cross sections
  mirror.py                   PEC ground plane via exact image theory
  tmat_io.py                  tmat.h5 reader
  test_*.py                   validation suite (see table below)
  run_demo.py                 full 49-frequency demo (periodic + finite arrays)
  run_mirror_demo.py          ground-plane variant
  treams_reference.py         independent cross-check via treams
  cst_direct/                 direct CST periodic reference simulation
  results/                    figures, CSV/NPZ spectra
  REPORT.md                   results and findings
  IMPLEMENTATION_GUIDE.md     full tutorial: how this was built from scratch
```

## Conventions (read this before touching anything)

All Stage-3 code follows the [tmat.h5 standard](https://doi.org/10.48550/arXiv.2404.10399)
exactly as declared by the data file:

* time dependence **e^(−iωt)**, outgoing radial functions **h_l^(1)**
* orthonormal spherical harmonics with **Condon–Shortley** phase
* Jackson-normalized VSWFs: `X = LY/√(l(l+1))`, `M = z_l X`, `N = (1/k)∇×M`
  — note N's radial term carries a factor **i** in this convention
* parity basis (TE/"magnetic" = M-waves, TM/"electric" = N-waves)
* `f = T a` (the file calls `f` "p"); mode order read from `/modes`, never assumed
* lmax = 3 → n = 2·lmax·(lmax+2) = 30 modes

## Quickstart

Requirements: Python with `numpy`, `scipy ≥ 1.15` (`sph_harm_y`), `h5py`,
`matplotlib`; optionally `treams` for the cross-check. CST is **not** needed
to run Stage 3.

```bash
cd aggregation
python test_vswf.py && python test_translate.py && python test_mirror.py   # validation suite
python run_demo.py        # ~8 min: 49 freqs, infinite lattice + finite arrays
python test_feature_fidelity.py
python plot_results.py    # figures + CSV into results/
```

Minimal API example — periodic array at normal incidence:

```python
from tmat_io import TMatrixData
from vswf import plane_wave_coeffs
from translate import lattice_sum_C, make_quad
from aggregate import solve_periodic
from sparams import sparams_normal, energy_balance

data = TMatrixData("../test/single/saw_gold_wl15p0025um.tmat.h5")
i = 21                                   # frequency index (~15 um)
k = data.k_at(i)                         # rad/um
C = lattice_sum_C(k, 2.0, data.modes, 0.8, make_quad())   # pitch 2 um
a_inc = plane_wave_coeffs([0, 0, 1], [1, 0, 0], data.modes)
a, f = solve_periodic(data.T[i], C, a_inc)
S = sparams_normal(k, 4.0, data.modes, f)                 # A_cell = 4 um^2
R, T, Aabs = energy_balance(S)
```

Finite arrays with arbitrary in-plane positions and per-site T-matrices:
`aggregate.build_finite_system` + `solve_finite`, then `sparams_normal` with
`A = N·A_cell`.

## Validation summary

| check | result |
|---|---|
| Mie sphere: optical theorem = ∫\|F\|² = Mie series | 3×10⁻¹⁵ |
| plane-wave reconstruction from VSWF expansion (E and H) | 3×10⁻⁸ |
| translation operator re-expansion / r₀-invariance | 2×10⁻⁷ / 3×10⁻⁷ |
| lattice-sum stability (taper set, quadrature, r₀) | ≤ 2×10⁻⁶ |
| input-file conventions: passivity max SV(I+2T) / reciprocity | 1.00007 / matches file's own metric |
| reciprocity of array S-params (S21 = S12) | 2×10⁻⁴ |
| finite arrays (5×5 → 13×13) → infinite-lattice limit | monotone ✓ |
| lossless array over PEC mirror: R = 1 (unitarity) | exact (1.000000) |
| **treams** (independent Ewald code), complex S, 49 freqs | ≤ 3.4×10⁻⁴ |
| **direct CST periodic simulation**, 8–20 µm | \|ΔS21\| ≤ 0.0011, \|ΔS11\| ≤ 0.0029 |

## Key physical findings for the demo cell

* The free-standing 2 µm-pitch array is featureless in-band
  (|S21| 0.99→0.94): the *isolated* resonator has no resonance in 8–20 µm.
  The designed λc = 15 µm response is a **ground-plane (MIM) resonance** —
  Stage 1 removes the ground plane by design, so this is correct physics,
  confirmed by three independent computations.
* Adding a PEC ground plane by image theory (`mirror.py`) restores an
  absorption resonance near the design band at the design spacing
  (`results/fig6_mirror_absorber.png`) — qualitative, since the real
  dielectric spacer is not part of the extracted T-matrix.
* Practical CST warning (cost us two wrong runs): with `"unit cell"`
  boundaries CST sizes the period to the **geometry bounding box** unless
  `UnitCellFitToBoundingBox "False"` + `UnitCellDs1/Ds2` are set — an
  island-type cell silently becomes a *touching*, near-totally-reflective
  mesh. See `aggregation/cst_direct/` and REPORT.md §4b.

## Known limitations (Stage 3, current state)

* Normal incidence only (Bloch phases and the 1/cosθ port factor are
  documented extension points, not wired in).
* Square lattices for the periodic path; finite arrays take arbitrary
  in-plane positions.
* Truncation inherited from the file (lmax = 3); at pitch/(2r_circ) = 1.39
  Wiscombe suggests checking L = 5 on the extraction side.
* Ground plane is an idealized PEC mirror with vacuum spacer; a layered
  substrate needs the Sommerfeld reflection operator (manual, Stage 2).

## Where to go next

* **Results and figures**: [`aggregation/REPORT.md`](aggregation/REPORT.md)
* **How it was built, from scratch, with all the derivations and war
  stories**: [`aggregation/IMPLEMENTATION_GUIDE.md`](aggregation/IMPLEMENTATION_GUIDE.md)

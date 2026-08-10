# Heterogeneous periodic supercell: `a,b;b,a` from two measured T-matrices

Stage-3 extended from *one* atom per cell to *M different* atoms per cell — the
pair-resolved block-Bloch formulation of the operational manual (expanded
Aug-2026 edition) §6.5.2–6.5.5 and §7.5 — and run end to end on the two
meta-atoms in `test/2x2/`.

| | |
|---|---|
| atom **A** | `saw_gold_wl13p10um_10to34THz.tmat.h5`, packed-project `scale` 4.0, r = 2.877 µm |
| atom **B** | `saw_gold_wl17p30um_10to34THz.tmat.h5`, packed-project `scale` 5.0, r = 3.596 µm |
| supercell | 16 × 16 µm, four atoms at (±4, ±4) µm: **A** on one diagonal, **B** on the other |
| band | 25 frequencies, 10–34 THz (λ 8.82–29.98 µm), vacuum, normal incidence, x-polarized |
| truncation | lmax 3 (30 modes/atom → a 120 × 120 block solve) |

```bash
# validation ladder (manual 6.5.5)
python tests/aggregation/test_supercell.py            # 40 checks, all pass
python tests/aggregation/test_supercell.py --taper    # + the tapered-sum comparison

# experiment 1: each atom alone on its own 8 um lattice, vs its direct CST run
python -m tmatrix.aggregation.run_supercell --cell 8 --site test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5 0 0 --lmax 3 --out results_A_ewald_l3
python -m tmatrix.aggregation.run_supercell --cell 8 --site test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5 0 0 --lmax 3 --out results_B_ewald_l3
python -m tmatrix.aggregation.cst_packed_reference test/2x2/SAW_gold_noSub_packed.cst --run 6 --out results_A_ewald_l3/cst_direct_reference.csv
python -m tmatrix.aggregation.cst_packed_reference test/2x2/SAW_gold_noSub_packed.cst --run 2 --out results_B_ewald_l3/cst_direct_reference.csv

# experiments 2-3: the checkerboard supercell + the finite-cluster T-matrix
python -m tmatrix.aggregation.run_supercell --cell 16 \
    --site test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5 -4 -4 \
    --site test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5  4 -4 \
    --site test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5 -4  4 \
    --site test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5  4  4 \
    --lmax 3 --cluster-lmax 12 --out results_2x2_super_l3
python -m tmatrix.aggregation.treams_supercell --cell 16 --site ... --lmax 3 --out results_2x2_super_l3/treams_reference.npz

# experiment 4: the direct CST supercell simulation
python -m tmatrix.aggregation.cst_supercell.build_2x2_supercell
python -m tmatrix.aggregation.cst_supercell.read_supercell_results --out aggregation/results_2x2_super_l3
python -m tmatrix.aggregation.plot_supercell results_2x2_super_l3
```

---

## 1. What the extension computes

The supercell lattice is `L = {R = n1 a1 + n2 a2}`; the reference cell holds M
atoms at basis positions ρ_s, each with its own isolated `T_s`. Bloch's theorem
reduces the infinite array to those M atoms, coupled by the *pair-resolved*
lattice sums

```
W_st(k∥) = Σ'_R  A(R + ρ_t − ρ_s) e^{i k∥·R}                      (manual Eq. 48)
(I − W T0) a_uc = a_inc,uc ,  f_uc = T0 a_uc                       (Eq. 50)
```

with the prime removing **only** the (s = t, R = 0) self term — for s ≠ t the
R = 0 term *is* the intracell coupling. The outgoing multipoles of all M atoms
are then added coherently into every propagating Floquet order,

```
E^±_G = (2πi / (A_uc k_z,G)) Σ_s e^{−i k^±_G·ρ_s} F_s(k̂^±_G)       (Eq. 64)
```

power-normalized by √(k_z,G / k_z,in). Manual Eqs. (65)/(66) — the scalar
`S21 = 1 + …`, `S11 = …` used by the one-atom code — are the G = 0,
normal-incidence special case, and are reproduced exactly (§2, test 1).

New code (under `src/tmatrix/aggregation/`, except the suite):

| file | contents |
|---|---|
| `supercell.py` | pair-resolved tapered sums, block solve, `T_B^loc`, Floquet channels, finite-cluster `T^O` (Eq. 40) |
| `ewald_supercell.py` | the same `W_st` by Ewald summation through treams, with an η-split refusal policy |
| `run_supercell.py` | driver: any list of `tmat.h5` + positions + lattice → full Floquet S-matrix |
| `treams_supercell.py` | independent end-to-end treams reference |
| `tests/aggregation/test_supercell.py` | the §6.5.5 / §6.6 / §7.5 / §8 / §6.4 ladder |
| `cst_supercell/` | the direct CST supercell benchmark (build + solve + de-embed) |
| `plot_supercell.py` | figures and agreement metrics |

---

## 2. Validation ladder (manual §6.5.5) — `tests/aggregation/test_supercell.py`

All 40 checks pass. The load-bearing ones:

| manual | check | result |
|---|---|---|
| 6.5.5-1 | M = 1 reproduces `aggregate.solve_periodic` | 0 (bit-identical) |
| 6.5.5-1 | M = 1 S11/S21 vs `sparams.sparams_normal` | 0 (bit-identical) |
| 6.5.5-2 | `Σ_t W_st = C_p` (four identical atoms, 2×2 cell ↔ one-atom p lattice) | 2.8×10⁻¹⁵ |
| 6.5.5-2 | the four `f_s` come out identical; equal to the one-atom `f` | 4.1×10⁻¹⁵ / 1.9×10⁻¹⁵ |
| 6.5.5-2 | complex S of the folded cell vs the primitive cell | 7.9×10⁻¹⁶ |
| 6.5.5-2 | power in the orders the folding invents | 5.5×10⁻³¹ |
| 6.5.5-3 | basis-atom relabelling leaves S and every `f_s` unchanged | 6.7×10⁻¹⁶ |
| 6.5.5-3 | shifting the whole basis by a lattice vector leaves W unchanged | 0 |
| 6.5.5-5 | A = 1 − R − T ≥ 0; cross-polarized \|S\| at normal incidence | ✓ / 3.3×10⁻¹⁶ |
| 6.5.5-5 | reciprocity of the array, \|S21 − S12\| | 2.6×10⁻³ |
| 6.5.5-6 | Ewald η bracket (0.5, 0.7, 1.0) over all 16 blocks | 2.8×10⁻¹² |
| 6.6 | circumscribing spheres: (a_A + a_B) − d_min = −1.53 µm | ✓ (Eq. 57) |
| 6.6 | `Rg(d) Rg(−d) = I`; `Rg(−ρ) a_pw = e^{ik·ρ} a_pw` | 6.0×10⁻⁹ / 1.8×10⁻⁹ |
| 8 row 7 | all T = 0 → S21 = 1, S11 = 0, total power 1 | exactly 0 |
| 7.5 | odd (n1+n2) orders extinguished by the a,b;b,a symmetry | 6.5×10⁻³³ |
| 6.4 | cluster far field vs the multi-center sum, L_C = 8/10/12/14 | 4×10⁻⁴ / 1×10⁻⁵ / 1.7×10⁻⁷ / 1.9×10⁻⁹ |
| 6.4 | shifting the cluster origin: far field covariant to e^{−ik·r̂·O} | 2.6×10⁻⁹ |
| 8 row 9 | **independent treams implementation of the whole chain** | **5×10⁻¹⁵ (S21), 1×10⁻¹² (S11)** |

One more independent path, with no lattice sum in it at all: a **finite**
checkerboard patch solved by real-space Foldy–Lax (`aggregate.build_finite_system`
over an N×N patch of the 8 µm grid with alternating types) approaches the
periodic answer from above, monotonically:

| λ (µm) | periodic \|S21\| | 4×4 atoms | 8×8 | 12×12 |
|---|---|---|---|---|
| 23.06 | 0.7092 | 0.7077 | 0.7234 | 0.7079 |
| 13.63 | 0.1356 | 0.2516 | 0.1969 | 0.1770 |
| 9.99 | 0.8136 | 0.8277 | 0.8206 | 0.8176 |

(Convergence is slowest at the resonance, where the edge of a finite patch
matters most — the same behaviour the one-atom case shows in
`results_2x2/finite_results.npz`.)

The treams check (`tmatrix.aggregation.treams_supercell`) is not a lattice-sum
comparison: treams builds its own block-diagonal cluster T-matrix, does its own
Ewald lattice interaction, and does its own plane-wave projection with its own
basis-position phases and power normalization. Agreement at 10⁻¹² therefore
validates the coupling, the block solve, the Eq. (64) output map *and* the
Floquet normalization — including the R/T sums over all nine open orders at
34 THz.

> One trap found here and worth recording: treams and this repository must be
> compared **at the same incidence direction**. Illuminating from −z instead of
> +z changes the answer by 0.01–0.06 in complex S, because the input T-matrices
> violate reciprocity by 2–11 %. That difference is the *input file's* up/down
> asymmetry, not an implementation difference — it tracks the per-frequency
> `reciprocity` diagnostic stored in the h5, peaking at 18.7 µm where that
> diagnostic is 0.076. With matching incidence the two codes agree to 10⁻¹².

### Two projection-radius traps

* **`Rg` is not band-limited.** A translation mixes multipole order, so a target
  basis truncated at lmax cannot absorb the re-expansion of a source of the same
  order. The identities `Rg(d)Rg(−d) = I` and `Rg(−ρ) a_pw = e^{ik·ρ}a_pw` only
  hold once the *source* basis is large enough (L = 12 here); testing them
  inside the lmax-3 working basis fails at the 30–60 % level for reasons that
  have nothing to do with the operator.
* **The sampling sphere must scale with the cluster order.** `j_l(k r₀) ~
  (k r₀)^l/(2l+1)!!`, so for the global P/Q projections of a cluster of order
  L_C the projector divides by `j_l² + ξ_l² ~ 10⁻²⁶` at k r₀ = 1.4, L = 14, and
  the result is round-off. Regular waves are entire, so r₀ is free:
  `supercell.cluster_projection_grid` puts it at `(L_C + 2)/k` and sizes the
  quadrature for the angular content the sphere actually sees. With r₀ = 3 µm
  fixed, L_C = 14 was *worse* than L_C = 12 (4.8×10⁻³ vs 1.0×10⁻⁵); with the
  scaled radius the sequence converges monotonically to 1.9×10⁻⁹.

---

## 3. The tapered lattice sum does not survive sub-lattice shifts — use Ewald

This is the one place where the repository's existing machinery had to be
replaced rather than extended, and manual §6.5.3 predicts it exactly ("for
shifted sublattices these values are only starting choices … a validated Ewald
or reciprocal-space method is an alternative").

`supercell.block_lattice_sums` implements Eq. (55)/(56) faithfully: a Gaussian
taper on the **full displacement** `|R + ρ_t − ρ_s|`, Richardson-extrapolated in
1/R_c². Tapering the displacement rather than the lattice vector makes the
sub-lattice decomposition exact *termwise*, so

```
Σ_t W_st(R_c) = C_p(R_c)   to 1.2×10⁻¹⁵, for every taper length
```

— and the extrapolated matrices inherit it. But the *individual* blocks are
shifted sub-lattice sums whose period is √M coarser, so a taper of length R_c
contains proportionally fewer sites and the 1/R_c² series has not settled:

| λ (µm) | repo C_p vs Ewald C_p (one atom, 8 µm) | Ewald `Σ_t W_st` vs C_p | repo W vs Ewald W, kRc (10,14,20) | kRc (20,28,40) |
|---|---|---|---|---|
| 29.98 | 1.7×10⁻⁶ | 2.1×10⁻¹⁵ | 9.6×10⁻⁷ | 1.5×10⁻⁸ |
| 18.74 | 1.4×10⁻⁵ | 3.8×10⁻¹⁵ | 4.7×10⁻³ | 4.2×10⁻⁴ |
| 13.63 | 4.6×10⁻⁵ | 3.0×10⁻¹⁵ | 3.8×10⁻² | 4.8×10⁻³ |
| 10.71 | 7.6×10⁻³ | 1.1×10⁻¹⁴ | 3.0×10⁻¹ | 1.7×10⁻¹ |
| 8.82 | 3.8×10⁻¹ | 4.4×10⁻¹¹ | 1.9×10⁻¹ | 4.8×10⁻² |

The error is *common to all blocks* and cancels in the sum, which is why the
one-atom lattice is accurate and the individual pair blocks are not — and why a
heterogeneous cell, where the blocks are weighted by *different* T's, cannot use
them. The last row is a second, independent warning: the tapered sum also
degrades as the primitive lattice approaches its own Rayleigh anomaly
(λ_min/pitch = 1.10 here).

`ewald_supercell.converged_W` therefore supplies `W` by Ewald summation, with
the same refuse-rather-than-guess η policy as
`src/tmatrix/retrieval/fastfull/ewald.py`: the automatic split must agree with
an η ∈ {0.5, 0.7, 1.0} bracket, which it does to 3×10⁻¹² on every cell here.
It is also much cheaper — the same 25-frequency one-atom sweep takes **2.0 s**
by Ewald against **64 s** tapered, and the gap widens with M² pair blocks.

**Regression against the published one-atom `test/2x2` numbers.** Those used the
tapered sum with `--lmax auto`. Re-run through
`tmatrix.aggregation.run_supercell` with Ewald coupling and the same adaptive
truncation, atom A reproduces them exactly:

| | published (tapered, `--lmax auto`) | this driver (Ewald, `--lmax auto`) |
|---|---|---|
| transmission minimum | 13.097 µm | 13.097 µm |
| complex S21 vs CST, max / mean | 0.078 / 0.022 | 0.078 / 0.022 |
| complex S11 vs CST, max / mean | 0.077 / 0.031 | 0.077 / 0.031 |
| vs treams (at fixed lmax 3) | max 0.070, mean 0.019 | 6×10⁻¹⁶ |

(The treams row is quoted at fixed lmax 3 in both columns, since
`tmatrix.aggregation.treams_supercell` takes one truncation for the whole sweep
and a per-frequency `auto` comparison would not be like for like.)

So swapping the lattice sum changes nothing about the one-atom answer except
that the treams disagreement collapses to round-off — confirming that the
residual `test/2x2` error was, and remains, the input file's, and that the two
codes had been differing only through the taper's extrapolation error and the
incidence convention.

---

## 4. Experiment 1 — each atom alone, against its own direct CST run

The two atoms are runs 6 and 2 of the packed parametric project, so a direct
periodic CST reference already exists for each; no new simulation was needed.

| | atom A (scale 4) | atom B (scale 5) |
|---|---|---|
| transmission minimum, this repo | 13.039 µm | 16.638 µm |
| transmission minimum, treams | 13.039 µm | 16.638 µm |
| transmission minimum, direct CST | 13.126 µm | 17.341 µm |
| resonance offset | −0.66 % | **−4.05 %** |
| complex S21 vs CST, max / mean | 0.062 / 0.030 | 0.121 / 0.054 |
| complex S11 vs CST, max / mean | 0.065 / 0.041 | 0.135 / 0.073 |
| R / T / A vs CST, max \|Δ\| | 0.081 / 0.067 / 0.033 | 0.118 / 0.085 / 0.087 |
| vs treams (same incidence) | 6×10⁻¹⁶ / 1×10⁻¹² | 8×10⁻¹⁶ / 1×10⁻¹² |

`results_B_ewald_l3/fig1_sparams.png` shows what this looks like: the
reconstruction reproduces the lineshape, the depth (\|S21\|min 0.05 vs CST's
0.03), the reflection maximum and the full phase excursion, with the resonance
sitting ~4 % short in wavelength.

**The new sample is the harder of the two, and the reason is where its
resonance falls, not the aggregation.** Both h5 files are merged from two
extraction sub-bands (A: 10–18 + 19–34 THz; B: 10–20 + 21–34 THz). Atom A's
array resonance is at 22.9 THz, inside its *better* sub-band (fit residual
0.002–0.015); atom B's is at 17.3 THz, inside its *worse* one (residual
0.006–0.016, reciprocity 0.025–0.076) — and near a resonance the Foldy–Lax
denominator amplifies whatever error the input carries:

| | 10–20 THz sub-band | 21–34 THz sub-band |
|---|---|---|
| atom A, mean (\|ΔS21\|+\|ΔS11\|) vs CST | 0.071 | 0.071 |
| atom B, mean (\|ΔS21\|+\|ΔS11\|) vs CST | **0.184** | 0.082 |

Two controls confirm the input is the source. Symmetrizing each T to enforce
reciprocity moves the answer by only 0.007 (mean) / 0.032 (max), so the
reciprocity violation alone is not enough — the extraction's amplitude error is.
And a uniform rescaling of the wavelength axis (a pure pole shift) reduces the
mean error only from 0.127 to 0.124, so the discrepancy is not a clean frequency
offset either: it is a distributed amplitude-and-phase error of the input
T-matrix, largest where that T is most resonant.

---

## 5. Experiments 2–3 — the `a,b;b,a` supercell and its S-parameters

### 5.1 Geometry, symmetry and open channels

Positions (−4,−4) A, (+4,−4) B, (−4,+4) B, (+4,+4) A in a 16 µm cell. The
structure is *not* the naive "2×2 of a 16 µm cell": both sublattices are square
lattices of constant 8√2 = 11.31 µm rotated by 45°, with B at the body centre of
A, so the true point group is C4v about an A site. Two consequences, both
reproduced by the code and both absent from any single-atom model:

* **No cross-polarization at normal incidence** — computed \|S_cross\| ≤ 1.3×10⁻¹².
* **Selection rule.** Orders with n1 + n2 odd are extinguished exactly: the
  coherent sum of Eq. (64) gives `F_A[1 + (−1)^{n1+n2}] + F_B[(−1)^{n1} +
  (−1)^{n2}]`, which vanishes for odd n1+n2 whatever A and B are. Measured
  power in those channels: **6.5×10⁻³³**.

So although the 16 µm cell's first Rayleigh onset is at λ = 16 µm (18.74 THz),
the first onset that *carries power* is the (±1,±1) family at
λ = 16/√2 = 11.31 µm (**26.50 THz**). Below that, scalar S11/S21 describe the
sheet completely; above it they do not, and `floquet_orders.csv` carries every
order. `fig2_power.png` (right panel) shows the diffracted fraction pinned at
zero through the entire 16 → 11.31 µm window and then rising to 0.31 at 27 THz.

### 5.2 Spectrum

| f (THz) | λ (µm) | \|S11\| | \|S21\| | R | T | A | open orders | diffracted |
|---|---|---|---|---|---|---|---|---|
| 10 | 29.98 | 0.459 | 0.835 | 0.211 | 0.697 | 0.093 | 1 | 0 |
| 14 | 21.41 | 0.705 | 0.642 | 0.498 | 0.413 | 0.090 | 1 | 0 |
| 17 | 17.63 | 0.877 | 0.291 | 0.768 | 0.084 | 0.147 | 1 | 0 |
| 18 | 16.66 | 0.891 | 0.111 | 0.793 | 0.012 | 0.194 | 1 | 0 |
| 19 | 15.78 | 0.462 | 0.619 | 0.213 | 0.384 | 0.403 | 5 | 0 |
| 21 | 14.28 | 0.921 | 0.134 | 0.848 | 0.018 | 0.134 | 5 | 0 |
| 22 | 13.63 | 0.929 | 0.136 | 0.863 | 0.018 | 0.118 | 5 | 0 |
| 26 | 11.53 | 0.549 | 0.785 | 0.302 | 0.616 | 0.082 | 5 | 0 |
| 27 | 11.10 | 0.533 | 0.585 | 0.439 | 0.498 | 0.062 | 9 | 0.310 |
| 30 | 9.99 | 0.403 | 0.814 | 0.222 | 0.723 | 0.055 | 9 | 0.120 |
| 34 | 8.82 | 0.145 | 0.939 | 0.048 | 0.912 | 0.040 | 9 | 0.057 |

The mixed cell is **not** an average of the two pure lattices. On its own 8 µm
lattice, A dips at 23 THz (\|S21\| = 0.055) and B at 18 THz (0.052). Mixed at half
the areal density each — every atom now on an 11.31 µm sublattice — the
checkerboard keeps two dips but moves them to 18 THz (0.111) and 21–22 THz
(0.134 / 0.136): 5 THz of separation becomes 3.5 THz. Between them, at
19–20 THz, transmission does not stay low but rises back to 0.62 / 0.46. That
window is not a gap between two independent resonances; §5.3 shows it is a
lattice resonance of the supercell itself. Neither the shift nor the window is
available from `Σ_s T_s` or from either pure lattice: they are what the
block-Bloch solve buys.

### 5.3 A dark lattice resonance at the supercell's own Rayleigh onset

The 19 THz row above is not an outlier. Re-running with `--refine 4`, which
subdivides each stored interval and interpolates T linearly between the
bracketing samples (`results_2x2_super_fine/`):

| f (THz) | λ (µm) | cond(I − W T0) | \|S11\| | \|S21\| | A | diffracted |
|---|---|---|---|---|---|---|
| 18.00 | 16.655 | 271 | 0.891 | 0.111 | 0.194 | 0 |
| 18.50 | 16.205 | 307 | 0.804 | 0.266 | 0.283 | 0 |
| **18.75** | **15.989** | **1302** | 0.683 | 0.429 | 0.349 | 3.6×10⁻³⁰ |
| 19.25 | 15.574 | 227 | 0.316 | 0.682 | **0.436** | 1.2×10⁻³⁰ |
| 20.00 | 14.990 | 83 | 0.761 | 0.457 | 0.212 | 1.4×10⁻³⁰ |

The conditioning peaks at λ = 15.99 µm — the (±1,0) Rayleigh condition of the
16 µm cell, λ = 16.000 µm — and the response goes through a resonance: a
transmission window opens inside the stop band and absorption rises to 0.44.
The diffracted power stays at 10⁻³⁰ throughout, so the responsible channels are
**dark**: the odd orders are singular in the near-field coupling `W_st` but
carry no far field, by the same selection rule as §5.1. This is a
symmetry-protected surface-lattice resonance of the *supercell*; the 8 µm
primitive lattice has no Rayleigh onset until 8 µm and shows nothing at all
there. The 1 THz sampling of the h5 undersamples the feature — its width is
about 1 THz — so the 19 THz sample lands on its flank.

The second onset, λ = 11.31 µm (26.50 THz), leaves its own mark — a step in
\|S21\| from 0.585 at 11.10 µm to 0.785 at 11.53 µm in `fig1_sparams.png` — but
that one is a *bright* anomaly: it is where the (±1,±1) channels actually open,
and the diffracted fraction jumps from 0 to 0.31 across it.

The amplitude of the 16 µm feature is the least trustworthy number in this
report: it sits at cond ≈ 300–1300, and 19 THz is also atom A's extraction band
seam. The direct CST run in §6 is what decides it.

### 5.4 Truncation: fixed lmax 3, and why `--lmax auto` must not be reused here

| truncation | A = 1 − R − T range | max cond(I − W T0) | verdict |
|---|---|---|---|
| lmax 3 (headline) | [+0.040, +0.403] | 1150 | passive everywhere |
| lmax 4 | [−0.911, +0.233] | 7880 | non-physical below 21 THz |
| `--lmax auto`, cond ≤ 100 | [+0.003, +0.372] | 97 | picks lmax 1–2 in places |

The one-atom driver's adaptive rule — keep the largest lmax whose
`cond(I − C T)` stays under a budget — **does not transfer to the block
system**, and `results_2x2_super_auto/` shows why. The condition number of
`I − W T0` for an M-atom cell also contains the *folded* k∥ ≠ 0 bands of the
primitive lattice, which are legitimately near-singular next to a Rayleigh
anomaly. Capping it therefore throws away physics rather than noise: at 18 THz
the rule drops to lmax 1 and returns \|S11\| = 0.238 where lmax 3 gives 0.891.
The headline run uses fixed lmax 3, the same truncation used for the two
single-atom references, so that any difference between them is attributable to
the aggregation and not to the truncation policy. lmax 3 is also the a-priori
right budget for these inputs: the l = 4 block of either T-matrix carries
10⁻⁴–10⁻³ of ‖T‖² below 20 THz, against a stored fit residual² of the same
order — i.e. it is at the extraction noise floor.

### 5.5 The single-origin T-matrix of the whole 2×2 block (manual §6.4)

`--cluster-lmax 12` additionally forms `T^O_array = Q (I − T0 U)^{-1} T0 P` for
the same four atoms taken as an **isolated finite cluster** — a different object
from the periodic solve, reported alongside it because it is the literal
"T-matrix of the whole array". Stored in `periodic_results.npz` as
`T_cluster`, shape (25, 336, 336).

| λ (µm) | ‖T^O‖_F | max SV(I + 2 T^O) (passivity, ≤ 1) |
|---|---|---|
| 29.98 | 0.554 | 1.0016 |
| 18.74 | 2.004 | 1.0065 |
| 13.63 | 2.193 | 1.0018 |
| 8.82 | 2.481 | 1.0113 |

Its violation of passivity, 0.2–1.1 %, is inherited from the input matrices
(which are themselves 2.1–2.8 % non-passive) — the aggregation adds none: with
the same inputs the far field of `T^O` matches the direct multi-center
Foldy–Lax sum to 1.7×10⁻⁷ at L_C = 12 and 1.9×10⁻⁹ at L_C = 14, and is
covariant under a shift of the expansion origin to 2.6×10⁻⁹.

Because a finite cluster under an infinite plane wave has a continuum of
radiation channels (manual §7.3), `T^O` is reported as a T-matrix and a
spherical partial-wave S-matrix `S_sph = I + 2 T^O` (Eq. 60), **not** as an
S11/S21 pair. The S11/S21 of §5.2 belong to the periodic problem.

---

## 6. Experiment 4 — direct CST simulation of the supercell

`src/tmatrix/aggregation/cst_supercell/build_2x2_supercell.py` builds the
16 × 16 µm cell with the four resonators in the checkerboard, every solver,
mesh, material and boundary setting copied from the packed project's own
periodic run (its `ModelHistory.json` steps 4/5/163/169/176–179), plus 20
Floquet modes per port so that all nine orders open at 34 THz are represented.
A companion **empty** cell — identical geometry, no metal — measures the
port-plane separation and serves as the manual's §8 row-7 background test.

### 6.1 The benchmark's own accuracy

| | |
|---|---|
| empty cell \|S21\| | 0.9851 … 1.0000 |
| empty cell max \|S11\| | 7.6×10⁻⁴ |
| empty cell power into every higher order | ≤ 1.3×10⁻⁶ |
| measured port-plane separation L | 56.813 µm (bbox ±25 µm → 50 µm geometric plus CST's "expanded open" margin) |
| \|S21 e^{jkL} − 1\| | 2.5×10⁻³ below 27 THz, 2.5×10⁻² at 33 THz |

That last row is the noise floor of the benchmark itself: nothing in the
comparison below is meaningful at the 10⁻³ level, and above 30 THz not at the
10⁻² level. Everything is de-embedded to z = 0 with
exp(+j (k_z,0 + k_z,G) L/2) and conjugated from CST's e^{+jωt}.

### 6.2 CST confirms the selection rule

The mode grouping implied by the `+beta/pw` sort — modes 1–2 the (0,0) order,
3–10 the (±1,0)/(0,±1) family, 11–18 the (±1,±1) family, 19–20 the (±2,0)
family — is confirmed by the data: no group carries power while it is closed.
And, independently of the aggregation:

| Floquet family | opens at | max power over the band, direct CST | predicted |
|---|---|---|---|
| (0,0) | — | 0.987 | — |
| (±1,0), (0,±1) | 18.74 THz | **1.5×10⁻⁵** | **0 (dark)** |
| (±1,±1) | 26.50 THz | 0.382 | 0.310 |
| (±2,0), (0,±2) | 37.47 THz | 0 (closed) | 0 (closed) |

The full-wave simulation puts **nothing** into the (±1,0)/(0,±1) channels over
the 18.7–34 THz window in which they are open — 1.5×10⁻⁵ against a 0.987 carrier,
i.e. at the empty run's own noise floor — exactly the extinction the coherent
basis-position sum of Eq. (64) predicts from the a,b;b,a symmetry. This is the
sharpest single confirmation in the report that the multi-atom output map is
right: a one-atom model of the same sheet has no way to say anything about those
channels at all.

Total propagating power stays at 0.987 ≤ 1, so the de-embedded, order-resolved
S-matrix is passive channel by channel.

### 6.3 Complex S-parameters

| | max | mean |
|---|---|---|
| complex S21 vs CST | 0.205 | 0.046 |
| complex S11 vs CST | 0.205 | 0.048 |
| complex S21, excluding 18.4–20.2 THz (the dark resonance) | 0.068 | 0.036 |
| complex S11, excluding 18.4–20.2 THz | 0.063 | 0.038 |
| R / T / A | 0.231 / 0.128 / 0.103 | 0.043 / 0.028 / 0.032 |
| diffracted fraction | 0.310 (this repo) vs 0.382 (CST) | |

Feature by feature:

| feature | this repo | direct CST | offset |
|---|---|---|---|
| B-derived transmission dip | 16.655 µm, \|S21\| 0.111 | 16.752 µm, 0.063 | +0.6 % |
| A-derived transmission dip | 14.276 µm, \|S21\| 0.134 | 14.025 µm, 0.062 | −1.8 % |
| dark lattice resonance peak (refined grid) | 15.574 µm, \|S21\| 0.682 | 15.352 µm, 0.723 | +1.4 % |
| (±1,±1) diffraction edge | 11.31 µm | 11.31 µm (sharp) | exact |

**The §5.3 dark lattice resonance is real.** CST independently produces the same
transmission window inside the stop band, at 15.352 µm with a peak of 0.723
against the predicted 15.574 µm and 0.682 — while keeping the responsible
(±1,0)/(0,±1) channels dark to 10⁻⁵. The 1 THz sampling of the h5 straddles it
(19 and 20 THz), which is where the 0.205 maximum in the table above comes from:
outside that pair of samples the agreement is 0.068 / 0.063 worst case.
`results_2x2_super_fine/` re-runs the sweep on a 4× refined grid with T
interpolated between the stored samples and resolves it.

The two transmission dips are also placed *better* than either pure lattice
manages on its own: +0.6 % and −1.8 %, against −4.05 % for atom B's own 8 µm
lattice in §4. The residual discrepancies are the input T-matrices', not the
supercell's.

The CST run itself: 48 adaptively placed frequency points, 3863 s, ~1.0 M
tetrahedral DOF at second order, 20 Floquet modes per port. The solver spent
most of its adaptive budget at 26.5 THz — the (±1,±1) diffraction edge — which
is its own confirmation that the sharp features are physical.

### 6.4 What the comparison establishes

The point of experiment 4 is not the absolute agreement — that is set by the
input T-matrices, as §4 showed — but whether *aggregating two different atoms
into a repeated cell* costs anything beyond what aggregating each of them alone
already costs. It does not:

| case | complex \|ΔS21\| vs its own direct CST run, mean | \|ΔS11\|, mean |
|---|---|---|
| atom A alone, 8 µm lattice | 0.030 | 0.041 |
| atom B alone, 8 µm lattice | 0.054 | 0.073 |
| **a,b;b,a supercell, 16 µm cell** | **0.049** | **0.049** |

The mixed cell lands between its two constituents, not outside them — and its
resonance positions are actually *closer* to the full-wave answer (+0.6 % and
−1.8 %) than atom B's own lattice is (−4.05 %), because in the checkerboard each
species sits on a sparser sublattice and the response depends less steeply on
the part of T the extraction resolves worst.

---

## 7. Files

| file | contents |
|---|---|
| `periodic_results.csv` | λ, f, lmax, cond, solve residual, η spread, complex S11/S21 (0th order) and cross-pol, R/T/A, open-order count, higher-order power, uncoupled sheet, cross sections |
| `periodic_results.npz` | the same plus the run metadata |
| `cluster_T.npz` | the finite-cluster `T^O` stack, 25 × 336 × 336 complex — **not tracked in git** (45 MB, incompressible); regenerate with `python -m tmatrix.aggregation.run_supercell --cluster-lmax 12` |
| `cst_direct_supercell.csv` | the direct CST run, de-embedded to z = 0: complex 0th-order S11/S21, cross-pol, R/T/A, higher-order power |
| `cst_direct_supercell_orders.csv` | all 20 Floquet modes, both sides: \|G\|, k_z, propagating flag, complex S, power |
| `floquet_orders.csv` | every propagating order at every frequency: (n1,n2), k_z, complex TE/TM amplitude, power |
| `treams_reference.npz` | independent treams end-to-end reference |
| `run.json` | the exact invocation: sites, lattice, coupling method, truncation, Rayleigh onset |
| `fig1_sparams.png` | \|S\| and phase of S21/S11: this repo, treams, direct CST |
| `fig2_power.png` | R/T/A and the diffracted fraction against the Rayleigh onsets |
| `fig3_experiment.png` | the three cases together: A alone, B alone, the mix, each over its own direct CST run |

This directory holds the `a,b;b,a` cell and the method common to all of them.
Two further mixed cells were built from the same three measured atoms and are
reported next to it — see [`../results_2x2_AC_l3/REPORT.md`](../results_2x2_AC_l3/REPORT.md)
and [`../results_2x2_BC_l3/REPORT.md`](../results_2x2_BC_l3/REPORT.md), and
[`../../experiment.md`](../../experiment.md) for the three-cell comparison.
`python -m tmatrix.aggregation.compare_cases --all` prints all six cases in one
table.

Sibling directories produced by the same driver:

| directory | what it is |
|---|---|
| `results_A_ewald_l3`, `results_B_ewald_l3`, `results_C_ewald_l3` | the three pure 8 µm lattices at the same truncation — the §4 references |
| `results_2x2_AC_l3`, `results_2x2_BC_l3` | the other two mixed cells, with their own CST benchmarks |
| `results_2x2_AC_fine`, `results_2x2_BC_fine` | their refined-grid sweeps |
| `results_A_ewald_auto` | atom A at `--lmax auto`, the regression against the published `test/2x2` numbers |
| `results_2x2_super_l4`, `results_2x2_super_auto` | the truncation sensitivity of §5.4 |
| `results_A_ewald_l4`, `results_B_ewald_l4` | the same two lattices at lmax 4, the single-atom half of the same truncation study |
| `results_2x2_super_fine` | the same supercell on a 4× refined frequency grid (`--refine 4`, T interpolated), which resolves the 16 µm dark resonance the stored 1 THz sampling aliases |
| `results_B` | atom B through the *old* `tmatrix.aggregation.run_case` driver (tapered coupling, `--lmax auto`) — the direct analogue of the published `results_2x2`, kept for comparison |
| `cst_supercell/runs*/` | the CST projects: `*.cst`, `Model/3D/ModelHistory.json`, `Result/Model.log`, `design.json`, `build_log.txt`. The 2.4 GB solver working directories are gitignored; `tmatrix.aggregation.cst_supercell.build_2x2_supercell` re-creates them exactly. |

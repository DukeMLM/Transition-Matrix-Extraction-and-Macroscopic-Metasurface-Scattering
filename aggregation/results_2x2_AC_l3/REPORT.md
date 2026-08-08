# `a,c;c,a` — atom A (scale 4.00) and atom C (scale 3.25)

The second heterogeneous supercell. Method, derivation and the full validation
ladder are in [`../results_2x2_super_l3/REPORT.md`](../results_2x2_super_l3/REPORT.md);
this file records only what is specific to this cell.

| | |
|---|---|
| atom **A** | `saw_gold_wl13p10um_10to34THz.tmat.h5`, `scale` 4.00, r = 2.877 µm, CST run 6 |
| atom **C** | `saw_gold_wl10p90um_10to34THz.tmat.h5`, `scale` 3.25, r = 2.338 µm, CST run 10 |
| cell | 16 × 16 µm; A at (−4, −4) and (+4, +4), C at (+4, −4) and (−4, +4) |
| truncation | lmax 3, Ewald coupling, 120 × 120 block solve |
| circumscribing margin | (a_A + a_C) − 8 µm = **−2.79 µm** (manual Eq. 57 satisfied) |

```bash
python run_supercell.py --cell 16 \
    --site ../test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5 -4 -4 \
    --site ../test/2x2/saw_gold_wl13p10um_10to34THz.tmat.h5  4  4 \
    --site ../test/2x2/saw_gold_wl10p90um_10to34THz.tmat.h5  4 -4 \
    --site ../test/2x2/saw_gold_wl10p90um_10to34THz.tmat.h5 -4  4 \
    --lmax 3 --cluster-lmax 12 --out results_2x2_AC_l3
python cst_supercell/build_2x2_supercell.py --pair AC --only supercell
python cst_supercell/read_supercell_results.py --run cst_supercell/runs_AC/supercell/supercell.cst \
    --empty cst_supercell/runs/empty/empty.cst --out results_2x2_AC_l3
python plot_supercell.py results_2x2_AC_l3 --fine results_2x2_AC_fine
python error_budget.py results_2x2_AC_l3
```

## Validation

| check | result |
|---|---|
| independent treams implementation, complex S | max 2.6×10⁻¹⁴ (S21), 8.8×10⁻¹³ (S11) |
| cross-polarized \|S\| at normal incidence | 2.1×10⁻¹² |
| power in the dark (n1+n2 odd) orders, this repo | ≤ 3.8×10⁻³¹ |
| power in the dark orders, **direct CST** | **1.5×10⁻⁵** against a 1.002 carrier |
| linear-solve residual | 1.1×10⁻¹⁵ |
| A = 1 − R − T | [+0.011, +0.356], non-negative everywhere |

## Against the direct CST supercell run

61 min, 48 adaptive frequency points, ~0.66 M tetrahedral DOF (a smaller mesh
than `a,b;b,a` because C is the smallest of the three atoms).

| | max | mean |
|---|---|---|
| complex S21 | 0.079 | **0.019** |
| complex S11 | 0.081 | 0.020 |
| … excluding 20.4–21.6 THz (the A–C hybrid) | 0.064 | 0.017 |
| R / T / A | 0.080 / 0.037 / 0.044 | 0.017 / 0.010 / 0.011 |

| feature | this repo (refined grid) | direct CST | offset |
|---|---|---|---|
| A-derived transmission dip | 14.448 µm, \|S21\| 0.207 | 14.363 µm, 0.164 | −0.59 % |
| C-derived transmission dip | 11.757 µm, \|S21\| 0.080 | 11.810 µm, 0.065 | +0.45 % |
| 16 µm dark-resonance shoulder | 16.427 µm, \|S21\| 0.743 | 16.487 µm, 0.745 | — |
| diffracted power, maximum | 0.216 | 0.208 | 4 % relative |

This is the closest agreement of the three mixed cells, and for the expected
reason: A and C are the two atoms whose own lattices reconstruct best
(mean \|ΔS21\| 0.030 and 0.017 against 0.054 for B).

## What is specific to this cell

* **Both resonances red-shift** when the atoms move from their own 8 µm lattice
  to the sparser 11.31 µm sublattice of the mixed cell: A from 13.04 to
  14.42 µm, C from 10.90 to 11.84 µm.
* **An A–C hybrid resonance at 21.25 THz.** On the refined grid \|S21\| runs
  0.207 → 0.283 → 0.532 → 0.732 → 0.815 between 20.75 and 21.75 THz while
  absorption peaks at **0.422**, with cond(I − W T₀) only ~78 throughout — so it
  is a genuine hybridized mode, not a conditioning artifact. It is the one place
  where the 1 THz sampling of the h5 files aliases badly, and the source of the
  0.079 maximum above.
* **C's own resonance sits inside the diffracting region.** At 10.90 µm it is
  below the (±1,±1) Rayleigh onset at 11.31 µm, so in the mixed cell that
  resonance is pushed out to 11.84 µm — just above the onset — and the diffracted
  fraction peaks at 0.216 immediately below it.
* The **16 µm dark lattice resonance** appears here too: on the refined grid
  cond(I − W T₀) peaks at 594 at exactly λ = 15.989 µm while the responsible
  channels stay at 3.8×10⁻³¹. It is a property of the cell and the two-species
  symmetry, not of which atoms fill it.

## Files

Same layout as `../results_2x2_super_l3/`: `periodic_results.csv` / `.npz`,
`floquet_orders.csv`, `cst_direct_supercell*.csv`, `treams_reference.npz`,
`run.json`, `fig1_sparams.png`, `fig2_power.png`, and `cluster_T.npz` (not
tracked in git; regenerate with `--cluster-lmax 12`).
`../results_2x2_AC_fine/` holds the same sweep on a 4× refined frequency grid.

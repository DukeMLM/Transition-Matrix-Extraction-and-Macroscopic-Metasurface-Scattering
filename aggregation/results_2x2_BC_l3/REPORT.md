# `b,c;c,b` — atom B (scale 5.00) and atom C (scale 3.25)

The third heterogeneous supercell, and the one with the widest separation
between its two constituent resonances. Method, derivation and the full
validation ladder are in
[`../results_2x2_super_l3/REPORT.md`](../results_2x2_super_l3/REPORT.md); this
file records only what is specific to this cell.

| | |
|---|---|
| atom **B** | `saw_gold_wl17p30um_10to34THz.tmat.h5`, `scale` 5.00, r = 3.596 µm, CST run 2 |
| atom **C** | `saw_gold_wl10p90um_10to34THz.tmat.h5`, `scale` 3.25, r = 2.338 µm, CST run 10 |
| cell | 16 × 16 µm; B at (−4, −4) and (+4, +4), C at (+4, −4) and (−4, +4) |
| truncation | lmax 3, Ewald coupling, 120 × 120 block solve |
| circumscribing margin | (a_B + a_C) − 8 µm = **−2.07 µm** (manual Eq. 57 satisfied) |

```bash
python run_supercell.py --cell 16 \
    --site ../test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5 -4 -4 \
    --site ../test/2x2/saw_gold_wl17p30um_10to34THz.tmat.h5  4  4 \
    --site ../test/2x2/saw_gold_wl10p90um_10to34THz.tmat.h5  4 -4 \
    --site ../test/2x2/saw_gold_wl10p90um_10to34THz.tmat.h5 -4  4 \
    --lmax 3 --cluster-lmax 12 --out results_2x2_BC_l3
python cst_supercell/build_2x2_supercell.py --pair BC --only supercell
python cst_supercell/read_supercell_results.py --run cst_supercell/runs_BC/supercell/supercell.cst \
    --empty cst_supercell/runs/empty/empty.cst --out results_2x2_BC_l3
python plot_supercell.py results_2x2_BC_l3 --fine results_2x2_BC_fine
python error_budget.py results_2x2_BC_l3
```

## Validation

| check | result |
|---|---|
| independent treams implementation, complex S | max 1.8×10⁻¹⁴ (S21), 9.0×10⁻¹³ (S11) |
| cross-polarized \|S\| at normal incidence | 2.0×10⁻¹² |
| power in the dark (n1+n2 odd) orders, this repo | ≤ 2.1×10⁻³⁰ |
| linear-solve residual | 1.5×10⁻¹⁵ |
| A = 1 − R − T | [+0.028, +0.208], non-negative everywhere |

## What is specific to this cell

* **The widest resonance separation of the three cells.** B alone dips at
  16.64 µm (18.0 THz) and C alone at 10.90 µm (27.5 THz); mixed, they sit at
  16.71 µm and 12.44 µm — B barely moves while C red-shifts by 1.5 µm onto the
  sparser 11.31 µm sublattice.
* **A near-transparent window between them.** At 21 THz \|S21\| = 0.907 with
  \|S11\| = 0.129 — the cell is almost invisible between its two stop bands.
* **The strongest diffraction of the three cells: 0.433** at 28 THz, against
  0.310 for `a,b;b,a` and 0.216 for `a,c;c,a`. C's resonance sits closest to the
  (±1,±1) Rayleigh onset at 11.31 µm, so the mixed cell is still strongly
  resonant where the diffraction channels open.
* The **16 µm dark lattice resonance** appears here too: on the refined grid
  cond(I − W T₀) peaks at 1198 at exactly λ = 15.989 µm while the responsible
  channels stay at 2.1×10⁻³⁰, and absorption reaches 0.209.

## Against the direct CST supercell run

*Solving at the time of writing — this section is completed when it lands. The
model is built by `build_2x2_supercell.py --pair BC` with the same settings as
the other two cells, and shares the `runs/empty` companion run for de-embedding.*

## Files

Same layout as `../results_2x2_super_l3/`: `periodic_results.csv` / `.npz`,
`floquet_orders.csv`, `treams_reference.npz`, `run.json`, `fig1_sparams.png`,
`fig2_power.png`, and `cluster_T.npz` (not tracked in git; regenerate with
`--cluster-lmax 12`). `../results_2x2_BC_fine/` holds the same sweep on a 4×
refined frequency grid.

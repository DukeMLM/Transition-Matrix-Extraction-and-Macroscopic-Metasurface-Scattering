# `a,c;d,b` — the tie-breaker: geometry, not symmetry

The third and last distinct arrangement of A, B, C, D on the 2×2 site square,
run to settle [`../../OPEN_QUESTIONS.md`](../../OPEN_QUESTIONS.md) §1. Its
partners are [`../results_2x2_ABCD_l3/REPORT.md`](../results_2x2_ABCD_l3/REPORT.md)
(`a,b;c,d`) and [`../results_2x2_ADBC_l3/REPORT.md`](../results_2x2_ADBC_l3/REPORT.md)
(`a,d;b,c`), which describe the physics all three share — odd diffraction
orders switching on, birefringence, vanishing cross-polarization.

```
        A  C          A at (−4, +4)   C at (+4, +4)
        D  B          D at (−4, −4)   B at (+4, −4)
```

| | |
|---|---|
| cell | 16 × 16 µm, atoms 8 µm apart, lmax 3, Ewald coupling, 25-point grid |
| worst pair | **B–D, ρ = 0.944, gap 0.448 µm** — *identical* to `a,b;c,d` |
| diagonal pairs | A–B, C–D (against A–D, B–C for `a,b;c,d`) |

## Why this cell was run

`a,d;b,c` reconstructs 14× better than `a,b;c,d` from the same four atoms, and
two explanations fit that equally well and are confounded in those two cells:
its worst pair is looser (ρ 0.854 against 0.944), *and* it is the more nearly
mirror-symmetric. `a,c;d,b` breaks the tie because it reproduces `a,b;c,d`'s
pair geometry **exactly** — same worst pair, same ρ, same 0.448 µm gap — while
placing a different pair on the diagonal. The predictors are computed by
[`../arrangement_predictors.py`](../arrangement_predictors.py).

## Verdict: the convergence ratio is the driver

| cell | ρ | MSE, complex S21 | mean \|ΔS21\| | deepest-dip misplacement |
|---|---|---|---|---|
| `a,b;c,d` | 0.944 | 0.1654 | 0.308 | **−5.51 µm** |
| **`a,c;d,b`** | **0.944** | **0.1207** | **0.244** | **+4.92 µm** |
| `a,d;b,c` | 0.854 | **0.0118** | **0.080** | **+0.11 µm** |

`a,c;d,b` lands with `a,b;c,d`, not with `a,d;b,c`. The two cells that share
ρ = 0.944 are within 1.4× of each other and both fail; the cell with ρ = 0.854
is 10–14× better than either. **Arrangement symmetry is not what made `a,d;b,c`
accurate — its wider tightest gap is.**

The clearest single statement is where each cell puts its deepest transmission
dip. Both ρ = 0.944 cells misplace it by about 5 µm — in *opposite* directions,
so this is not a systematic bias the pipeline could be corrected for — while
the ρ = 0.854 cell places it to 0.11 µm:

| cell | deepest predicted | deepest CST | shift |
|---|---|---|---|
| `a,b;c,d` | 0.081 @ 17.63 µm | 0.058 @ 23.15 µm | −5.51 µm |
| `a,c;d,b` | 0.190 @ 23.06 µm | 0.063 @ 18.14 µm | +4.92 µm |
| `a,d;b,c` | 0.152 @ 17.63 µm | 0.050 @ 17.53 µm | +0.11 µm |

The residual 0.1207 against 0.1654 is a 27 % difference between the two failing
cells. It is not nothing, but against the 10× step to `a,d;b,c` it is second
order, and ρ is identical across it — so whatever produces it, it is not the
quantity that separates a usable cell from an unusable one.

### One caveat on how far this falsifies "symmetry"

The symmetry score depends on which mirrors are admitted, and the two choices
disagree about this cell:

| cell | nearest diagonal mirror | nearest of all four mirrors |
|---|---|---|
| `a,b;c,d` | 27.3 % (A–D) | 18.8 % (axis x: A–C, B–D) |
| `a,c;d,b` | 20.0 % (A–B) | 18.8 % (axis y: A–C, B–D) |
| `a,d;b,c` | 9.1 % (B–D) | 9.1 % (diag y=−x: B–D) |

Under the diagonal-mirror metric quoted in `OPEN_QUESTIONS.md`, symmetry
predicted `a,c;d,b` would sit strictly between the other two, and it does not —
that version is falsified. Under the all-mirrors metric, `a,b;c,d` and
`a,c;d,b` tie at 18.8 %, so that version predicts they behave alike, which they
do, and it is *not* tested by this run. What is established either way is the
useful half: **ρ predicts accuracy, and a cell with ρ = 0.944 cannot be trusted
regardless of how symmetric it looks.**

## Validation

| check | result |
|---|---|
| independent treams implementation, complex S11 and S21 | max 8.4×10⁻¹³ |
| linear-solve residual | 2.2×10⁻¹⁵ |
| A = 1 − R − T | [+0.046, +0.230], non-negative everywhere |
| max cross-polarized \|S\| | 3.1×10⁻¹² |
| max cond(I − W T0) | 961 |
| CST channel check: power in any closed order | 0.00×10⁰ |
| CST passivity: max \|S11\|²+\|S21\|²+higher | 1.0034 |
| empty-cell de-embedding, L | 56.8131 µm (branch-resolved) |

The aggregation is therefore not what fails here. treams reproduces this
repository's answer to 8×10⁻¹³ having built its own cluster T-matrix, its own
Ewald lattice sum and its own plane-wave projection; both implementations agree
on an answer that is wrong against full-wave by 0.24 in mean \|ΔS21\|. The error
is in the truncated outgoing→regular translation that both share, exactly where
ρ → 1 says it should be.

## Against the direct CST supercell run

5 972 s (1.66 h), 1 005 in-band samples after interpolation, sharing the
`runs/empty` companion run. Built by `build_2x2_supercell.py --pair ACDB`, every
solver, mesh, material and boundary setting identical to the other nine
benchmarks.

| | MSE | max \|Δ\| | mean \|Δ\| |
|---|---|---|---|
| complex S21 | **0.1207** | 0.817 | 0.244 |
| complex S11 | 0.1192 | 0.807 | 0.242 |
| R / T / A | — | 0.722 / 0.741 / 0.100 | 0.158 / 0.162 / 0.030 |
| … S21 below 20 THz | 0.2881 | | |
| … S21 above 20 THz | **0.0090** | | |

The error is concentrated below 20 THz in all three four-atom cells — the band
where atom D resonates and where its ρ = 0.944 pair coupling is least
converged. Above 20 THz this cell reaches MSE 0.0090, within a factor 2–5 of
the well-behaved two-species cells.

**The diffraction channels are fine even though the specular one is not.**
Power into the higher orders peaks at 0.407 predicted against 0.403 measured,
1.0 % relative — the best agreement of the three arrangements, in the cell with
the second-worst specular error. Whatever the truncated translation is getting
wrong, it is not the aggregate power leaving the sheet; it is where the
0th-order resonances sit.

### Solve cost

| cell | solver time | note |
|---|---|---|
| `a,c;d,b` | 5 972 s | this run |
| `a,b;c,d` | 6 568 s | |
| `a,d;b,c` | 17 469 s | its D-derived resonance sits on the 18.74 THz Rayleigh edge, which the adaptive sweep has to resolve |

## Files

Same layout as `../results_2x2_super_l3/`; `../results_2x2_ACDB_fine/` holds the
same sweep on a 4× refined grid, and the Jones-matrix data for all three
four-atom arrangements is in `../results_2x2_ABCD_l3/jones_xy.npz`, regenerated
by [`../jones_xy.py`](../jones_xy.py).

## Incidental: birefringence tracks B–D adjacency

Not what the run was for, but it completes the set. Putting the two largest
atoms side by side makes a strong linear polarizer; putting them on a diagonal
makes a nearly isotropic sheet:

| cell | B and D | max ‖t_xx\|−\|t_yy‖ | mean | max \|t_xy\| |
|---|---|---|---|---|
| `a,b;c,d` | adjacent | 0.603 | 0.176 | 1.8×10⁻¹² |
| `a,c;d,b` | adjacent | 0.556 | 0.149 | 3.1×10⁻¹² |
| `a,d;b,c` | diagonal | 0.168 | 0.037 | 1.5×10⁻¹² |

Cross-polarization vanishes for a third time in a cell whose point group is C1
and where nothing forbids it — `OPEN_QUESTIONS.md` §2 is now three for three.

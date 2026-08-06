"""Feature-fidelity check: a synthetic resonant T-matrix pushed through the
SAME aggregation + S-parameter pipeline must show the resonance in S21/S11.

This separates 'pipeline flattens features' (a bug) from 'the demo input has
no in-band feature' (physics of the isolated resonator).  The lattice-sum
matrices C are reused from the demo run.

Synthetic scatterer: passive Lorentzian electric-dipole response in the
(l=1, m=+/-1, electric) channels, resonant at 15 um:
    T_res(w) = -gr / (i (w0 - w) + gr + gnr)
which satisfies |1 + 2 T| <= 1 (passivity) for gr, gnr >= 0.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tmat_io import TMatrixData, C0
from vswf import ModeBasis, ELECTRIC, plane_wave_coeffs
from aggregate import solve_periodic
from sparams import sparams_normal, energy_balance

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "test", "single", "saw_gold_wl15p0025um.tmat.h5")
OUT = os.path.join(HERE, "results")
PITCH = 2.0


def main():
    data = TMatrixData(DEMO)
    modes = data.modes
    lam = data.wavelength_um
    d = np.load(os.path.join(OUT, "periodic_results.npz"))
    C_stack = d["C"]

    w = 2 * np.pi * data.freq
    w0 = 2 * np.pi * C0 / 15e-6
    gr, gnr = 0.02 * w0, 0.005 * w0
    t_lor = -gr / (1j * (w0 - w) + gr + gnr)
    assert np.abs(1 + 2 * t_lor).max() <= 1.0 + 1e-12

    e_x = np.array([1.0, 0, 0])
    a_inc = plane_wave_coeffs([0, 0, 1], e_x, modes)
    idx = [modes.index(1, -1, ELECTRIC), modes.index(1, 1, ELECTRIC)]

    S21c, S11c, S21u, Ab = [], [], [], []
    for i in range(len(lam)):
        k = data.k_at(i)
        T = np.zeros((modes.n, modes.n), dtype=complex)
        for j in idx:
            T[j, j] = t_lor[i]
        a, f = solve_periodic(T, C_stack[i], a_inc)
        S = sparams_normal(k, PITCH ** 2, modes, f)
        R_, T_, A_ = energy_balance(S)
        S21c.append(abs(S["S21_co"]))
        S11c.append(abs(S["S11_co"]))
        Ab.append(A_)
        S0 = sparams_normal(k, PITCH ** 2, modes, T @ a_inc)
        S21u.append(abs(S0["S21_co"]))
        assert -0.02 <= A_ <= 1.0, f"energy violation at {lam[i]}: A={A_}"

    i_min = int(np.argmin(S21c))
    print(f"synthetic resonance (single-particle at 15 um): coupled "
          f"(Foldy-Lax) dip at lam = {lam[i_min]:.2f} um, min |S21| = "
          f"{S21c[i_min]:.3f}, max |S11| = {max(S11c):.3f};  A range "
          f"[{min(Ab):.3f}, {max(Ab):.3f}]")
    print("  (the shift from 15 um is the collective lattice resonance of the "
          "dense array; the uncoupled sheet is not a meaningful reference "
          "here because a bare strong resonant sheet violates unitarity -- "
          "the Foldy-Lax radiative coupling is what renormalizes it)")
    # the pipeline must transfer a sharp feature into S21
    assert S21c[i_min] < 0.7, "resonance not visible in coupled S21!"
    # external validation: treams' Ewald-summed lattice coupling must place
    # the collective dip at the same wavelength with the same depth
    tr_path = os.path.join(OUT, "treams_synthetic.npz")
    if os.path.exists(tr_path):
        tr = np.load(tr_path)
        j_min = int(np.argmin(tr["S21_mag"]))
        print(f"  treams external check: dip at {tr['lam_um'][j_min]:.2f} um, "
              f"min |S21| = {tr['S21_mag'][j_min]:.3f}")
        assert abs(tr["lam_um"][j_min] - lam[i_min]) < 0.51
        assert abs(tr["S21_mag"][j_min] - S21c[i_min]) < 0.02

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(lam, S21c, "-o", ms=3, lw=1.5, label="|S21| coupled (Foldy-Lax)")
    ax.plot(lam, np.asarray(S21u) / 10, "--", lw=1.2, alpha=0.6,
            label=r"|S21| uncoupled sheet $\times$0.1 (unphysical: no "
                  "radiative self-consistency)")
    ax.plot(lam, S11c, "-s", ms=3, lw=1.5, label="|S11| coupled")
    ax.plot(lam, Ab, "-^", ms=3, lw=1.2, label="A = 1-R-T")
    ax.axvline(15.0, color="gray", lw=0.8, ls=":")
    ax.annotate("single-particle\nresonance (15 um)", xy=(15.0, 1.0),
                xytext=(12.6, 0.86), fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    i_min = int(np.argmin(S21c))
    ax.annotate("collective lattice\nresonance", xy=(lam[i_min], S21c[i_min]),
                xytext=(16.6, 0.45), fontsize=8, color="C0",
                arrowprops=dict(arrowstyle="->", color="C0", lw=0.8))
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("|S| / power")
    ax.set_ylim(0, 1.1)
    ax.set_title("Feature fidelity: synthetic 15-um Lorentzian dipole through "
                 "the same pipeline", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    fig.savefig(os.path.join(OUT, "fig5_feature_fidelity.png"), dpi=200)
    print("PASS: pipeline transfers spectral features faithfully")


if __name__ == "__main__":
    main()

"""Demo: single-unit-cell T-matrix (CST extraction) -> metasurface S-parameters.

Pipeline per frequency:
  1. C = lattice sum of translation operators (square lattice, pitch 2 um,
     normal incidence, Richardson-extrapolated Gaussian taper).
  2. Periodic Foldy-Lax solve: a = (I - C T)^{-1} a_inc,  f = T a.
  3. S11/S21 from the 0th-order plane-wave projection of the per-cell far
     field (subwavelength pitch: no diffraction orders in the band).

Also computed: uncoupled sheet (C = 0) for comparison, single-scatterer cross
sections, T-matrix diagnostics, reciprocity spot checks, and finite N x N
Foldy-Lax arrays at selected frequencies to show convergence to the periodic
result.
"""
import os
import time

import numpy as np

from tmatrix.aggregation.tmat_io import TMatrixData
from tmatrix.aggregation.vswf import ModeBasis, plane_wave_coeffs, RegularProjector
from tmatrix.aggregation.translate import lattice_sum_C, make_quad
from tmatrix.aggregation.aggregate import solve_periodic, build_finite_system, solve_finite
from tmatrix.aggregation.sparams import sparams_normal, energy_balance, cross_sections

from tmatrix.paths import AGG_RESULTS, DEMO_TMAT


DEMO = DEMO_TMAT
OUT = AGG_RESULTS
PITCH = 2.0          # um (unit cell of the absorber design)
R0 = 0.8             # projection sphere radius, um
KRC = (10.0, 14.0, 20.0)

os.makedirs(OUT, exist_ok=True)


def tmatrix_diagnostics(data):
    print("=== T-matrix diagnostics (convention checks) ===")
    sv_max = 0.0
    for i in range(len(data.freq)):
        S = np.eye(data.modes.n) + 2 * data.T[i]
        sv = np.linalg.svd(S, compute_uv=False).max()
        sv_max = max(sv_max, sv)
    print(f"  passivity: max singular value of S = I + 2T over band: "
          f"{sv_max:.6f} (must be <= 1 + eps)")
    # reciprocity in this basis: T[l'm'p', lmp] = (-1)^(m+m') T[l,-m,p ; l',-m',p']
    modes = data.modes
    perm = np.array([modes.index(modes.l[i], -modes.m[i], modes.pol[i])
                     for i in range(modes.n)])
    sign = (-1.0) ** (modes.m[:, None] + modes.m[None, :])
    errs = []
    for i in range(0, len(data.freq), 6):
        T = data.T[i]
        Trec = sign * T[np.ix_(perm, perm)].T
        errs.append(np.linalg.norm(T - Trec) / np.linalg.norm(T))
    print(f"  reciprocity residual |T - T_recip|/|T|: "
          f"{np.min(errs):.3f} .. {np.max(errs):.3f} "
          f"(file's own metric: ~0.006-0.012)")
    return sv_max


def main():
    data = TMatrixData(DEMO)
    modes = data.modes
    nf = len(data.freq)
    lam = data.wavelength_um
    print(f"Loaded {os.path.basename(DEMO)}: {nf} freqs, "
          f"lambda {lam.min():.1f}-{lam.max():.1f} um, {modes.n} modes "
          f"(lmax={modes.lmax})")
    tmatrix_diagnostics(data)

    quad = make_quad(16, 32)
    proj = RegularProjector(modes, quad)
    e_x = np.array([1.0, 0, 0])
    A_cell = PITCH ** 2

    rows = []
    C_stack = np.empty((nf, modes.n, modes.n), dtype=complex)
    t0 = time.time()
    for i in range(nf):
        k = data.k_at(i)
        T = data.T[i]
        a_inc = plane_wave_coeffs([0, 0, 1], e_x, modes)

        C = lattice_sum_C(k, PITCH, modes, R0, quad, kRc=KRC, projector=proj)
        C_stack[i] = C
        a, f = solve_periodic(T, C, a_inc)
        S = sparams_normal(k, A_cell, modes, f)
        R, Tr, Ab = energy_balance(S)

        f0 = T @ a_inc                          # uncoupled sheet (C = 0)
        S0 = sparams_normal(k, A_cell, modes, f0)
        R0_, T0_, A0_ = energy_balance(S0)

        sext, ssca, sabs = cross_sections(k, modes, f0, [0, 0, 1], e_x)
        rows.append(dict(
            lam=lam[i], f_THz=data.freq[i] / 1e12,
            S11=S["S11_co"], S21=S["S21_co"],
            S11x=S["S11_cross"], S21x=S["S21_cross"],
            R=R, T=Tr, A=Ab,
            S11_nc=S0["S11_co"], S21_nc=S0["S21_co"],
            R_nc=R0_, T_nc=T0_, A_nc=A0_,
            sig_ext=sext, sig_sca=ssca, sig_abs=sabs))
        print(f"  [{i+1:2d}/{nf}] lam={lam[i]:6.2f} um  "
              f"|S11|={abs(S['S11_co']):.3f} |S21|={abs(S['S21_co']):.3f} "
              f"A={Ab:.3f}  (uncoupled A={A0_:.3f})  "
              f"sig_abs={sabs:.3f} um^2  [{time.time()-t0:5.1f} s]",
              flush=True)

    np.savez(os.path.join(OUT, "periodic_results.npz"),
             lam=lam, freq=data.freq, C=C_stack,
             **{key: np.array([r[key] for r in rows])
                for key in rows[0] if key not in ("lam", "f_THz")})

    # ---- reciprocity spot check: illuminate from -z, S12 must equal S21
    print("=== reciprocity spot check (S21 vs S12) ===")
    for i in (10, 24, 40):
        k = data.k_at(i)
        a_inc_b = plane_wave_coeffs([0, 0, -1], e_x, modes)
        a_b, f_b = solve_periodic(data.T[i], C_stack[i], a_inc_b)
        from tmatrix.aggregation.vswf import far_field_amplitude
        F_b = far_field_amplitude(k, f_b, modes, np.array([[0, 0, -1.0]]))[0]
        S12 = 1.0 + 2j * np.pi / (k * A_cell) * np.vdot(e_x, F_b)
        S21 = rows[i]["S21"]
        print(f"  lam={lam[i]:6.2f}: S21={S21:.4f}  S12={S12:.4f}  "
              f"|diff|={abs(S21 - S12):.2e}")

    # ---- finite arrays at selected frequencies
    print("=== finite Foldy-Lax arrays vs periodic ===")
    sel = range(0, nf, 6)
    fin = {N: [] for N in (5, 9, 13)}
    for i in sel:
        k = data.k_at(i)
        a_inc = plane_wave_coeffs([0, 0, 1], e_x, modes)
        for N in fin:
            g = (np.arange(N) - (N - 1) / 2) * PITCH
            X, Y = np.meshgrid(g, g)
            pos = np.stack([X.ravel(), Y.ravel(), np.zeros(N * N)], axis=1)
            M, T_list = build_finite_system(k, data.T[i], pos, modes, R0, quad)
            a_blocks = np.tile(a_inc, (N * N, 1))   # normal incidence: equal
            _, f_sites = solve_finite(M, T_list, a_blocks)
            f_tot = f_sites.sum(axis=0)             # in-plane: no extra phase
            S = sparams_normal(k, N * N * A_cell, modes, f_tot)
            fin[N].append((lam[i], S["S11_co"], S["S21_co"]))
        print(f"  lam={lam[i]:6.2f}: |S21| periodic={abs(rows[i]['S21']):.3f}  "
              + "  ".join(f"{N}x{N}={abs(fin[N][-1][2]):.3f}" for N in fin),
              flush=True)
    np.savez(os.path.join(OUT, "finite_results.npz"),
             **{f"N{N}": np.array([[l, s11, s21] for l, s11, s21 in v],
                                  dtype=complex) for N, v in fin.items()})
    print(f"done in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()

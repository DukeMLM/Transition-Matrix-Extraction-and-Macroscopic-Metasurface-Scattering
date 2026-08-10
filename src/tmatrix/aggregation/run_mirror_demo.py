"""Stage-2-lite demo: the extracted T-matrix over a PEC ground plane
(image theory), periodic 2-um lattice, normal incidence.

Geometry from the design sketch: spacer t_d = 300 nm, resonator thickness
100 nm -> mirror distance h = t_d + t_MM/2 = 350 nm from the resonator
center plane.  The dielectric spacer is treated as vacuum (its permittivity
is not in the tmat.h5 file), and gold ground is idealized as PEC, so the
resonance position is expected to shift relative to the actual design;
this is a qualitative demonstration that the aggregation machinery
reproduces the absorber physics once the ground plane is included.

Reuses the cached in-plane lattice sums C from run_demo.py.
"""
import os
import time

import numpy as np
from tmatrix.plotting import plt                    # noqa: E402

from tmatrix.aggregation.tmat_io import TMatrixData
from tmatrix.aggregation.vswf import RegularProjector
from tmatrix.aggregation.translate import make_quad
from tmatrix.aggregation.mirror import mirror_parity_signs, image_lattice_sum, \
    sparams_mirror_periodic
from tmatrix.aggregation.plot_results import thz_axis

from tmatrix.paths import AGG_RESULTS, DEMO_TMAT


DEMO = DEMO_TMAT
OUT = AGG_RESULTS
PITCH = 2.0
HS = (0.35, 0.55)      # mirror distances to compute (um)


def main():
    data = TMatrixData(DEMO)
    modes = data.modes
    lam = data.wavelength_um
    nf = len(lam)
    quad = make_quad(16, 32)
    proj = RegularProjector(modes, quad)
    s = mirror_parity_signs(modes)
    C_stack = np.load(os.path.join(OUT, "periodic_results.npz"))["C"]

    results = {}
    t0 = time.time()
    for h in HS:
        S11 = np.empty(nf, dtype=complex)
        A = np.empty(nf)
        for i in range(nf):
            k = data.k_at(i)
            C_im = image_lattice_sum(k, PITCH, modes, h, 0.6, quad,
                                     projector=proj)
            S = sparams_mirror_periodic(k, h, PITCH ** 2, modes, data.T[i],
                                        C_stack[i], C_im, s)
            S11[i], A[i] = S["S11_co"], S["A"]
            if i % 8 == 0:
                print(f"  h={h}: [{i+1:2d}/{nf}] lam={lam[i]:6.2f} "
                      f"|S11|={abs(S11[i]):.4f} A={A[i]:.4f} "
                      f"[{time.time()-t0:5.0f} s]", flush=True)
        results[h] = (S11, A)
        i_pk = int(np.argmax(A))
        print(f"h={h} um: peak absorption A={A[i_pk]:.3f} at "
              f"lam={lam[i_pk]:.2f} um")

    np.savez(os.path.join(OUT, "mirror_results.npz"), lam=lam,
             **{f"S11_h{int(h*1000)}": results[h][0] for h in HS},
             **{f"A_h{int(h*1000)}": results[h][1] for h in HS})

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for h, color in zip(HS, ("C2", "C4")):
        S11, A = results[h]
        ax.plot(lam, A, "-o", ms=3, lw=1.6, color=color,
                label=f"A, PEC mirror at h = {h*1000:.0f} nm")
        ax.plot(lam, np.abs(S11), "--", lw=1.1, color=color, alpha=0.5,
                label=f"|S11|, h = {h*1000:.0f} nm")
    d0 = np.load(os.path.join(OUT, "periodic_results.npz"))
    ax.plot(lam, d0["A"], ":", lw=1.4, color="gray",
            label="A, free-standing array (no mirror)")
    ax.axvline(15.0, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.text(15.05, 0.9, "design $\\lambda_c$", fontsize=8, color="gray")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("|S11| / Absorption")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Ground-plane coupling (image theory): absorber response "
                 "emerges", fontsize=11)
    thz_axis(ax)
    fig.savefig(os.path.join(OUT, "fig6_mirror_absorber.png"), dpi=200)
    print("saved fig6_mirror_absorber.png")


if __name__ == "__main__":
    main()

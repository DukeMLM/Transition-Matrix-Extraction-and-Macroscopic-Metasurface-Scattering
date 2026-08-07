"""Figures + agreement metrics for a run_case.py output directory.

Overlays whatever is present in the directory:
  periodic_results.npz        this repo's Foldy-Lax reconstruction (required)
  treams_reference.npz        independent Ewald cross-check (treams_case.py)
  cst_direct_reference.csv    direct CST periodic run (cst_packed_reference.py)

    python plot_case.py results_2x2 --title "spoke-and-wheel, pitch 8 um"
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C0 = 299.792458      # um * THz


def thz_axis(ax):
    top = ax.secondary_xaxis("top", functions=(lambda l: C0 / np.maximum(l, 1e-9),
                                               lambda f: C0 / np.maximum(f, 1e-9)))
    top.set_xlabel("Frequency (THz)")


def parabola_min(lam, y):
    """Sub-sample wavelength of a minimum, fitted in frequency.

    These sweeps are sampled uniformly in frequency, and a resonance line is
    symmetric in f, not in lambda -- fitting the parabola in lambda biases the
    peak toward the long-wavelength side by several percent.
    """
    i = int(np.argmin(y))
    if i in (0, len(y) - 1):
        return lam[i]
    f0, f1, f2 = C0 / np.asarray(lam[i - 1:i + 2], dtype=float)
    y0, y1, y2 = y[i - 1:i + 2]
    a, b = f1 - f0, f1 - f2
    den = a * (y1 - y2) - b * (y1 - y0)
    if abs(den) < 1e-300:
        return lam[i]
    f_min = f1 - 0.5 * (a * a * (y1 - y2) - b * b * (y1 - y0)) / den
    return C0 / f_min


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    out = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(
        HERE, args.out_dir)

    d = np.load(os.path.join(out, "periodic_results.npz"))
    lam, S11, S21 = d["lam"], d["S11"], d["S21"]
    R, T, A = d["R"], d["T"], d["A"]

    tr = cst = None
    p = os.path.join(out, "treams_reference.npz")
    if os.path.exists(p):
        tr = np.load(p, allow_pickle=True)
    p = os.path.join(out, "cst_direct_reference.csv")
    if os.path.exists(p):
        raw = np.loadtxt(p, delimiter=",", skiprows=1)
        m = raw[:, 0] > 0
        cst = dict(lam=raw[m, 1], S11=raw[m, 2] + 1j * raw[m, 3],
                   S21=raw[m, 4] + 1j * raw[m, 5],
                   R=raw[m, 8], T=raw[m, 9], A=raw[m, 10])

    # ---- fig 1: |S11|, |S21| ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, key, mine in ((axes[0], "S21", S21), (axes[1], "S11", S11)):
        if cst is not None:
            ax.plot(cst["lam"], np.abs(cst[key]), "-", lw=1.8, color="0.6",
                    label=f"|{key}| CST direct (packed .cst in folder)")
        ax.plot(lam, np.abs(mine), "o-", ms=5, lw=1.6, color="C0",
                label=f"|{key}| T-matrix -> Foldy-Lax (this repo)")
        if tr is not None:
            ax.plot(tr["lam_um"], np.abs(tr[key]), "s--", ms=4, lw=1.2,
                    color="C1", alpha=0.85, label=f"|{key}| treams (Ewald)")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel(f"|{key}|")
        ax.set_xlim(lam.min(), lam.max())
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
        thz_axis(ax)
    fig.suptitle(args.title or os.path.basename(out), fontsize=11)
    f1 = os.path.join(out, "fig1_sparams.png")
    fig.savefig(f1, dpi=180)

    # ---- fig 2: power balance + truncation ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    if cst is not None:
        for y, c in ((cst["R"], "C3"), (cst["T"], "C2"), (cst["A"], "C4")):
            ax.plot(cst["lam"], y, "-", lw=1.4, color=c, alpha=0.35)
    ax.plot(lam, R, "o-", ms=4, color="C3", label="R (reflectance)")
    ax.plot(lam, T, "o-", ms=4, color="C2", label="T (transmittance)")
    ax.plot(lam, A, "o-", ms=4, color="C4", label="A = 1 - R - T")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("power fraction")
    ax.set_xlim(lam.min(), lam.max())
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("thin lines: CST direct run in the same folder", fontsize=9)
    thz_axis(ax)

    ax = axes[1]
    ax.plot(lam, d["lmax"], "o-", color="C0", label="lmax used")
    ax.set_ylabel("multipole truncation lmax", color="C0")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylim(0, max(d["lmax"]) + 1)
    ax2 = ax.twinx()
    ax2.semilogy(lam, d["cond"], "s--", ms=4, color="C1")
    ax2.set_ylabel("cond(I - C T)", color="C1")
    ax.grid(alpha=0.3)
    ax.set_title("adaptive truncation and Foldy-Lax conditioning", fontsize=9)
    f2 = os.path.join(out, "fig2_balance_truncation.png")
    fig.savefig(f2, dpi=180)

    # ---- metrics ------------------------------------------------------------
    print(f"transmission minimum (parabolic fit):")
    print(f"  this repo   : {parabola_min(lam, np.abs(S21)):.3f} um")
    if tr is not None:
        print(f"  treams      : {parabola_min(tr['lam_um'], np.abs(tr['S21'])):.3f} um")
    if cst is not None:
        m = (cst["lam"] >= lam.min()) & (cst["lam"] <= lam.max())
        print(f"  CST direct  : {parabola_min(cst['lam'][m], np.abs(cst['S21'][m])):.3f} um")
    if tr is not None:
        d21 = np.abs(np.abs(S21) - np.abs(tr["S21"]))
        d11 = np.abs(np.abs(S11) - np.abs(tr["S11"]))
        print(f"vs treams: max ||S21|-|S21|| = {d21.max():.4f} "
              f"(mean {d21.mean():.4f}); max ||S11|-|S11|| = {d11.max():.4f} "
              f"(mean {d11.mean():.4f})")
    if cst is not None:
        s21c = np.interp(lam, cst["lam"][::-1], np.abs(cst["S21"])[::-1])
        s11c = np.interp(lam, cst["lam"][::-1], np.abs(cst["S11"])[::-1])
        print(f"vs CST direct: max ||S21|-|S21|| = "
              f"{np.abs(s21c - np.abs(S21)).max():.4f}; max ||S11|-|S11|| = "
              f"{np.abs(s11c - np.abs(S11)).max():.4f}")
    print(f"saved {f1}\n      {f2}")


if __name__ == "__main__":
    main()

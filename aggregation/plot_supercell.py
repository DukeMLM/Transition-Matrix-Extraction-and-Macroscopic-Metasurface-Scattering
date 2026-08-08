"""Figures and agreement metrics for a run_supercell.py output directory.

Overlays whatever it finds:
  periodic_results.npz            this repo's block-Bloch reconstruction
  floquet_orders.csv              its per-order Floquet amplitudes
  treams_reference.npz            independent treams/Ewald cross-check
  cst_direct_reference.csv        direct CST run, one atom per cell
  cst_direct_supercell.csv        direct CST run, heterogeneous supercell

    python plot_supercell.py results_2x2_super_l3 --title "a,b;b,a, 16 um cell"
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C0 = 299.792458


def thz_axis(ax):
    top = ax.secondary_xaxis(
        "top", functions=(lambda l: C0 / np.maximum(l, 1e-9),
                          lambda f: C0 / np.maximum(f, 1e-9)))
    top.set_xlabel("Frequency (THz)")


def parabola_min(lam, y):
    """Sub-sample wavelength of a minimum, fitted in frequency."""
    i = int(np.argmin(y))
    if i in (0, len(y) - 1):
        return lam[i]
    f0, f1, f2 = C0 / np.asarray(lam[i - 1:i + 2], float)
    y0, y1, y2 = y[i - 1:i + 2]
    a, b = f1 - f0, f1 - f2
    den = a * (y1 - y2) - b * (y1 - y0)
    if abs(den) < 1e-300:
        return lam[i]
    return C0 / (f1 - 0.5 * (a * a * (y1 - y2) - b * b * (y1 - y0)) / den)


def load_cst(out):
    for name in ("cst_direct_supercell.csv", "cst_direct_reference.csv"):
        p = os.path.join(out, name)
        if os.path.exists(p):
            raw = np.genfromtxt(p, delimiter=",", names=True)
            m = raw["f_THz"] > 0
            d = dict(lam=raw["lam_um"][m],
                     S11=raw["Re_S11"][m] + 1j * raw["Im_S11"][m],
                     S21=raw["Re_S21"][m] + 1j * raw["Im_S21"][m],
                     R=raw["R_total"][m] if "R_total" in raw.dtype.names
                     else raw["R"][m],
                     T=raw["T_total"][m] if "T_total" in raw.dtype.names
                     else raw["T"][m],
                     A=raw["A"][m], name=name)
            for key in ("R_higher", "T_higher"):
                if key in raw.dtype.names:
                    d[key] = raw[key][m]
            return d
    return None


def interp_c(lam, cl, cy):
    o = np.argsort(cl)
    return (np.interp(lam, cl[o], cy[o].real)
            + 1j * np.interp(lam, cl[o], cy[o].imag))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--title", default="")
    ap.add_argument("--fine", default=None,
                    help="a second run_supercell.py directory on a refined "
                         "frequency grid (--refine N), drawn as a thin line so "
                         "the lattice resonances the stored grid aliases are "
                         "visible")
    args = ap.parse_args()
    out = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(
        HERE, args.out_dir)
    fine = None
    if args.fine:
        fp = args.fine if os.path.isabs(args.fine) else os.path.join(HERE,
                                                                     args.fine)
        fine = np.load(os.path.join(fp, "periodic_results.npz"),
                       allow_pickle=True)

    d = np.load(os.path.join(out, "periodic_results.npz"), allow_pickle=True)
    lam, S11, S21 = d["lam"], d["S11"], d["S21"]
    R, T, A = d["R"], d["T"], d["A"]
    a1, a2 = d["a1"], d["a2"]
    period = float(np.linalg.norm(a1))

    tr = None
    p = os.path.join(out, "treams_reference.npz")
    if os.path.exists(p):
        tr = np.load(p, allow_pickle=True)
    cst = load_cst(out)

    # ---- fig 1: complex S ---------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    for col, (key, mine) in enumerate((("S21", S21), ("S11", S11))):
        for row, part in enumerate(("abs", "phase")):
            ax = axes[row, col]
            def y(v):
                return np.abs(v) if part == "abs" else np.unwrap(np.angle(v))
            if cst is not None:
                ax.plot(cst["lam"], y(cst[key]), "-", lw=1.8, color="0.6",
                        label=f"direct CST ({cst['name'].split('_')[-1][:-4]})")
            if fine is not None:
                ax.plot(fine["lam"], y(fine[key]), "-", lw=1.0, color="C0",
                        alpha=0.55,
                        label="this repo, refined frequency grid")
            ax.plot(lam, y(mine), "o-", ms=5, lw=1.6, color="C0",
                    label="T-matrix -> block-Bloch (this repo)")
            if tr is not None:
                ax.plot(tr["lam_um"], y(tr[key]), "s--", ms=4, lw=1.1,
                        color="C1", alpha=0.85, label="treams (independent)")
            ax.set_xlabel("Wavelength (um)")
            ax.set_ylabel(f"|{key}|" if part == "abs" else f"arg {key} (rad)")
            ax.set_xlim(lam.min(), lam.max())
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(frameon=False, fontsize=8.5)
                thz_axis(ax)
    fig.suptitle(args.title or os.path.basename(out), fontsize=11)
    f1 = os.path.join(out, "fig1_sparams.png")
    fig.savefig(f1, dpi=170)
    plt.close(fig)

    # ---- fig 2: power balance and diffraction -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    ax = axes[0]
    if cst is not None:
        for yv, c in ((cst["R"], "C3"), (cst["T"], "C2"), (cst["A"], "C4")):
            ax.plot(cst["lam"], yv, "-", lw=1.4, color=c, alpha=0.35)
    ax.plot(lam, R, "o-", ms=4, color="C3", label="R (all open orders)")
    ax.plot(lam, T, "o-", ms=4, color="C2", label="T (all open orders)")
    ax.plot(lam, A, "o-", ms=4, color="C4", label="A = 1 - R - T")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("power fraction")
    ax.set_xlim(lam.min(), lam.max())
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("thin lines: direct CST", fontsize=9)
    thz_axis(ax)

    ax = axes[1]
    hi = d["R_hi"] + d["T_hi"]
    ax.plot(lam, hi, "o-", ms=4, color="C5",
            label="power into higher orders (this repo)")
    if cst is not None and "R_higher" in cst:
        ax.plot(cst["lam"], cst["R_higher"] + cst["T_higher"], "-", lw=1.4,
                color="0.5", label="direct CST")
    for n in ((1, 0), (1, 1), (2, 0)):
        lr = period / np.hypot(*n)
        if lam.min() <= lr <= lam.max():
            ax.axvline(lr, color="k", ls=":", lw=0.9)
            ax.text(lr, ax.get_ylim()[1] * 0.92, f"  {n} opens",
                    fontsize=8, rotation=90, va="top")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("diffracted power fraction")
    ax.set_xlim(lam.min(), lam.max())
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Rayleigh onsets of the {period:g} um cell", fontsize=9)
    thz_axis(ax)
    f2 = os.path.join(out, "fig2_power.png")
    fig.savefig(f2, dpi=170)
    plt.close(fig)

    # ---- metrics ------------------------------------------------------------
    print(f"=== {os.path.basename(out)} ===")
    print(f"  cell {a1} x {a2} um, {len(lam)} frequencies, "
          f"lmax {d['lmax'].min()}-{d['lmax'].max()}")
    print(f"  transmission minimum (parabolic fit in frequency):")
    print(f"    this repo  : {parabola_min(lam, np.abs(S21)):.3f} um")
    if tr is not None:
        print(f"    treams     : {parabola_min(tr['lam_um'], np.abs(tr['S21'])):.3f} um")
    if cst is not None:
        m = (cst["lam"] >= lam.min()) & (cst["lam"] <= lam.max())
        print(f"    direct CST : {parabola_min(cst['lam'][m], np.abs(cst['S21'][m])):.3f} um")
    if tr is not None:
        for key, mine, ref in (("S21", S21, tr["S21"]), ("S11", S11, tr["S11"])):
            dd = np.abs(mine - ref)
            print(f"  vs treams  complex {key}: max {dd.max():.3e}  "
                  f"mean {dd.mean():.3e}")
    if cst is not None:
        for key, mine in (("S21", S21), ("S11", S11)):
            ref = interp_c(lam, cst["lam"], cst[key])
            dd = np.abs(mine - ref)
            print(f"  vs CST     complex {key}: max {dd.max():.4f}  "
                  f"mean {dd.mean():.4f}")
        o = np.argsort(cst["lam"])
        for key, mine in (("R", R), ("T", T), ("A", A)):
            ref = np.interp(lam, cst["lam"][o], cst[key][o])
            print(f"  vs CST     {key}: max |d| {np.abs(mine-ref).max():.4f}  "
                  f"mean {np.abs(mine-ref).mean():.4f}")
    print(f"  saved {f1}\n        {f2}")


if __name__ == "__main__":
    main()

"""Smoke tests (i)-(v) for precompute_C.py (streaming Bloch-sum cache).

Run per stage (each stage fits well under a 10-minute budget; stages (ii),
(iii8), (iii12) recompute lattice sums from scratch and are the slow ones):

  python test_precompute.py --stage i       # theta=0 vs stored run_demo C
  python test_precompute.py --stage ii      # (30,22.5) vs fresh bloch sum
  python test_precompute.py --stage iii8    # (60,45) vs fresh sum, lam=8
  python test_precompute.py --stage iii12   # (60,45) vs fresh sum, lam=12
  python test_precompute.py --stage iv      # resume/byte-identity, freq 48
  python test_precompute.py --stage v       # timing + full-run extrapolation
  python test_precompute.py --stage quick   # i only (no recompute)

Smoke frequencies (nearest lam = 8 and 12 um): indices 48 and 32.  Stages
(i)-(iv) require the checkpoints results/C_bloch/freq_48.npz and freq_32.npz
(create with `python precompute_C.py --freqs 48,32 --angles all`).

Gates:
  (i)   C[angle (0,0)] vs aggregation/results/periodic_results.npz key "C"
        at the same frequency index: rel Frobenius <= 1e-12.
  (ii)  C[angle (30,22.5)] vs a fresh lattice_sum_C_bloch(kRc(30)) call:
        rel Frobenius <= 1e-13 (same shells, same tapers; only float
        regrouping differs).
  (iii) C[angle (60,45)] vs fresh lattice_sum_C_bloch(kRc(60)): <= 1e-13.
  (iv)  delete freq_48.npz, re-run the same command path, np.array_equal on
        C/have/kpar/kRc/r_max (byte-identical resume).
  (v)   report measured per-freq wall time and extrapolate the full 49-freq
        serial cost and the 4-worker cost (informational, always PASS).
"""
import argparse
import os
import sys
import time


import numpy as np

from tmatrix.aggregation.tmat_io import TMatrixData
from tmatrix.aggregation.vswf import RegularProjector
from tmatrix.aggregation.translate import make_quad, square_lattice_shells
from tmatrix.retrieval.bloch_lattice import lattice_sum_C_bloch
from tmatrix.retrieval import precompute_C as pc
from tmatrix.numerics import rel_frob
from tmatrix.paths import AGG_RESULTS

SMOKE_FREQS = (48, 32)            # lam = 8.00 and 12.00 um exactly
PERIODIC_NPZ = os.path.join(AGG_RESULTS, "periodic_results.npz")

RESULTS = []


def record(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}",
          flush=True)




def load_ck(ifreq):
    path = pc.checkpoint_path(pc.OUT_DEFAULT, ifreq)
    d = pc.load_checkpoint(path)
    if d is None:
        raise SystemExit(f"missing/invalid checkpoint {path} -- run "
                         f"`python precompute_C.py --freqs {ifreq}` first")
    return d


# ---------------------------------------------------------------- stage i
def stage_i(data):
    stored = np.load(PERIODIC_NPZ)["C"]
    for ifreq in SMOKE_FREQS:
        d = load_ck(ifreq)
        if not d["have"][0]:
            record(f"(i) theta=0 vs run_demo C, ifreq={ifreq}", False,
                   "angle 0 missing from checkpoint")
            continue
        err = rel_frob(d["C"][0], stored[ifreq])
        record(f"(i) theta=0 entry vs stored run_demo C, ifreq={ifreq} "
               f"(lam={data.wavelength_um[ifreq]:.2f} um)",
               err <= 1e-12,
               f"rel Frobenius = {err:.3e} (gate 1e-12; stored C from "
               f"translate.lattice_sum_C, kRc=(10,14,20))")


# ------------------------------------------------------------- stage ii/iii
def fresh_compare(data, ifreq, iang, gate, tag):
    d = load_ck(ifreq)
    if not d["have"][iang]:
        record(tag, False, f"angle {iang} missing from checkpoint")
        return
    k = data.k_at(ifreq)
    modes = data.modes
    quad = make_quad(*pc.QUAD_SPEC)
    proj = RegularProjector(modes, quad)
    kp = pc.kpar_table(k)[iang]
    t0 = time.time()
    C_fresh = lattice_sum_C_bloch(k, pc.PITCH, modes, pc.R0, quad, kp,
                                  kRc=pc.KRC_TABLE[iang], projector=proj)
    dt = time.time() - t0
    err = rel_frob(d["C"][iang], C_fresh)
    record(tag, err <= 1e-13,
           f"rel Frobenius = {err:.3e} (gate 1e-13); fresh call: "
           f"kRc={tuple(np.round(pc.KRC_TABLE[iang], 3))}, "
           f"|k_par|={np.linalg.norm(kp):.5f}, {dt:.1f} s; "
           f"|C|max={np.abs(C_fresh).max():.4g}")


def stage_ii(data):
    for ifreq in SMOKE_FREQS:
        fresh_compare(
            data, ifreq, 5, 1e-13,
            f"(ii) (30,22.5) vs fresh lattice_sum_C_bloch(kRc(30)), "
            f"ifreq={ifreq} (lam={data.wavelength_um[ifreq]:.2f} um)")


def stage_iii(data, ifreq):
    fresh_compare(
        data, ifreq, 12, 1e-13,
        f"(iii) (60,45) vs fresh lattice_sum_C_bloch(kRc(60)), "
        f"ifreq={ifreq} (lam={data.wavelength_um[ifreq]:.2f} um)")


# ---------------------------------------------------------------- stage iv
def stage_iv(data):
    ifreq = 48
    path = pc.checkpoint_path(pc.OUT_DEFAULT, ifreq)
    d_old = load_ck(ifreq)
    if not d_old["have"].all():
        record("(iv) resume byte-identity, freq 48", False,
               "checkpoint incomplete -- run all 17 angles first")
        return
    os.remove(path)
    print(f"    deleted {path}; re-running the same compute path "
          f"(all 17 angles) ...", flush=True)
    quad = make_quad(*pc.QUAD_SPEC)
    proj = RegularProjector(data.modes, quad)
    t0 = time.time()
    pc.process_freq(data, ifreq, list(range(pc.N_ANGLES)), proj, quad,
                    pc.OUT_DEFAULT, chunk=48)
    dt = time.time() - t0
    d_new = load_ck(ifreq)
    checks = {key: bool(np.array_equal(d_old[key], d_new[key]))
              for key in ("C", "have", "kpar", "kRc", "r_max",
                          "theta_deg", "phi_deg")}
    ok = all(checks.values())
    worst = ""
    if not checks["C"]:
        worst = (f"; max|dC| = {np.abs(d_old['C'] - d_new['C']).max():.3e}"
                 f" (byte-compare failed -> report nondeterminism)")
    record("(iv) resume: delete freq_48.npz, re-run, np.array_equal",
           ok, f"array_equal per key: {checks}; re-run {dt:.1f} s{worst}")


# ---------------------------------------------------------------- stage v
def stage_v(data):
    # measured per-shell rates from the smoke checkpoints
    rates = []
    lines = []
    for ifreq in SMOKE_FREQS:
        d = load_ck(ifreq)
        ns = int(d["nshell_last_run"])
        el = float(d["elapsed_last_run_s"])
        rates.append(el / ns * 1000.0)
        lines.append(f"ifreq={ifreq} (lam={float(d['lam_um']):.2f} um): "
                     f"nshell={ns}, wall={el:.1f} s "
                     f"({el / ns * 1000:.2f} ms/shell incl. assembly)")
    ms = float(np.mean(rates))
    # exact shell counts for all 49 at the full 17-angle superset
    t0 = time.time()
    counts = []
    for i in range(49):
        k = data.k_at(i)
        r_max_sup = pc.r_max_table(k).max()
        radii, _ = square_lattice_shells(pc.PITCH, r_max_sup)
        counts.append(len(radii))
    t_enum = time.time() - t0
    counts = np.array(counts)
    serial_s = counts.sum() * ms / 1000.0
    lines.append(f"exact shell counts for all 49 freqs enumerated in "
                 f"{t_enum:.1f} s: min={counts.min()} (lam=8), "
                 f"max={counts.max()} (lam=20), total={counts.sum()}")
    lines.append(f"extrapolated FULL precompute (17 angles x 49 freqs) at "
                 f"{ms:.2f} ms/shell: serial = {serial_s / 3600:.2f} h; "
                 f"4 workers ~= {serial_s / 4 / 3600:.2f} h (frequency-"
                 f"parallel, workers fully independent)")
    lines.append(f"worst single freq (lam=20 um, ifreq=0): "
                 f"{counts[0] * ms / 1000 / 60:.1f} min -- exceeds a 10-min "
                 f"call budget; drive it via --freqs batches or --angles "
                 f"splits (merge is exact) or run detached")
    record("(v) timing + extrapolation", True, "; ".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["i", "ii", "iii8", "iii12", "iv", "v", "quick"])
    args = ap.parse_args(argv)
    data = TMatrixData(pc.DATA)
    print(f"stage {args.stage}: smoke freqs {SMOKE_FREQS} "
          f"(lam = {data.wavelength_um[SMOKE_FREQS[0]]:.2f}, "
          f"{data.wavelength_um[SMOKE_FREQS[1]]:.2f} um)", flush=True)
    if args.stage in ("i", "quick"):
        stage_i(data)
    elif args.stage == "ii":
        stage_ii(data)
    elif args.stage == "iii8":
        stage_iii(data, 48)
    elif args.stage == "iii12":
        stage_iii(data, 32)
    elif args.stage == "iv":
        stage_iv(data)
    elif args.stage == "v":
        stage_v(data)
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n=== SUMMARY (test_precompute --stage {args.stage}) ===")
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("ALL TESTS PASSED" if nfail == 0 else f"{nfail} TEST(S) FAILED")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()

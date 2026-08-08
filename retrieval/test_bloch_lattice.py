"""Tests (a)-(d) for bloch_lattice.py (retrieval deliverable 1).

Run:  python test_bloch_lattice.py       (conda env cst_inference)

(a) k_par = 0 reproduces translate.lattice_sum_C (gate 1e-12 rel Frobenius;
    the code path is the same arithmetic, so near machine-exact expected).
(b) Nonzero-k_par correctness gates: brute-force per-site sum using
    translate.translation_shells + translate.rotate_inplane (gate 1e-13),
    the conjugate-phase identity for C(-k_par), and the lattice-inversion
    identity C(-k_par) = rotate_inplane(C(+k_par), pi).
(c) Richardson/taper stability at lam~14 um, th=30 deg, ph=22.5 deg with the
    doc-scaled taper set (base x 2.0) vs a 1.3x larger set.
(d) th=60 deg, lam~20 um worst case: BENCHMARK first, then build the largest
    affordable shell superset (<= ~15 min projected) and measure taper
    stability as a function of the kRc scaling factor, using prefix slices of
    the one superset (slice_shells) so every taper set costs assembly only.

All numbers are printed; the summary lists PASS/FAIL per test and the exit
code is nonzero on any failure.
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "aggregation"))

import numpy as np

from tmat_io import TMatrixData
from vswf import RegularProjector
from translate import (lattice_sum_C, make_quad, square_lattice_shells,
                       translation_shells, rotate_inplane)
from bloch_lattice import (lattice_sum_C_bloch, build_shells, slice_shells,
                           assemble_shell_sum_bloch)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "test", "single",
                    "saw_gold_wl15p0025um.tmat.h5")
PITCH = 2.0
R0 = 0.8
KRC_BASE = np.array([10.0, 14.0, 20.0])

RESULTS = []


def record(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}", flush=True)


def rel_frob(A, B):
    """|A - B|_F / |B|_F."""
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def masked_rel_frob(CA, CB, floor_frac=1e-6):
    """Relative Frobenius difference restricted to entries with
    |CA| > floor_frac * max|CA| (the metric prescribed for test (c))."""
    mask = np.abs(CA) > floor_frac * np.abs(CA).max()
    return (float(np.linalg.norm((CA - CB)[mask]) /
                  np.linalg.norm(CA[mask])), int(mask.sum()))


def entrywise_metric(CA, CB, k, pitch):
    """test_translate-style metric: max |dC| / (|C| + 2 pi/(k A)) -- relative
    for large entries, absolute against the radiative scale for small ones."""
    scale = 2 * np.pi / (k * pitch ** 2)
    return float((np.abs(CA - CB) / (np.abs(CA) + scale)).max())


def richardson(Cs, Rcs):
    """Same Lagrange extrapolation in 1/Rc^2 as translate.assemble_shell_sum."""
    x = 1.0 / np.asarray(Rcs, dtype=float) ** 2
    C = np.zeros_like(Cs[0])
    for i in range(len(Rcs)):
        L = 1.0
        for j in range(len(Rcs)):
            if j != i:
                L *= x[j] / (x[j] - x[i])
        C += L * Cs[i]
    return C


def brute_bloch(A_s, radii, angles, modes, Rcs, kpar, sign):
    """Explicit per-site reference sum:
    sum_R w(|R|) rotate_inplane(A_shell, phi_R) e^{sign * i kpar . R},
    Richardson-extrapolated with the same weights.  Independent of the
    vectorized unique-dm assembly -- this is the critical bookkeeping gate."""
    n = modes.n
    Cs = []
    for Rc in Rcs:
        Ct = np.zeros((n, n), dtype=complex)
        for s, (r, ph) in enumerate(zip(radii, angles)):
            w = np.exp(-(r / Rc) ** 2)
            if w < 1e-16:
                continue
            for p in np.atleast_1d(ph):
                Rx, Ry = r * np.cos(p), r * np.sin(p)
                phase = np.exp(sign * 1j * (kpar[0] * Rx + kpar[1] * Ry))
                Ct += w * phase * rotate_inplane(A_s[s], p, modes)
        Cs.append(Ct)
    return richardson(Cs, Rcs)


def nearest_idx(data, lam_target):
    return int(np.argmin(np.abs(data.wavelength_um - lam_target)))


# ---------------------------------------------------------------- test (a)
def test_a(data, quad):
    modes = data.modes
    for lam_t in (8.0, 20.0):
        idx = nearest_idx(data, lam_t)
        k = data.k_at(idx)
        lam = data.wavelength_um[idx]
        t0 = time.time()
        C_ref = lattice_sum_C(k, PITCH, modes, R0, quad, kRc=tuple(KRC_BASE))
        t_ref = time.time() - t0
        cache = {}
        t0 = time.time()
        C_b = lattice_sum_C_bloch(k, PITCH, modes, R0, quad, (0.0, 0.0),
                                  kRc=KRC_BASE, shells_cache=cache)
        t_b = time.time() - t0
        err = rel_frob(C_b, C_ref)
        t0 = time.time()
        C_b2 = lattice_sum_C_bloch(k, PITCH, modes, R0, quad, (0.0, 0.0),
                                   kRc=KRC_BASE, shells_cache=cache)
        t_b2 = time.time() - t0
        cache_same = bool(np.array_equal(C_b, C_b2))
        record(f"(a) k_par=0 == lattice_sum_C at lam={lam:.2f} um",
               err <= 1e-12 and cache_same,
               f"rel Frobenius = {err:.3e} (gate 1e-12); "
               f"|C|max = {np.abs(C_ref).max():.3g}; "
               f"cache-reuse bit-identical = {cache_same}; "
               f"t(translate) = {t_ref:.1f} s, t(bloch, fresh shells) = "
               f"{t_b:.1f} s, t(bloch, cached shells) = {t_b2:.2f} s")


# ---------------------------------------------------------------- test (b)
def test_b(data, quad):
    modes = data.modes
    idx = nearest_idx(data, 14.0)
    k = data.k_at(idx)
    r_max = 7.3                       # first 8 square-lattice shells
    radii, angles = square_lattice_shells(PITCH, r_max)
    proj = RegularProjector(modes, quad)
    A_s = translation_shells(k, radii, modes, R0, quad, projector=proj)
    Rcs = np.array([3.0, 4.0, 5.0])   # strong taper, all shells contribute
    th, ph = np.deg2rad(37.0), np.deg2rad(17.0)   # generic, no symmetry
    kp = k * np.sin(th) * np.array([np.cos(ph), np.sin(ph)])

    C_p = assemble_shell_sum_bloch(A_s, radii, angles, modes, Rcs, kp)
    C_m = assemble_shell_sum_bloch(A_s, radii, angles, modes, Rcs, -kp)
    B_p = brute_bloch(A_s, radii, angles, modes, Rcs, kp, +1)
    B_m = brute_bloch(A_s, radii, angles, modes, Rcs, kp, -1)

    e1 = rel_frob(C_p, B_p)
    e2 = rel_frob(C_m, B_m)
    e3 = rel_frob(rotate_inplane(C_p, np.pi, modes), C_m)
    nsites = sum(len(np.atleast_1d(a)) for a in angles)
    record("(b1) vectorized assembly == brute-force per-site sum at +k_par",
           e1 <= 1e-13,
           f"nshell={len(radii)}, nsites={nsites}, k={k:.4f}, "
           f"|k_par|={np.linalg.norm(kp):.4f}: rel Frobenius = {e1:.3e} "
           f"(gate 1e-13)")
    record("(b2) C(-k_par) == brute-force sum with conjugate phases "
           "e^{-i k_par.R}", e2 <= 1e-13,
           f"rel Frobenius = {e2:.3e} (gate 1e-13)")
    record("(b3) lattice inversion: C(-k_par) == rotate_inplane(C(+k_par), pi)",
           e3 <= 1e-12, f"rel Frobenius = {e3:.3e} (gate 1e-12)")


# ---------------------------------------------------------------- test (c)
def test_c(data, quad):
    modes = data.modes
    idx = nearest_idx(data, 14.0)
    k = data.k_at(idx)
    lam = data.wavelength_um[idx]
    th, ph = np.deg2rad(30.0), np.deg2rad(22.5)
    kp = k * np.sin(th) * np.array([np.cos(ph), np.sin(ph)])
    scale = 1.0 / (1.0 - np.sin(th))          # doc par. 2 scaling = 2.0
    setA = KRC_BASE * scale                   # (20, 28, 40)
    setB = setA * 1.3                         # (26, 36.4, 52)
    proj = RegularProjector(modes, quad)
    t0 = time.time()
    radii, angles, A_s = build_shells(k, PITCH, modes, R0, quad,
                                      3.5 * setB.max() / k, projector=proj)
    t_build = time.time() - t0
    rA, aA, AA = slice_shells(radii, angles, A_s, PITCH, 3.5 * setA.max() / k)
    t0 = time.time()
    C_A = assemble_shell_sum_bloch(AA, rA, aA, modes, setA / k, kp)
    C_B = assemble_shell_sum_bloch(A_s, radii, angles, modes, setB / k, kp)
    t_asm = time.time() - t0
    m_rel, nmask = masked_rel_frob(C_A, C_B)
    e_ent = entrywise_metric(C_A, C_B, k, PITCH)
    u_rel = rel_frob(C_B, C_A)
    # context: identical taper sets at k_par = 0
    C_A0 = assemble_shell_sum_bloch(AA, rA, aA, modes, setA / k, (0.0, 0.0))
    C_B0 = assemble_shell_sum_bloch(A_s, radii, angles, modes, setB / k,
                                    (0.0, 0.0))
    m0 = masked_rel_frob(C_A0, C_B0)[0]
    record(f"(c) Richardson stability, lam={lam:.2f} um, th=30, ph=22.5",
           m_rel <= 1e-3,
           f"kRc {tuple(setA)} vs {tuple(np.round(setB, 1))}: "
           f"masked rel Frobenius = {m_rel:.3e} (gate 1e-3; mask |C| > "
           f"1e-6 max|C|, {nmask}/{modes.n ** 2} entries); "
           f"unmasked rel = {u_rel:.3e}; entrywise max|dC|/(|C|+2pi/kA) = "
           f"{e_ent:.3e}; same sets at k_par=0: masked = {m0:.3e}; "
           f"nshell = {len(radii)} (setA slice {len(rA)}), "
           f"build {t_build:.0f} s, 4 assemblies {t_asm:.1f} s")


# ---------------------------------------------------------------- test (d)
def test_d(data, quad):
    modes = data.modes
    idx = nearest_idx(data, 20.0)
    k = data.k_at(idx)
    lam = data.wavelength_um[idx]
    th, ph = np.deg2rad(60.0), np.deg2rad(22.5)
    kp = k * np.sin(th) * np.array([np.cos(ph), np.sin(ph)])
    s_doc = 1.0 / (1.0 - np.sin(th))          # 7.464
    print(f"    worst case: lam={lam:.2f} um, th=60 deg -> doc scaling "
          f"x{s_doc:.3f}, kRc = {tuple(np.round(KRC_BASE * s_doc, 1))}",
          flush=True)

    # ---- mandated benchmark BEFORE committing to any large build
    proj = RegularProjector(modes, quad)
    radii_b, _ = square_lattice_shells(PITCH, 3.5 * KRC_BASE.max() / k)
    nb = min(240, len(radii_b))
    t0 = time.time()
    translation_shells(k, radii_b[:nb], modes, R0, quad, projector=proj)
    ms_shell = (time.time() - t0) / nb * 1000.0
    print(f"    benchmark: translation_shells = {ms_shell:.2f} ms/shell "
          f"({nb}-shell sample)", flush=True)

    def profile(s):
        r_max = 3.5 * KRC_BASE.max() * s / k
        t0 = time.time()
        rr, _ = square_lattice_shells(PITCH, r_max)
        t_enum = time.time() - t0
        t_proj = ms_shell * len(rr) / 1000.0
        mem = len(rr) * modes.n ** 2 * 16 / 1e9
        return r_max, len(rr), t_enum, t_proj, mem

    candidates = [s_doc, 5.0, 4.0, 3.0, 2.0]
    print("    projected costs (enumeration measured, projection projected "
          "from benchmark):", flush=True)
    prof = {}
    for s in candidates + [1.3 * s_doc]:
        r_max, nsh, t_enum, t_proj, mem = profile(s)
        prof[s] = (r_max, nsh, t_enum, t_proj, mem)
        tag = " (1.3x companion of doc scaling; NOT a candidate)" \
            if s > s_doc + 1e-9 else ""
        print(f"      s={s:6.3f}: r_max={r_max:7.1f} um, nshell={nsh:7d}, "
              f"enum={t_enum:5.2f} s, projected build={t_proj / 60:6.2f} min, "
              f"A_s memory={mem:5.2f} GB{tag}", flush=True)

    BUDGET_S, MEM_CAP_GB = 900.0, 6.0
    s_sup = None
    for s in candidates:                      # descending: largest affordable
        if prof[s][3] <= BUDGET_S and prof[s][4] <= MEM_CAP_GB:
            s_sup = s
            break
    if s_sup is None:
        record("(d) th=60 taper-stability study", False,
               "no candidate scaling fits the 15-min/6-GB budget "
               f"(smallest projected: {prof[candidates[-1]][3] / 60:.1f} min)")
        return
    print(f"    -> chosen superset scaling s={s_sup:.3f} "
          f"(largest with projected build <= {BUDGET_S / 60:.0f} min and "
          f"A_s <= {MEM_CAP_GB:.0f} GB)", flush=True)

    # ---- one superset build; all taper sets below are prefix slices
    r_max_sup = 3.5 * KRC_BASE.max() * s_sup / k
    t0 = time.time()
    radii, angles = square_lattice_shells(PITCH, r_max_sup)
    t_enum = time.time() - t0
    t0 = time.time()
    A_s = translation_shells(k, radii, modes, R0, quad, projector=proj)
    t_proj = time.time() - t0
    print(f"    superset built: nshell={len(radii)}, enum {t_enum:.1f} s, "
          f"projection {t_proj / 60:.2f} min "
          f"(projected {prof[s_sup][3] / 60:.2f} min)", flush=True)

    # ---- stability (set vs 1.3x set) as a function of the scaling factor
    s_los = [s for s in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
             if 1.3 * s <= s_sup + 1e-9]
    if all(abs(s_sup / 1.3 - s) > 1e-6 for s in s_los):
        s_los.append(s_sup / 1.3)
    s_los.sort()
    rows = []
    C_prev = None
    print("    s_lo    kRc set A         k(1-sin th)Rc   masked relF(A,1.3xA)"
          "  entrywise   relF(A(s_prev),A(s))  t_asm", flush=True)
    for s in s_los:
        setA = KRC_BASE * s
        setB = setA * 1.3
        rA, aA, AA = slice_shells(radii, angles, A_s, PITCH,
                                  3.5 * setA.max() / k)
        rB, aB, AB = slice_shells(radii, angles, A_s, PITCH,
                                  3.5 * setB.max() / k)
        t0 = time.time()
        C_A = assemble_shell_sum_bloch(AA, rA, aA, modes, setA / k, kp)
        C_B = assemble_shell_sum_bloch(AB, rB, aB, modes, setB / k, kp)
        t_asm = time.time() - t0
        m_rel, _ = masked_rel_frob(C_A, C_B)
        e_ent = entrywise_metric(C_A, C_B, k, PITCH)
        succ = masked_rel_frob(C_prev, C_A)[0] if C_prev is not None \
            else float("nan")
        xpar = KRC_BASE * s * (1.0 - np.sin(th))
        rows.append((s, m_rel, e_ent, succ))
        print(f"    {s:5.2f}  ({setA[0]:6.1f},{setA[1]:6.1f},{setA[2]:6.1f})"
              f"  {xpar.min():5.2f}..{xpar.max():5.2f}      "
              f"{m_rel:.3e}         {e_ent:.3e}   {succ:.3e}"
              f"        {t_asm:4.1f} s", flush=True)
        C_prev = C_A

    # context: normal-incidence baseline with the base set, same superset
    rA, aA, AA = slice_shells(radii, angles, A_s, PITCH,
                              3.5 * KRC_BASE.max() / k)
    rB, aB, AB = slice_shells(radii, angles, A_s, PITCH,
                              3.5 * 1.3 * KRC_BASE.max() / k)
    C_A0 = assemble_shell_sum_bloch(AA, rA, aA, modes, KRC_BASE / k,
                                    (0.0, 0.0))
    C_B0 = assemble_shell_sum_bloch(AB, rB, aB, modes, 1.3 * KRC_BASE / k,
                                    (0.0, 0.0))
    m0 = masked_rel_frob(C_A0, C_B0)[0]
    e0 = entrywise_metric(C_A0, C_B0, k, PITCH)
    print(f"    baseline th=0, base kRc {tuple(KRC_BASE)} vs 1.3x: "
          f"masked relF = {m0:.3e}, entrywise = {e0:.3e}", flush=True)

    # ---- verdict
    st_first, st_last = rows[0][1], rows[-1][1]
    improving = st_last < st_first
    converged = st_last <= 1e-3
    doc_reached = abs(s_sup - s_doc) < 1e-6
    verdict = []
    verdict.append(f"stability at s={rows[-1][0]:.2f} (largest tested pair, "
                   f"vs 1.3x = s={1.3 * rows[-1][0]:.2f}): "
                   f"masked relF = {st_last:.3e}, entrywise = "
                   f"{rows[-1][2]:.3e}; at s=1: {st_first:.3e}")
    if doc_reached:
        verdict.append(f"doc scaling x{s_doc:.2f} WAS reached by the "
                       f"superset (its stability pair is the last row)")
    else:
        verdict.append(f"doc scaling x{s_doc:.2f} NOT affordable "
                       f"(projected {prof[s_doc][3] / 60:.1f} min, "
                       f"{prof[s_doc][4]:.1f} GB); largest tested "
                       f"s={s_sup:.2f}")
    if converged and improving:
        verdict.append("VERDICT: tapered sums CONVERGE at th=60 with scaled "
                       "kRc; Ewald fallback not required at this angle")
    elif improving:
        verdict.append("VERDICT: stability improves with scaling but has not "
                       f"reached 1e-3 at the largest affordable scaling -> "
                       "use larger kRc (cost above) or the doc's Ewald "
                       "(treams) fallback near grazing")
    else:
        verdict.append("VERDICT: stability does NOT improve with scaling -> "
                       "tapered sums degrade at th=60; use the doc's Ewald "
                       "(treams) fallback")
    record("(d) th=60 taper-stability study", improving,
           "; ".join(verdict))


def main():
    print(f"data: {os.path.abspath(DATA)}")
    print(f"numpy {np.__version__}, pitch={PITCH}, r0={R0}, "
          f"quad=make_quad(16,32), base kRc={tuple(KRC_BASE)}")
    data = TMatrixData(DATA)
    print(f"loaded: {len(data.freq)} freqs, lam "
          f"{data.wavelength_um.min():.1f}-{data.wavelength_um.max():.1f} um, "
          f"{data.modes.n} modes (lmax={data.modes.lmax})")
    quad = make_quad(16, 32)
    for fn in (test_a, test_b, test_c, test_d):
        print(f"\n=== {fn.__name__} ===", flush=True)
        t0 = time.time()
        try:
            fn(data, quad)
        except Exception:
            traceback.print_exc()
            record(fn.__name__ + " (exception)", False,
                   "crashed -- see traceback above")
        print(f"    ({time.time() - t0:.1f} s total)", flush=True)
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n=== SUMMARY (test_bloch_lattice) ===")
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("ALL TESTS PASSED" if nfail == 0 else f"{nfail} TEST(S) FAILED")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()

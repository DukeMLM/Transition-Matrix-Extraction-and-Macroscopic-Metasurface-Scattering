"""Tests (e)-(f) for sparams_oblique.py (retrieval deliverable 2).

Run:  python test_sparams_oblique.py       (conda env cst_inference)

(e) theta -> 0 consistency against sparams.sparams_normal at k_hat = +z_hat
    (theta=0, phi=0) at 3 frequencies (band edges + middle), asserting the
    exact sign mapping documented in sparams_oblique.py's module docstring
    (all S21 entries and the S11 TE row map with +, the S11 TM row with -,
    from e_TM^(r) -> -x_hat at theta -> 0 along phi = 0); gate 1e-12.
    Plus continuity |S(theta=1e-4) - S(theta=0)| <= 1e-3 relative (mid-band).
(f) Full oblique forward map at theta=30 deg, phi=22.5 deg, mid-band:
    C2 consistency S(th, ph) == S(th, ph+180) (uses C(k, -k_par) from
    deliverable 1; residual ~ the reference T's C4-violation noise), and
    energy conservation R + T < 1 with A = 1 - R - T in (0, 1) per incident
    polarization (lossy gold).

Lattice sums are cached in test_cache_C.npz next to this file so re-runs are
fast; delete that file to force recomputation.
"""
import os
import sys
import time
import traceback

import numpy as np

from tmatrix.aggregation.tmat_io import TMatrixData
from tmatrix.aggregation.vswf import plane_wave_coeffs
from tmatrix.aggregation.translate import lattice_sum_C, make_quad
from tmatrix.aggregation.aggregate import solve_periodic
from tmatrix.aggregation.sparams import sparams_normal
from tmatrix.retrieval.bloch_lattice import lattice_sum_C_bloch
from tmatrix.retrieval.sparams_oblique import sparams_oblique, energy_balance_oblique, pol_basis
from tmatrix.numerics import nearest_idx
from tmatrix.paths import DEMO_TMAT, RETRIEVAL_DATA

DATA = str(DEMO_TMAT)
#: cached Bloch sums; a run product, so it belongs with the retrieval data
CACHE_PATH = os.path.join(RETRIEVAL_DATA, "test_cache_C.npz")
PITCH = 2.0
A_CELL = PITCH ** 2
R0 = 0.8
KRC_BASE = np.array([10.0, 14.0, 20.0])

RESULTS = []


def record(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}", flush=True)


def load_cache():
    if os.path.exists(CACHE_PATH):
        with np.load(CACHE_PATH) as z:
            return {key: z[key] for key in z.files}
    return {}


def get_C(cache, key, builder):
    if key in cache:
        print(f"    {key}: from cache", flush=True)
    else:
        t0 = time.time()
        cache[key] = builder()
        np.savez(CACHE_PATH, **cache)
        print(f"    {key}: computed in {time.time() - t0:.1f} s (cached)",
              flush=True)
    return cache[key]




# ---------------------------------------------------------------- test (e)
def test_e(data, quad, cache):
    modes = data.modes
    e_x = np.array([1.0, 0.0, 0.0])
    e_y = np.array([0.0, 1.0, 0.0])

    # basis sanity at theta=0, phi=0 (documents the sign mapping's origin)
    te_t, tm_t = pol_basis(0.0, 0.0, +1)
    te_r, tm_r = pol_basis(0.0, 0.0, -1)
    print(f"    basis at th=0, ph=0: e_TE^t={te_t}, e_TM^t={tm_t}, "
          f"e_TE^r={te_r}, e_TM^r={tm_r}  (e_TM^r = -x_hat)", flush=True)

    worst = 0.0
    details = []
    for lam_t in (8.0, 14.0, 20.0):
        idx = nearest_idx(data, lam_t)
        k = data.k_at(idx)
        lam = data.wavelength_um[idx]
        T = data.T[idx]
        C = get_C(cache, f"Cn_{idx}",
                  lambda k=k: lattice_sum_C(k, PITCH, modes, R0, quad,
                                            kRc=tuple(KRC_BASE)))
        f_x = solve_periodic(T, C,
                             plane_wave_coeffs([0, 0, 1.0], e_x, modes))[1]
        f_y = solve_periodic(T, C,
                             plane_wave_coeffs([0, 0, 1.0], e_y, modes))[1]
        Sn_x = sparams_normal(k, A_CELL, modes, f_x, e_co=e_x, e_cross=e_y)
        Sn_y = sparams_normal(k, A_CELL, modes, f_y, e_co=e_y, e_cross=e_x)
        res = sparams_oblique(k, A_CELL, modes, T, C, 0.0, 0.0, +1)
        S11, S21 = res["S11"], res["S21"]
        # normative mapping (see sparams_oblique module docstring):
        checks = [
            ("S21[TE,TE] = +Sn_y[S21_co]", S21[0, 0], Sn_y["S21_co"]),
            ("S21[TM,TE] = +Sn_y[S21_cross]", S21[1, 0], Sn_y["S21_cross"]),
            ("S21[TE,TM] = +Sn_x[S21_cross]", S21[0, 1], Sn_x["S21_cross"]),
            ("S21[TM,TM] = +Sn_x[S21_co]", S21[1, 1], Sn_x["S21_co"]),
            ("S11[TE,TE] = +Sn_y[S11_co]", S11[0, 0], Sn_y["S11_co"]),
            ("S11[TM,TE] = -Sn_y[S11_cross]", S11[1, 0], -Sn_y["S11_cross"]),
            ("S11[TE,TM] = +Sn_x[S11_cross]", S11[0, 1], Sn_x["S11_cross"]),
            ("S11[TM,TM] = -Sn_x[S11_co]", S11[1, 1], -Sn_x["S11_co"]),
        ]
        m = max(abs(got - want) for _, got, want in checks)
        worst = max(worst, m)
        details.append(f"lam={lam:.1f}: max|dS|={m:.2e}")
        if m > 1e-12:
            for nm, got, want in checks:
                print(f"      {nm}: got {got:.6e}, want {want:.6e}, "
                      f"|d|={abs(got - want):.2e}", flush=True)
    record("(e) theta=0 mapping vs sparams_normal "
           "(S21 rows +, S11 TE row +, S11 TM row -)",
           worst <= 1e-12,
           "; ".join(details) + " (gate 1e-12, 8 complex entries per freq)")

    # continuity at mid-band
    idx = nearest_idx(data, 14.0)
    k = data.k_at(idx)
    T = data.T[idx]
    C0 = cache[f"Cn_{idx}"]
    th_small = 1e-4
    kp = k * np.sin(th_small) * np.array([1.0, 0.0])
    C1 = get_C(cache, f"Cth1em4_{idx}",
               lambda: lattice_sum_C_bloch(k, PITCH, modes, R0, quad, kp,
                                           kRc=KRC_BASE))
    Sa = sparams_oblique(k, A_CELL, modes, T, C0, 0.0, 0.0, +1)
    Sb = sparams_oblique(k, A_CELL, modes, T, C1, th_small, 0.0, +1)
    smax = max(np.abs(Sa["S21"]).max(), np.abs(Sa["S11"]).max())
    d = max(np.abs(Sb["S21"] - Sa["S21"]).max(),
            np.abs(Sb["S11"] - Sa["S11"]).max()) / smax
    record("(e) continuity S(theta=1e-4) vs S(theta=0), lam=14 um",
           d <= 1e-3,
           f"max|dS| / max|S| = {d:.3e} (gate 1e-3; max|S| = {smax:.3f})")


# ---------------------------------------------------------------- test (f)
def test_f(data, quad, cache):
    modes = data.modes
    idx = nearest_idx(data, 14.0)
    k = data.k_at(idx)
    lam = data.wavelength_um[idx]
    T = data.T[idx]
    th = np.deg2rad(30.0)
    ph1 = np.deg2rad(22.5)
    ph2 = ph1 + np.pi
    kRc = KRC_BASE / (1.0 - np.sin(th))       # doc scaling x2 at th=30
    kp1 = k * np.sin(th) * np.array([np.cos(ph1), np.sin(ph1)])
    kp2 = k * np.sin(th) * np.array([np.cos(ph2), np.sin(ph2)])   # = -kp1
    shells = {}
    C1 = get_C(cache, f"Cb_{idx}_th30_ph22p5",
               lambda: lattice_sum_C_bloch(k, PITCH, modes, R0, quad, kp1,
                                           kRc=kRc, shells_cache=shells))
    C2 = get_C(cache, f"Cb_{idx}_th30_ph202p5",
               lambda: lattice_sum_C_bloch(k, PITCH, modes, R0, quad, kp2,
                                           kRc=kRc, shells_cache=shells))
    S1 = sparams_oblique(k, A_CELL, modes, T, C1, th, ph1, +1)
    S2 = sparams_oblique(k, A_CELL, modes, T, C2, th, ph2, +1)

    print(f"    lam={lam:.2f} um, th=30, kRc={tuple(kRc)}; "
          f"|k_par|={np.linalg.norm(kp1):.4f}", flush=True)
    for tag, S in (("ph=22.5 ", S1), ("ph=202.5", S2)):
        print(f"    {tag}: |S11|=[[{abs(S['S11'][0, 0]):.4f}, "
              f"{abs(S['S11'][0, 1]):.4f}], [{abs(S['S11'][1, 0]):.4f}, "
              f"{abs(S['S11'][1, 1]):.4f}]]  |S21|=[[{abs(S['S21'][0, 0]):.4f}, "
              f"{abs(S['S21'][0, 1]):.4f}], [{abs(S['S21'][1, 0]):.4f}, "
              f"{abs(S['S21'][1, 1]):.4f}]]", flush=True)
    d11 = np.abs(S1["S11"] - S2["S11"]).max()
    d21 = np.abs(S1["S21"] - S2["S21"]).max()
    record("(f) C2 consistency: S(th,ph) == S(th,ph+180) via C(k,-k_par)",
           max(d11, d21) <= 1e-2,
           f"max|dS11| = {d11:.3e}, max|dS21| = {d21:.3e} (gate 1e-2; "
           f"exact for a C2-symmetric T -- residual measures the reference "
           f"T's C4-violation noise ~1.5e-3 propagated through the map)")

    ok_all = True
    lines = []
    for tag, S in (("ph=22.5", S1), ("ph=202.5", S2)):
        R, Tt, A = energy_balance_oblique(S["S11"], S["S21"])
        for b, pol in enumerate(("TE", "TM")):
            lines.append(f"{tag} inc {pol}: R={R[b]:.6f} T={Tt[b]:.6f} "
                         f"A={A[b]:.6f}")
            ok_all &= bool((0.0 < A[b] < 1.0) and (R[b] + Tt[b] < 1.0))
    record("(f) energy conservation R+T<1, A in (0,1) (lossy gold)",
           ok_all, "; ".join(lines))


def main():
    print(f"data: {os.path.abspath(DATA)}")
    print(f"numpy {np.__version__}, pitch={PITCH}, A_cell={A_CELL}, r0={R0}, "
          f"quad=make_quad(16,32), base kRc={tuple(KRC_BASE)}")
    data = TMatrixData(DATA)
    quad = make_quad(16, 32)
    cache = load_cache()
    for fn in (test_e, test_f):
        print(f"\n=== {fn.__name__} ===", flush=True)
        t0 = time.time()
        try:
            fn(data, quad, cache)
        except Exception:
            traceback.print_exc()
            record(fn.__name__ + " (exception)", False,
                   "crashed -- see traceback above")
        print(f"    ({time.time() - t0:.1f} s total)", flush=True)
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n=== SUMMARY (test_sparams_oblique) ===")
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("ALL TESTS PASSED" if nfail == 0 else f"{nfail} TEST(S) FAILED")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()

"""Oblique treams cross-check of the retrieval forward map
(INVERSE_TMATRIX_FROM_FLOQUET.md par. 6, validation-ladder step 1).

Generalizes aggregation/treams_reference.py (k_par = 0 only) to oblique
incidence.  Angles: (0,0), (0,45), (20,0), (20,45), (40,0), (40,45), both
incident polarizations, all four complex Jones entries of S11 and S21.

Illumination convention (matches the existing k_par=0 reference): incidence
from the +z side travelling DOWN (-z), phase reference z = 0 on both sides.
In OUR model that is sparams_oblique(..., direction=-1) with the same
(theta, phi) and the cached C at k_par = k sin th (cos ph, sin ph)
(C is direction-independent).

treams path per (frequency, angle):
    tm_lat = tm.latticeinteraction.solve(Lattice.square(2.0), kpar)
    basis  = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, 1e-9)
    smat   = treams.SMatrices.from_array(tm_lat, basis)
    M_up/M_down = treams.efield([0,0,0], basis, k0, material, "parity",
                                modetype="up"/"down")     # (3, nmodes)
Incident coefficients: the reference's [1,0,0] lstsq trick is
normal-incidence-only (doc par. 6 warning); instead the desired incident
polarization e_b = sparams_oblique.pol_basis(theta, phi, -1)[b] (0=TE, 1=TM)
is lstsq-solved from M_down @ c = e_b (e_b is transverse to k_hat, so it
lies in the 2-mode span; residual asserted <= 1e-10 in the sweep).  Receive
projection uses OUR bases on both sides, mirroring jones_blocks for
direction=-1 exactly: transmitted (down-going) receive basis =
pol_basis(theta, phi, -1); reflected (UP-going, k_hat_r = k_hat - 2 k_z z)
receive basis = pol_basis(theta, phi, +1).  S[a, b] = e_a* . E.

The SIGN of kpar passed to treams is not documented -> determined
EMPIRICALLY (--sign-test): at (20,0), mid-band frequency index 24
(lam = 14 um), both kpar = +/- k sin th (cos ph, sin ph) are tried; the sign
whose full 8-entry complex match against our forward map is <= 1e-3 is kept
and then CONFIRMED at (40,45) before the sweep (nothing else is tuned
per-angle).  The wrong sign also fails physically: e_TM is not transverse to
the wrong-sign plane wave, so its lstsq residual is O(sin 2*theta) -- both
residuals are recorded as evidence.  The diffr_orders mode ordering at
k_par != 0 is re-verified by printing the basis (kx, ky, pol) rows rather
than assuming the k_par=0 order.

C values come from the precompute_C.py checkpoints (results/C_bloch/);
build the needed subset first, e.g. in batches:
    python precompute_C.py --angles 0,13-16 --freqs 0,4,8,12
    python precompute_C.py --angles 0,13-16 --freqs 16,20,24,28,36,40,44
(freqs 32 and 48 already hold all 17 angles from the smoke tests).

Usage:
    python treams_oblique_validate.py --sign-test        # writes sign file
    python treams_oblique_validate.py --stride 4         # 13-freq sweep
    python treams_oblique_validate.py --stride 1         # full 49 (later)

By default the sweep runs the six VAL_ANGLES entries above and writes
results/treams_oblique_check.npz (the recorded, closed step-1 gate -- that
default is frozen so the record stays reproducible).  --angles selects
CAMPAIGN angles instead, by index into the normative 17-row
precompute_C.ANGLES_DEG table (same comma-list/range idiom as
`precompute_C.py --angles`), from which theta/phi are DERIVED, and --out
sends the record to a different npz, e.g.

    python treams_oblique_validate.py --stride 1 --angles 5,10-12 \
        --out results/treams_oblique_check_campaign.npz

Gate: max complex |dS| <= 1e-3 over all entries/angles/freqs in the subset.
When the selection contains the (0, 0) angle it is additionally compared
against the stored aggregation/results/treams_reference.npz and our stored
periodic results.
Results -> results/treams_oblique_check.npz (or --out).
"""
import argparse
import os
import sys
import time


import numpy as np

import treams
import treams.io

from tmatrix import treams_compat

from tmatrix.retrieval.forward import ForwardModel
from tmatrix.retrieval.sparams_oblique import pol_basis
from tmatrix.retrieval import precompute_C as pc

from tmatrix.paths import AGG_RESULTS, RETRIEVAL_RESULTS

TMAT_FILE = r"D:/Claude/T matrix/test/single/saw_gold_wl15p0025um.tmat.h5"
OUT_FILE = os.path.join(RETRIEVAL_RESULTS, "treams_oblique_check.npz")
SIGN_FILE = os.path.join(RETRIEVAL_RESULTS, "treams_kpar_sign.npz")
PITCH_UM = 2.0

treams_compat.apply()

# ----------------------------------------------------------------------------
# validation angle set: (theta_deg, phi_deg, cache angle index)
# theta=0 entries reuse the k_par=0 cache C (index 0); phi enters only
# through the polarization bases.
# ----------------------------------------------------------------------------
VAL_ANGLES = [
    (0.0, 0.0, 0),
    (0.0, 45.0, 0),
    (20.0, 0.0, 13),
    (20.0, 45.0, 14),
    (40.0, 0.0, 15),
    (40.0, 45.0, 16),
]

_LATTICE = treams.Lattice.square(PITCH_UM)


def check_angle_table(val_angles):
    """Assert each (theta, phi, cache_idx) triple agrees with the normative
    precompute_C.ANGLES_DEG row it names, so the two can never disagree.

    theta must ALWAYS match the table exactly (it sets the taper scaling
    s(theta) the cached C was built with).  phi must match too, with one
    documented exception: at theta = 0 the Bloch vector is k_par = 0 for
    every phi, so the cached row is phi-independent and the default
    VAL_ANGLES (0, 45) entry legitimately reuses cache row 0 (tabulated as
    (0, 0)) while being evaluated with phi = 45 polarization bases.
    """
    for th, phd, cidx in val_angles:
        if not 0 <= cidx < pc.N_ANGLES:
            raise SystemExit(f"cache angle index {cidx} out of range "
                             f"0..{pc.N_ANGLES - 1}")
        th_t = float(pc.ANGLES_DEG[cidx, 0])
        ph_t = float(pc.ANGLES_DEG[cidx, 1])
        assert abs(th - th_t) < 1e-9, (
            f"angle-table disagreement: requested theta={th} but cache row "
            f"{cidx} is theta={th_t} (precompute_C.ANGLES_DEG)")
        assert abs(phd - ph_t) < 1e-9 or abs(th) < 1e-12, (
            f"angle-table disagreement: requested phi={phd} but cache row "
            f"{cidx} is phi={ph_t} at theta={th_t} != 0 "
            f"(precompute_C.ANGLES_DEG)")


def angles_from_cache_indices(idx):
    """(theta_deg, phi_deg, cache_idx) triples DERIVED from the normative
    precompute_C.ANGLES_DEG table -- the two can then not disagree."""
    out = [(float(pc.ANGLES_DEG[i, 0]), float(pc.ANGLES_DEG[i, 1]), int(i))
           for i in idx]
    check_angle_table(out)
    return out


def basis_rows(basis):
    """(kx, ky, pol) rows of a PlaneWaveBasisByComp, defensively."""
    try:
        kx = np.asarray(basis.kx, dtype=float)
        ky = np.asarray(basis.ky, dtype=float)
        pol = np.asarray(basis.pol, dtype=int)
        return list(zip(kx.tolist(), ky.tolist(), pol.tolist()))
    except AttributeError:
        return [tuple(row) for row in basis]


def treams_jones(tm, kpar, theta_deg, phi_deg, tmlat_cache=None,
                 resid_tol=None):
    """Our-convention Jones blocks from treams at one (freq, angle).

    kpar : 2-sequence passed straight to treams (sign under test).
    Returns dict with S11, S21 (2,2, rows=receive 0=TE/1=TM, cols=incident),
    lstsq residuals per incident polarization, basis rows, timings.
    """
    k0 = tm.k0
    key = (round(float(kpar[0]), 12), round(float(kpar[1]), 12))
    t0 = time.time()
    if tmlat_cache is not None and key in tmlat_cache:
        tm_lat = tmlat_cache[key]
        t_lat = 0.0
    else:
        tm_lat = tm.latticeinteraction.solve(_LATTICE, list(kpar))
        t_lat = time.time() - t0
        if tmlat_cache is not None:
            tmlat_cache[key] = tm_lat
    basis = treams.PlaneWaveBasisByComp.diffr_orders(list(kpar), _LATTICE,
                                                     1e-9)
    smat = treams.SMatrices.from_array(tm_lat, basis)
    M_up = np.asarray(
        treams.efield([0, 0, 0.0], basis=basis, k0=k0, material=tm.material,
                      poltype="parity", modetype="up"))
    M_down = np.asarray(
        treams.efield([0, 0, 0.0], basis=basis, k0=k0, material=tm.material,
                      poltype="parity", modetype="down"))

    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    e_inc = pol_basis(th, ph, -1)     # incident, down-going (0=TE, 1=TM)
    e_t = pol_basis(th, ph, -1)       # transmitted receive basis (down)
    e_r = pol_basis(th, ph, +1)       # reflected receive basis (UP-going),
    #                                   = jones_blocks' basis for direction=-1
    S11 = np.zeros((2, 2), dtype=complex)
    S21 = np.zeros((2, 2), dtype=complex)
    resid = np.zeros(2)
    S_dn = np.asarray(smat[1, 1])
    S_up = np.asarray(smat[0, 1])
    for b in range(2):
        c, *_ = np.linalg.lstsq(M_down, e_inc[b].astype(complex), rcond=None)
        resid[b] = float(np.linalg.norm(M_down @ c - e_inc[b]))
        if resid_tol is not None and resid[b] > resid_tol:
            raise AssertionError(
                f"incident polarization {('TE', 'TM')[b]} not representable "
                f"in the treams down-mode span: lstsq residual = "
                f"{resid[b]:.3e} > {resid_tol:g} (theta={theta_deg}, "
                f"phi={phi_deg}, kpar={list(kpar)})")
        E_t = M_down @ (S_dn @ c)     # transmitted (down) total field at 0
        E_r = M_up @ (S_up @ c)       # reflected (up) field at 0
        for a in range(2):
            S21[a, b] = np.vdot(e_t[a], E_t)
            S11[a, b] = np.vdot(e_r[a], E_r)
    return dict(S11=S11, S21=S21, resid=resid, basis=basis_rows(basis),
                t_lat=t_lat)


def our_jones(fm, ifreq, theta_deg, phi_deg, cache_idx):
    """Our forward map, direction = -1 (down-going incidence).

    Calls sparams_oblique directly with the REQUESTED (theta, phi) and the
    cached C[ifreq, cache_idx] -- not fm.predict, whose angles come from the
    cache table: the (0, 45) validation entry reuses the k_par = 0 cache
    entry (index 0, tabulated as (0, 0)) but must be evaluated with phi=45
    polarization bases.  For the five entries whose table angles equal the
    requested ones this is identical to fm.predict.
    """
    from tmatrix.retrieval.sparams_oblique import sparams_oblique
    T = fm.data.T[ifreq]
    if not fm.have[ifreq, cache_idx]:
        raise KeyError(f"C not cached for ifreq={ifreq}, angle {cache_idx}")
    res = sparams_oblique(fm.k[ifreq], fm.A_cell, fm.modes, T,
                          fm.C[ifreq, cache_idx], np.deg2rad(theta_deg),
                          np.deg2rad(phi_deg), -1)
    return res["S11"], res["S21"]


def eight_entry_delta(S11_a, S21_a, S11_b, S21_b):
    return max(np.abs(S11_a - S11_b).max(), np.abs(S21_a - S21_b).max())


# ----------------------------------------------------------------------------
def sign_test(fm, tms, ifreq=24):
    """Empirical kpar-sign determination at (20,0), confirmation at (40,45)."""
    lam = fm.lam_um[ifreq]
    tm = tms[ifreq]
    k = fm.k[ifreq]
    print(f"=== kpar sign test at ifreq={ifreq} (lam={lam:.2f} um, "
          f"k={k:.6f} 1/um) ===", flush=True)
    evidence = []
    verdicts = {}
    for th, phd, cidx in ((20.0, 0.0, 13), (40.0, 45.0, 16)):
        kmag = k * np.sin(np.deg2rad(th))
        kpar0 = kmag * np.array([np.cos(np.deg2rad(phd)),
                                 np.sin(np.deg2rad(phd))])
        S11_o, S21_o = our_jones(fm, ifreq, th, phd, cidx)
        row = {}
        for sgn in (+1, -1):
            r = treams_jones(tm, sgn * kpar0, th, phd)
            d = eight_entry_delta(r["S11"], r["S21"], S11_o, S21_o)
            row[sgn] = (d, r["resid"], r["basis"])
            line = (f"  ({th:g},{phd:g}) kpar sign {sgn:+d}: "
                    f"max|dS| = {d:.3e}; lstsq resid TE={r['resid'][0]:.2e} "
                    f"TM={r['resid'][1]:.2e}; basis rows (kx,ky,pol) = "
                    f"{[(round(kx, 6), round(ky, 6), p) for kx, ky, p in r['basis']]}")
            print(line, flush=True)
            evidence.append(line)
            if sgn == +1 and abs(th - 20.0) < 1e-9:
                print(f"    our S11 = {np.round(S11_o, 6).tolist()}",
                      flush=True)
                print(f"    our S21 = {np.round(S21_o, 6).tolist()}",
                      flush=True)
                print(f"    trm S11 = {np.round(r['S11'], 6).tolist()}",
                      flush=True)
                print(f"    trm S21 = {np.round(r['S21'], 6).tolist()}",
                      flush=True)
        ok_signs = [s for s in (+1, -1) if row[s][0] <= 1e-3]
        verdicts[(th, phd)] = ok_signs
        print(f"  -> ({th:g},{phd:g}): sign(s) passing 1e-3: {ok_signs}",
              flush=True)
    sel = verdicts[(20.0, 0.0)]
    conf = verdicts[(40.0, 45.0)]
    if len(sel) == 1 and sel == conf:
        sign = sel[0]
        print(f"=== determined kpar sign: {sign:+d} (kpar_treams = "
              f"{sign:+d} * k sin th (cos ph, sin ph)); confirmed at "
              f"(40,45) ===", flush=True)
        os.makedirs(os.path.dirname(SIGN_FILE), exist_ok=True)
        np.savez(SIGN_FILE, sign=sign, ifreq=ifreq,
                 evidence="\n".join(evidence))
        print(f"saved {SIGN_FILE}", flush=True)
        return sign
    print(f"=== AMBIGUOUS/FAILED sign determination: (20,0) -> {sel}, "
          f"(40,45) -> {conf}; NOT saving a sign file ===", flush=True)
    return None


# ----------------------------------------------------------------------------
def sweep(fm, tms, sign, stride, val_angles=None, out_file=OUT_FILE):
    val_angles = VAL_ANGLES if val_angles is None else list(val_angles)
    check_angle_table(val_angles)
    freq_idx = [i for i in range(0, fm.nf, stride)]
    # availability check
    missing = []
    for i in freq_idx:
        for _, _, cidx in val_angles:
            if not fm.have[i, cidx]:
                missing.append((i, cidx))
    if missing:
        raise SystemExit(
            f"C cache incomplete for the sweep -- missing (ifreq, angle) "
            f"pairs: {sorted(set(missing))}; build with e.g.\n"
            f"  python precompute_C.py --angles "
            f"{','.join(str(a) for a in sorted(set(m[1] for m in missing)))} "
            f"--freqs "
            f"{','.join(str(i) for i in sorted(set(m[0] for m in missing)))}")
    print(f"C cache: all {len(freq_idx)} x {len(val_angles)} (freq, angle) "
          f"entries present (fm.have)", flush=True)

    na = len(val_angles)
    nf = len(freq_idx)
    S_ours = np.empty((nf, na, 2, 2, 2), dtype=complex)
    S_trm = np.empty_like(S_ours)
    resids = np.empty((nf, na, 2))
    basis_note = None
    t0 = time.time()
    for jf, i in enumerate(freq_idx):
        tm = tms[i]
        k = fm.k[i]
        tmlat_cache = {}
        for ja, (th, phd, cidx) in enumerate(val_angles):
            kmag = k * np.sin(np.deg2rad(th))
            kpar = sign * kmag * np.array([np.cos(np.deg2rad(phd)),
                                           np.sin(np.deg2rad(phd))])
            S11_o, S21_o = our_jones(fm, i, th, phd, cidx)
            r = treams_jones(tm, kpar, th, phd, tmlat_cache=tmlat_cache,
                             resid_tol=1e-10)
            S_ours[jf, ja, 0], S_ours[jf, ja, 1] = S11_o, S21_o
            S_trm[jf, ja, 0], S_trm[jf, ja, 1] = r["S11"], r["S21"]
            resids[jf, ja] = r["resid"]
            if basis_note is None:
                basis_note = f"first basis rows (kx,ky,pol): {r['basis']}"
        d_f = np.abs(S_ours[jf] - S_trm[jf]).max()
        print(f"  [{jf + 1:2d}/{nf}] ifreq={i:2d} lam={fm.lam_um[i]:6.2f} um"
              f"  max|dS| over {na} angles = {d_f:.3e}  "
              f"[{time.time() - t0:5.1f} s]", flush=True)

    dS = np.abs(S_ours - S_trm)
    worst_per_angle = dS.reshape(nf, na, -1).max(axis=(0, 2))
    worst_per_freq = dS.reshape(nf, -1).max(axis=1)
    worst = float(dS.max())
    gate = 1e-3
    print(f"\nper-angle worst complex |dS| over {nf} freqs "
          f"(gate {gate:g}):", flush=True)
    print(f"  {'theta':>6} {'phi':>6} {'cidx':>5} {'worst|dS|':>12}  "
          f"{'at lam_um':>9}  verdict")
    for ja, (th, phd, cidx) in enumerate(val_angles):
        v = "PASS" if worst_per_angle[ja] <= gate else "FAIL"
        jf_w = int(np.argmax(dS[:, ja].reshape(nf, -1).max(axis=1)))
        print(f"  {th:6.1f} {phd:6.1f} {cidx:5d} {worst_per_angle[ja]:12.3e}"
              f"  {fm.lam_um[freq_idx[jf_w]]:9.2f}  {v}", flush=True)
    jf_worst = int(np.argmax(worst_per_freq))
    print(f"  overall worst |dS| = {worst:.3e} -> "
          f"{'PASS' if worst <= gate else 'FAIL'} (gate {gate:g}); worst "
          f"frequency: ifreq={freq_idx[jf_worst]} "
          f"lam={fm.lam_um[freq_idx[jf_worst]]:.2f} um", flush=True)
    print(f"  worst lstsq incident residual = {resids.max():.3e} "
          f"(assert tol 1e-10)", flush=True)
    print(f"\nper-frequency worst complex |dS| (all {na} angles):", flush=True)
    for jf, i in enumerate(freq_idx):
        per_a = " ".join(f"{dS[jf, ja].max():9.2e}" for ja in range(na))
        print(f"  ifreq={i:2d} lam={fm.lam_um[i]:6.2f} um  "
              f"max={worst_per_freq[jf]:9.2e}  | per angle: {per_a}",
              flush=True)

    # ---- theta=0 extra checks vs stored references (only if (0,0) selected)
    idx = np.array(freq_idx)
    ja0 = next((j for j, (th, phd, _) in enumerate(val_angles)
                if abs(th) < 1e-12 and abs(phd) < 1e-12), None)
    if ja0 is None:
        theta0_note = ("theta=0 (0,0) angle not in this selection -> the "
                       "stored-reference cross-checks were skipped.")
        print(f"\n{theta0_note}", flush=True)
    else:
        stored_t = np.load(os.path.join(AGG_RESULTS, "treams_reference.npz"))
        stored_p = np.load(os.path.join(AGG_RESULTS, "periodic_results.npz"))
        # new treams at (0,0), our-basis blocks vs stored x-pol treams
        # reference: TM column: S21[1,1] = +S21_stored,
        # S11[1,1] = -S11_stored (exact); TE diagonal equals the same numbers
        # through C4 (noise-level residual).
        d_trm_tm = np.max(np.abs(np.stack([
            S_trm[:, ja0, 1, 1, 1] - stored_t["S21"][idx],
            S_trm[:, ja0, 0, 1, 1] + stored_t["S11"][idx]])))
        d_trm_te = np.max(np.abs(np.stack([
            S_trm[:, ja0, 1, 0, 0] - stored_t["S21"][idx],
            S_trm[:, ja0, 0, 0, 0] - stored_t["S11"][idx]])))
        # our (0,0) direction=-1 vs stored periodic direction=+1 (reciprocity
        # for S21; z-mirror symmetry of the flat cell for S11):
        d_our_tm = np.max(np.abs(np.stack([
            S_ours[:, ja0, 1, 1, 1] - stored_p["S21"][idx],
            S_ours[:, ja0, 0, 1, 1] + stored_p["S11"][idx]])))
        theta0_note = (f"theta=0 checks: treams vs stored ref "
                       f"{d_trm_tm:.2e}, TE-diagonal (C4) {d_trm_te:.2e}, "
                       f"ours vs stored periodic {d_our_tm:.2e}.")
        print(f"\ntheta=0 cross-checks over the subset:")
        print(f"  new treams vs stored treams_reference.npz "
              f"(S21[TM,TM]-S21_ref, S11[TM,TM]+S11_ref): max = "
              f"{d_trm_tm:.3e} (expected ~exact)", flush=True)
        print(f"  new treams TE diagonal vs same stored x-pol numbers (C4): "
              f"max = {d_trm_te:.3e} (reference-T C4 noise level)", flush=True)
        print(f"  our (0,0) dir=-1 vs stored periodic dir=+1 "
              f"(S21 reciprocity, -S11 z-symmetry): max = {d_our_tm:.3e}",
              flush=True)

    notes = (
        "Oblique treams cross-check (doc par. 6 step 1). Angles "
        f"{[(a, b) for a, b, _ in val_angles]} (cache angle indices "
        f"{[c for _, _, c in val_angles]} of the normative 17-row "
        f"precompute_C.ANGLES_DEG table), freq indices {freq_idx} "
        f"(stride {stride}, {nf} of {fm.nf} frequencies, lam "
        f"{fm.lam_um[idx].min():.2f}-{fm.lam_um[idx].max():.2f} um). "
        f"Worst complex |dS| = {worst:.3e} vs gate {gate:g} -> "
        f"{'PASS' if worst <= gate else 'FAIL'}; per-angle worst "
        f"{[float(f'{w:.3e}') for w in worst_per_angle]}; worst at ifreq="
        f"{freq_idx[jf_worst]} (lam={fm.lam_um[freq_idx[jf_worst]]:.2f} um). "
        f"{theta0_note} Illumination: down-going from +z, phase ref "
        "z=0; our model = sparams_oblique(direction=-1) with cached "
        "C(k, k_par); treams = latticeinteraction.solve + "
        "PlaneWaveBasisByComp.diffr_orders + SMatrices.from_array, incident "
        "polarization built by lstsq of pol_basis(th, ph, -1)[b] on the "
        "down-mode efield columns, receive projections on pol_basis(-1) "
        "(transmitted) / pol_basis(+1) (reflected). kpar_treams = "
        f"{sign:+d} * k sin th (cos ph, sin ph) (empirical, "
        "see treams_kpar_sign.npz). S[angle_block, a, b]: a = receive "
        "(0=TE, 1=TM), b = incident. " + (basis_note or ""))
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    np.savez(out_file,
             S_ours=S_ours, S_treams=S_trm, absdiff=dS,
             worst_per_angle=worst_per_angle,
             theta_deg=np.array([a[0] for a in val_angles]),
             phi_deg=np.array([a[1] for a in val_angles]),
             cache_angle_idx=np.array([a[2] for a in val_angles]),
             freq_idx=idx, lam_um=fm.lam_um[idx], kpar_sign=sign,
             lstsq_resid=resids, notes=notes)
    print(f"\nsaved {out_file}", flush=True)
    return worst <= gate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4,
                    help="frequency stride for the sweep (4 -> 13 freqs; "
                         "1 -> all 49 once the campaign cache exists)")
    ap.add_argument("--sign-test", action="store_true",
                    help="run only the empirical kpar-sign determination")
    ap.add_argument("--sign", choices=["auto", "pos", "neg"], default="auto",
                    help="kpar sign for the sweep (auto = read/derive)")
    ap.add_argument("--angles", default=None,
                    help="comma list / ranges of CACHE angle indices "
                         f"0..{pc.N_ANGLES - 1} into precompute_C.ANGLES_DEG "
                         "(theta/phi are derived from that table); default = "
                         "the module's frozen VAL_ANGLES validation set. "
                         "Campaign obliques: 5,10-12")
    ap.add_argument("--out", default=OUT_FILE,
                    help="output npz (default results/treams_oblique_check"
                         ".npz -- use a different path so an --angles run "
                         "does not overwrite the recorded default sweep)")
    args = ap.parse_args(argv)

    if args.angles is None:
        val_angles = VAL_ANGLES
    else:
        idx = pc.parse_int_list(args.angles, pc.N_ANGLES, "angle")
        if not idx:
            raise SystemExit("--angles selected no angles")
        val_angles = angles_from_cache_indices(idx)
    check_angle_table(val_angles)
    if (args.angles is not None
            and os.path.abspath(args.out) == os.path.abspath(OUT_FILE)):
        raise SystemExit(
            "refusing to overwrite the recorded default sweep "
            f"{OUT_FILE} with an --angles selection; pass --out")

    print(f"angle selection ({len(val_angles)}): "
          f"{[(th, ph, ci) for th, ph, ci in val_angles]} "
          f"(theta, phi, cache index)", flush=True)
    print(f"loading {TMAT_FILE} via treams.io (lunit=um) ...", flush=True)
    tms = treams.io.load_hdf5(TMAT_FILE, lunit="um")
    assert len(tms) == 49, f"expected 49 frequencies, got {len(tms)}"
    assert tms[0].poltype == "parity"
    fm = ForwardModel()
    print(f"C cache: available freqs {fm.available_freqs}", flush=True)

    if args.sign_test:
        sign = sign_test(fm, tms)
        sys.exit(0 if sign is not None else 1)

    if args.sign == "pos":
        sign = +1
    elif args.sign == "neg":
        sign = -1
    else:
        if os.path.exists(SIGN_FILE):
            sign = int(np.load(SIGN_FILE)["sign"])
            print(f"kpar sign {sign:+d} from {SIGN_FILE}", flush=True)
        else:
            sign = sign_test(fm, tms)
            if sign is None:
                sys.exit(1)
    ok = sweep(fm, tms, sign, args.stride, val_angles=val_angles,
               out_file=args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

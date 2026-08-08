"""Gates for retrieval/fastfull: lattice, transforms, symmetry.

Run:  python test_fastfull_core.py          (from retrieval/, env cst_inference)

Every gate here is a re-derivation against something already validated, not a
self-consistency check:

  (a) lattice reciprocity identity A B^T = 2 pi I, and fractional round trip
  (b) Lattice2D.shells reproduces translate.square_lattice_shells exactly on
      a square lattice, and enumerate_orders reproduces a brute-force scan
  (c) the par. 6 seed design (26.0 x 33.8 um, k_B = 0.090 b1 - 0.460 b2,
      lambda = 20 um) really gives 8 orders / 32 channels
  (d) farfield_basis contracts to vswf.far_field_amplitude
  (e) plane_wave_coeffs_batch equals vswf.plane_wave_coeffs
  (f) THE BIG ONE: in a cell where only one order propagates, the
      flux-normalized W T_eff A + S_empty reproduces sparams_oblique's S11
      and S21 blocks, in BOTH illumination directions.  This locks the new
      normalization to the campaign-validated forward map.
  (g) the CST gauge sign table reproduces the sparams_oblique docstring's
      entry-by-entry theta -> 0 mapping (the S11 TM-row sign that took the
      campaign's chi^2_red from 658.9 to 2.49)
  (h) empty_modal_S is gauge independent and port symmetric
  (i) coupling.lattice_sum_C on a square lattice equals the validated
      bloch_lattice.lattice_sum_C_bloch
  (j) D4h: sigma_h numeric == closed form; ranks 58 / 40 match the character
      prediction; sector table matches proposal par. 3.1; parity block
      structure; the reference wheel T is D4h consistent at its known noise
      floor
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "aggregation"))
sys.path.insert(0, HERE)

import numpy as np

import fastfull  # noqa: F401  (sets up sys.path)
from fastfull import lattice as lt
from fastfull import transforms as xf
from fastfull import symmetry as sym
from fastfull import coupling as cp

from vswf import ModeBasis, far_field_amplitude, plane_wave_coeffs
from translate import square_lattice_shells, make_quad
from sparams_oblique import sparams_oblique, pol_basis
from bloch_lattice import lattice_sum_C_bloch
from tmat_io import TMatrixData

RESULTS = []


def record(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s\n         %s" % ("PASS" if ok else "FAIL", name, detail),
          flush=True)


# --------------------------------------------------------------- (a) lattice

def test_lattice_identities():
    for lat in (lt.Lattice2D.square(2.0),
                lt.Lattice2D.rectangular(26.0, 33.8),
                lt.Lattice2D.rectangular(11.0, 17.0, alpha_deg=31.0),
                lt.Lattice2D.oblique(13.0, 19.0, 71.0, 17.0)):
        d = np.abs(lat.A @ lat.B.T - 2 * np.pi * np.eye(2)).max()
        f = np.array([0.137, -0.421])
        rt = np.abs(lat.fractional(lat.bloch(*f)) - f).max()
        area_direct = abs(lat.a1[0] * lat.a2[1] - lat.a1[1] * lat.a2[0])
        da = abs(lat.area - area_direct)
        record("(a) reciprocity + fractional round trip: %s" % lat.name,
               d < 1e-12 and rt < 1e-12 and da < 1e-12,
               "max|A B^T - 2pi I| = %.2e, |f round trip| = %.2e, "
               "area resid %.2e" % (d, rt, da))


def test_rotation_invariance():
    """A rotated lattice must give the same physics: rotating both the cell
    and the Bloch vector rotates every order's azimuth by alpha and leaves
    |q|, kz and the retained count untouched."""
    k = 2 * np.pi / 8.0
    f = (0.09, -0.46)
    base = lt.Lattice2D.rectangular(26.0, 33.8)
    rot = lt.Lattice2D.rectangular(26.0, 33.8, alpha_deg=37.0)
    o0 = lt.enumerate_orders(base, k, f_bloch=f, kz_min_frac=0.0)
    o1 = lt.enumerate_orders(rot, k, f_bloch=f, kz_min_frac=0.0)
    same_g = np.array_equal(o0.g, o1.g)
    dq = np.abs(np.linalg.norm(o0.q, axis=1)
                - np.linalg.norm(o1.q, axis=1)).max()
    dphi = np.abs(((np.rad2deg(o1.phi) - np.rad2deg(o0.phi) - 37.0 + 180.0)
                   % 360.0) - 180.0).max()
    record("(a) rotating the cell rotates the orders and nothing else",
           same_g and dq < 1e-12 and dphi < 1e-9,
           "identical g list = %s, max|d|q|| = %.2e, max azimuth error "
           "= %.2e deg" % (same_g, dq, dphi))


# ------------------------------------------------------- (b) enumeration

def test_shells_vs_square():
    lat = lt.Lattice2D.square(2.0)
    for r_max in (7.5, 21.0):
        r_ref, a_ref = square_lattice_shells(2.0, r_max)
        r_new, a_new = lat.shells(r_max)
        ok = len(r_ref) == len(r_new)
        dr = np.abs(r_ref - r_new).max() if ok else np.inf
        dn = (max(abs(len(x) - len(y)) for x, y in zip(a_ref, a_new))
              if ok else -1)
        dang = 0.0
        if ok and dn == 0:
            for x, y in zip(a_ref, a_new):
                dang = max(dang, float(np.abs(np.sort(x) - np.sort(y)).max()))
        record("(b) Lattice2D.shells == translate.square_lattice_shells "
               "(r_max = %g)" % r_max,
               ok and dr < 1e-12 and dn == 0 and dang < 1e-12,
               "%d vs %d shells, max|dr| = %.2e, site-count diff = %s, "
               "max|dphi| = %.2e" % (len(r_ref), len(r_new), dr, dn, dang))


def test_enumerate_vs_bruteforce():
    lat = lt.Lattice2D.oblique(13.0, 19.0, 71.0, 17.0)
    k = 2 * np.pi / 9.0
    f = (0.11, -0.37)
    kB = lat.bloch(*f)
    o = lt.enumerate_orders(lat, k, f_bloch=f, kz_min_frac=0.0,
                            q_scan=1.6)
    # brute force over a deliberately oversized integer box
    N = 40
    gg = np.arange(-N, N + 1)
    G1, G2 = np.meshgrid(gg, gg, indexing="ij")
    g = np.stack([G1.ravel(), G2.ravel()], axis=1)
    q = kB[None, :] + g @ lat.B
    qa = np.linalg.norm(q, axis=1)
    ref = {(int(a), int(b)) for (a, b), r in zip(g, qa) if r < k}
    got = {(int(a), int(b)) for a, b in o.g[o.prop]}
    kz_ok = np.allclose(o.kz[o.prop] ** 2 + o.qabs[o.prop] ** 2, k ** 2)
    th_ok = np.allclose(np.sin(o.theta[o.prop]) * k, o.qabs[o.prop])
    record("(b) propagating set == brute force over a 81x81 box",
           ref == got and kz_ok and th_ok,
           "%d propagating orders, sets equal = %s, kz^2 + q^2 = k^2 %s, "
           "sin(theta) k = |q| %s" % (len(got), ref == got, kz_ok, th_ok))


def test_seed_design():
    lat = lt.Lattice2D.rectangular(26.0, 33.8)
    k = 2 * np.pi / 20.0
    o = lt.enumerate_orders(lat, k, f_bloch=(0.090, -0.460), kz_min_frac=0.0)
    ch = lt.ChannelSet(o)
    record("(c) proposal par. 6 seed design reproduces 8 orders / 32 channels",
           o.n_prop == 8 and ch.n == 32 and abs(lat.area - 878.8) < 1e-9,
           "n_prop = %d, n_channels = %d, area = %.4f um^2 (proposal: 8, 32, "
           "878.8)" % (o.n_prop, ch.n, lat.area))


# ------------------------------------------------------- (d, e) primitives

def test_farfield_basis():
    modes = ModeBasis.standard(3)
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(7, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    k = 0.7854
    FF = xf.farfield_basis(k, modes, pts)
    worst = 0.0
    for _ in range(3):
        f = rng.normal(size=modes.n) + 1j * rng.normal(size=modes.n)
        worst = max(worst, float(np.abs(
            np.tensordot(f, FF, axes=(0, 0))
            - far_field_amplitude(k, f, modes, pts)).max()))
    record("(d) farfield_basis contracts to vswf.far_field_amplitude",
           worst < 1e-13, "max|dF| = %.3e over 3 random coefficient vectors"
           % worst)


def test_plane_wave_batch():
    modes = ModeBasis.standard(3)
    rng = np.random.default_rng(2)
    kh = rng.normal(size=(9, 3))
    kh /= np.linalg.norm(kh, axis=1, keepdims=True)
    eh = rng.normal(size=(9, 3))
    eh -= (eh * kh).sum(axis=1, keepdims=True) * kh
    eh /= np.linalg.norm(eh, axis=1, keepdims=True)
    got = xf.plane_wave_coeffs_batch(kh, eh, modes)
    ref = np.stack([plane_wave_coeffs(kh[i], eh[i], modes)
                    for i in range(len(kh))], axis=1)
    d = float(np.abs(got - ref).max())
    record("(e) plane_wave_coeffs_batch == vswf.plane_wave_coeffs",
           d < 1e-13, "max|da| = %.3e over 9 directions" % d)


# --------------------------------------------- (f) the normalization anchor

def _forward_model():
    from forward import ForwardModel
    return ForwardModel()


def test_specular_equivalence():
    """W T_eff A + S_empty == sparams_oblique in a single-order cell."""
    fm = _forward_model()
    modes = fm.modes
    lat = lt.Lattice2D.square(fm.pitch)
    for ifreq, (th_deg, ph_deg) in ((48, (60.0, 22.5)), (32, (30.0, 0.0)),
                                    (48, (0.0, 0.0))):
        k = fm.k[ifreq]
        T0 = fm.data.T[ifreq]
        ia = fm.angle_index(th_deg, ph_deg)
        if not fm.have[ifreq, ia]:
            record("(f) specular equivalence ifreq=%d (%g, %g)"
                   % (ifreq, th_deg, ph_deg), False,
                   "C not cached -- cannot run this gate")
            continue
        C = fm.C[ifreq, ia]
        th, ph = np.deg2rad(th_deg), np.deg2rad(ph_deg)
        kB = k * np.sin(th) * np.array([np.cos(ph), np.sin(ph)])
        o = lt.enumerate_orders(lat, k, k_bloch=kB, kz_min_frac=0.0,
                                wood_margin=0.0)
        if o.n_prop != 1:
            record("(f) specular equivalence ifreq=%d" % ifreq, False,
                   "cell is not single-order (n_prop = %d)" % o.n_prop)
            continue
        ch = lt.ChannelSet(o)
        kin = xf.check_channel_kinematics(ch)
        A = xf.build_A(k, ch, modes, gauge=xf.GAUGE_PHYSICAL)
        W = xf.build_W(k, ch, modes, gauge=xf.GAUGE_PHYSICAL)
        S = xf.modal_S(W, xf.t_effective(T0, C), A, xf.empty_modal_S(ch))
        lab = ch.labels()

        def ix(side, pol):
            s = "Zmax" if side > 0 else "Zmin"
            p = "TE" if pol == 0 else "TM"
            return [i for i, L in enumerate(lab) if L[0] == s and L[3] == p][0]

        worst = 0.0
        for direction in (+1, -1):
            ref = sparams_oblique(k, lat.area, modes, T0, C, th, ph,
                                  direction)
            side_inc = -direction
            S11 = np.array([[S[ix(side_inc, a), ix(side_inc, b)]
                             for b in (0, 1)] for a in (0, 1)])
            S21 = np.array([[S[ix(-side_inc, a), ix(side_inc, b)]
                             for b in (0, 1)] for a in (0, 1)])
            worst = max(worst, float(np.abs(S11 - ref["S11"]).max()),
                        float(np.abs(S21 - ref["S21"]).max()))
        record("(f) flux-normalized W T_eff A == sparams_oblique "
               "(ifreq=%d, theta=%g, phi=%g, both directions)"
               % (ifreq, th_deg, ph_deg),
               worst < 1e-12 and kin < 1e-12,
               "max|dS| = %.3e (gate 1e-12); channel kinematics resid %.2e"
               % (worst, kin))


# ------------------------------------------------------------ (g) the gauge

def test_cst_gauge_table():
    """Reproduce the sparams_oblique docstring's theta -> 0 mapping table.

    For illumination with direction = +1 (entering at Zmin), the CST gauge
    must leave every S21 entry alone and flip the TM RECEIVE row of S11.
    """
    k = 2 * np.pi / 12.0
    lat = lt.Lattice2D.square(2.0)
    th, ph = np.deg2rad(23.0), np.deg2rad(41.0)
    kB = k * np.sin(th) * np.array([np.cos(ph), np.sin(ph)])
    o = lt.enumerate_orders(lat, k, k_bloch=kB, kz_min_frac=0.0,
                            wood_margin=0.0)
    ch = lt.ChannelSet(o)
    s_in_p, s_out_p = xf.gauge_signs(ch, xf.GAUGE_PHYSICAL)
    s_in_c, s_out_c = xf.gauge_signs(ch, xf.GAUGE_CST)
    lab = ch.labels()

    def ix(side, pol):
        s = "Zmax" if side > 0 else "Zmin"
        p = "TE" if pol == 0 else "TM"
        return [i for i, L in enumerate(lab) if L[0] == s and L[3] == p][0]

    direction = +1                      # docstring's case
    side_inc = -direction               # Zmin
    ratio11 = np.array([[s_in_c[ix(side_inc, b)] * s_out_c[ix(side_inc, a)]
                         for b in (0, 1)] for a in (0, 1)])
    ratio21 = np.array([[s_in_c[ix(side_inc, b)] * s_out_c[ix(-side_inc, a)]
                         for b in (0, 1)] for a in (0, 1)])
    want11 = np.array([[+1.0, +1.0], [-1.0, -1.0]])   # TM receive row flips
    want21 = np.ones((2, 2))
    ok = (np.array_equal(ratio11, want11) and np.array_equal(ratio21, want21)
          and np.all(s_in_p == 1) and np.all(s_out_p == 1))
    record("(g) CST gauge reproduces the sparams_oblique theta->0 table",
           ok, "S11 sign matrix %s (want [[+,+],[-,-]]), S21 %s (want all +); "
               "physical gauge is all +1" % (ratio11.tolist(),
                                             ratio21.tolist()))

    # a mode gauge is a similarity D S D and cannot touch a co-polar diagonal
    rng = np.random.default_rng(5)
    mg = rng.choice([-1.0, 1.0], size=ch.n)
    s_in_m, s_out_m = xf.gauge_signs(ch, xf.GAUGE_CST, mode_gauge=mg)
    diag_same = all(s_in_m[i] * s_out_m[i] == s_in_c[i] * s_out_c[i]
                    for i in range(ch.n))
    record("(g) a per-mode gauge leaves every co-polar diagonal invariant",
           diag_same, "checked all %d channels (deembed.py's "
           "label-hypothesis argument)" % ch.n)


def test_empty_matrix():
    k = 2 * np.pi / 20.0
    lat = lt.Lattice2D.rectangular(26.0, 33.8)
    o = lt.enumerate_orders(lat, k, f_bloch=(0.09, -0.46), kz_min_frac=0.0)
    ch = lt.ChannelSet(o)
    S = xf.empty_modal_S(ch)
    unitary = float(np.abs(S.conj().T @ S - np.eye(ch.n)).max())
    lab = ch.labels()
    ok_map = True
    for c in range(ch.n):
        r = np.flatnonzero(np.abs(S[:, c]) > 0)
        if len(r) != 1:
            ok_map = False
            break
        a, b = lab[int(r[0])], lab[c]
        ok_map &= (a[0] != b[0] and a[1:] == b[1:])
    ph = xf.port_plane_phase(ch, 0.0)
    record("(h) empty_modal_S is unitary, port-swapping and order/pol "
           "preserving", unitary < 1e-15 and ok_map
           and np.abs(ph - 1).max() == 0.0,
           "max|S^H S - I| = %.2e, mapping ok = %s, L=0 phase is exactly 1"
           % (unitary, ok_map))


# ---------------------------------------------------------- (i) coupling

def test_coupling_matches_bloch():
    """The generalized Bravais lattice sum reduces to the validated square
    implementation, entry for entry."""
    modes = ModeBasis.standard(3)
    quad = make_quad(12, 24)
    k = 2 * np.pi / 12.0
    pitch, r0 = 2.0, 0.8
    lat = lt.Lattice2D.square(pitch)
    th, ph = np.deg2rad(30.0), np.deg2rad(22.5)
    k_par = k * np.sin(th) * np.array([np.cos(ph), np.sin(ph)])
    kRc = (6.0, 8.0, 11.0)          # small, so the gate runs in seconds
    C_ref = lattice_sum_C_bloch(k, pitch, modes, r0, quad, k_par, kRc=kRc)
    C_new = cp.lattice_sum_C(lat, k, modes, r0, quad, k_par, kRc=kRc)
    d = float(np.abs(C_new - C_ref).max())
    rel = d / float(np.abs(C_ref).max())
    record("(i) coupling.lattice_sum_C == bloch_lattice.lattice_sum_C_bloch "
           "on a square lattice", rel < 1e-12,
           "max|dC| = %.3e, relative %.3e (|C|max = %.3e)"
           % (d, rel, np.abs(C_ref).max()))


def test_coupling_refuses_big_cell():
    modes = ModeBasis.standard(3)
    quad = make_quad(8, 16)
    k = 2 * np.pi / 20.0
    lat = lt.Lattice2D.rectangular(26.0, 33.8)
    st = cp.shell_statistics(lat, k)
    C, info = cp.converged_C(lat, k, modes, 0.8, quad, np.array([0.01, 0.02]))
    record("(i) converged_C refuses the large diffractive cell instead of "
           "guessing", C is None and not info["converged"],
           "sites inside the smallest taper = %d (Rc = %.2f um, pitch "
           "26/33.8 um); reason: %s"
           % (st["sites_in_min_taper"], st["Rc_min"],
              info["reasons"][0] if info["reasons"] else "-"))


# ------------------------------------------------------------ (j) symmetry

def test_d4h_basis():
    modes = ModeBasis.standard(3)
    B, meta = sym.build_d4h_reciprocity_basis(modes)
    record("(j) sigma_h numeric derivation == closed form (S1)",
           meta["sigma_h_numeric_err"] < 1e-10,
           "max|D_num - D_exact| = %.2e; lstsq E residual %.2e; independent "
           "pseudovector H residual %.2e"
           % (meta["sigma_h_numeric_err"], meta["sigma_h_lstsq_resid_E"],
              meta["sigma_h_resid_H"]))

    record("(j) D4h group is closed, unitary, and commutes with reciprocity",
           max(meta["closure_resid"], meta["unitarity_resid"],
               meta["comm_resid"], meta["sigma_hv_commute_resid"],
               meta["sigma_h_sq_resid"]) < 1e-12,
           "closure %.1e, unitarity %.1e, [P_grp, P_rec] %.1e, "
           "[sigma_h, sigma_v] %.1e, sigma_h^2 - I %.1e"
           % (meta["closure_resid"], meta["unitarity_resid"],
              meta["comm_resid"], meta["sigma_hv_commute_resid"],
              meta["sigma_h_sq_resid"]))

    record("(j) commutant ranks 58 / 40 match the character prediction",
           meta["rank_d4h"] == meta["rank_d4h_predicted"] == 58
           and meta["rank_full"] == meta["rank_full_predicted"] == 40
           and meta["character_integrality_resid"] < 1e-9,
           "dim commutant(D4h) = %d (predicted %d), with reciprocity = %d "
           "(predicted %d); character integrality %.1e"
           % (meta["rank_d4h"], meta["rank_d4h_predicted"],
              meta["rank_full"], meta["rank_full_predicted"],
              meta["character_integrality_resid"]))

    want = dict(even=dict(A1=1, A2=2, B1=2, B2=2, E=4),
                odd=dict(A1=2, A2=1, B1=2, B2=2, E=4))
    got = meta["sector_multiplicities"]
    record("(j) sector table matches proposal par. 3.1",
           got == want and meta["n_even"] == meta["n_odd"] == 15,
           "even %s / odd %s; 15 even + 15 odd modes"
           % (got["even"], got["odd"]))

    record("(j) basis is orthonormal, projector invariant, parity block "
           "diagonal",
           max(meta["orthonormality_resid"], meta["basis_invariance_resid"],
               meta["parity_leak"], meta["eigenvalue_purity"],
               meta["vec_apply_resid"],
               meta["basis_projection_resid"]) < 1e-12,
           "orthonormality %.1e, invariance %.1e, parity leak %.1e, "
           "eigenvalue purity %.1e"
           % (meta["orthonormality_resid"], meta["basis_invariance_resid"],
              meta["parity_leak"], meta["eigenvalue_purity"]))
    return B, meta


def test_d4h_vs_c4v(B):
    """The D4h space must be a strict subspace of the validated C4v space."""
    import parametrize as pz
    modes = ModeBasis.standard(3)
    Bc, mc = pz.build_c4v_reciprocity_basis(modes, verify_numeric=False)
    Pc = mc["P_full"]
    leak = 0.0
    for Bk in B:
        v = Bk.reshape(-1)
        leak = max(leak, float(np.abs(Pc @ v - v).max()))
    record("(j) every D4h basis element lies inside the validated 68-dim "
           "C4v x reciprocity space", leak < 1e-12 and mc["rank_full"] == 68,
           "max|P_c4v(B) - B| = %.2e; C4v rank %d, D4h rank %d"
           % (leak, mc["rank_full"], len(B)))


def test_reference_is_d4h(B):
    path = os.path.join(HERE, "..", "test", "single",
                        "saw_gold_wl15p0025um.tmat.h5")
    if not os.path.exists(path):
        record("(j) reference wheel T is D4h consistent", False,
               "reference file not found: %s" % path)
        return
    data = TMatrixData(path)
    res = sym.symmetry_residual(data.T, B)
    # the campaign established the file's own C4v-violation noise at ~3e-3
    ok = float(res.max()) < 2e-2
    record("(j) reference wheel tmat.h5 is D4h consistent at its known noise "
           "floor", ok,
           "relative ||T - P_D4h(T)||_F: median %.3e, worst %.3e over %d "
           "frequencies (the file's own C4v-violation noise is ~3e-3; gate "
           "2e-2)" % (float(np.median(res)), float(res.max()), len(res)))


def test_passive_ensemble(B):
    rng = np.random.default_rng(11)
    T, c = sym.random_passive_d4h(B, rng, n_draw=6, target_fro=0.2)
    resid = float(np.abs(sym.symmetry_residual(T, B)).max())
    pas = sym.passivity_max_sv(T)
    record("(j) passive D4h ensemble draws stay in the space and are passive",
           resid < 1e-12 and pas <= 1.0 + 1e-8 and T.shape == (6, 30, 30),
           "worst symmetry residual %.2e, max SV(I + 2T) = %.9f (gate "
           "<= 1 + 1e-8), ||T||_F = %.3f" % (resid, pas,
                                             np.linalg.norm(T[0])))


# ------------------------------------------------------------------- driver

def main():
    print("=== fastfull core gates ===", flush=True)
    test_lattice_identities()
    test_rotation_invariance()
    test_shells_vs_square()
    test_enumerate_vs_bruteforce()
    test_seed_design()
    test_farfield_basis()
    test_plane_wave_batch()
    test_specular_equivalence()
    test_cst_gauge_table()
    test_empty_matrix()
    test_coupling_matches_bloch()
    test_coupling_refuses_big_cell()
    B, _ = test_d4h_basis()
    test_d4h_vs_c4v(B)
    test_reference_is_d4h(B)
    test_passive_ensemble(B)

    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n=== SUMMARY (test_fastfull_core) ===")
    for name, ok, _ in RESULTS:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("ALL %d TESTS PASSED" % len(RESULTS) if nfail == 0
          else "%d of %d TEST(S) FAILED" % (nfail, len(RESULTS)))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())

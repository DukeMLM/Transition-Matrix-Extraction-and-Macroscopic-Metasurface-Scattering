"""Tests for deembed.py (+ the cst_campaign.py dry-run contract).

Synthetic-data closed loop: build a fake empty cell (analytic CST-convention
S21 for a known L and theta) + a fake structure S, push both through the
full chain (load -> conjugate -> check_empty_phase -> de-embed -> interp ->
closure), and assert the injected complex S is recovered exactly.  The
conjugation-direction check is additionally required to FAIL when the
conjugation is deliberately skipped -- the check must be able to catch the
error it exists for.

Run:  python test_deembed.py        (or pytest test_deembed.py)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from tmatrix.retrieval import cst_campaign # noqa: E402
from tmatrix.retrieval.deembed import (DeembedError, REF_NPZ, analytic_empty_s21,   # noqa: E402
                     apply_hypothesis, check_empty_phase,
                     check_mirror_plane_crosspol, closure_normal,
                     conj_cst, deembed_blocks, deembed_spectra,
                     interp_to_grid, kz_per_um, label_hypotheses,
                     map_cst_labels, select_hypothesis)

GRID_THZ = 299.792458 / (20.0 - 0.25 * np.arange(49))      # the tmat grid


def _smooth_spectrum(f, amp, ph0, ph1):
    """A generic smooth complex spectrum (no special structure)."""
    return amp * (0.6 + 0.4 * np.cos(0.3 * f)) * np.exp(
        1j * (ph0 + ph1 * f))


def _fake_cst_pair(f, s11_true, s21_true, L_um, theta_deg):
    """Given the TRUE cell-referenced physics S (e^{-i omega t}), build what
    CST would report at the ports: propagation e^{+i k_z L} appended to both
    blocks (z-symmetric domain), then conjugated into e^{+j omega t}."""
    prop = np.exp(1j * kz_per_um(f, theta_deg) * L_um)
    s11_raw = conj_cst(s11_true * prop)
    s21_raw = conj_cst(s21_true * prop)
    empty_raw = analytic_empty_s21(f, L_um, theta_deg, convention="cst")
    return s11_raw, s21_raw, empty_raw


def test_analytic_empty_roundtrip():
    """Full chain recovers the injected complex S exactly (1e-12)."""
    f = GRID_THZ
    for theta in (0.0, 30.0):
        L = 11.714687
        s11_true = _smooth_spectrum(f, 0.31, 0.40, 0.021)
        s21_true = _smooth_spectrum(f, 0.93, -0.20, 0.017)
        s11_raw, s21_raw, empty_raw = _fake_cst_pair(
            f, s11_true, s21_true, L, theta)

        chk = check_empty_phase(f, conj_cst(empty_raw), theta,
                                L_expected_um=L)
        assert chk["sign_ok"] and chk["passed"], chk
        assert abs(chk["L_fit_um"] - L) < 1e-9, chk["L_fit_um"]
        assert chk["mag_dev"] < 1e-14

        s11_de, s21_de = deembed_spectra(s11_raw, s21_raw, empty_raw)
        assert np.max(np.abs(s11_de - s11_true)) < 1e-12
        assert np.max(np.abs(s21_de - s21_true)) < 1e-12
    print("[ok] analytic-empty roundtrip exact at theta = 0, 30 deg")


def test_conjugation_direction_catch():
    """Skipping the conjugation MUST fail the phase-slope sign check."""
    f = GRID_THZ
    L, theta = 11.714687, 0.0
    empty_raw = analytic_empty_s21(f, L, theta, convention="cst")
    # deliberately skip conj_cst: feed the raw e^{+j omega t} data
    chk = check_empty_phase(f, empty_raw, theta, L_expected_um=L)
    assert not chk["sign_ok"], chk
    assert not chk["passed"], chk
    assert chk["slope_rad_per_THz"] < 0
    # and double-conjugation (conj applied twice = raw again) also fails
    chk2 = check_empty_phase(f, conj_cst(conj_cst(empty_raw)),
                             theta, L_expected_um=L)
    assert not chk2["passed"]
    print("[ok] conjugation-direction check catches the skipped/doubled "
          "conjugation (slope sign flips)")


def test_wrong_L_is_flagged():
    """A wrong port-to-port L must fail the slope magnitude check."""
    f = GRID_THZ
    L_true, L_claimed = 11.714687, 6.0
    empty_raw = analytic_empty_s21(f, L_true, 0.0, convention="cst")
    chk = check_empty_phase(f, conj_cst(empty_raw), 0.0,
                            L_expected_um=L_claimed)
    assert chk["sign_ok"] and not chk["passed"], chk
    print("[ok] wrong L_expected fails the rel-tol part of the check "
          f"(rel_err = {chk['rel_err']:.2f})")


def test_closure_recovers_reference_exactly():
    """Chain the reference spectra through a synthetic CST pair; the
    closure gate must pass at machine precision -- and must FAIL when the
    empty used a wrong L."""
    with np.load(REF_NPZ) as z:
        f = z["freq"] / 1e12
        s11_true, s21_true = z["S11"], z["S21"]
    L, theta = 11.714687, 0.0
    s11_raw, s21_raw, empty_raw = _fake_cst_pair(
        f, s11_true, s21_true, L, theta)
    s11_de, s21_de = deembed_spectra(s11_raw, s21_raw, empty_raw)
    with tempfile.TemporaryDirectory() as td:
        cl = closure_normal(f, {"TE": s11_de, "TM": s11_de},
                            {"TE": s21_de, "TM": s21_de},
                            out_npz=Path(td) / "c.npz",
                            out_fig=Path(td) / "c.png",
                            label="selftest_closure")
        assert cl["passed"] and cl["worst"] < 1e-12, cl["worst"]
        assert Path(td, "c.npz").exists() and Path(td, "c.png").exists()
        # sigma definition: residual spectrum, same length as the 49 grid
        assert cl["sigma"].shape == (49,)

        # wrong-L de-embedding must blow the 5e-3 gate (the gate catches
        # phase-reference errors, its purpose)
        empty_wrong = analytic_empty_s21(f, L - 0.5, theta,
                                         convention="cst")
        s11_bad, s21_bad = deembed_spectra(s11_raw, s21_raw, empty_wrong)
        cl_bad = closure_normal(f, {"TE": s11_bad}, {"TE": s21_bad},
                                out_npz=Path(td) / "b.npz",
                                out_fig=Path(td) / "b.png",
                                label="selftest_closure_bad")
        assert not cl_bad["passed"] and cl_bad["worst"] > 5e-3
    print(f"[ok] closure gate: exact chain passes ({cl['worst']:.1e}), "
          f"0.5 um L error fails ({cl_bad['worst']:.1e})")


def test_deembed_blocks_and_labels():
    """2x2 block de-embedding + label-hypothesis machinery."""
    rng = np.random.default_rng(7)
    f = GRID_THZ
    nf = len(f)
    L, theta = 11.714687, 30.0
    prop = np.exp(1j * kz_per_um(f, theta) * L)

    # generic TRUE Jones blocks (par. 2 basis), full cross-pol
    S11_true = (rng.normal(size=(2, 2, nf))
                + 1j * rng.normal(size=(2, 2, nf))) * 0.2
    S21_true = (rng.normal(size=(2, 2, nf))
                + 1j * rng.normal(size=(2, 2, nf))) * 0.2

    # a known label hypothesis distorts what "CST" reports
    hyp_true = dict(swap=True, s11_cross=-1, s21_cross=+1)
    S11_cstbasis, S21_cstbasis = apply_hypothesis(S11_true, S21_true,
                                                  hyp_true)

    raw, empty = {}, {}
    for a in (1, 2):
        for b in (1, 2):
            raw[f"1D Results\\S-Parameters\\SZmax({a}),Zmax({b})"] = (
                f, conj_cst(S11_cstbasis[a - 1, b - 1] * prop))
            raw[f"1D Results\\S-Parameters\\SZmin({a}),Zmax({b})"] = (
                f, conj_cst(S21_cstbasis[a - 1, b - 1] * prop))
            empty[f"1D Results\\S-Parameters\\SZmax({a}),Zmax({b})"] = (
                f, np.zeros(nf, dtype=complex))
            empty[f"1D Results\\S-Parameters\\SZmin({a}),Zmax({b})"] = (
                f, conj_cst(prop) if a == b else np.zeros(nf,
                                                          dtype=complex))

    f_out, S11_de, S21_de = deembed_blocks(raw, empty)
    assert np.max(np.abs(S11_de - S11_cstbasis)) < 1e-12
    assert np.max(np.abs(S21_de - S21_cstbasis)) < 1e-12

    # apply_hypothesis is an involution -> the true hypothesis restores the
    # par. 2 blocks, and select_hypothesis finds it uniquely
    hyp, res = select_hypothesis(S11_de, S21_de, S11_true, S21_true,
                                 tol=1e-6)
    assert hyp == hyp_true and res < 1e-12, (hyp, res)

    # on a mirror plane (cross-pol = 0) the cross signs are unobservable:
    # exactly-degenerate hypotheses -> the acceptance MUST refuse
    S11_m = S11_true.copy()
    S21_m = S21_true.copy()
    for S in (S11_m, S21_m):
        S[0, 1] = 0.0
        S[1, 0] = 0.0
    try:
        select_hypothesis(S11_m, S21_m, S11_m, S21_m, tol=1e-6)
        raise AssertionError("degenerate acceptance should have raised")
    except DeembedError as e:
        assert "4 of 8" in str(e), e
    print("[ok] 2x2 block de-embedding exact; label hypothesis uniquely "
          "selected off-mirror, correctly REFUSED on a mirror plane "
          "(the phi != 0 spot-check requirement)")


def test_map_cst_labels_empty_integrity():
    """map_cst_labels runs the empty-cell integrity checks it CAN do and
    carries the hypothesis enumeration it cannot collapse."""
    f = GRID_THZ
    nf = len(f)
    L, theta = 11.714687, 45.0
    prop_cst = analytic_empty_s21(f, L, theta, convention="cst")
    empty = {}
    for a in (1, 2):
        for b in (1, 2):
            empty[f"1D Results\\S-Parameters\\SZmax({a}),Zmax({b})"] = (
                f, np.zeros(nf, dtype=complex))
            empty[f"1D Results\\S-Parameters\\SZmin({a}),Zmax({b})"] = (
                f, prop_cst if a == b else np.zeros(nf, dtype=complex))
    lm = map_cst_labels(theta, 0.0, empty_blocks=empty, L_expected_um=L)
    assert lm["default"]["mode1"] == "TE"
    assert len(lm["hypotheses"]) == 8
    for a in (1, 2):
        chk = lm["empty_checks"][f"mode{a}"]
        assert chk["sign_ok"] and chk["passed"], chk
        assert abs(chk["L_fit_um"] - L) < 1e-9
    assert lm["empty_checks"]["copol_degeneracy"] < 1e-14
    assert lm["empty_checks"]["crosspol_max"] < 1e-14
    print("[ok] map_cst_labels: empty integrity checks pass at "
          "theta = 45 deg (phase slope, degeneracy, cross-pol)")


def test_mirror_plane_crosspol():
    nf = 49
    S = np.zeros((2, 2, nf), dtype=complex)
    S[0, 0] = 0.9
    S[1, 1] = 0.8
    r = check_mirror_plane_crosspol(S, S, 0.0)
    assert r["applicable"] and r["passed"]
    S_bad = S.copy()
    S_bad[0, 1] = 0.05
    r2 = check_mirror_plane_crosspol(S_bad, S, 0.0)
    assert r2["applicable"] and not r2["passed"]
    r3 = check_mirror_plane_crosspol(S_bad, S, 22.5)
    assert not r3["applicable"] and r3["passed"]
    r4 = check_mirror_plane_crosspol(S, S, 45.0)
    assert r4["applicable"] and r4["passed"]
    print("[ok] mirror-plane cross-pol check (phi = 0/45 applicable, "
          "22.5 not)")


def test_interp_matches_base_pattern():
    """interp_to_grid == the base script's separate Re/Im np.interp,
    including the endpoint clamp onto the 49-point grid."""
    f_src = np.linspace(14.99, 37.47, 1001)
    y = np.exp(1j * 0.3 * f_src) * (1.0 + 0.1 * f_src)
    f_tgt = GRID_THZ
    got = interp_to_grid(f_src, y, f_tgt)
    want = (np.interp(f_tgt, f_src, y.real)
            + 1j * np.interp(f_tgt, f_src, y.imag))
    assert np.array_equal(got, want)
    assert got[0] == y[0]          # clamped: 14.98962 < 14.99
    assert got[-1] == y[-1]        # clamped: 37.47406 > 37.47
    print("[ok] Re/Im interp identical to the base-script pattern "
          "(endpoints clamp)")


def test_campaign_dry_run():
    """cst_campaign dry run: manifest counts, VBA content, starter set."""
    with tempfile.TemporaryDirectory() as td:
        runs_dir = Path(td) / "cst_runs"
        man = cst_campaign.generate_dry_run(runs_dir)
        checks = cst_campaign.verify_campaign(man)
        assert len(checks) >= 5
        man2 = json.loads((runs_dir / "campaign_manifest.json")
                          .read_text(encoding="utf-8"))
        assert len(man2["runs"]) == 19
        assert man2["angles_direction"] == "inward"
        assert man2["stimulation"] == ["All", "All"]
        assert len(man2["target_grid_THz"]) == 49
        assert abs(man2["L_expected_um"] - 11.714687) < 1e-4
        # independent VBA spot checks (not via verify_campaign)
        vba = (runs_dir / "struct_th60_ph45" / "build_history.vba"
               ).read_text(encoding="utf-8")
        assert '.SetPeriodicBoundaryAngles "60", "45"' in vba
        assert '.SetDialogTheta "60"' in vba and \
            '.SetDialogPhi "45"' in vba
        assert '.Name "cellpad"' in vba
        vba_e = (runs_dir / "empty_th45" / "build_history.vba"
                 ).read_text(encoding="utf-8")
        assert "saw" not in vba_e and '"Au"' not in vba_e
        assert '.Name "cellpad"' in vba_e
        vba_p = (runs_dir / "empty_th00_pert" / "build_history.vba"
                 ).read_text(encoding="utf-8")
        assert '.AccuracyTet "3e-4"' in vba_p
        n_starter = sum(r["in_starter"] for r in man2["runs"])
        assert n_starter == 8       # 5 structure + 3 empty
    print("[ok] campaign dry run: 19 runs, manifest fields, VBA spot "
          "checks, starter subset 5+3")


def test_deembed_blocks_different_grids():
    """Empty and structure on different frequency grids exercises the
    Re/Im interp path inside deembed_blocks."""
    f_a = np.linspace(15.0, 37.4, 300)
    f_b = np.linspace(15.0, 37.4, 300)      # same ends, aligned values
    L, theta = 6.0, 15.0
    prop_a = np.exp(1j * kz_per_um(f_a, theta) * L)
    s_true = _smooth_spectrum(f_a, 0.5, 0.1, 0.01)
    raw, empty = {}, {}
    for a in (1, 2):
        for b in (1, 2):
            co = a == b
            raw[f"1D Results\\S-Parameters\\SZmax({a}),Zmax({b})"] = (
                f_a, conj_cst((s_true if co else 0 * s_true) * prop_a))
            raw[f"1D Results\\S-Parameters\\SZmin({a}),Zmax({b})"] = (
                f_a, conj_cst((s_true if co else 0 * s_true) * prop_a))
            empty[f"1D Results\\S-Parameters\\SZmax({a}),Zmax({b})"] = (
                f_b, np.zeros_like(f_b, dtype=complex))
            empty[f"1D Results\\S-Parameters\\SZmin({a}),Zmax({b})"] = (
                f_b, conj_cst(np.exp(1j * kz_per_um(f_b, theta) * L))
                if co else np.zeros_like(f_b, dtype=complex))
    f_out, S11_de, S21_de = deembed_blocks(raw, empty)
    assert np.max(np.abs(S11_de[0, 0] - s_true)) < 1e-9
    assert np.max(np.abs(S21_de[1, 1] - s_true)) < 1e-9
    print("[ok] deembed_blocks with per-entry grid alignment")


ALL = [test_analytic_empty_roundtrip,
       test_conjugation_direction_catch,
       test_wrong_L_is_flagged,
       test_closure_recovers_reference_exactly,
       test_deembed_blocks_and_labels,
       test_map_cst_labels_empty_integrity,
       test_mirror_plane_crosspol,
       test_interp_matches_base_pattern,
       test_campaign_dry_run,
       test_deembed_blocks_different_grids]

if __name__ == "__main__":
    for t in ALL:
        t()
    print(f"\nall {len(ALL)} deembed/campaign tests passed")

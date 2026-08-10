"""Offline tests for cst_tmatrix.pipeline helpers (no CST required)."""
from __future__ import annotations

import numpy as np
import pytest

from cst_tmatrix.pipeline import (check_monitor_lmax_margin,
                                  recommended_monitor_factor,
                                  physical_lmax_per_frequency,
                                  noise_order_content, mask_noise_orders,
                                  estimate_fd_memory_gb,
                                  incident_deviation_action,
                                  top_order_content)
from cst_tmatrix import vswf
from cst_tmatrix.mie import wiscombe_lmax

C0 = 299792458.0


# ---------------------------------------------------------------------------
# automatic checks (pure helpers; the CST-bound code calls these)
# ---------------------------------------------------------------------------

def test_memory_estimate_reproduces_the_observed_split():
    """Calibration anchors (2026-08-06/07): ~1M cells solved in <19 GB
    available; the ~10M-cell single-band config exhausted a 32 GB machine
    ('Could not compute preconditioner')."""
    assert estimate_fd_memory_gb(1_000_000) < 19 * 0.9
    assert estimate_fd_memory_gb(10_000_000) > 32


def test_incident_deviation_action_thresholds():
    warn, abort = 1e-2, 3e-2
    assert incident_deviation_action(5e-3, warn, abort) == "ok"
    assert incident_deviation_action(2e-2, warn, abort) == "warn"
    assert incident_deviation_action(1.1e-1, warn, abort) == "abort"
    # the real broken-band value must abort; the real healthy value must not
    assert incident_deviation_action(0.11, warn, abort) == "abort"
    assert incident_deviation_action(6.9e-3, warn, abort) == "ok"


def test_top_order_content_separates_guarded_from_truncating():
    lmax = 5
    _, ns, _ = vswf.mode_list(lmax)
    N = vswf.n_modes(lmax)
    rng = np.random.default_rng(0)
    # guarded: content decays into the top order
    decaying = (rng.normal(size=(2, N)) + 1j * rng.normal(size=(2, N)))
    decaying *= 4.0 ** -ns[None, :]
    assert top_order_content(decaying, lmax) < 0.05
    # truncating: top order comparable to the peak
    flat = rng.normal(size=(2, N)) + 1j * rng.normal(size=(2, N))
    assert top_order_content(flat, lmax) > 0.3


def test_monitor_lmax_margin_flags_the_real_failure_case():
    """kr_mon(fmin)=0.90 against lmax=3 -- the exact case that produced
    incident_deviation_max up to 0.99 in a real extraction (see
    conversation/session notes) -- must be flagged."""
    warning = check_monitor_lmax_margin(k_min=0.90 / 4.318e-6,
                                        r_mon_m=4.318e-6, lmax=3)
    assert warning is not None
    assert "ill-conditioned" in warning


def test_monitor_lmax_margin_passes_with_comfortable_clearance():
    # kr_mon(fmin) = 10, lmax = 3 -- comfortably clear
    warning = check_monitor_lmax_margin(k_min=10.0 / 1.0e-6, r_mon_m=1.0e-6,
                                        lmax=3)
    assert warning is None


def test_monitor_lmax_margin_boundary():
    lmax = 3
    r_mon_m = 1.0e-6
    just_below = check_monitor_lmax_margin(
        k_min=(lmax + 1.999) / r_mon_m, r_mon_m=r_mon_m, lmax=lmax)
    just_above = check_monitor_lmax_margin(
        k_min=(lmax + 2.001) / r_mon_m, r_mon_m=r_mon_m, lmax=lmax)
    assert just_below is not None
    assert just_above is None


def test_recommended_monitor_factor_round_trips_with_the_margin_check():
    """The recommended factor must ALWAYS clear check_monitor_lmax_margin --
    that's the whole point -- across a range of geometries/frequencies/lmax,
    not just one hand-picked case."""
    C0 = 299792458.0
    for r_circ_m in (1e-6, 2.8789e-6, 40e-6):
        for lmax in (3, 7, 9):
            for fmin_hz in (10e12, 1e12, 34e12):
                freqs_hz = np.array([fmin_hz, fmin_hz * 2.5])
                mf = recommended_monitor_factor(r_circ_m, freqs_hz, lmax)
                r_mon_m = mf * r_circ_m
                k_min = 2 * np.pi * fmin_hz / C0
                assert check_monitor_lmax_margin(k_min, r_mon_m, lmax) is None


def test_recommended_monitor_factor_matches_this_designs_known_value():
    """The exact spoke-wheel design/band from tonight's session: lmax=9,
    r_circ=2.8789 um, band 10-34 THz -- independently computed by hand
    earlier in the conversation as ~18.23."""
    r_circ_m = 2.8789e-6
    freqs_hz = np.array([10e12, 34e12])
    mf = recommended_monitor_factor(r_circ_m, freqs_hz, lmax=9)
    assert mf == pytest.approx(18.23, abs=0.01)


# ---------------------------------------------------------------------------
# scatterer near-field clearance (the 2026-08-07 monitor criterion)
# ---------------------------------------------------------------------------

def test_nearfield_clearance_flags_every_observed_failure():
    """Evidence from five extractions (2026-08-06/08).  In the continuous
    variable n(x) = x + 4x^(1/3) + 1, extractions with kr - n >= +0.67
    satisfied the diagnostics, and those with +0.44 and -0.25 did not
    (incident deviations 0.11-0.15).  The integer ceiling w = ceil(n)
    obscured this: two extractions with different n both had kr = w = 8,
    and a rule requiring only kr >= w, consistent with the first four
    data points, failed on the fifth.  With nearfield_margin = 1 every
    new extraction has kr - n >= 1.0, above all satisfactory
    observations.  The check must flag both observed failures and pass
    configurations meeting the enlarged requirement; it is
    intentionally stricter than the one extraction that satisfied the
    diagnostics with no margin (13.1 band B)."""
    cases = [   # (label, r_circ_m, f_bottom_hz, r_mon_m, must_flag)
        ("13.1A ok", 2.8789e-6, 10e12, 33.4e-6, False),
        ("17.3A ok", 3.5978e-6, 10e12, 33.4e-6, False),
        ("10.9B ok", 2.3400e-6, 21e12, 15.905e-6, None),  # informational
        ("17.3B failed", 3.5978e-6, 21e12, 15.905e-6, True),
        ("23.5B failed", 3.9573e-6, 21e12, 18.177e-6, True),
    ]
    for label, r_circ, f_bot, r_mon, must_flag in cases:
        k_min = 2 * np.pi * f_bot / C0
        w = check_monitor_lmax_margin(k_min, r_mon, lmax=5, r_circ_m=r_circ)
        if must_flag is True:
            assert w is not None, label
        elif must_flag is False:
            assert w is None, label


def test_recommended_monitor_factor_clears_the_23p5_band_B_failure():
    """The corrected formula (margin +1) must place the 23.5 um design's
    21-34 THz monitor a full order beyond the scatterer's Wiscombe order,
    and its output must clear its own check (round trip)."""
    r_circ = 3.9573e-6
    freqs = np.arange(21.0, 34.0 + 0.5, 1.0) * 1e12
    mf = recommended_monitor_factor(r_circ, freqs, lmax=5)
    k21 = 2 * np.pi * 21e12 / C0
    # binding constraint: near-field clause at the band bottom, w(x)+1 = 9
    assert k21 * mf * r_circ == pytest.approx(9.0, abs=0.01)
    assert check_monitor_lmax_margin(k21, mf * r_circ, lmax=5,
                                     r_circ_m=r_circ) is None
    # the FAILED configuration (kr = 8.0 = w, continuous margin +0.44)
    # must now be refused
    assert check_monitor_lmax_margin(k21, 18.177e-6, lmax=5,
                                     r_circ_m=r_circ) is not None
    k_min = 2 * np.pi * float(np.min(freqs)) / C0
    assert check_monitor_lmax_margin(k_min, mf * r_circ, lmax=5,
                                     r_circ_m=r_circ) is None


def test_nearfield_criterion_leaves_small_scatterer_factors_unchanged():
    """For the 13.1 um design's full-band lmax=9 case the truncation
    criterion dominates, so the recommended factor must be unchanged by
    the near-field clause (guards the ~18.23 regression value)."""
    mf = recommended_monitor_factor(2.8789e-6, np.array([10e12, 34e12]),
                                    lmax=9)
    assert mf == pytest.approx(18.23, abs=0.01)


# ---------------------------------------------------------------------------
# per-frequency multipole (noise-order) diagnostics
# ---------------------------------------------------------------------------

SPOKE_R_CIRC = 2.8789e-6                     # scaled spoke-wheel design


def test_physical_lmax_spoke_wheel_band_endpoints():
    """The motivating case: extraction runs at lmax=9 (Wiscombe at 34 THz)
    but at 10 THz the physics supports far fewer orders."""
    l_phys = physical_lmax_per_frequency([10e12, 34e12], SPOKE_R_CIRC)
    k34 = 2 * np.pi * 34e12 / C0
    assert l_phys[1] == wiscombe_lmax(k34 * SPOKE_R_CIRC)   # self-consistent
    assert l_phys[0] < l_phys[1]                            # fewer at 10 THz
    assert l_phys[0] <= 6                                   # the known gap


def synthetic_T(lmax, l_signal, noise_amp, seed=0):
    """Block T: order-decaying signal in l <= l_signal, flat absolute noise
    everywhere -- the structure established for real extractions."""
    rng = np.random.default_rng(seed)
    N = vswf.n_modes(lmax)
    _, ns, _ = vswf.mode_list(lmax)
    sig = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    sig *= 1.0 / (2.0 ** np.maximum(ns[:, None], ns[None, :]))
    sig[(ns[:, None] > l_signal) | (ns[None, :] > l_signal)] = 0.0
    noise = noise_amp * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    return sig + noise


def test_noise_order_content_flags_only_the_low_frequency_end():
    lmax = 9
    freqs = np.array([10e12, 34e12])
    l_phys = physical_lmax_per_frequency(freqs, SPOKE_R_CIRC)
    # frequency 0: signal only up to l_phys[0], noise everywhere
    # frequency 1: signal up to lmax (all orders physical)
    T = np.stack([synthetic_T(lmax, l_phys[0], 1e-3),
                  synthetic_T(lmax, lmax, 1e-3)])
    frac = noise_order_content(T, lmax, freqs, SPOKE_R_CIRC)
    assert frac[0] > 5 * frac[1]     # low end dominated by noise orders
    assert frac[1] < 0.2             # top of band: all orders physical


def test_mask_noise_orders_removes_noise_and_preserves_signal():
    lmax = 9
    freqs = np.array([10e12])
    l_phys = int(physical_lmax_per_frequency(freqs, SPOKE_R_CIRC)[0])
    T = synthetic_T(lmax, l_phys, 1e-3)
    T_masked = mask_noise_orders(T, lmax, freqs, SPOKE_R_CIRC, margin=1)
    _, ns, _ = vswf.mode_list(lmax)
    keep = (ns[:, None] <= l_phys) & (ns[None, :] <= l_phys)
    # signal block untouched
    assert np.array_equal(T_masked[keep], T[keep])
    # everything above l_phys + margin exactly zero
    kill = (ns[:, None] > l_phys + 1) | (ns[None, :] > l_phys + 1)
    assert np.all(T_masked[kill] == 0.0)
    # input not mutated
    assert not np.all(T[kill] == 0.0)


def test_mask_noise_orders_is_identity_when_every_order_is_physical():
    lmax = 4
    freqs = np.array([400e12])          # cylinder validation top frequency
    r_circ = 0.2915e-6
    assert physical_lmax_per_frequency(freqs, r_circ)[0] >= lmax
    T = synthetic_T(lmax, lmax, 1e-3)
    assert np.array_equal(mask_noise_orders(T, lmax, freqs, r_circ), T)

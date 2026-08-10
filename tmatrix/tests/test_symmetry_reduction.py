"""Offline validation of symmetry-reduced illumination (no CST required).

The claim under test: for a scatterer with point group G, one solved
illumination yields |G| valid illumination columns via (D(g) a, D(g) f), so a
full extraction needs |G| times fewer solves.

The tests build an EXACTLY C4v-symmetric but otherwise RANDOM T (via the
group-average projection) rather than an isotropic Mie T.  An isotropic T is
invariant under the entire rotation group, so it would pass these tests even
if the C4v representations were wrong -- it cannot distinguish a correct
implementation from a trivially-symmetric one.
"""
from __future__ import annotations

import numpy as np
import pytest

from cst_tmatrix import vswf
from cst_tmatrix.quadrature import illumination_directions
from cst_tmatrix.tmatrix_solve import solve_tmatrix
from cst_tmatrix.postprocess.symmetry import (
    point_group_operations, orbit_directions, orbit_is_degenerate,
    expand_columns_by_symmetry, symmetrize_tmatrix, check_symmetry,
    reduced_illumination_directions, rotation_z_representation,
    mirror_representation)

LMAX = 3
C4V_MIRRORS = [0.0, 45.0, 90.0, 135.0]


def c4v_ops(lmax=LMAX):
    return point_group_operations(lmax, n_fold=4,
                                  mirror_phi0_deg=C4V_MIRRORS)


def random_c4v_tmatrix(lmax=LMAX, seed=0):
    """A random T projected onto the C4v-invariant subspace -- symmetric by
    construction, but with no other structure."""
    rng = np.random.default_rng(seed)
    N = vswf.n_modes(lmax)
    T = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    return symmetrize_tmatrix(T, c4v_ops(lmax))


# --------------------------------------------------------------------------
# group structure
# --------------------------------------------------------------------------

def test_c4v_has_eight_elements():
    assert len(c4v_ops()) == 8


def test_group_average_actually_produces_a_symmetric_tmatrix():
    """symmetrize_tmatrix must land in the invariant subspace: the projected
    T commutes with every group element."""
    T = random_c4v_tmatrix()
    for label, D, _spec in c4v_ops():
        assert check_symmetry(T, D) < 1e-12, label


def test_projection_is_idempotent():
    """A projection applied twice is the projection -- guards against the
    group list being an incomplete (non-closed) set of elements."""
    ops = c4v_ops()
    T = random_c4v_tmatrix()
    assert (np.linalg.norm(symmetrize_tmatrix(T, ops) - T)
            / np.linalg.norm(T)) < 1e-12


def test_random_unsymmetrized_tmatrix_is_NOT_c4v():
    """Control: the test would be vacuous if everything looked symmetric."""
    rng = np.random.default_rng(1)
    N = vswf.n_modes(LMAX)
    T = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    D = rotation_z_representation(LMAX, np.pi / 2)
    assert check_symmetry(T, D) > 0.1


# --------------------------------------------------------------------------
# orbit bookkeeping (the empirically established conventions)
# --------------------------------------------------------------------------

def test_orbit_direction_matches_plane_wave_coefficients():
    """The recorded physical direction of each generated column must agree
    with the actual plane-wave expansion at that direction.  This is the
    empirically-fixed convention (rotation uses D(-alpha), mirror carries a
    -1 for phi-polarization) -- pinned here so a regression is caught."""
    ops = c4v_ops()
    th, ph = 0.7, 1.1
    for pol in ("theta", "phi"):
        a0 = vswf.plane_wave_coefficients(th, ph, pol, LMAX)
        dirs = orbit_directions(th, ph, pol, ops)
        for (label, D, _spec), (lab2, th2, ph2, pol2, sign) in zip(ops, dirs):
            a_direct = vswf.plane_wave_coefficients(th2, ph2, pol2, LMAX)
            err = (np.linalg.norm(D @ a0 - sign * a_direct)
                   / np.linalg.norm(a_direct))
            assert err < 1e-10, f"{label} pol={pol}: {err:.3e}"


def test_generic_direction_has_full_eightfold_orbit():
    assert not orbit_is_degenerate(0.7, 1.1, "theta", c4v_ops())


def test_on_axis_direction_is_degenerate():
    """Normal incidence lies on the C4 axis: every rotation maps it to
    itself, so its orbit collapses and it cannot deliver an 8x gain."""
    assert orbit_is_degenerate(0.0, 0.0, "theta", c4v_ops())


def test_direction_in_a_mirror_plane_is_degenerate():
    """phi=0 lies in the sigma(0) plane, so that reflection is the identity
    on it."""
    assert orbit_is_degenerate(0.7, 0.0, "theta", c4v_ops())


# --------------------------------------------------------------------------
# the actual claim: fewer solves, same T
# --------------------------------------------------------------------------

def test_expanded_columns_satisfy_the_same_tmatrix():
    """Every symmetry-generated column must satisfy f = T a with the SAME T.
    This is the identity the whole speedup rests on."""
    ops = c4v_ops()
    T = random_c4v_tmatrix()
    th_i, ph_i = illumination_directions("fibonacci", 5)
    A = np.column_stack([vswf.plane_wave_coefficients(t, p, pol, LMAX)
                         for t, p in zip(th_i, ph_i)
                         for pol in ("theta", "phi")])
    F = T @ A
    A_exp, F_exp = expand_columns_by_symmetry(A, F, ops)
    assert A_exp.shape[1] == A.shape[1] * 8
    assert (np.linalg.norm(T @ A_exp - F_exp)
            / np.linalg.norm(F_exp)) < 1e-10


def test_reduced_illumination_set_recovers_the_full_tmatrix():
    """End to end: solve T from ~8x fewer illuminations plus their orbits,
    and compare against the true C4v T."""
    ops = c4v_ops()
    T_true = random_c4v_tmatrix(seed=3)
    N = vswf.n_modes(LMAX)
    needed = int(np.ceil(1.5 * N))

    th_all, ph_all = illumination_directions("fibonacci", 4 * N)
    idx, n_expected = reduced_illumination_directions(th_all, ph_all, ops,
                                                      needed)
    th_i, ph_i = th_all[idx], ph_all[idx]

    A = np.column_stack([vswf.plane_wave_coefficients(t, p, pol, LMAX)
                         for t, p in zip(th_i, ph_i)
                         for pol in ("theta", "phi")])
    F = T_true @ A
    A_exp, F_exp = expand_columns_by_symmetry(A, F, ops)
    T_rec, diag = solve_tmatrix(A_exp, F_exp)

    err = np.linalg.norm(T_rec - T_true) / np.linalg.norm(T_true)
    assert diag["rank"] == N, f"rank {diag['rank']} < {N}"
    assert err < 1e-8, f"rel err {err:.3e}"
    # the point of the exercise: far fewer solves than the unreduced scheme
    assert 2 * len(idx) <= needed / 4


def test_reduced_directions_give_a_well_conditioned_system():
    """The regression that produced a garbage T on real data (2026-08-06):
    an in-order pick of Fibonacci directions selects a polar cap; since the
    axial group ops preserve theta, the orbit-expanded system was rank
    38/48 (cond 4.2e5) and CST's ~1e-3 field noise destroyed the
    undetermined modes -- while all noise-free synthetic tests passed.
    Farthest-point selection must keep the noise-free condition number
    small; measured 24 for the real lmax=4/C4v case."""
    for lmax in (3, 4):
        N = vswf.n_modes(lmax)
        needed = int(np.ceil(1.5 * N))
        ops = point_group_operations(lmax, n_fold=4,
                                     mirror_phi0_deg=C4V_MIRRORS)
        th, ph = illumination_directions("fibonacci",
                                         int(np.ceil(1.5 * N / 2)))
        idx, _ = reduced_illumination_directions(th, ph, ops, needed)
        cols = [vswf.plane_wave_coefficients(float(th[i]), float(ph[i]),
                                             pol, lmax)
                for i in idx for pol in ("theta", "phi")]
        A = np.column_stack(cols)
        A_exp, _ = expand_columns_by_symmetry(A, A, ops)
        sv = np.linalg.svd(A_exp, compute_uv=False)
        assert sv[-1] > 0
        cond = sv[0] / sv[-1]
        assert cond < 100, f"lmax={lmax}: cond={cond:.3e}"
        # and genuinely full rank at a noise-relevant threshold
        assert int(np.sum(sv > 1e-3 * sv[0])) == N


def test_reduced_scheme_uses_far_fewer_solves_than_unreduced():
    """Quantifies the saving actually delivered at this lmax."""
    ops = c4v_ops()
    N = vswf.n_modes(LMAX)
    needed = int(np.ceil(1.5 * N))
    th_all, ph_all = illumination_directions("fibonacci", 4 * N)
    idx, n_expected = reduced_illumination_directions(th_all, ph_all, ops,
                                                      needed)
    n_solves_reduced = 2 * len(idx)
    assert n_expected >= needed
    assert n_solves_reduced < needed
    # record the ratio so a regression in orbit selection is visible
    assert n_solves_reduced <= needed / 4


# --------------------------------------------------------------------------
# denoising use of the same projection
# --------------------------------------------------------------------------

def test_projection_quantifies_and_removes_non_invariant_part():
    """project_onto_symmetry must return the invariant part together with
    the relative magnitude of what was removed, and the removed fraction
    must equal the known perturbation for a synthetic case."""
    ops = c4v_ops()
    T_true = random_c4v_tmatrix(seed=9)
    rng = np.random.default_rng(13)
    noise = rng.normal(size=T_true.shape) + 1j * rng.normal(size=T_true.shape)
    # make the perturbation purely non-invariant so its size is known
    from cst_tmatrix.postprocess.symmetry import project_onto_symmetry
    noise = noise - symmetrize_tmatrix(noise, ops)
    noise *= 0.02 * np.linalg.norm(T_true) / np.linalg.norm(noise)
    T_meas = T_true + noise
    T_proj, removed = project_onto_symmetry(T_meas, ops)
    assert np.allclose(T_proj, T_true, rtol=1e-10, atol=1e-12)
    expected = np.linalg.norm(noise) / np.linalg.norm(T_meas)
    assert removed == pytest.approx(expected, rel=1e-6)


def test_symmetrization_suppresses_non_symmetric_noise():
    """Extraction noise is generically not G-invariant, so the group average
    attenuates it while leaving the true symmetric T untouched."""
    ops = c4v_ops()
    T_true = random_c4v_tmatrix(seed=7)
    rng = np.random.default_rng(11)
    noise = (rng.normal(size=T_true.shape)
             + 1j * rng.normal(size=T_true.shape))
    noise *= 0.01 * np.linalg.norm(T_true) / np.linalg.norm(noise)

    before = np.linalg.norm(noise) / np.linalg.norm(T_true)
    after = (np.linalg.norm(symmetrize_tmatrix(T_true + noise, ops) - T_true)
             / np.linalg.norm(T_true))
    assert after < before / 2


def test_c4_only_group_is_weaker_than_c4v():
    """A C4 (rotations only) scatterer must NOT be forced to satisfy the
    mirror constraints -- guards against silently over-symmetrizing."""
    ops_c4 = point_group_operations(LMAX, n_fold=4)
    assert len(ops_c4) == 4
    T_c4 = symmetrize_tmatrix(
        np.random.default_rng(5).normal(size=(vswf.n_modes(LMAX),) * 2)
        + 1j * np.random.default_rng(6).normal(size=(vswf.n_modes(LMAX),) * 2),
        ops_c4)
    assert check_symmetry(T_c4, rotation_z_representation(LMAX, np.pi / 2)) < 1e-12
    # but generically NOT mirror-symmetric
    assert check_symmetry(T_c4, mirror_representation(LMAX, 0.0)) > 1e-3

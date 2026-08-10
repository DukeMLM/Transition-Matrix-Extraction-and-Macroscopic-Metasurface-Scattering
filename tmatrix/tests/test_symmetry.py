"""Offline tests for cst_tmatrix.postprocess.symmetry (no CST required).

These lock down the REPRESENTATION machinery (rotation_z_representation,
mirror_representation, check_symmetry) against independent physics, not
against any particular extracted T-matrix — the "does this scatterer
actually have this symmetry" question belongs to the extraction, not here.
"""
from __future__ import annotations

import numpy as np
import pytest

from cst_tmatrix.vswf import mode_list, block_index, n_modes
from cst_tmatrix.postprocess.symmetry import (
    rotation_z_representation, mirror_representation, check_symmetry,
    point_group_report, detect_axial_point_group)


@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_mirror_is_involution_and_unitary(lmax):
    N = n_modes(lmax)
    D = mirror_representation(lmax, phi0=np.radians(37.0))
    assert np.linalg.norm(D @ D - np.eye(N)) / np.sqrt(N) < 1e-8
    assert np.linalg.norm(D.conj().T @ D - np.eye(N)) / np.sqrt(N) < 1e-8


@pytest.mark.parametrize("phi0_deg", [0.0, 30.0, 45.0, 90.0])
def test_mirror_m0_dipole_parity(phi0_deg):
    """z-axis lies IN every vertical mirror plane, so it is always
    tangential: the true-vector (electric, N-type) m=0 dipole must be
    invariant, the axial-vector (magnetic, M-type) m=0 dipole must flip
    sign -- textbook mirror-image parity, independent of phi0."""
    lmax = 2
    half = n_modes(lmax) // 2
    idx_N10 = half + block_index(1, 0)
    idx_M10 = block_index(1, 0)
    D = mirror_representation(lmax, phi0=np.radians(phi0_deg))
    assert D[idx_N10, idx_N10] == pytest.approx(1.0, abs=1e-8)
    assert D[idx_M10, idx_M10] == pytest.approx(-1.0, abs=1e-8)
    # and each is otherwise decoupled from every other mode (m=0 is its
    # own mirror image up to that sign, for any phi0)
    col = D[:, idx_N10].copy(); col[idx_N10] = 0
    assert np.linalg.norm(col) < 1e-8


def test_check_symmetry_discriminates_synthetic_matrices():
    """A matrix explicitly symmetrized against D must show ~machine-
    precision violation; a generic random matrix must show O(1) violation.
    This is the ground-truth check that check_symmetry() + a representation
    actually detect symmetry/asymmetry, independent of any real scatterer."""
    lmax = 3
    N = n_modes(lmax)
    rng = np.random.default_rng(0)
    D = mirror_representation(lmax, phi0=0.0)

    T_random = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    assert check_symmetry(T_random, D) > 0.1

    T_symmetrized = 0.5 * (T_random + D @ T_random @ D)
    assert check_symmetry(T_symmetrized, D) < 1e-10


def test_rotation_representation_matches_mask_selection_rule():
    """rotation_z_representation + check_symmetry (general commutator form)
    must vanish exactly whenever T is constructed to strictly satisfy the
    m' - m == 0 (mod n_fold) selection rule (the simple index-mask form),
    cross-validating the two formulations of the same physical statement."""
    lmax = 3
    N = n_modes(lmax)
    pol, ns, ms = mode_list(lmax)
    rng = np.random.default_rng(1)
    n_fold = 4
    allowed = (ms[:, None] - ms[None, :]) % n_fold == 0
    T = np.where(allowed, rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)),
                0.0)
    D = rotation_z_representation(lmax, 2 * np.pi / n_fold)
    assert check_symmetry(T, D) < 1e-12


def test_point_group_report_shape():
    lmax = 2
    N = n_modes(lmax)
    T = np.eye(N, dtype=complex)          # identity commutes with everything
    report = point_group_report(T, lmax, n_fold=4, mirror_phi0_deg=[0, 45])
    assert set(report) == {"C4", "sigma_v(0 deg)", "sigma_v(45 deg)"}
    assert all(v < 1e-10 for v in report.values())


# ----------------------------------------------------------------------------
# CAD/geometry point-group detector (synthetic STL, no CST required)
# ----------------------------------------------------------------------------

def _flag_triangle(azimuth_deg, tip_r, base_r=0.3, half_width=0.08):
    """One triangular 'flag' pointing radially outward at `azimuth_deg`,
    mirror-symmetric about its own bisector by construction."""
    phi = np.radians(azimuth_deg)
    local = np.array([[tip_r, 0.0, 0.0],
                      [base_r, half_width, 0.0],
                      [base_r, -half_width, 0.0]])
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ R.T


def _write_stl(path, triangles):
    with open(path, "w") as fh:
        fh.write("solid test\n")
        for tri in triangles:
            fh.write("  facet normal 0 0 1\n    outer loop\n")
            for v in tri:
                fh.write(f"      vertex {v[0]:e} {v[1]:e} {v[2]:e}\n")
            fh.write("    endloop\n  endfacet\n")
        fh.write("endsolid test\n")


def test_detect_axial_point_group_c4v(tmp_path):
    tris = [_flag_triangle(az, tip_r=1.0) for az in (0, 90, 180, 270)]
    path = tmp_path / "c4v.stl"
    _write_stl(path, tris)
    report = detect_axial_point_group(path, n_candidates=(2, 3, 4, 5, 6, 8))
    assert report["rotation_order"] == 4
    assert report["label"] == "C4v"
    assert sorted(report["mirror_planes_deg"]) == [0.0, 45.0, 90.0, 135.0]
    assert report["axis_is_principal"]


def test_detect_axial_point_group_broken_symmetry_falls_back_to_c2v(tmp_path):
    """Four flags at 90 deg spacing but alternating tip radius: true C4 is
    broken (opposite flags still match under 180 deg rotation, but adjacent
    ones don't), so the detector must fall back to the actual subgroup
    rather than either falsely reporting C4v or collapsing to C1."""
    tris = [_flag_triangle(0, tip_r=1.0), _flag_triangle(90, tip_r=0.6),
           _flag_triangle(180, tip_r=1.0), _flag_triangle(270, tip_r=0.6)]
    path = tmp_path / "c2v.stl"
    _write_stl(path, tris)
    report = detect_axial_point_group(path, n_candidates=(2, 3, 4, 5, 6, 8))
    assert report["rotation_order"] == 2
    assert report["label"] == "C2v"
    assert sorted(report["mirror_planes_deg"]) == [0.0, 90.0]


def test_detect_axial_point_group_asymmetric_is_c1(tmp_path):
    tris = [_flag_triangle(0, tip_r=1.0), _flag_triangle(80, tip_r=0.7),
           _flag_triangle(210, tip_r=0.5)]
    path = tmp_path / "c1.stl"
    _write_stl(path, tris)
    report = detect_axial_point_group(path, n_candidates=(2, 3, 4, 5, 6, 8))
    assert report["rotation_order"] == 1
    assert report["label"] == "C1"
    assert report["mirror_planes_deg"] == []

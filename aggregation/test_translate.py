"""Layer-1 validation: translation operators and the periodic lattice sum."""
import numpy as np

from vswf import ModeBasis, vswf_fields
from translate import translation_matrix, translation_shells, rotate_inplane, \
    lattice_sum_C, make_quad


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_translation_reexpansion():
    """Outgoing mode at d, re-expanded about origin with generous lmax,
    must reproduce the direct field at interior test points.

    Note: the truncated addition theorem converges like (r/d)^(l - l_src),
    so exactness of the operator is checked at small test radius with a
    generous row basis; the demo itself only needs the l<=3 block, whose
    truncation is inherited from the T-matrix, not from this operator.
    """
    k = 1.1
    src_modes = ModeBasis.standard(3)
    row_modes = ModeBasis.standard(12)  # generous re-expansion basis
    quad = make_quad(36, 72)
    rng = np.random.default_rng(4)
    worst = 0.0
    for d in ([2.0, 0.0, 0.0], [1.4, -1.2, 0.8], [0.0, 2.5, 0.0]):
        d = np.asarray(d)
        r0 = 0.45 * np.linalg.norm(d)
        A = translation_matrix(k, d, row_modes, r0, quad)  # (n_row, n_row)
        # random combination of the 30 low-order source modes
        c = rng.normal(size=src_modes.n) + 1j * rng.normal(size=src_modes.n)
        c_full = np.zeros(row_modes.n, dtype=complex)
        for i in range(src_modes.n):
            j = row_modes.index(src_modes.l[i], src_modes.m[i], src_modes.pol[i])
            c_full[j] = c[i]
        a_reg = A @ c_full
        pts = rng.normal(size=(25, 3))
        pts /= np.linalg.norm(pts, axis=1)[:, None]
        pts *= 0.18 * np.linalg.norm(d) * rng.uniform(0.3, 1.0, size=(25, 1))
        Ereg, _ = vswf_fields(k, row_modes, pts, "regular")
        Erec = np.tensordot(a_reg, Ereg, axes=(0, 0))
        Eout, _ = vswf_fields(k, row_modes, pts, "outgoing", origin=d)
        Edir = np.tensordot(c_full, Eout, axes=(0, 0))
        worst = max(worst, rel_err(Erec, Edir))
    print(f"  translation re-expansion field check (r/d<=0.18): rel err {worst:.2e}")
    assert worst < 1e-6


def test_translation_r0_invariance():
    """The 30x30 operator block must not depend on the projection radius."""
    k = 1.1
    modes = ModeBasis.standard(3)
    quad = make_quad(24, 48)
    d = np.array([2.0, 0.7, -0.4])
    A1 = translation_matrix(k, d, modes, 0.3 * np.linalg.norm(d), quad)
    A2 = translation_matrix(k, d, modes, 0.6 * np.linalg.norm(d), quad)
    err = rel_err(A1, A2)
    print(f"  translation operator r0-invariance: rel err {err:.2e}")
    assert err < 1e-5


def test_rotation_identity():
    """A(Rot_phi d) == e^{i(m_nu - m_mu) phi} A(d) for in-plane rotations."""
    k = 0.9
    modes = ModeBasis.standard(3)
    quad = make_quad(16, 32)
    r = 2.3
    r0 = 0.8
    A0 = translation_shells(k, [r], modes, r0, quad)[0]     # d = r x_hat
    worst = 0.0
    for phi in (0.71, 2.2, -1.3):
        d = np.array([r * np.cos(phi), r * np.sin(phi), 0.0])
        A_direct = translation_matrix(k, d, modes, r0, quad)
        A_rot = rotate_inplane(A0, phi, modes)
        worst = max(worst, rel_err(A_rot, A_direct))
    print(f"  in-plane rotation identity: rel err {worst:.2e}")
    assert worst < 1e-9


def test_lattice_sum_convergence():
    """Richardson-extrapolated C must be stable against the choice of taper
    set and against quadrature density (worst case: largest wavelength)."""
    lam = 20.0    # worst case: smallest k in the demo band (um)
    k = 2 * np.pi / lam
    pitch = 2.0
    modes = ModeBasis.standard(3)
    r0 = 0.8
    q1, q2 = make_quad(16, 32), make_quad(24, 48)
    C1 = lattice_sum_C(k, pitch, modes, r0, q1, kRc=(10, 14, 20))
    C2 = lattice_sum_C(k, pitch, modes, r0, q1, kRc=(12, 17, 24))
    C3 = lattice_sum_C(k, pitch, modes, r0, q2, kRc=(10, 14, 20))
    # entrywise metric: relative for large entries, absolute against the
    # radiative scale 2 pi/(k A) for small ones (large quasi-static entries
    # would otherwise dominate a max-abs metric with irrelevant ~1e-8 jitter)
    scale = 2 * np.pi / (k * pitch ** 2)
    e_taper = (np.abs(C1 - C2) / (np.abs(C1) + scale)).max()
    e_quad = (np.abs(C1 - C3) / (np.abs(C1) + scale)).max()
    e_rel = rel_err(C1, C2)
    print(f"  lattice sum: |C|max={np.abs(C1).max():.3g}  taper-set stability "
          f"{e_taper:.2e} (entrywise) / {e_rel:.2e} (Frobenius)  "
          f"quadrature stability {e_quad:.2e}")
    assert e_taper < 1e-4 and e_quad < 1e-4


def test_lattice_sum_r0_invariance():
    """C is an expansion-coefficient object: must not depend on the
    projection sphere radius r0."""
    lam = 12.0
    k = 2 * np.pi / lam
    modes = ModeBasis.standard(3)
    quad = make_quad(16, 32)
    Ca = lattice_sum_C(k, 2.0, modes, 0.6, quad)
    Cb = lattice_sum_C(k, 2.0, modes, 1.0, quad)
    err = rel_err(Ca, Cb)
    print(f"  lattice sum r0-invariance: rel err {err:.2e}")
    assert err < 1e-4


if __name__ == "__main__":
    import time
    for t in (test_translation_reexpansion, test_translation_r0_invariance,
              test_rotation_identity, test_lattice_sum_convergence,
              test_lattice_sum_r0_invariance):
        print(t.__name__)
        t0 = time.time()
        t()
        print(f"  ({time.time() - t0:.1f} s)")
    print("ALL LAYER-1 TESTS PASSED")

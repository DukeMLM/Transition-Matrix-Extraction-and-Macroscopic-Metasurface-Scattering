"""Mirror (PEC ground plane) validation.

The killer test: a LOSSLESS scatterer over a PEC mirror reflects all power,
so |S11_co|^2 + |S11_cross|^2 = 1 must hold for the coupled periodic array.
Any sign/phase error in the image bookkeeping (parity signs, image lattice
sum, ground reflection phase, image far-field phase) breaks this.
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn

from tmatrix.aggregation import vswf
from tmatrix.aggregation.vswf import ModeBasis, RegularProjector
from tmatrix.aggregation.translate import make_quad, lattice_sum_C
from tmatrix.aggregation.mirror import mirror_parity_signs, image_lattice_sum, \
    sparams_mirror_periodic
from tmatrix.aggregation.tmat_io import TMatrixData
from tmatrix.paths import DEMO_TMAT


def mie_T(modes, k, radius, n_ref):
    """Diagonal lossless Mie T-matrix (same as test_vswf, f = T a)."""
    from tmatrix.aggregation.vswf import spherical_h1
    x = k * radius
    mx = n_ref * x

    def psi(l, z):
        return z * spherical_jn(l, z)

    def dpsi(l, z):
        return spherical_jn(l, z) + z * spherical_jn(l, z, derivative=True)

    def xi(l, z):
        return z * spherical_h1(l, z)

    def dxi(l, z):
        h = spherical_h1(l, z)
        dh = (spherical_jn(l, z, derivative=True)
              + 1j * spherical_yn(l, z, derivative=True))
        return h + z * dh

    T = np.zeros((modes.n, modes.n), dtype=complex)
    for i in range(modes.n):
        l = modes.l[i]
        if modes.pol[i] == vswf.ELECTRIC:
            num = n_ref * psi(l, mx) * dpsi(l, x) - psi(l, x) * dpsi(l, mx)
            den = n_ref * psi(l, mx) * dxi(l, x) - xi(l, x) * dpsi(l, mx)
        else:
            num = psi(l, mx) * dpsi(l, x) - n_ref * psi(l, x) * dpsi(l, mx)
            den = psi(l, mx) * dxi(l, x) - n_ref * xi(l, x) * dpsi(l, mx)
        T[i, i] = -num / den
    return T


def test_mirror_parity_signs():
    modes = ModeBasis.standard(3)
    s = mirror_parity_signs(modes)   # internal consistency asserted
    # analytic expectation: s = +(-1)^(l+m) for magnetic, -(-1)^(l+m) electric
    # (PEC image: tangential-E flip is parity-odd for the electric multipoles)
    exp = np.where(modes.pol == vswf.MAGNETIC,
                   ((-1.0) ** (modes.l + modes.m)),
                   -((-1.0) ** (modes.l + modes.m)))
    match = np.array_equal(s, exp)
    print(f"  mirror parity signs diagonal and +/-1: OK; matches analytic "
          f"(-1)^(l+m) pattern: {match}")
    assert match


def test_lossless_unitarity_periodic():
    """Lossless Mie array over PEC: |S11|^2 must be 1."""
    modes = ModeBasis.standard(3)
    pitch, h = 2.0, 0.6
    quad = make_quad(16, 32)
    s = mirror_parity_signs(modes)
    worst = 0.0
    for lam in (16.0, 11.0):
        k = 2 * np.pi / lam
        T = mie_T(modes, k, radius=0.55, n_ref=3.0)
        assert np.abs(np.abs(1 + 2 * np.diag(T)) - 1).max() < 1e-12
        C = lattice_sum_C(k, pitch, modes, 0.8, quad)
        C_im = image_lattice_sum(k, pitch, modes, h, 0.6, quad)
        S = sparams_mirror_periodic(k, h, pitch ** 2, modes, T, C, C_im, s)
        worst = max(worst, abs(S["R"] - 1.0))
        print(f"  lam={lam}: |S11_co|={abs(S['S11_co']):.6f} "
              f"R={S['R']:.6f}  (must be 1)")
    assert worst < 2e-3, f"unitarity violated: {worst}"


def test_energy_bounds_real_T():
    """Real (lossy) demo T over PEC: 0 <= A <= 1 required."""
    data = TMatrixData(str(DEMO_TMAT))
    modes = data.modes
    quad = make_quad(16, 32)
    s = mirror_parity_signs(modes)
    h = 0.35
    for i in (5, 25, 45):
        k = data.k_at(i)
        C = lattice_sum_C(k, 2.0, modes, 0.8, quad)
        C_im = image_lattice_sum(k, 2.0, modes, h, 0.6, quad)
        S = sparams_mirror_periodic(k, h, 4.0, modes, data.T[i], C, C_im, s)
        print(f"  lam={data.wavelength_um[i]:.2f}: |S11|="
              f"{abs(S['S11_co']):.4f}  A={S['A']:.4f}")
        assert -0.02 <= S["A"] <= 1.0


if __name__ == "__main__":
    for t in (test_mirror_parity_signs, test_lossless_unitarity_periodic,
              test_energy_bounds_real_T):
        print(t.__name__)
        t()
    print("ALL MIRROR TESTS PASSED")

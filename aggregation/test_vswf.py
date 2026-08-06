"""Layer-0 validation: VSWF evaluation, plane-wave expansion, far field, projection."""
import numpy as np

import vswf
from vswf import ModeBasis, vswf_fields, plane_wave_coeffs, far_field_amplitude, \
    sphere_quadrature, project_regular, spherical_h1


def rel_err(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def test_maxwell_curl():
    """N = (1/k) curl M and M = (1/k) curl N via central finite differences."""
    k = 1.3
    modes = ModeBasis.standard(3)
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(6, 3)) * 1.5 + np.array([2.0, 0.5, 1.0])
    h = 1e-5
    worst = 0.0
    for kind in ("regular", "outgoing"):
        E0, H0 = vswf_fields(k, modes, pts, kind)
        curl = np.zeros_like(E0)
        for ax in range(3):
            dp = pts.copy(); dp[:, ax] += h
            dm = pts.copy(); dm[:, ax] -= h
            Ep, _ = vswf_fields(k, modes, dp, kind)
            Em, _ = vswf_fields(k, modes, dm, kind)
            dE = (Ep - Em) / (2 * h)
            e1, e2 = (ax + 1) % 3, (ax + 2) % 3
            curl[:, :, e2] += dE[:, :, e1]
            curl[:, :, e1] -= dE[:, :, e2]
        # E = M(or N) => (1/k) curl E should equal the dual field = -i * Ht_here?
        # With Ht = i Z0 H and H = curl E/(i omega mu): Ht = curl E / k.
        worst = max(worst, rel_err(curl / k, H0))
    print(f"  curl consistency (Ht = curl E / k): rel err {worst:.2e}")
    assert worst < 1e-8


def test_plane_wave_expansion():
    """Reconstruct e_hat exp(ik.r) from regular VSWFs with lmax=12."""
    k = 2.0
    modes = ModeBasis.standard(12)
    rng = np.random.default_rng(1)
    worst = 0.0
    for trial in range(3):
        kh = rng.normal(size=3); kh /= np.linalg.norm(kh)
        # polarization orthogonal to k (complex combination allowed)
        t = np.cross(kh, [0.3, -0.9, 0.44]); t /= np.linalg.norm(t)
        u = np.cross(kh, t)
        eh = (t + 1j * 0.7 * u) / np.sqrt(1 + 0.49)
        a = plane_wave_coeffs(kh, eh, modes)
        pts = rng.normal(size=(40, 3)) * 0.45  # kr up to ~2.5
        E, Ht = vswf_fields(k, modes, pts, "regular")
        Erec = np.tensordot(a, E, axes=(0, 0))
        Eex = eh[None, :] * np.exp(1j * k * pts @ kh)[:, None]
        worst = max(worst, rel_err(Erec, Eex))
        # magnetic field: Ht = i k_hat x e_hat exp(ik.r)
        Hrec = np.tensordot(a, Ht, axes=(0, 0))
        Hex = 1j * np.cross(kh, eh)[None, :] * np.exp(1j * k * pts @ kh)[:, None]
        worst = max(worst, rel_err(Hrec, Hex))
    print(f"  plane-wave reconstruction (E and Ht): rel err {worst:.2e}")
    assert worst < 1e-6


def test_far_field():
    """far_field_amplitude vs direct outgoing-field evaluation at kr = 1e5."""
    k = 1.0
    modes = ModeBasis.standard(3)
    rng = np.random.default_rng(2)
    f = rng.normal(size=modes.n) + 1j * rng.normal(size=modes.n)
    rhat = rng.normal(size=(30, 3))
    rhat /= np.linalg.norm(rhat, axis=1, keepdims=True)
    R = 1e5
    E, _ = vswf_fields(k, modes, R * rhat, "outgoing")
    Edir = np.tensordot(f, E, axes=(0, 0))
    F = far_field_amplitude(k, f, modes, rhat)
    Efar = F * (np.exp(1j * k * R) / R)
    err = rel_err(Efar, Edir)
    print(f"  far-field amplitude vs direct at kr=1e5: rel err {err:.2e}")
    assert err < 1e-4


def test_projection_roundtrip():
    """project_regular recovers random regular-wave coefficients from (E, Ht)."""
    k = 0.5
    modes = ModeBasis.standard(3)
    rng = np.random.default_rng(3)
    c = rng.normal(size=modes.n) + 1j * rng.normal(size=modes.n)
    r0 = 0.8
    quad = sphere_quadrature(24, 48)
    TH, PH, W = quad
    pts = r0 * np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                         np.cos(TH)], axis=1)
    E, Ht = vswf_fields(k, modes, pts, "regular")
    Esum = np.tensordot(c, E, axes=(0, 0))
    Hsum = np.tensordot(c, Ht, axes=(0, 0))
    crec = project_regular(k, r0, Esum, Hsum, modes, quad)
    err = rel_err(crec, c)
    print(f"  sphere-projection roundtrip: rel err {err:.2e}")
    assert err < 1e-9


def test_optical_theorem_sphere():
    """Mie sphere via analytic coefficients: sigma_ext(optical theorem) ==
    sigma_sca(|F|^2 integral) for a lossless dielectric sphere.

    Mie T-matrix diagonal in this basis: T^M_l = -j_l(x) d/dx'[..] standard:
      a_l (electric), b_l (magnetic) with T^E = -a_l, T^M = -b_l in the
      convention f = T a  (checked by energy conservation herein).
    """
    from scipy.special import spherical_jn, spherical_yn
    k = 1.0
    x = 1.7          # size parameter
    n_ref = 1.9      # refractive index (lossless)
    mx = n_ref * x
    lmax = 8
    modes = ModeBasis.standard(lmax)

    def psi(l, z):  # Riccati-Bessel
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
        T[i, i] = -num / den   # f = T a
    a = plane_wave_coeffs([0, 0, 1], [1, 0, 0], modes)
    f = T @ a
    F_fwd = far_field_amplitude(k, f, modes, np.array([[0, 0, 1.0]]))[0]
    sigma_ext = 4 * np.pi / k * np.imag(np.vdot([1, 0, 0], F_fwd))
    TH, PH, W = sphere_quadrature(40, 80)
    rhat = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                     np.cos(TH)], axis=1)
    F = far_field_amplitude(k, f, modes, rhat)
    sigma_sca = np.sum(W * np.sum(np.abs(F) ** 2, axis=1))
    err = abs(sigma_ext - sigma_sca) / sigma_sca
    print(f"  Mie lossless sphere: sig_ext={sigma_ext:.6f} sig_sca={sigma_sca:.6f}"
          f" rel diff {err:.2e}")
    assert err < 1e-6
    # cross-check against classic Mie series sum (2l+1)(|a_l|^2+|b_l|^2)
    ssum = 0.0
    for l in range(1, lmax + 1):
        iE = modes.index(l, 1, vswf.ELECTRIC)
        iM = modes.index(l, 1, vswf.MAGNETIC)
        ssum += (2 * l + 1) * (abs(T[iE, iE]) ** 2 + abs(T[iM, iM]) ** 2)
    sigma_mie = 2 * np.pi / k ** 2 * ssum
    err2 = abs(sigma_mie - sigma_sca) / sigma_sca
    print(f"  vs classic Mie series sigma_sca={sigma_mie:.6f}: rel diff {err2:.2e}")
    assert err2 < 1e-6


if __name__ == "__main__":
    for t in (test_maxwell_curl, test_plane_wave_expansion, test_far_field,
              test_projection_roundtrip, test_optical_theorem_sphere):
        print(t.__name__)
        t()
    print("ALL LAYER-0 TESTS PASSED")

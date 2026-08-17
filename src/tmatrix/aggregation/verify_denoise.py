"""Machine-precision gate for denoise.py's constraint maps.

Three independent checks; exit code 0 iff every one passes:

1. treams lossless Mie sphere: every singular value of S = I + 2T equals 1
   in this convention -- pins the S = I + 2T normalization used by
   enforce_passivity.  (treams' own cluster/translate path is NOT used: its
   sw.translate ufunc dispatch is broken against this environment's
   scipy/numpy, and the sphere suffices for the normalization.)

2. Saxon amplitude reciprocity at FIELD level: for real transverse
   polarizations, any reciprocal scatterer satisfies
       e2 . F(s2 <- s1; e1) = e1 . F(-s1 <- -s2; e2).
   A dense random matrix projected by enforce_reciprocity must satisfy it to
   machine precision (built from this package's independently locked
   plane_wave_coeffs / far_field_amplitude primitives -- not from the map
   under test), and the unprojected matrix must violate it (control).
   This pins the (-1)^{m+m'}, m -> -m transpose convention physically.

3. Projection algebra: idempotence, exact constraint satisfaction, and the
   passivity clip commuting with the reciprocity symmetry.

Run:  python -m tmatrix.aggregation.verify_denoise
"""
import sys

import numpy as np


def main():
    from tmatrix.aggregation.vswf import (ModeBasis, far_field_amplitude,
                                          plane_wave_coeffs)
    from tmatrix.aggregation.denoise import (apply_denoise,
                                             enforce_passivity,
                                             enforce_reciprocity,
                                             reciprocity_residual)

    ok = True

    def check(name, val, tol):
        nonlocal ok
        good = val <= tol
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: {val:.3e} (tol {tol:g})")

    # --- 1. S = I + 2T normalization against an exact treams sphere ---------
    print("1. treams lossless Mie sphere (lmax 4, x ~ 1.3):")
    import treams
    treams.config.POLTYPE = "parity"
    sph = treams.TMatrix.sphere(4, 2 * np.pi / 10.0, 2.0,
                                [treams.Material(6.0), treams.Material()])
    T = np.asarray(sph)
    sv = np.linalg.svd(np.eye(len(T)) + 2 * T, compute_uv=False)
    check("max |sv(I+2T) - 1|  (S = I + 2T normalization)",
          float(np.abs(sv - 1).max()), 1e-10)

    # --- 2. Saxon field-level reciprocity on a dense projected matrix -------
    print("2. Saxon amplitude reciprocity (field level, dense random T):")
    mb = ModeBasis.standard(4)
    rng = np.random.default_rng(11)
    X = rng.normal(size=(mb.n, mb.n)) + 1j * rng.normal(size=(mb.n, mb.n))
    Tr = enforce_reciprocity(X, mb)
    k = 2 * np.pi / 9.0

    def saxon(T, s1, e1, s2, e2):
        F = far_field_amplitude(k, T @ plane_wave_coeffs(s1, e1, mb), mb,
                                np.array([s2]))[0]
        Fr = far_field_amplitude(k, T @ plane_wave_coeffs(-s2, e2, mb), mb,
                                 np.array([-s1]))[0]
        return np.dot(e2, F), np.dot(e1, Fr)      # e real: plain dot

    def rand_dir_pol():
        th = rng.uniform(0.4, np.pi - 0.4)
        ph = rng.uniform(0.0, 2 * np.pi)
        s = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                      np.cos(th)])
        t_hat = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph),
                          -np.sin(th)])
        p_hat = np.array([-np.sin(ph), np.cos(ph), 0.0])
        al = rng.uniform(0.0, 2 * np.pi)
        return s, np.cos(al) * t_hat + np.sin(al) * p_hat

    worst_proj, least_raw = 0.0, np.inf
    for _ in range(5):
        s1, e1 = rand_dir_pol()
        s2, e2 = rand_dir_pol()
        lhs, rhs = saxon(Tr, s1, e1, s2, e2)
        worst_proj = max(worst_proj, abs(lhs - rhs) / (abs(lhs) + abs(rhs)))
        lhs0, rhs0 = saxon(X, s1, e1, s2, e2)
        least_raw = min(least_raw, abs(lhs0 - rhs0) / (abs(lhs0) + abs(rhs0)))
    check("projected T satisfies Saxon reciprocity (5 direction pairs)",
          worst_proj, 1e-11)
    ctrl = least_raw > 1e-3
    ok = ok and ctrl
    print(f"  {'PASS' if ctrl else 'FAIL'}  control: raw random T violates "
          f"it (least violation {least_raw:.3e}, must exceed 1e-3)")

    # --- 3. projection algebra ----------------------------------------------
    print("3. projection algebra:")
    Xr = enforce_reciprocity(X, mb)
    check("projection is idempotent",
          float(np.linalg.norm(enforce_reciprocity(Xr, mb) - Xr)
                / np.linalg.norm(X)), 1e-13)
    check("projected matrix satisfies the mode-level constraint exactly",
          float(reciprocity_residual(Xr, mb)), 1e-13)
    Xp = apply_denoise(X / np.linalg.norm(X), mb,
                       ["reciprocity", "passivity"])
    svp = np.linalg.svd(np.eye(len(Xp)) + 2 * Xp, compute_uv=False)
    check("after reciprocity+passivity: passive", float((svp - 1).max()),
          1e-12)
    check("after reciprocity+passivity: still reciprocal (clip commutes)",
          float(reciprocity_residual(Xp, mb)), 1e-12)

    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

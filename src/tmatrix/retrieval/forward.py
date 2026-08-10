"""Thin, fast forward-map layer over the Bloch lattice-sum cache
(retrieval deliverable: the layer fit.py / observability.py build on).

ForwardModel loads the precomputed C(k_i, k_par(theta, phi)) cache produced
by precompute_C.py -- either the consolidated results/C_bloch.npz or the
per-frequency checkpoint directory results/C_bloch/ (partially filled dirs
are supported; see available_freqs / angles_available) -- plus the mode
basis, frequencies and k values from the tmat.h5 file, and holds the
normative 17-entry angle table.

With C cached, one predict() call is a handful of 30x30 solves plus far-field
projections: microseconds-to-milliseconds per angle, so finite-difference
Jacobians over ~10-70 T-matrix unknowns are cheap.

Conventions (normative, from sparams_oblique.py / the design doc par. 2):
  - index order 0 = TE, 1 = TM in every 2x2 Jones block;
  - S[angle, block, a, b]: block 0 = S11 (reflection), 1 = S21
    (transmission); row a = receive polarization, column b = incident;
  - direction=+1 (incidence from -z travelling UP) is the DEFAULT;
    direction=-1 (incidence from +z travelling DOWN, the treams/CST
    illumination) is accepted everywhere.  C is direction-INDEPENDENT:
    k_hat and its specular reflection k_hat_r share the same in-plane
    k_par = k sin th (cos ph, sin ph), which is all the lattice sum sees,
    so one cache serves both directions.

packed-vector layout (pack_S): for a complex S-stack of shape
(n_angles, 2, 2, 2), v = S.ravel(order="C") (length 8*n_angles, angle
slowest, then block S11/S21, then receive row, then incident column), and
the packed real vector is np.concatenate([v.real, v.imag]) (length
16*n_angles: ALL real parts first, then ALL imaginary parts, same order).
With weights w (one non-negative weight per COMPLEX observable, length
8*n_angles), both the real and imaginary components are multiplied by
sqrt(w), so that

    || pack_S(S_meas, w) - pack_S(S_pred, w) ||^2  =  sum_i w_i |dS_i|^2

matches the par. 4 weighted least-squares objective for the LM driver.

Selftest:  python -m tmatrix.retrieval.forward --selftest   (needs
checkpoints for the smoke frequencies 48 and 32; see
tests/retrieval/test_precompute.py)
"""
import os
import sys


import numpy as np

from tmatrix.aggregation.tmat_io import TMatrixData
from tmatrix.retrieval.sparams_oblique import sparams_oblique
from tmatrix.retrieval import precompute_C as pc

from tmatrix.paths import AGG_RESULTS

DEFAULT_CACHE = pc.OUT_DEFAULT                # results/C_bloch (dir)
DEFAULT_TMAT = pc.DATA


class ForwardModel:
    """Cache-backed par.-2 forward map S(T0; theta_i, phi_i) per frequency.

    Parameters
    ----------
    cache_path : str
        Either the consolidated results/C_bloch.npz or the checkpoint
        directory results/C_bloch (default).  Partially filled directories
        are supported: `available_freqs` lists the frequency indices with at
        least one cached angle and `angles_available(ifreq)` the per-angle
        mask; predict() raises KeyError on a missing (ifreq, angle).
    tmat_path : str
        tmat.h5 file providing modes / frequencies / k (and the reference T).
    """

    def __init__(self, cache_path=DEFAULT_CACHE, tmat_path=DEFAULT_TMAT):
        self.data = TMatrixData(tmat_path)
        self.modes = self.data.modes
        self.nf = len(self.data.freq)
        self.k = np.array([self.data.k_at(i) for i in range(self.nf)])
        self.lam_um = self.data.wavelength_um
        self.theta_deg = pc.THETA_DEG.copy()
        self.phi_deg = pc.PHI_DEG.copy()
        self.n_angles = pc.N_ANGLES
        self.A_cell = pc.A_CELL
        self.pitch = pc.PITCH
        n = self.modes.n
        self.C = np.full((self.nf, self.n_angles, n, n), np.nan,
                         dtype=complex)
        self.have = np.zeros((self.nf, self.n_angles), dtype=bool)
        self.cache_path = cache_path
        if os.path.isdir(cache_path):
            for i in range(self.nf):
                d = pc.load_checkpoint(pc.checkpoint_path(cache_path, i),
                                       k=self.k[i], n=n)
                if d is None:
                    continue
                self.C[i] = d["C"]
                self.have[i] = d["have"]
        elif os.path.isfile(cache_path):
            with np.load(cache_path) as z:
                if not (np.array_equal(z["theta_deg"], self.theta_deg)
                        and np.array_equal(z["phi_deg"], self.phi_deg)):
                    raise ValueError("cache angle table does not match the "
                                     "normative 17-angle set")
                if z["C"].shape != self.C.shape:
                    raise ValueError(f"cache C shape {z['C'].shape} != "
                                     f"expected {self.C.shape}")
                self.C[:] = z["C"]
                self.have[:] = True
        else:
            raise FileNotFoundError(cache_path)

    # ------------------------------------------------------------ inventory
    @property
    def available_freqs(self):
        """Sorted frequency indices with at least one cached angle."""
        return [int(i) for i in np.nonzero(self.have.any(axis=1))[0]]

    def angles_available(self, ifreq):
        """Boolean mask (n_angles,) of cached angles at frequency ifreq."""
        return self.have[ifreq].copy()

    def angle_index(self, theta_deg, phi_deg):
        """Index into the normative table for an exact (theta, phi) pair."""
        hit = np.nonzero((np.abs(self.theta_deg - theta_deg) < 1e-9)
                         & (np.abs(self.phi_deg - phi_deg) < 1e-9))[0]
        if len(hit) == 0:
            raise KeyError(f"({theta_deg}, {phi_deg}) not in the angle table")
        return int(hit[0])

    # ------------------------------------------------------------ forward map
    def predict(self, T0, ifreq, angle_indices=None, direction=+1):
        """S(T0) at cached angles of one frequency.

        Returns complex array (n_angles_sel, 2, 2, 2) indexed
        [angle, block (0=S11, 1=S21), receive pol a, incident pol b],
        0 = TE / 1 = TM (normative).  direction=+1 by default; -1 accepted
        (C is direction-independent -- same k_par for k_hat and k_hat_r).
        The per-angle loop is deliberately a plain loop for clarity; with C
        cached each iteration is a 30x30 solve + far-field projection.
        """
        if angle_indices is None:
            angle_indices = list(range(self.n_angles))
        k = self.k[ifreq]
        out = np.empty((len(angle_indices), 2, 2, 2), dtype=complex)
        for j, ia in enumerate(angle_indices):
            if not self.have[ifreq, ia]:
                raise KeyError(f"C not cached for ifreq={ifreq}, "
                               f"angle {ia} ({self.theta_deg[ia]:g}, "
                               f"{self.phi_deg[ia]:g})")
            res = sparams_oblique(
                k, self.A_cell, self.modes, T0, self.C[ifreq, ia],
                np.deg2rad(self.theta_deg[ia]),
                np.deg2rad(self.phi_deg[ia]), direction)
            out[j, 0] = res["S11"]
            out[j, 1] = res["S21"]
        return out

    # ------------------------------------------------------------ packing
    @staticmethod
    def pack_S(S_stack, weights=None):
        """Complex S-stack (..., 2, 2, 2) -> 1-D real vector.

        Layout: v = S_stack.ravel(order='C'); packed =
        [v.real..., v.imag...] (all real parts first, then all imaginary
        parts, in identical C order).  weights: optional per-COMPLEX-entry
        non-negative array broadcastable to S_stack.shape; both components
        are scaled by sqrt(weights) so squared 2-norms of packed differences
        equal sum_i w_i |dS_i|^2.
        """
        v = np.asarray(S_stack, dtype=complex).ravel(order="C")
        if weights is not None:
            sw = np.sqrt(np.broadcast_to(
                np.asarray(weights, dtype=float),
                np.asarray(S_stack).shape)).ravel(order="C")
            v = v * sw
        return np.concatenate([v.real, v.imag])

    @staticmethod
    def unpack_S(packed, n_angles):
        """Inverse of pack_S (unweighted): 1-D real (16*n_angles,) ->
        complex (n_angles, 2, 2, 2)."""
        m = 8 * n_angles
        v = packed[:m] + 1j * packed[m:]
        return v.reshape(n_angles, 2, 2, 2)

    def predict_packed(self, T0, ifreq, angle_indices=None, weights=None,
                       direction=+1):
        """pack_S(predict(...)) for the LM driver.  weights: per-complex-
        observable, broadcastable to (n_angles_sel, 2, 2, 2) (e.g. shape
        (n_angles_sel, 1, 1, 1) for per-angle 1/sigma^2 weights)."""
        S = self.predict(T0, ifreq, angle_indices, direction)
        return self.pack_S(S, weights)


# ---------------------------------------------------------------- selftest
def _selftest():
    """Smoke tests at the smoke frequencies (checkpoints 48 and 32):
    (1) predict at (0,0) reproduces the theta->0 (e)-mapping of run_demo's
        stored x-polarized S11/S21 (periodic_results.npz) to <= 1e-10:
        stored data has incidence e = x_hat = e_TM(0,0,+1), so column b=TM:
          S21[TE,TM] = +S21x   S21[TM,TM] = +S21   (stored co/cross)
          S11[TE,TM] = +S11x   S11[TM,TM] = -S11   (S11 TM-row sign)
    (2) predict at (30,22.5) equals a direct sparams_oblique call.
    """
    results = []

    def record(name, ok, detail):
        results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}",
              flush=True)

    fm = ForwardModel()
    print(f"cache: {fm.cache_path}; available freqs: {fm.available_freqs}",
          flush=True)
    stored = np.load(os.path.join(AGG_RESULTS, "periodic_results.npz"))
    for ifreq in (48, 32):
        T0 = fm.data.T[ifreq]
        S = fm.predict(T0, ifreq, angle_indices=[0], direction=+1)[0]
        checks = [
            ("S21[TE,TM] = +S21x", S[1, 0, 1], stored["S21x"][ifreq]),
            ("S21[TM,TM] = +S21 ", S[1, 1, 1], stored["S21"][ifreq]),
            ("S11[TE,TM] = +S11x", S[0, 0, 1], stored["S11x"][ifreq]),
            ("S11[TM,TM] = -S11 ", S[0, 1, 1], -stored["S11"][ifreq]),
        ]
        worst = max(abs(got - want) for _, got, want in checks)
        for nm, got, want in checks:
            print(f"      ifreq={ifreq} {nm}: got {got:+.10f}, "
                  f"want {want:+.10f}, |d|={abs(got - want):.2e}",
                  flush=True)
        record(f"(fwd-1) (0,0) vs stored run_demo S (theta->0 mapping, "
               f"ifreq={ifreq}, lam={fm.lam_um[ifreq]:.2f} um)",
               worst <= 1e-10, f"max|dS| = {worst:.3e} (gate 1e-10; "
               f"4 complex entries, x-pol = TM column)")

        ia = fm.angle_index(30.0, 22.5)
        S_fm = fm.predict(T0, ifreq, angle_indices=[ia], direction=+1)[0]
        res = sparams_oblique(fm.k[ifreq], fm.A_cell, fm.modes, T0,
                              fm.C[ifreq, ia], np.deg2rad(30.0),
                              np.deg2rad(22.5), +1)
        d = max(np.abs(S_fm[0] - res["S11"]).max(),
                np.abs(S_fm[1] - res["S21"]).max())
        record(f"(fwd-2) (30,22.5) predict == direct sparams_oblique "
               f"(ifreq={ifreq})", d == 0.0,
               f"max|dS| = {d:.3e} (identical code path; expect exactly 0)")

    # pack/unpack round trip + weight semantics
    rng = np.random.default_rng(0)
    S = rng.normal(size=(3, 2, 2, 2)) + 1j * rng.normal(size=(3, 2, 2, 2))
    w = rng.uniform(0.5, 2.0, size=(3, 2, 2, 2))
    rt = np.abs(ForwardModel.unpack_S(ForwardModel.pack_S(S), 3) - S).max()
    lhs = np.sum(ForwardModel.pack_S(S, w) ** 2)
    rhs = np.sum(w * np.abs(S) ** 2)
    record("(fwd-3) pack_S round trip + weighted-norm identity",
           rt == 0.0 and abs(lhs - rhs) <= 1e-12 * rhs,
           f"round-trip max|d| = {rt:.1e}; "
           f"|sum w|S|^2 mismatch| = {abs(lhs - rhs):.2e}")

    # timing: predict all 17 angles at ifreq 48
    import time
    T0 = fm.data.T[48]
    t0 = time.time()
    nrep = 20
    for _ in range(nrep):
        fm.predict(T0, 48)
    dt = (time.time() - t0) / nrep
    print(f"  timing: predict(all 17 angles) = {dt * 1e3:.1f} ms "
          f"({dt / 17 * 1e3:.2f} ms/angle)", flush=True)

    nfail = sum(1 for _, ok, _ in results if not ok)
    print("\n=== SUMMARY (forward --selftest) ===")
    for name, ok, _ in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("ALL TESTS PASSED" if nfail == 0 else f"{nfail} TEST(S) FAILED")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)

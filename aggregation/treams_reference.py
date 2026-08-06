"""Independent reference: S-parameters of a square-lattice (pitch 2 um) metasurface
computed with treams (v0.4.7) from a tmat.h5 T-matrix file.

Input : D:/Claude/T matrix/test/single/saw_gold_wl15p0025um.tmat.h5
        (49 frequencies, lmax 3, parity basis, embedding vacuum,
         tmat.h5 convention: exp(-i omega t) time dependence)
Output: D:/Claude/T matrix/aggregation/results/treams_reference.npz
        arrays lam_um (49,), S11 (49,) complex, S21 (49,) complex, notes (str)

Definitions (documented also in the saved 'notes' string):
- Normal incidence from the +z side, i.e. the incident plane wave propagates in -z
  ("down" in treams' convention; "up" = +z propagation).
- Incident field: E_inc = x_hat * exp(-i k z) (x-polarized), phase reference z = 0
  (the plane of the particle centers).
- S21 = complex co-polarized (x) amplitude of the total transmitted 0th-order wave
  below the array (includes the direct/unity term: treams' SMatrices.from_array
  builds S[1,1] = 1 + P_down A_down).
- S11 = complex co-polarized (x) amplitude of the reflected 0th-order wave above
  the array, referenced to the same z = 0 plane.

treams API path used:
    treams.io.load_hdf5(file, lunit="um")           -> 49 TMatrix (parity poltype)
    treams.Lattice.square(2.0)                       -> square lattice, pitch 2 um
    tm.latticeinteraction.solve(lattice, [0, 0])     -> lattice-coupled T-matrix
                                                        (Ewald-summed multiple
                                                        scattering, kpar = 0)
    treams.PlaneWaveBasisByComp.diffr_orders([0,0], lattice, bmax)
                                                     -> 0th-order plane-wave basis
                                                        (lambda = 8..20 um >> 2 um
                                                        pitch, only the 0th order
                                                        propagates)
    treams.SMatrices.from_array(tm_lat, basis)       -> slab S-matrix
    treams.efield([0,0,0], basis=..., modetype=...)  -> per-mode E field at origin,
                                                        used to map the parity
                                                        (TM=electric pol=1 /
                                                        TE=magnetic pol=0) modes to
                                                        linear x polarization.

Windows/numpy2 workarounds (do not change the physics):
- treams.io.FREQUENCIES lacks the plain "Hz" key -> added ("Hz": 1.0).
- The compiled gufuncs (treams.sw / .pw / .cw / .lattice) declare integer inputs
  as C long = int32 on Windows, while numpy 2.x defaults to int64 -> all integer
  ufunc arguments are cast to int32 (values are small quantum numbers, lossless),
  and real/complex arguments to the complex variant of the ufunc signature.
"""

import os

import numpy as np
import treams
import treams.io
from treams import sw, pw, cw
import treams.lattice as la

HERE = os.path.dirname(os.path.abspath(__file__))
TMAT_FILE = r"D:/Claude/T matrix/test/single/saw_gold_wl15p0025um.tmat.h5"
OUT_FILE = os.path.join(HERE, "results", "treams_reference.npz")
PITCH_UM = 2.0

# ----------------------------------------------------------------------------
# Windows / numpy>=2 compatibility patches (casting only, no physics change)
# ----------------------------------------------------------------------------
treams.io.FREQUENCIES.setdefault("Hz", 1.0)


def _cast_wrap(uf):
    """Wrap a compiled gufunc: cast int args to int32, floats to the complex
    signature, so numpy 2.x on Windows (default int64) can call it."""
    sigs = uf.types
    ins = sigs[-1].split("->")[0]  # last signature = complex variant

    def f(*args, **kw):
        cast = []
        for a, t in zip(args, ins):
            if t == "l":
                cast.append(np.asarray(a, np.int32))
            elif t == "D":
                cast.append(np.asarray(a, np.complex128))
            elif t == "d":
                cast.append(np.asarray(a, np.float64))
            else:
                cast.append(a)
        cast.extend(args[len(ins):])
        return uf(*cast, **kw)

    f.types = sigs
    return f


for _mod in (sw, pw, cw, la):
    for _n in dir(_mod):
        _uf = getattr(_mod, _n)
        if isinstance(_uf, np.ufunc) and any(
            "l" in s.split("->")[0] for s in _uf.types
        ):
            setattr(_mod, _n, _cast_wrap(_uf))

# ----------------------------------------------------------------------------
# Load T-matrices (lunit um -> k0 in um^-1, lattice lengths in um)
# ----------------------------------------------------------------------------
tms = treams.io.load_hdf5(TMAT_FILE, lunit="um")
nfreq = len(tms)
assert nfreq == 49, f"expected 49 frequencies, got {nfreq}"

lattice = treams.Lattice.square(PITCH_UM)
kpar = [0.0, 0.0]  # normal incidence

lam_um = np.array([2 * np.pi / tm.k0 for tm in tms])
S11 = np.empty(nfreq, complex)
S21 = np.empty(nfreq, complex)
S11_cross = np.empty(nfreq, complex)  # y-pol content, diagnostics only
S21_cross = np.empty(nfreq, complex)

for i, tm in enumerate(tms):
    assert tm.poltype == "parity"
    k0 = tm.k0

    # Lattice-coupled effective T-matrix (multiple scattering in the array)
    tm_lat = tm.latticeinteraction.solve(lattice, kpar)

    # Plane-wave basis: only 0th diffraction order needed (lambda >> pitch);
    # a single-layer S-matrix does not mix diffraction channels afterwards.
    basis = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, 1e-9)
    # basis order at kpar=0: [(kx=0, ky=0, pol=1=TM/electric), (0, 0, pol=0=TE/magnetic)]

    smat = treams.SMatrices.from_array(tm_lat, basis)

    # Per-mode E field (Cartesian) at the origin. modetype fixes the sign of kz.
    # Rows: (Ex, Ey, Ez); columns: basis modes.
    M_up = np.asarray(
        treams.efield([0, 0, 0.0], basis=basis, k0=k0, material=tm.material,
                      poltype="parity", modetype="up")
    )
    M_down = np.asarray(
        treams.efield([0, 0, 0.0], basis=basis, k0=k0, material=tm.material,
                      poltype="parity", modetype="down")
    )

    # Incident wave: down-going (from +z side), E = x_hat at z=0.
    # Solve M_down @ c = [1, 0, 0] for the mode coefficients c.
    c_inc, *_ = np.linalg.lstsq(M_down, np.array([1.0, 0.0, 0.0]), rcond=None)
    resid = np.linalg.norm(M_down @ c_inc - np.array([1.0, 0.0, 0.0]))
    assert resid < 1e-12, f"x-pol decomposition failed at index {i}: {resid}"

    # S-matrix block convention (treams SMatrices):
    #   field_up   = S[0,0] @ illu_up + S[0,1] @ illu_down
    #   field_down = S[1,0] @ illu_up + S[1,1] @ illu_down
    # Down illumination -> transmission below = S[1,1] @ c, reflection above = S[0,1] @ c
    b_t = np.asarray(smat[1, 1]) @ c_inc
    b_r = np.asarray(smat[0, 1]) @ c_inc

    E_t = M_down @ b_t  # transmitted (down-going) field at origin
    E_r = M_up @ b_r    # reflected (up-going) field at origin

    S21[i] = E_t[0]        # co-pol (x) transmission, includes direct unity term
    S11[i] = E_r[0]        # co-pol (x) reflection
    S21_cross[i] = E_t[1]  # cross-pol (y), diagnostics
    S11_cross[i] = E_r[1]

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
notes = (
    "treams v0.4.7 independent reference for square-lattice metasurface, pitch 2 um, "
    "vacuum embedding. T-matrix: saw_gold_wl15p0025um.tmat.h5 (49 freqs, lmax 3, "
    "parity basis, exp(-i omega t)). Normal incidence from the +z side (propagation "
    "along -z = treams modetype 'down'), incident E = x_hat exp(-i k z), phase "
    "reference z=0 (particle-center plane). S21 = complex co-pol (x) amplitude of the "
    "total transmitted 0th-order wave (includes direct unity term via "
    "SMatrices.from_array S[1,1] = 1 + P_d A_d). S11 = complex co-pol (x) amplitude "
    "of the reflected 0th-order wave, same phase plane. Linear x-pol was built from "
    "the parity plane-wave modes using treams.efield at the origin: at kpar=0 the "
    "pol=1 (TM/'electric') mode has E = +x_hat (down) / -x_hat (up) and the pol=0 "
    "(TE/'magnetic') mode has E = -i y_hat, so incidence = [1, 0] in the (TM, TE) "
    "mode order and reflected co-pol picks up a sign from the up-going TM mode. "
    "API path: treams.io.load_hdf5(lunit='um') -> tm.latticeinteraction.solve("
    "Lattice.square(2), [0,0]) -> PlaneWaveBasisByComp.diffr_orders([0,0], lattice, "
    "bmax~0) -> SMatrices.from_array. Windows/numpy2 patches: FREQUENCIES['Hz']=1; "
    "compiled gufunc integer args cast int64->int32 (casting only, no physics). "
    "Cross-pol amplitudes stored as S11_cross/S21_cross (diagnostics)."
)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
np.savez(
    OUT_FILE,
    lam_um=lam_um,
    S11=S11,
    S21=S21,
    S11_cross=S11_cross,
    S21_cross=S21_cross,
    notes=notes,
)
print(f"saved {OUT_FILE}")

# ----------------------------------------------------------------------------
# Table (every 4th frequency)
# ----------------------------------------------------------------------------
print(f"\n{'lam_um':>8} {'|S11|':>10} {'|S21|':>10} {'1-|S11|^2-|S21|^2':>18}")
for i in range(0, nfreq, 4):
    a = 1 - abs(S11[i]) ** 2 - abs(S21[i]) ** 2
    print(f"{lam_um[i]:8.4f} {abs(S11[i]):10.6f} {abs(S21[i]):10.6f} {a:18.6f}")
print(f"\nmax |cross-pol|: S11 {abs(S11_cross).max():.3e}, "
      f"S21 {abs(S21_cross).max():.3e}")

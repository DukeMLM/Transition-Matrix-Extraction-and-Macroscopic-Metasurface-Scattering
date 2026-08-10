"""External validation of the collective lattice shift seen in
test_feature_fidelity.py: the same synthetic Lorentzian electric-dipole
T-matrix is pushed through treams' Ewald-summed lattice coupling.

If treams also moves the collective resonance from the single-particle
15 um to ~18 um at pitch 2 um, the shift is real multiple-scattering
physics, not an artifact of our lattice sum.
"""
import numpy as np
import treams

from tmatrix import treams_compat
from tmatrix.paths import AGG_RESULTS, DEMO_TMAT
from tmatrix.units import C0_M_S

treams_compat.apply()

TMAT_FILE = DEMO_TMAT
OUT_FILE = AGG_RESULTS / "treams_synthetic.npz"
PITCH_UM = 2.0


def main():
    tms = treams.io.load_hdf5(str(TMAT_FILE), lunit="um")
    lattice = treams.Lattice.square(PITCH_UM)
    kpar = [0.0, 0.0]

    lam = np.array([2 * np.pi / tm.k0 for tm in tms])
    w = 2 * np.pi * C0_M_S / (lam * 1e-6)
    w0 = 2 * np.pi * C0_M_S / 15e-6
    gr, gnr = 0.02 * w0, 0.005 * w0
    t_lor = -gr / (1j * (w0 - w) + gr + gnr)

    basis_sw = tms[0].basis
    l_arr = np.array([b[1] for b in basis_sw])
    m_arr = np.array([b[2] for b in basis_sw])
    p_arr = np.array([b[3] for b in basis_sw])

    for pol_choice in (1, 0):
        sel = (l_arr == 1) & (np.abs(m_arr) == 1) & (p_arr == pol_choice)
        S21_mag = []
        for i, tm in enumerate(tms):
            data = np.zeros((len(l_arr), len(l_arr)), dtype=complex)
            data[np.where(sel)[0], np.where(sel)[0]] = t_lor[i]
            tm_s = treams.TMatrix(data, k0=tm.k0, basis=basis_sw,
                                  material=tm.material, poltype="parity")
            tm_lat = tm_s.latticeinteraction.solve(lattice, kpar)
            basis = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice,
                                                            1e-9)
            smat = treams.SMatrices.from_array(tm_lat, basis)
            M_down = np.asarray(treams.efield(
                [0, 0, 0.0], basis=basis, k0=tm.k0, material=tm.material,
                poltype="parity", modetype="down"))
            c_inc, *_ = np.linalg.lstsq(M_down, np.array([1.0, 0, 0]),
                                        rcond=None)
            b_t = np.asarray(smat[1, 1]) @ c_inc
            S21_mag.append(abs((M_down @ b_t)[0]))
        S21_mag = np.array(S21_mag)
        i_min = int(np.argmin(S21_mag))
        tag = "electric(pol=1)" if pol_choice == 1 else "magnetic(pol=0)"
        print(f"treams synthetic dipole [{tag}]: min |S21| = "
              f"{S21_mag[i_min]:.3f} at lam = {lam[i_min]:.2f} um")
        if pol_choice == 1:
            OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(OUT_FILE, lam_um=lam, S21_mag=S21_mag)


if __name__ == "__main__":
    main()

"""Reader for tmat.h5 files (the unified T-matrix data format)."""
import h5py
import numpy as np

from tmatrix.aggregation.vswf import ModeBasis, ELECTRIC, MAGNETIC
from tmatrix.units import C0_M_S as C0



class TMatrixData:
    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.freq = f["frequency"][:]                    # Hz
            self.T = f["tmatrix"][:]                         # (nf, n, n)
            l = f["modes/l"][:]
            m = f["modes/m"][:]
            pol_raw = f["modes/polarization"][:]
            pol = [p.decode() if isinstance(p, bytes) else str(p)
                   for p in pol_raw]
            pol = [ELECTRIC if p.lower().startswith("e") else MAGNETIC
                   for p in pol]
            self.modes = ModeBasis(l, m, pol)
            self.name = f.attrs.get("name", "")
            self.description = f.attrs.get("description", "")
            emb = f["embedding"]
            self.eps_emb = complex(emb["relative_permittivity"][()])
            self.mu_emb = complex(emb["relative_permeability"][()])

    @property
    def wavelength_um(self):
        return C0 / self.freq * 1e6

    def k_at(self, i):
        """Embedding-medium wavenumber in rad/um at frequency index i."""
        n_emb = np.sqrt(self.eps_emb * self.mu_emb)
        if abs(n_emb.imag) > 1e-9 * abs(n_emb.real):
            raise NotImplementedError(
                "lossy embedding media are not supported by the real-k "
                "aggregation pipeline")
        return 2 * np.pi / self.wavelength_um[i] * n_emb.real

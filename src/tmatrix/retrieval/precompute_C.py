"""Checkpointed, streaming precomputation of the 17-angle Bloch lattice-sum
cache C(k_i, k_par(theta, phi)) (retrieval deliverable: results/C_bloch.npz).

For each of the 49 frequencies of test/single/saw_gold_wl15p0025um.tmat.h5
this script computes the Bloch lattice sums

    C(k, k_par) = sum_{R != 0} A(R) e^{+i k_par . R}

at the 17 normative (theta, phi) pairs (campaign 13 + treams-validation 4,
see ANGLES_DEG below), with the per-theta taper scaling of
INVERSE_TMATRIX_FROM_FLOQUET.md par. 2:

    s(theta)   = 1 / (1 - sin theta)
    kRc(theta) = (10, 14, 20) * s(theta)
    r_max(theta) = 3.5 * max(kRc(theta)) / k

STREAMING single-pass design (the theta=60 superset at lam=20 um is ~152k
shells ~ 2.2 GB if materialized -- it never is):  per frequency the shells are
enumerated ONCE up to the superset r_max (largest requested theta), the
translation matrices are built chunk by chunk (translate.translation_shells,
chunk ~ 48), and each chunk updates n_angle x 3 accumulators (angle x taper)
using Bloch-phased per-site sums (the site_phase_sums arithmetic of
bloch_lattice, vectorized over angles).  For each (angle, taper) BOTH the
taper weight w(r; Rc) (with translate's w < 1e-16 -> 0 skip) AND the exact
cutoff mask r <= r_max(theta) are applied; the mask replicates
bloch_lattice.slice_shells' float rule (keep shells with
rint((r/pitch)^2) * pitch^2 <= r_max(theta)^2), so each accumulated sum
equals a fresh lattice_sum_C_bloch(kRc(theta)) call up to floating-point
regrouping (verified <= 1e-13 rel in test_precompute.py).  The three taper
sums per angle are Richardson-extrapolated at the end with the same
Lagrange-in-1/Rc^2 code as translate/bloch_lattice.  One RegularProjector is
reused throughout a process.

Checkpointing: one npz per frequency index under results/C_bloch/
(freq_XX.npz), storing C (17, n, n) with a per-angle `have` mask, the angle
table, kRc scalings, k and meta.  On start, frequency indices whose file
already contains all requested angles are skipped.  Runs restricted with
--angles merge exactly into existing checkpoints: an angle's sum depends only
on shells with r <= r_max(theta), which any superset covering that angle
enumerates identically (bloch_lattice.slice_shells logic), so per-angle
results are independent of which other angles were requested (up to
float-regrouping noise of the chunk boundaries; a given command line is
deterministic and byte-reproducible).

Usage
-----
  python precompute_C.py                       # all 49 freqs, all 17 angles
  python precompute_C.py --freqs 48,32         # explicit frequency indices
  python precompute_C.py --stride 4            # indices 0,4,...,48
  python precompute_C.py --angles 0,13-16      # angle subset (e.g. treams set)
  python precompute_C.py --workers 4           # multiprocessing over freqs
  python precompute_C.py --consolidate         # write results/C_bloch.npz
  python precompute_C.py --dry-run             # shell counts + projected cost

Progress lines (per-freq shell counts and timings) go to stdout (flushed) and
are appended to results/C_bloch/progress.log.
"""
import argparse
import os
import sys
import time


import numpy as np

from tmatrix.aggregation.tmat_io import TMatrixData
from tmatrix.aggregation.vswf import RegularProjector
from tmatrix.aggregation.translate import make_quad, square_lattice_shells, translation_shells

from tmatrix.paths import BENCHMARK_SINGLE, RETRIEVAL_RESULTS
from tmatrix.numerics import richardson

DATA = os.path.join(BENCHMARK_SINGLE, "saw_gold_wl15p0025um.tmat.h5")
OUT_DEFAULT = os.path.join(RETRIEVAL_RESULTS, "C_bloch")
CONSOLIDATED = os.path.join(RETRIEVAL_RESULTS, "C_bloch.npz")

PITCH = 2.0                       # um (normative)
A_CELL = PITCH ** 2               # 4.0 um^2
R0 = 0.8                          # projection sphere radius, um
KRC_BASE = np.array([10.0, 14.0, 20.0])
RMAX_FACTOR = 3.5
QUAD_SPEC = (16, 32)              # make_quad(16, 32) (normative)

# Normative angle set (degrees): campaign 13 first, then treams-validation 4.
ANGLES_DEG = np.array([
    (0.0, 0.0),
    (15.0, 0.0), (15.0, 22.5), (15.0, 45.0),
    (30.0, 0.0), (30.0, 22.5), (30.0, 45.0),
    (45.0, 0.0), (45.0, 22.5), (45.0, 45.0),
    (60.0, 0.0), (60.0, 22.5), (60.0, 45.0),
    (20.0, 0.0), (20.0, 45.0),
    (40.0, 0.0), (40.0, 45.0),
])
N_ANGLES = len(ANGLES_DEG)        # 17

THETA_DEG = ANGLES_DEG[:, 0]
PHI_DEG = ANGLES_DEG[:, 1]
S_THETA = 1.0 / (1.0 - np.sin(np.deg2rad(THETA_DEG)))       # taper scaling
KRC_TABLE = KRC_BASE[None, :] * S_THETA[:, None]            # (17, 3)


def kpar_table(k):
    """(17, 2) in-plane Bloch vectors k sin th (cos ph, sin ph)."""
    th = np.deg2rad(THETA_DEG)
    ph = np.deg2rad(PHI_DEG)
    return k * np.sin(th)[:, None] * np.stack(
        [np.cos(ph), np.sin(ph)], axis=1)


def r_max_table(k):
    """(17,) per-angle shell cutoffs 3.5 * max(kRc(theta)) / k."""
    return RMAX_FACTOR * KRC_TABLE.max(axis=1) / k


def checkpoint_path(out_dir, ifreq):
    return os.path.join(out_dir, f"freq_{ifreq:02d}.npz")




def log_line(out_dir, msg):
    print(msg, flush=True)
    try:
        with open(os.path.join(out_dir, "progress.log"), "a",
                  encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


def load_checkpoint(path, k=None, n=None):
    """Load and validate a checkpoint; returns dict of arrays or None."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            d = {key: z[key] for key in z.files}
    except Exception as e:                                    # noqa: BLE001
        print(f"  warning: unreadable checkpoint {path}: {e}", flush=True)
        return None
    need = ("C", "have", "theta_deg", "phi_deg", "kpar", "kRc", "r_max", "k")
    if any(key not in d for key in need):
        return None
    if d["C"].shape[0] != N_ANGLES or d["have"].shape != (N_ANGLES,):
        return None
    if not (np.array_equal(d["theta_deg"], THETA_DEG)
            and np.array_equal(d["phi_deg"], PHI_DEG)):
        return None
    if k is not None and abs(float(d["k"]) - k) > 1e-12 * k:
        return None
    if n is not None and d["C"].shape[1:] != (n, n):
        return None
    return d


def save_checkpoint(path, payload):
    """Atomic npz write (tmp + replace)."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def stream_bloch_sums(k, modes, proj, quad, angle_idx, chunk=48,
                      progress=None):
    """Single streaming pass: returns (C (n_req, n, n), stats dict).

    C[i] equals lattice_sum_C_bloch(k, PITCH, modes, R0, quad,
    kpar_table(k)[angle_idx[i]], kRc=KRC_TABLE[angle_idx[i]]) up to
    floating-point regrouping (same shells, same taper weights, same
    Richardson extrapolation; only the summation chunking differs).
    """
    req = np.asarray(angle_idx, dtype=int)
    n = modes.n
    kpar = kpar_table(k)[req]                     # (nreq, 2)
    Rcs = KRC_TABLE[req] / k                      # (nreq, 3) lengths
    r_max = r_max_table(k)[req]                   # (nreq,)
    r_max_sup = float(r_max.max())

    t0 = time.time()
    radii, site_ang = square_lattice_shells(PITCH, r_max_sup)
    t_enum = time.time() - t0
    nshell = len(radii)
    nsites = sum(len(np.atleast_1d(a)) for a in site_ang)

    # exact slice_shells cutoff rule per angle
    r2int = np.rint((radii / PITCH) ** 2).astype(np.int64)
    keep = r2int[None, :] * PITCH ** 2 <= r_max[:, None] ** 2   # (nreq, nshell)

    dm = modes.m[None, :] - modes.m[:, None]      # (mu, nu): m_nu - m_mu
    dms = np.unique(dm)
    dm_idx = np.searchsorted(dms, dm)
    kx, ky = kpar[:, 0], kpar[:, 1]

    acc = np.zeros((len(req), 3, n, n), dtype=complex)
    t_proj = 0.0
    t_asm = 0.0
    t_start = time.time()
    for i0 in range(0, nshell, chunk):
        i1 = min(i0 + chunk, nshell)
        rs = radii[i0:i1]
        t0 = time.time()
        A_c = translation_shells(k, rs, modes, R0, quad, projector=proj,
                                 chunk=chunk)
        t_proj += time.time() - t0
        t0 = time.time()
        # per-shell Bloch phase sums, all requested angles at once:
        # Q[s, a, d] = sum_{R in shell s} e^{i dms[d] phi_R} e^{+i kpar_a . R}
        Q = np.empty((i1 - i0, len(req), len(dms)), dtype=complex)
        for j in range(i0, i1):
            phis = np.atleast_1d(site_ang[j])
            r = radii[j]
            B = np.exp(1j * r * (np.cos(phis)[:, None] * kx[None, :]
                                 + np.sin(phis)[:, None] * ky[None, :]))
            E = np.exp(1j * np.outer(phis, dms))
            Q[j - i0] = (B[:, :, None] * E[:, None, :]).sum(axis=0)
        for a_i in range(len(req)):
            # taper weights (translate's w<1e-16 skip) AND exact cutoff mask
            w = np.exp(-(rs[:, None] / Rcs[a_i][None, :]) ** 2)   # (nc, 3)
            w[w < 1e-16] = 0.0
            w[~keep[a_i, i0:i1], :] = 0.0
            G = A_c * Q[:, a_i, dm_idx]                           # (nc, n, n)
            acc[a_i] += np.einsum("sj,smn->jmn", w, G)
        t_asm += time.time() - t0
        if progress is not None and (i1 == nshell or (i0 // chunk) % 400 == 0):
            el = time.time() - t_start
            eta = el / max(i1, 1) * (nshell - i1)
            progress(f"    shells {i1}/{nshell}  elapsed {el:6.1f} s  "
                     f"eta {eta:6.1f} s")

    C = np.empty((len(req), n, n), dtype=complex)
    for a_i in range(len(req)):
        C[a_i] = richardson(acc[a_i], Rcs[a_i])
    stats = dict(nshell=nshell, nsites=nsites, r_max_sup=r_max_sup,
                 t_enum=t_enum, t_proj=t_proj, t_asm=t_asm)
    return C, stats


def process_freq(data, ifreq, angle_idx, proj, quad, out_dir, chunk=48,
                 verbose=True):
    """Compute/merge the checkpoint for one frequency index. Returns a
    summary string ('' if skipped)."""
    k = data.k_at(ifreq)
    modes = data.modes
    n = modes.n
    path = checkpoint_path(out_dir, ifreq)
    existing = load_checkpoint(path, k=k, n=n)
    req_all = sorted(set(int(a) for a in angle_idx))
    if existing is not None:
        missing = [a for a in req_all if not existing["have"][a]]
    else:
        missing = req_all
    lam = data.wavelength_um[ifreq]
    if not missing:
        msg = (f"freq {ifreq:02d} lam={lam:6.2f} um: checkpoint complete "
               f"({int(existing['have'].sum())}/{N_ANGLES} angles) -- skip")
        log_line(out_dir, msg)
        return msg
    if verbose:
        log_line(out_dir,
                 f"freq {ifreq:02d} lam={lam:6.2f} um k={k:.6f}: computing "
                 f"{len(missing)} angle(s) {missing} "
                 f"(superset r_max={r_max_table(k)[missing].max():.1f} um)")
    t0 = time.time()
    progress = (lambda m: print(m, flush=True)) if verbose else None
    C_new, st = stream_bloch_sums(k, modes, proj, quad, missing, chunk=chunk,
                                  progress=progress)
    elapsed = time.time() - t0

    if existing is not None:
        C = existing["C"]
        have = existing["have"].copy()
    else:
        C = np.full((N_ANGLES, n, n), np.nan, dtype=complex)
        have = np.zeros(N_ANGLES, dtype=bool)
    for j, a in enumerate(missing):
        C[a] = C_new[j]
        have[a] = True
    payload = dict(
        C=C, have=have,
        theta_deg=THETA_DEG, phi_deg=PHI_DEG,
        kpar=kpar_table(k), kRc=KRC_TABLE, s_theta=S_THETA,
        kRc_base=KRC_BASE, r_max=r_max_table(k),
        k=k, lam_um=lam, freq_hz=data.freq[ifreq], ifreq=ifreq,
        pitch=PITCH, r0=R0, quad_n_theta=QUAD_SPEC[0],
        quad_n_phi=QUAD_SPEC[1], rmax_factor=RMAX_FACTOR,
        nshell_last_run=st["nshell"], nsites_last_run=st["nsites"],
        r_max_sup_last_run=st["r_max_sup"], elapsed_last_run_s=elapsed,
    )
    save_checkpoint(path, payload)
    msg = (f"freq {ifreq:02d} lam={lam:6.2f} um k={k:.6f}: "
           f"{len(missing)} angle(s), nshell={st['nshell']}, "
           f"nsites={st['nsites']}, enum={st['t_enum']:.1f} s, "
           f"proj={st['t_proj']:.1f} s, asm={st['t_asm']:.1f} s, "
           f"total={elapsed:.1f} s -> {os.path.basename(path)} "
           f"[{int(have.sum())}/{N_ANGLES} angles]")
    log_line(out_dir, msg)
    return msg


def _worker(job):
    """Multiprocessing worker: fully independent per frequency index."""
    ifreq, angle_idx, out_dir, chunk = job
    data = TMatrixData(DATA)
    quad = make_quad(*QUAD_SPEC)
    proj = RegularProjector(data.modes, quad)
    return process_freq(data, ifreq, angle_idx, proj, quad, out_dir,
                        chunk=chunk, verbose=False)


def consolidate(out_dir):
    """Once all 49 checkpoints hold all 17 angles, write the doc-named
    consolidated cache results/C_bloch.npz."""
    data = TMatrixData(DATA)
    n = data.modes.n
    nf = len(data.freq)
    C = np.empty((nf, N_ANGLES, n, n), dtype=complex)
    kpar = np.empty((nf, N_ANGLES, 2))
    ks = np.empty(nf)
    missing = []
    for i in range(nf):
        d = load_checkpoint(checkpoint_path(out_dir, i), k=data.k_at(i), n=n)
        if d is None or not d["have"].all():
            got = 0 if d is None else int(d["have"].sum())
            missing.append((i, got))
            continue
        C[i] = d["C"]
        kpar[i] = d["kpar"]
        ks[i] = float(d["k"])
    if missing:
        print("cannot consolidate -- incomplete checkpoints "
              f"(ifreq, n_angles): {missing}", flush=True)
        return False
    notes = (
        "Bloch lattice-sum cache C(k_i, k_par(theta, phi)) for the retrieval "
        "campaign. C[ifreq, iangle] = sum_{R != 0} A(R) e^{+i k_par . R}, "
        "square lattice pitch 2 um, modes/order from "
        "saw_gold_wl15p0025um.tmat.h5 (30 modes, lmax 3), r0=0.8, "
        "quad=make_quad(16,32), tapered shells with per-theta scaling "
        "kRc = (10,14,20)/(1-sin th), r_max = 3.5*max(kRc)/k, Richardson "
        "extrapolation in 1/Rc^2 (translate.py conventions). Angle order: "
        "campaign 13 [(0,0),(15,0),(15,22.5),(15,45),(30,0),(30,22.5),"
        "(30,45),(45,0),(45,22.5),(45,45),(60,0),(60,22.5),(60,45)] then "
        "treams-validation 4 [(20,0),(20,45),(40,0),(40,45)]. Built by "
        "retrieval/precompute_C.py (streaming, checkpointed)."
    )
    np.savez(CONSOLIDATED, C=C, k=ks, kpar=kpar, lam_um=data.wavelength_um,
             freq_hz=data.freq, theta_deg=THETA_DEG, phi_deg=PHI_DEG,
             kRc=KRC_TABLE, s_theta=S_THETA, kRc_base=KRC_BASE,
             pitch=PITCH, r0=R0, quad_n_theta=QUAD_SPEC[0],
             quad_n_phi=QUAD_SPEC[1], rmax_factor=RMAX_FACTOR, notes=notes)
    print(f"consolidated cache written: {CONSOLIDATED} "
          f"({os.path.getsize(CONSOLIDATED) / 1e6:.1f} MB)", flush=True)
    return True


def dry_run(freq_indices, angle_idx):
    """Per-freq shell counts and projected cost (no projections run)."""
    data = TMatrixData(DATA)
    ms_per_shell = 5.2            # measured (test_bloch_lattice.py test (d))
    print(f"{'ifreq':>5} {'lam_um':>7} {'k':>9} {'r_max_sup':>10} "
          f"{'nshell':>8} {'proj_est_min':>12}")
    tot = 0.0
    for i in freq_indices:
        k = data.k_at(i)
        r_max_sup = r_max_table(k)[list(angle_idx)].max()
        radii, _ = square_lattice_shells(PITCH, r_max_sup)
        est = len(radii) * ms_per_shell / 1000.0
        tot += est
        print(f"{i:5d} {data.wavelength_um[i]:7.2f} {k:9.6f} "
              f"{r_max_sup:10.1f} {len(radii):8d} {est / 60:12.2f}",
              flush=True)
    print(f"total projected projection time: {tot / 60:.1f} min "
          f"(at {ms_per_shell} ms/shell; assembly overhead extra)")


def parse_int_list(spec, nmax, what):
    """'a,b,c' with 'a-b' ranges, or 'all'."""
    if spec is None or spec.strip().lower() == "all":
        return list(range(nmax))
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    out = sorted(set(out))
    if out and (out[0] < 0 or out[-1] >= nmax):
        raise SystemExit(f"{what} indices out of range 0..{nmax - 1}: {spec}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freqs", default=None,
                    help="comma list / ranges of frequency indices "
                         "(default: all 49)")
    ap.add_argument("--stride", type=int, default=None,
                    help="use frequency indices 0, stride, 2*stride, ...")
    ap.add_argument("--angles", default="all",
                    help="comma list / ranges of angle indices 0..16 "
                         "(default all; treams-validation set: 0,13-16)")
    ap.add_argument("--workers", type=int, default=1,
                    help="multiprocessing over frequency indices (default 1)")
    ap.add_argument("--chunk", type=int, default=48,
                    help="shell chunk size for the streaming pass")
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help="checkpoint directory (default results/C_bloch)")
    ap.add_argument("--consolidate", action="store_true",
                    help="write results/C_bloch.npz from complete checkpoints"
                         " and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print per-freq shell counts and projected cost")
    args = ap.parse_args(argv)

    if args.consolidate:
        ok = consolidate(args.out)
        sys.exit(0 if ok else 1)

    if args.stride is not None and args.freqs is not None:
        raise SystemExit("--freqs and --stride are mutually exclusive")
    if args.stride is not None:
        freqs = list(range(0, 49, args.stride))
    else:
        freqs = parse_int_list(args.freqs, 49, "frequency")
    angles = parse_int_list(args.angles, N_ANGLES, "angle")

    if args.dry_run:
        dry_run(freqs, angles)
        return

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    header = (f"precompute_C: freqs={freqs} angles={angles} "
              f"workers={args.workers} chunk={args.chunk} "
              f"pitch={PITCH} r0={R0} quad={QUAD_SPEC} "
              f"kRc_base={tuple(KRC_BASE)}")
    log_line(args.out, header)

    if args.workers > 1:
        import multiprocessing as mp
        jobs = [(i, angles, args.out, args.chunk) for i in freqs]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for msg in pool.imap_unordered(_worker, jobs):
                print(f"[worker] {msg}", flush=True)
    else:
        data = TMatrixData(DATA)
        quad = make_quad(*QUAD_SPEC)
        proj = RegularProjector(data.modes, quad)
        for i in freqs:
            process_freq(data, i, angles, proj, quad, args.out,
                         chunk=args.chunk)
    log_line(args.out, f"precompute_C: done {len(freqs)} freq(s) in "
                       f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

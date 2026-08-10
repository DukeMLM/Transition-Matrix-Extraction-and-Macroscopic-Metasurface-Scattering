"""Read the direct-simulation reference straight out of a packed CST project.

A ``*.cst`` archive is a ZIP variant: the local/central headers carry the
signature ``DE`` instead of ``PK`` and use 32-bit DOS time and date fields
(so headers are 34 / 50 bytes instead of 30 / 46).  Inside, ``Result/Model.res``
is the result tree (``treepath`` -> signal file name) and ``Result/Storage.sdb``
is a SQLite database holding the signal samples.

That is enough to pull the Floquet-port S-parameters of a periodic unit-cell
run without opening CST.

Parametric sweeps: a project can hold many runs.  Signals are versioned by
``choice``, which is the 3D Run ID shown in CST's Result Navigator, and
``Result/db.parmap`` (also SQLite) maps that ID to the parameter combination.
``--list`` prints the table; ``--run`` picks one.  Taking the newest signal
blindly gives whichever run happened to be last -- not necessarily the one you
want.

Reference plane: CST places the Floquet ports on the Zmin/Zmax boundaries, so
the raw S-parameters carry the free-space phase from the port planes to the
array, and that distance changes with the model's z extent (i.e. from run to
run).  ``deembed_length`` recovers it and shifts both S11 and S21 back to
z = 0 for a z-symmetric model.  CST is e^{+j omega t}; this repo is
e^{-i omega t}, so the de-embedded values are conjugated to match.

    python -m tmatrix.aggregation.cst_packed_reference \
        test/2x2/SAW_gold_noSub_packed.cst --list
    python -m tmatrix.aggregation.cst_packed_reference \
        test/2x2/SAW_gold_noSub_packed.cst \
        --run 6 --out results_2x2/cst_direct_reference.csv
"""
import argparse
import os
import re
import sqlite3
import struct
import tempfile
import zlib

import numpy as np
from tmatrix.paths import AGG_DATA

C0_UM_THZ = 299.792458      # c in um*THz


# ---------------------------------------------------------------- archive ---
def read_packed_cst(path, wanted):
    """Extract the members whose name contains any string in `wanted`."""
    blob = open(path, "rb").read()
    eocd = blob.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError(f"{path}: no ZIP end-of-central-directory record")
    ntot, _cdsize, cdoff = struct.unpack("<HII", blob[eocd + 10:eocd + 20])

    out, p = {}, cdoff
    for _ in range(ntot):
        (_sig, _vmb, _vn, _flags, comp, _t, _d, _crc, csize, _usize,
         nlen, elen, clen, _disk, _ia, _ea, off) = struct.unpack(
            "<4s4H2I3I5H2I", blob[p:p + 50])
        name = blob[p + 50:p + 50 + nlen].decode("utf8", "replace")
        p += 50 + nlen + elen + clen
        if not any(w in name for w in wanted):
            continue
        (_s, _vn, _fl, lcomp, _t, _d, _crc, lcsize, _us, lnlen,
         lelen) = struct.unpack("<4s3H2I3I2H", blob[off:off + 34])
        start = off + 34 + lnlen + lelen
        raw = blob[start:start + lcsize]
        out[name] = zlib.decompress(raw, -15) if lcomp == 8 else raw
    return out


def result_tree(model_res):
    """treepath -> signal file name, from Result/Model.res."""
    txt = model_res.decode("latin1")
    tree = {}
    for block in txt.split("\r\n\r\n"):
        m = re.search(r"treepath=s:(.*)", block)
        if not m:
            continue
        files = [f.strip() for f in re.findall(r"files=s:(.*)", block)]
        if files:
            tree[m.group(1).strip()] = files[0]
    return tree


def run_table(parmap_db):
    """3D Run ID -> {parameter name: value} from Result/db.parmap."""
    names = {r[0]: r[1] for r in
             parmap_db.execute("select par_id,name from ParaList")}
    cols = [c[1] for c in
            parmap_db.execute("PRAGMA table_info(ParaCombinations)")]
    runs = {}
    for row in parmap_db.execute(
            f"select {','.join(cols)} from ParaCombinations order by combi_id"):
        runs[row[0]] = {names[int(c[3:])]: v
                        for c, v in zip(cols[1:], row[1:]) if c.startswith("par")}
    return runs


def read_signal(db, name, choice=None):
    """Sample set stored under `name` in Storage.sdb -> (x, y complex).

    `choice` is the 3D Run ID; None takes the highest one that has samples.
    """
    if choice is None:
        ids = [r[0] for r in db.execute(
            "select sig_id from SigHeader where name=? order by choice desc",
            (name,))]
    else:
        ids = [r[0] for r in db.execute(
            "select sig_id from SigHeader where name=? and choice=?",
            (name, choice))]
        if not ids:
            raise KeyError(f"no signal {name!r} for run {choice}")
    for sid in ids:
        row = db.execute("select bData from SigData6 where sig_id=?",
                         (sid,)).fetchone()
        if row is None:
            continue
        b = row[0]
        n, _size, rec = struct.unpack("<3q", b[:24])
        x = np.frombuffer(b, dtype="<f8", count=n, offset=24)
        y = np.frombuffer(b, dtype="<f8", count=n * (rec // 8), offset=24 + n * 8)
        return x, (y[0::2] + 1j * y[1::2]) if rec == 16 else y.astype(complex)
    raise KeyError(f"no sample data for signal {name!r}")


# ------------------------------------------------------------- de-embed ----
def deembed_length(f_thz, s11_raw, s21_raw):
    """Port-plane separation L [um] from the thin-sheet gauge.

    A planar sheet much thinner than the wavelength scatters as a surface
    current: at its own plane S11 = r and S21 = 1 + r, so S21 - S11 = 1
    *exactly*, whatever the loss or the resonance.  Both raw S-parameters carry
    the same port phase e^{-j k L} (z-symmetric model: S21 sees d1 + d2 = L,
    S11 sees 2 d1 = L), so arg(S21_raw - S11_raw) = -k L.  Fitting that slope
    through the origin uses the whole band instead of just the transparent
    low-frequency tail, which is where a fit to arg(S21) alone goes wrong --
    the first samples sit at f ~ 0 where the phase carries almost no
    information and the sheet is not quite transparent yet.
    """
    sel = f_thz > 0
    k = 2 * np.pi * f_thz[sel] / C0_UM_THZ
    ph = np.unwrap(np.angle(s21_raw[sel] - s11_raw[sel]))
    return float(-np.sum(k * ph) / np.sum(k * k))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cst", help="packed .cst project of the periodic run")
    ap.add_argument("--out", help="output CSV (not needed with --list)")
    ap.add_argument("--run", type=int, default=None,
                    help="3D Run ID of a parametric sweep (default: newest -- "
                         "run --list first, the newest is rarely the one you want)")
    ap.add_argument("--list", action="store_true",
                    help="print the parametric run table and exit")
    ap.add_argument("--port", default="Zmin(1)",
                    help="excited Floquet mode (default TE(0,0) at Zmin)")
    ap.add_argument("--deembed", type=float, default=None,
                    help="port-plane separation um (default: fitted)")
    args = ap.parse_args()
    if not args.list and not args.out:
        ap.error("--out is required unless --list is given")

    members = read_packed_cst(
        os.path.abspath(args.cst),
        ("Result/Model.res", "Result/Storage.sdb", "Result/db.parmap"))
    tree = result_tree(members["Result/Model.res"])

    tmpdir = tempfile.mkdtemp()
    tmp = os.path.join(tmpdir, "Storage.sdb")
    open(tmp, "wb").write(members["Result/Storage.sdb"])
    db = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)

    runs = {}
    if "Result/db.parmap" in members:
        pmp = os.path.join(tmpdir, "db.parmap")
        open(pmp, "wb").write(members["Result/db.parmap"])
        try:
            runs = run_table(sqlite3.connect(f"file:{pmp}?mode=ro", uri=True))
        except sqlite3.Error:
            runs = {}

    if args.list:
        if not runs:
            print("no parametric run table (single-run project)")
            return
        keys = sorted(next(iter(runs.values())))
        print("3D Run ID  " + "  ".join(f"{k:>10}" for k in keys))
        for rid in sorted(runs):
            print(f"{rid:9d}  " + "  ".join(f"{runs[rid][k]:10.6g}"
                                            for k in keys))
        return

    exc = args.port
    refl_key = f"1D Results\\S-Parameters\\S{exc},{exc}"
    trans_key = f"1D Results\\S-Parameters\\SZmax({exc.split('(')[1].rstrip(')')}),{exc}"
    for key in (refl_key, trans_key):
        if key not in tree:
            raise KeyError(f"{key!r} not in the result tree; available:\n  " +
                           "\n  ".join(k for k in tree if "S-Param" in k))

    f_thz, s11_raw = read_signal(db, tree[refl_key], args.run)
    _, s21_raw = read_signal(db, tree[trans_key], args.run)

    L = (args.deembed if args.deembed is not None
         else deembed_length(f_thz, s11_raw, s21_raw))
    k = 2 * np.pi * f_thz / C0_UM_THZ
    # port planes -> z = 0 (z-symmetric model: S11 sees 2*(L/2) = L as well),
    # then e^{+j w t} -> e^{-i w t}
    s11 = np.conj(s11_raw * np.exp(1j * k * L))
    s21 = np.conj(s21_raw * np.exp(1j * k * L))

    print(f"{os.path.basename(args.cst)}: {len(f_thz)} samples, "
          f"{f_thz.min():.3f}-{f_thz.max():.3f} THz, run "
          f"{args.run if args.run is not None else 'newest'}")
    if runs and args.run in runs:
        print("  parameters: " + ", ".join(
            f"{k_}={v:g}" for k_, v in sorted(runs[args.run].items())))
    print(f"  de-embed port separation L = {L:.3f} um "
          f"({'fitted' if args.deembed is None else 'given'})")
    gauge = np.abs(s21 - s11 - 1.0)
    print(f"  thin-sheet gauge max |S21 - S11 - 1| = {gauge.max():.4f} "
          f"(0 for an ideal zero-thickness sheet at its own plane)")
    print(f"  max |S21|^2 + |S11|^2 = {(np.abs(s21)**2 + np.abs(s11)**2).max():.4f} "
          f"(<= 1 for a passive cell)")
    lo = f_thz <= 0.05 * f_thz.max()
    print(f"  low-f check: max |S21 - 1| = "
          f"{np.abs(s21[lo] - 1).max():.2e}, max |S11| = {np.abs(s11[lo]).max():.2e}")

    out = args.out if os.path.isabs(args.out) else os.path.join(AGG_DATA,
                                                                args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lam = np.where(f_thz > 0, C0_UM_THZ / np.where(f_thz > 0, f_thz, 1), np.inf)
    with open(out, "w") as fh:
        fh.write("f_THz,lam_um,Re_S11,Im_S11,Re_S21,Im_S21,absS11,absS21,R,T,A\n")
        for i in range(len(f_thz)):
            R, T = abs(s11[i]) ** 2, abs(s21[i]) ** 2
            fh.write(",".join(f"{v:.10g}" for v in [
                f_thz[i], lam[i], s11[i].real, s11[i].imag,
                s21[i].real, s21[i].imag, abs(s11[i]), abs(s21[i]),
                R, T, 1 - R - T]) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

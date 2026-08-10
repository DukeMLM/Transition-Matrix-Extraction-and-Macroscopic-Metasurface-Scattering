"""Windows / numpy>=2 compatibility patch for treams' compiled gufuncs.

treams' gufuncs declare their integer inputs as C `long`, which is int32 on
Windows, while numpy 2.x defaults integer arrays to int64; calling them then
fails to find a matching loop.  The fix is to cast the arguments to the types
the last (complex) signature declares.  Quantum numbers are small, so the
int32 cast is lossless -- this changes no physics.

treams.io.FREQUENCIES also lacks a plain "Hz" key, which load_hdf5 needs.

Five modules used to carry their own copy of this, three of them without an
idempotence flag, so importing two of them double-wrapped every ufunc.  Call
`apply()` once at the top of any module that touches treams; it is idempotent
and cheap on repeat calls.
"""
import warnings

import numpy as np

import treams
import treams.io
import treams.lattice as _tl
from treams import cw, pw, sw

__all__ = ["apply"]

_PATCH_FLAG = "_tmatrix_cast_wrapped"


def _cast_wrap(uf):
    """Wrap a compiled gufunc so numpy 2.x on Windows can call it."""
    sigs = uf.types
    ins = sigs[-1].split("->")[0]          # last signature = complex variant

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
    setattr(f, _PATCH_FLAG, True)
    return f


def apply():
    """Patch treams in place.  Idempotent; safe to call from every module."""
    # treams calls its gufuncs with `where=` and no `out=`; numpy 2 warns that
    # the masked-out entries are uninitialized.  treams fills them itself, and
    # the warning fires from inside _cast_wrap rather than from treams, so it
    # is filtered here rather than left to bury the run log.
    warnings.filterwarnings(
        "ignore", message=".*'where' used without 'out'.*",
        category=UserWarning)
    treams.io.FREQUENCIES.setdefault("Hz", 1.0)
    for mod in (sw, pw, cw, _tl):
        for name in dir(mod):
            uf = getattr(mod, name)
            if getattr(uf, _PATCH_FLAG, False):
                continue
            if isinstance(uf, np.ufunc) and any(
                    "l" in s.split("->")[0] for s in uf.types):
                setattr(mod, name, _cast_wrap(uf))

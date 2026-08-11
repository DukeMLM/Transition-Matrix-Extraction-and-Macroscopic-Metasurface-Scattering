"""Complex de-embedding of periodic CST Floquet S-parameters
(checklist item 8, de-embedding half; INVERSE_TMATRIX_FROM_FLOQUET.md par. 7).

Nothing complex-valued existed in the repo before this module
(plot_cst_comparison.py / REPORT par. 4b validated magnitudes only).

Conventions (normative)
-----------------------
* CST S-parameters arrive in the e^{+j omega t} convention; the tmat.h5 /
  aggregation / retrieval stack uses e^{-i omega t}.  CONJUGATE ON LOAD
  (conj_cst), before any physics.
* Empty-cell phase check (par. 7, pins THREE choices at once): after
  conjugation, the empty cell of port-to-port distance L at scan angle theta
  must satisfy

      S21_empty = e^{+i k_z L},   k_z = k cos(theta)

  i.e. arg(S21_empty) must ADVANCE with frequency.  check_empty_phase fits
  the phase slope; a NEGATIVE slope means either the conjugation was skipped
  (wrong time convention) or the scan-angle direction convention is not the
  par. 7 "inward" one.  At oblique angles the same check pins the k_par
  sign.  Also reported: max| |S21_empty| - 1 | and max |S11_empty| (par. 7
  noise-floor calibration source (i)).
* De-embedding (z-symmetric pinned domain, par. 7):

      S21_de = conj(S21_raw) / conj(S21_empty)
      S11_de = conj(S11_raw) / conj(S21_empty)

  The SAME empty S21 divides both blocks: the cell sits at z = 0 in a
  z-symmetric domain with ports at z = +/- L/2, so the reflected path
  (L/2 down + L/2 back = L) equals the port-to-port transmission path.
  For 2x2 Jones blocks every entry (a,b) is divided by the empty co-polar
  transmission of the RECEIVE mode a (vacuum propagation is polarization
  degenerate, so the per-mode empties differ only by solver noise; using the
  mode-matched one keeps that noise symmetric).
* Frequency interpolation onto the 49-point tmat grid: SEPARATE Re/Im
  np.interp -- the exact pattern of build_saw_unitcell.py ~line 317.
  The CST band (14.99-37.47 THz) clips the grid endpoints by <= 0.004 THz;
  np.interp clamps there (recorded in the campaign manifest).
* Closure gate (par. 7, NEW deliverable): closure_normal compares
  de-embedded complex S at theta = 0 (both polarizations) against
  aggregation/results/periodic_results.npz to <= 5e-3.  Whatever residual
  spectrum the closure yields IS the fit sigma (the 3e-3 on record is
  magnitude-only and is NOT assumed for phase).

TE/TM label mapping (par. 7 channel dictionary)
-----------------------------------------------
CST sorts Floquet modes with SetSortCode "+beta/pw"; the default hypothesis
is mode 1 = TE, mode 2 = TM in the par. 2 Jones basis of
sparams_oblique.py.  What the EMPTY cell can and cannot pin down:

* it CAN pin the propagation phase, the conjugation direction, the "inward"
  angle convention, and (integrity) that its own cross-pol entries vanish
  and its two co-pol entries are degenerate;
* it CANNOT orient the labels: vacuum specular propagation is polarization
  DEGENERATE (S21_empty = e^{+i k_z L} I in EVERY orthonormal transverse
  basis), so no empty-cell S-parameter distinguishes mode swap or signs.

The label hypothesis is therefore fixed by two structure-run checks:
(1) check_mirror_plane_crosspol -- on the mirror planes phi in {0, 45} deg
    the physical cross-pol vanishes (doc par. 3), so a label ROTATION would
    show up as nonzero cross-pol without any reference model;
(2) the par. 7 acceptance test at an oblique angle: select_hypothesis
    requires the de-embedded blocks to match the forward model with the
    reference T under EXACTLY ONE discrete label hypothesis (the spot-check
    hook covers one phi != 0 case as well).
map_cst_labels packages the integrity checks + the hypothesis enumeration.

S11 TM-ROW SIGN (`r11_tm`) -- the EXTENDED family, added 2026-08-07 after
the real campaign refused
-------------------------------------------------------------------------
The 8 hypotheses above are all PORT GAUGES: a per-mode orientation flip
acts as S -> D S D with D = diag(s_1, s_2), so it multiplies S[a, b] by
s_a s_b and therefore can NEVER change a CO-POLAR diagonal entry (a = b,
s_a^2 = 1).  The (theta=60, phi=22.5) acceptance run measured a defect that
IS on a co-polar diagonal -- S11[TM, TM] disagreeing with the model by
|d| = 2|S| (a sign, not a phase or a scale) -- so no member of the 8 could
express it and the acceptance correctly refused 0-of-8.

The cause is a CONVENTION CONSEQUENCE, not a free parameter.  The par.-2
polarization basis of sparams_oblique.py evaluates e_TM SEPARATELY for
k_hat_t and k_hat_r, and the reflected one is transversally opposite
(its docstring derives e_TM^(r) = -x_hat at theta -> 0, and forward.py's
selftest already encodes S11[TM, TM] = -stored S11 and passes at 1e-10
against run_demo.py).  CST's Floquet port S11 is an ordinary waveguide
S-parameter referred to the SAME transverse mode pattern for the incident
and the reflected wave.  The two conventions therefore differ by a sign on
the TM RECEIVE ROW of the REFLECTION block only; the transmission block is
unaffected.

`r11_tm` in {+1, -1} carries that sign.  It is applied AFTER the swap, so
it always acts on the TM row in the par.-2 Jones basis.  It is deterministic
physics -- we keep it as a hypothesis dimension only so the acceptance test
stays FAIL-SAFE (it must be able to refuse, not to assume).

Measured on the real campaign checkpoints (sigma = 2.633e-3 from the
par.-7 normal-incidence closure, n_obs = 8 x 49 = 392, reduced chi2 of the
best member):
  theta=60, phi=22.5 : 8-member family  chi2_red 658.58, margin z = 0.91
                       -> REFUSED;
                       16-member family chi2_red   1.595, margin z = 5.77
                       -> ACCEPTED, winner swap=False, s11_cross=-1,
                       s21_cross=-1, r11_tm=-1
  theta=0  (control) : 16-member family chi2_red   1.134 but all 8
                       r11_tm=-1 members tie (z = 0.15) -- normal incidence
                       is structurally degenerate (C4 forces S_TE = S_TM
                       and the cross-pol vanishes), which is exactly why
                       par. 7 puts the acceptance at an OBLIQUE angle.
Controls that rule out alternatives: flipping the S21 TM row instead gives
chi2_red 71585; flipping both gives 70929.

DEFAULT UNCHANGED: label_hypotheses() still returns the 8 port gauges;
pass extended=True for the 16.  apply_hypothesis accepts the r11_tm key
whether or not it is present.

Legacy data (run_v3 / run_empty) diagnostics: see legacy_diagnostics() --
measurement, NOT a gate; those runs predate the pinned-domain edit.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from tmatrix.cst_env import CST_PYTHON_LIB, ensure_on_path
from tmatrix.paths import AGG_RESULTS, CST_DIRECT_DATA, RETRIEVAL_RESULTS
from tmatrix.units import C_UM_THZ

RESULTS_DIR = RETRIEVAL_RESULTS
REF_NPZ = AGG_RESULTS / "periodic_results.npz"
RUN_V3 = CST_DIRECT_DATA / "run_v3"
RUN_EMPTY = CST_DIRECT_DATA / "run_empty"

PITCH_UM = 2.0
CLOSURE_GATE = 5e-3


class DeembedError(RuntimeError):
    pass


# ===========================================================================
# Readers
# ===========================================================================

def read_sparams_csv(path):
    """run_v3-style s_params_complex.csv reader.

    Columns: freq_THz, Re_S11, Im_S11, Re_S21, Im_S21.  Returns
    (f_THz, S11, S21) with S RAW, i.e. still in CST's e^{+j omega t}
    convention -- call conj_cst before any physics.  (Provenance note:
    run_v3's csv was verified offline to be the SZmax(2),Zmax(2) /
    SZmin(2),Zmax(2) pair, i.e. Floquet mode 2.)"""
    rows = np.loadtxt(path, delimiter=",", skiprows=1)
    f = rows[:, 0]
    s11 = rows[:, 1] + 1j * rows[:, 2]
    s21 = rows[:, 3] + 1j * rows[:, 4]
    return f, s11, s21


def import_cst_results(extra_paths=()):
    """Import cst.results with a clear message on machines without CST."""
    ensure_on_path()
    for p in map(str, extra_paths):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import cst.results as cstres  # noqa: PLC0415
        return cstres
    except ImportError as e:
        raise DeembedError(
            "cst.results is not importable on this machine (looked in "
            f"{CST_PYTHON_LIB}).  Offline .cst reading needs the CST python "
            "libraries; use the csv reader or run on the CST host.  "
            f"Original error: {e}") from e


def read_project_sparams(cst_path, entries=None):
    """Offline extractor: complex S spectra from a solved .cst project.

    entries: list of S-tree paths (default: the 8 par. 7 channel-dictionary
    entries SZmax(a),Zmax(b) + SZmin(a),Zmax(b)).  Returns
    dict entry -> (f_THz, S_raw) with S RAW (e^{+j omega t}).  Absolute-path
    and run-id footguns handled per nir/cst_helpers.py's documentation."""
    cstres = import_cst_results()
    cst_path = Path(cst_path)
    if not cst_path.exists():
        raise DeembedError(f"project not found: {cst_path}")
    proj = cstres.ProjectFile(str(cst_path.resolve()),
                              allow_interactive=True)
    p3 = proj.get_3d()
    try:
        run_ids = sorted(p3.get_all_run_ids())
    except Exception:
        run_ids = [0]
    if entries is None:
        entries = [f"1D Results\\S-Parameters\\SZmax({a}),Zmax({b})"
                   for a in (1, 2) for b in (1, 2)]
        entries += [f"1D Results\\S-Parameters\\SZmin({a}),Zmax({b})"
                    for a in (1, 2) for b in (1, 2)]
    tree = set(p3.get_tree_items())
    out = {}
    for entry in entries:
        if entry not in tree:
            raise DeembedError(
                f"{cst_path.name}: S-tree entry missing: {entry} "
                f"(project not solved, or solve failed -- cf. the legacy "
                f"run_empty 'Could not read mesh' failure)")
        item = None
        for rid in sorted(run_ids, reverse=True):
            try:
                ri = p3.get_result_item(entry, rid)
                if ri.get_xdata():
                    item = ri
                    break
            except Exception:
                continue
        if item is None:
            raise DeembedError(f"{cst_path.name}: no run id has data "
                               f"for {entry}")
        out[entry] = (np.asarray(item.get_xdata(), dtype=float),
                      np.asarray(item.get_ydata(), dtype=complex))
    return out


def load_manifest(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_domain_z_extent(log_tet_path, pitch_um=PITCH_UM):
    """Port-to-port z-extent from a CST tet-mesh log's bounding-box diagonal.

    CADSurf logs 'Bounding box diagonal = <d>' for the model+background box
    (transverse size = the unit cell p x p), so L = sqrt(d^2 - 2 p^2).
    Returns L in um.  Raises DeembedError if the log or line is absent
    (e.g. the legacy run_empty project, whose mesh was never built)."""
    log_tet_path = Path(log_tet_path)
    if not log_tet_path.exists():
        raise DeembedError(f"no tet-mesh log at {log_tet_path} "
                           "(mesh never built?)")
    text = log_tet_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Bounding box diagonal\s*=\s*([0-9.eE+-]+)", text)
    if not m:
        raise DeembedError(f"no 'Bounding box diagonal' in {log_tet_path}")
    diag = float(m.group(1))
    return float(np.sqrt(diag ** 2 - 2.0 * pitch_um ** 2))


# ===========================================================================
# Convention step + empty-cell phase check
# ===========================================================================

def conj_cst(S):
    """e^{+j omega t} (CST) -> e^{-i omega t} (tmat/aggregation): conjugate."""
    return np.conjugate(S)


def kz_per_um(f_THz, theta_deg):
    """k_z = (2 pi f / c) cos(theta) in rad/um."""
    return (2.0 * np.pi * np.asarray(f_THz) / C_UM_THZ
            * np.cos(np.deg2rad(theta_deg)))


def analytic_empty_s21(f_THz, L_um, theta_deg, convention="physics"):
    """Analytic empty-cell transmission for port-to-port distance L.

    convention='physics' (e^{-i omega t}):  S21 = e^{+i k_z L}
    convention='cst'     (e^{+j omega t}):  S21 = e^{-j k_z L}  (conjugate)
    """
    s = np.exp(1j * kz_per_um(f_THz, theta_deg) * L_um)
    if convention == "physics":
        return s
    if convention == "cst":
        return np.conjugate(s)
    raise ValueError("convention must be 'physics' or 'cst'")


def check_empty_phase(f_THz, s21_empty_physics, theta_deg,
                      L_expected_um=None, rel_tol=0.2,
                      s11_empty_physics=None):
    """Verify the par. 7 empty-cell phase-slope convention (NORMATIVE).

    Input S21 must ALREADY be conjugated to the e^{-i omega t} convention
    (conj_cst on CST data).  Fits arg(S21_empty) = a + slope * f and checks

        slope == d(k_z L)/df = 2 pi cos(theta) L / c  > 0.

    The SIGN is the hard gate: a negative fitted slope means the conjugation
    was skipped/doubled or the scan-angle "inward" convention is violated.
    If L_expected_um is given, additionally require the fitted L to match
    within rel_tol (default 20 %).  Returns a dict:

      slope_rad_per_THz, slope_expected_rad_per_THz (None if no L_expected),
      L_fit_um, sign_ok, rel_err (None if no L_expected),
      mag_dev = max| |S21|-1 |, s11_max (None if not given), passed
    """
    f = np.asarray(f_THz, dtype=float)
    s21 = np.asarray(s21_empty_physics)
    phase = np.unwrap(np.angle(s21))
    slope, _ = np.polyfit(f, phase, 1)
    per_um = 2.0 * np.pi * np.cos(np.deg2rad(theta_deg)) / C_UM_THZ
    L_fit = slope / per_um
    sign_ok = bool(slope > 0)
    slope_expected = rel_err = None
    if L_expected_um is not None:
        slope_expected = per_um * L_expected_um
        rel_err = float(abs(slope - slope_expected) / abs(slope_expected))
    mag_dev = float(np.max(np.abs(np.abs(s21) - 1.0)))
    s11_max = (float(np.max(np.abs(s11_empty_physics)))
               if s11_empty_physics is not None else None)
    passed = sign_ok and (rel_err is None or rel_err <= rel_tol)
    return dict(slope_rad_per_THz=float(slope),
                slope_expected_rad_per_THz=slope_expected,
                L_fit_um=float(L_fit), sign_ok=sign_ok, rel_err=rel_err,
                mag_dev=mag_dev, s11_max=s11_max, passed=bool(passed))


# ===========================================================================
# De-embedding
# ===========================================================================

def deembed_spectra(s11_raw_cst, s21_raw_cst, s21_empty_cst):
    """par. 7 complex de-embedding for one (receive, incident) channel.

        S21_de = conj(S21_raw) / conj(S21_empty)
        S11_de = conj(S11_raw) / conj(S21_empty)

    The same empty S21 divides both blocks: the pinned domain is z-symmetric
    (cell at z = 0, ports at +/- L/2), so the reflected path L/2 + L/2
    equals the port-to-port transmission path L.  Inputs are RAW CST
    (e^{+j omega t}); output is de-embedded, e^{-i omega t}, referenced to
    the cell plane z = 0."""
    e = conj_cst(np.asarray(s21_empty_cst))
    return (conj_cst(np.asarray(s11_raw_cst)) / e,
            conj_cst(np.asarray(s21_raw_cst)) / e)


def deembed_blocks(raw, empty):
    """De-embed full 2x2 Jones blocks read by read_project_sparams.

    raw, empty: dict entry -> (f_THz, S_raw) for one structure run and its
    matching empty run (same theta).  Every structure entry (receive mode a,
    incident mode b) is divided by the empty co-polar transmission of the
    receive mode a, SZmin(a),Zmax(a) (see module docstring).  Frequencies
    are aligned with the Re/Im interp pattern when the grids differ.
    Returns (f_THz, S11_de, S21_de): S11_de/S21_de of shape (2, 2, nf),
    index [a-1, b-1] (CST mode numbers; map to TE/TM downstream)."""
    def entry(d, side, a, b):
        return d[f"1D Results\\S-Parameters\\S{side}({a}),{'Zmax'}({b})"]

    f_ref = entry(raw, "Zmax", 1, 1)[0]
    nf = len(f_ref)
    S11 = np.empty((2, 2, nf), dtype=complex)
    S21 = np.empty((2, 2, nf), dtype=complex)
    for a in (1, 2):
        f_e, s21_e = entry(empty, "Zmin", a, a)
        s21_e = interp_to_grid(f_e, s21_e, f_ref)
        for b in (1, 2):
            f_r, s11_r = entry(raw, "Zmax", a, b)
            s11_r = interp_to_grid(f_r, s11_r, f_ref)
            f_t, s21_r = entry(raw, "Zmin", a, b)
            s21_r = interp_to_grid(f_t, s21_r, f_ref)
            S11[a - 1, b - 1], S21[a - 1, b - 1] = deembed_spectra(
                s11_r, s21_r, s21_e)
    return f_ref, S11, S21


def interp_to_grid(f_src_THz, S, f_tgt_THz):
    """Separate Re/Im np.interp (build_saw_unitcell.py ~line 317 pattern)."""
    f_src = np.asarray(f_src_THz, dtype=float)
    f_tgt = np.asarray(f_tgt_THz, dtype=float)
    S = np.asarray(S)
    if len(f_src) == len(f_tgt) and np.allclose(f_src, f_tgt):
        return S.copy()
    return (np.interp(f_tgt, f_src, S.real)
            + 1j * np.interp(f_tgt, f_src, S.imag))


# ===========================================================================
# TE/TM label mapping (par. 7 channel dictionary)
# ===========================================================================

def label_hypotheses(extended=False):
    """The discrete label hypotheses relating CST mode numbers to the
    par. 2 Jones basis (index 0 = TE, 1 = TM, sparams_oblique.py).

    extended=False (DEFAULT, unchanged): the 8 PORT GAUGES

      swap        : False -> mode1=TE, mode2=TM (the "+beta/pw" default);
                    True  -> mode1=TM, mode2=TE
      s11_cross   : +/-1 sign carried by the CROSS-pol entries of the
                    reflection block (a per-mode orientation flip at the
                    Zmax port flips cross-pol only: D S D with
                    D = diag(+1,-1) leaves co-pol invariant)
      s21_cross   : same for the transmission block (the Zmin port's mode
                    orientation is independent of Zmax's)

    extended=True: those 8 x r11_tm in {+1, -1} = 16, adding

      r11_tm      : +/-1 sign on the TM RECEIVE ROW of the REFLECTION block
                    (both S11[TM, TE] and S11[TM, TM]), applied AFTER the
                    swap so it always acts on the TM row of the par.-2
                    Jones basis.  This is NOT a port gauge -- no port gauge
                    can touch a co-polar diagonal -- but the documented
                    difference between our reflected-TM basis vector and
                    CST's waveguide port convention.  See the module
                    docstring for the physics and for the campaign
                    measurement that made it necessary.

    The first 8 entries of the extended list are exactly the base 8 with
    r11_tm = +1, so index-based references to the base family survive.
    """
    base = [dict(swap=sw, s11_cross=s1, s21_cross=s2)
            for sw in (False, True)
            for s1 in (+1, -1)
            for s2 in (+1, -1)]
    if not extended:
        return base
    return [dict(h, r11_tm=r) for r in (+1, -1) for h in base]


def apply_hypothesis(S11, S21, hyp, inverse=False):
    """Map de-embedded blocks from CST mode indices to the par. 2 Jones
    basis under one label hypothesis.  S11/S21: (2, 2, ...) arrays indexed
    [receive, incident] in CST mode order.

    The forward map is the composition (in this order)

        swap  ->  cross-pol signs  ->  r11_tm row sign,

    r11_tm defaulting to +1 when the key is absent (so the 8-member family
    behaves exactly as before).  inverse=True applies the three steps in
    REVERSE order, which is the true inverse map.

    NOTE: with r11_tm = -1 AND swap = True the forward map is NOT an
    involution (the row sign is applied in Jones indices, so it does not
    commute with the swap); applying it twice negates the whole S11 block.
    Use inverse=True to undo it.  For every member of the base 8 family
    the two directions coincide, as before.
    """
    S11 = np.asarray(S11).copy()
    S21 = np.asarray(S21).copy()
    r11 = hyp.get("r11_tm", 1)

    def do_swap():
        nonlocal S11, S21
        if hyp["swap"]:
            S11 = S11[::-1, ::-1]
            S21 = S21[::-1, ::-1]

    def do_cross():
        for S, key in ((S11, "s11_cross"), (S21, "s21_cross")):
            if hyp[key] == -1:
                S[0, 1] = -S[0, 1]
                S[1, 0] = -S[1, 0]

    def do_row():
        if r11 == -1:
            S11[1] = -S11[1]

    if inverse:
        do_row()
        do_cross()
        do_swap()
    else:
        do_swap()
        do_cross()
        do_row()
    return S11, S21


def select_hypothesis(S11_meas, S21_meas, S11_pred, S21_pred, tol=1e-2,
                      extended=False):
    """par. 7 acceptance: EXACTLY ONE hypothesis must bring all 8
    de-embedded complex numbers within tol of the forward-model prediction
    (reference T).  Predictions are in the par. 2 Jones basis.  Returns
    (winning_hypothesis, residual); raises DeembedError when zero or
    several hypotheses pass (with the residual table in the message).

    extended: passed to label_hypotheses (False -> the 8 port gauges,
    unchanged default; True -> the 16 including the r11_tm row sign that
    the real campaign needed -- see the module docstring).

    This is the doc-literal MAX statistic.  It is nearly powerless once the
    noise floor approaches the hypothesis separation; the campaign path
    uses the pooled chi2 discriminant in
    validate_against_reference.channel_dictionary_acceptance instead.
    """
    hyps = label_hypotheses(extended)
    rows = []
    for hyp in hyps:
        S11_h, S21_h = apply_hypothesis(S11_meas, S21_meas, hyp)
        r = max(float(np.max(np.abs(S11_h - S11_pred))),
                float(np.max(np.abs(S21_h - S21_pred))))
        rows.append((hyp, r))
    winners = [(h, r) for h, r in rows if r <= tol]
    if len(winners) == 1:
        return winners[0]
    table = "\n".join(f"  {h}: residual {r:.3e}" for h, r in rows)
    raise DeembedError(
        f"label-hypothesis acceptance failed: {len(winners)} of {len(hyps)} "
        f"hypotheses within tol={tol:g} (need exactly 1).\n{table}")


def check_mirror_plane_crosspol(S11_de, S21_de, phi_deg, tol=5e-3):
    """On the mirror planes phi in {0, 45} deg the physical cross-pol
    vanishes identically (doc par. 3); measured cross-pol above tol at
    those angles indicts the label ORIENTATION (a rotated CST mode basis
    mixes co- into cross-pol) -- a reference-model-free orientation check.
    Returns dict(applicable, max_crosspol, passed)."""
    applicable = float(phi_deg) % 45.0 == 0.0
    xmax = max(float(np.max(np.abs(np.asarray(S11_de)[0, 1]))),
               float(np.max(np.abs(np.asarray(S11_de)[1, 0]))),
               float(np.max(np.abs(np.asarray(S21_de)[0, 1]))),
               float(np.max(np.abs(np.asarray(S21_de)[1, 0]))))
    return dict(applicable=applicable, max_crosspol=xmax,
                passed=bool((not applicable) or xmax <= tol))


def map_cst_labels(theta_deg, phi_deg, empty_blocks=None,
                   L_expected_um=None, extended=False):
    """Fix the (mode1, mode2) <-> (TE, TM) dictionary at one scan angle.

    empty_blocks: optional dict entry -> (f, S_raw) of the matching EMPTY
    run (read_project_sparams).  What the empty cell CAN pin (and this
    function checks): conjugation/phase direction (check_empty_phase per
    co-pol mode), co-pol mode DEGENERACY, and vanishing empty cross-pol.
    What it CANNOT pin (vacuum is polarization degenerate; module
    docstring): swap/sign orientation -- returned as the 8-hypothesis
    enumeration, to be collapsed by check_mirror_plane_crosspol on the
    phi in {0,45} structure runs and by select_hypothesis against the
    reference forward model at (theta=30, phi=0) [par. 7 acceptance], with
    one phi != 0 spot check.  Returns a dict with the default map, the
    hypothesis list, and the empty-cell integrity results."""
    out = dict(theta_deg=float(theta_deg), phi_deg=float(phi_deg),
               default=dict(mode1="TE", mode2="TM",
                            basis="par. 2 Jones (sparams_oblique.pol_basis)",
                            provenance='SetSortCode "+beta/pw"'),
               hypotheses=label_hypotheses(extended),
               empty_checks=None,
               degeneracy_note=("vacuum specular S is polarization "
                                "degenerate: the empty cell cannot orient "
                                "the labels; it pins phase/conjugation "
                                "only"))
    if empty_blocks is not None:
        checks = {}
        co = {}
        for a in (1, 2):
            f, s21_raw = empty_blocks[
                f"1D Results\\S-Parameters\\SZmin({a}),Zmax({a})"]
            s21 = conj_cst(s21_raw)
            co[a] = s21
            _, s11_raw = empty_blocks[
                f"1D Results\\S-Parameters\\SZmax({a}),Zmax({a})"]
            checks[f"mode{a}"] = check_empty_phase(
                f, s21, theta_deg, L_expected_um=L_expected_um,
                s11_empty_physics=conj_cst(s11_raw))
        checks["copol_degeneracy"] = float(np.max(np.abs(co[1] - co[2])))
        xmax = 0.0
        for a, b in ((1, 2), (2, 1)):
            for side in ("SZmin", "SZmax"):
                _, s = empty_blocks[
                    f"1D Results\\S-Parameters\\{side}({a}),Zmax({b})"]
                xmax = max(xmax, float(np.max(np.abs(s))))
        checks["crosspol_max"] = xmax
        out["empty_checks"] = checks
    return out


# ===========================================================================
# Closure gate (par. 7 NEW deliverable) -- its residual IS the fit sigma
# ===========================================================================

def closure_normal(f_THz, S11_by_pol, S21_by_pol, ref_npz=REF_NPZ,
                   gate=CLOSURE_GATE, out_npz=None, out_fig=None,
                   label="closure_normal"):
    """Compare de-embedded complex S at theta = 0 (both polarizations)
    against the aggregation reference (periodic_results.npz) to the <= 5e-3
    complex gate.  The returned residual spectrum is DEFINED as the fit
    sigma (par. 7): sigma_i = |S_de - S_ref|(f_i), reported per observable
    and pooled.

    S11_by_pol / S21_by_pol: dict pol_name -> complex spectrum on f_THz
    (e.g. {"TE": ..., "TM": ...}; at theta = 0 the C4 model forces
    S_TE = S_TM = the reference co-pol spectrum, so both are compared to
    the same reference).  Emits out_npz (default
    retrieval/results/closure_normal.npz) + a two-panel figure."""
    with np.load(ref_npz) as z:
        f_ref = z["freq"] / 1e12
        ref = {"S11": z["S11"], "S21": z["S21"]}
    res = {}
    de = {}
    for pol in S11_by_pol:
        s11 = interp_to_grid(f_THz, S11_by_pol[pol], f_ref)
        s21 = interp_to_grid(f_THz, S21_by_pol[pol], f_ref)
        de[f"S11_{pol}"] = s11
        de[f"S21_{pol}"] = s21
        res[f"S11_{pol}"] = np.abs(s11 - ref["S11"])
        res[f"S21_{pol}"] = np.abs(s21 - ref["S21"])
    allres = np.stack(list(res.values()))
    sigma = allres.max(axis=0)               # per-frequency pooled floor
    worst = float(allres.max())
    passed = bool(worst <= gate)

    mag = {k: float(np.max(np.abs(np.abs(de[k]) - np.abs(ref[k[:3]]))))
           for k in de}
    ph = {k: float(np.max(np.abs(np.angle(de[k] / ref[k[:3]]))))
          for k in de}

    out_npz = Path(out_npz or RESULTS_DIR / f"{label}.npz")
    out_fig = Path(out_fig or RESULTS_DIR / f"fig_{label}.png")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, f_THz=f_ref, sigma=sigma, gate=gate,
             passed=passed, ref_S11=ref["S11"], ref_S21=ref["S21"],
             **{f"de_{k}": v for k, v in de.items()},
             **{f"res_{k}": v for k, v in res.items()})

    from tmatrix.plotting import plt
    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ref_labeled = set()
    for k in de:
        r = ref[k[:3]]
        ax[0].plot(f_ref, np.abs(de[k]), label=f"|{k}| de-embedded")
        lbl = (f"|{k[:3]}| reference"
               if k[:3] not in ref_labeled else None)
        ref_labeled.add(k[:3])
        ax[0].plot(f_ref, np.abs(r), "--", lw=1, label=lbl)
        ax[1].semilogy(f_ref, res[k], label=f"|d{k}| complex residual")
    ax[1].axhline(gate, color="k", ls=":", label=f"gate {gate:g}")
    ax[0].set_ylabel("|S|")
    ax[0].legend(fontsize=7)
    ax[1].set_ylabel("complex residual (= fit sigma)")
    ax[1].set_xlabel("f (THz)")
    ax[1].legend(fontsize=7)
    ax[0].set_title(f"{label}: worst {worst:.2e} "
                    f"({'PASS' if passed else 'FAIL'} at {gate:g})")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=130)
    plt.close(fig)

    return dict(f_THz=f_ref, sigma=sigma, residuals=res, worst=worst,
                gate=gate, passed=passed, mag_dev=mag, phase_dev_rad=ph,
                out_npz=str(out_npz), out_fig=str(out_fig))


# ===========================================================================
# Campaign entry point (consumes cst_campaign.py's manifest)
# ===========================================================================

def deembed_campaign(manifest_path, theta_deg=0.0, phi_deg=0.0):
    """De-embed one structure run of the campaign against its matching
    empty run (same theta), running the full convention chain:
    read -> conjugate -> check_empty_phase (normative) -> de-embed ->
    map_cst_labels.  Returns (f, S11_de, S21_de, label_map, phase_check).
    At theta = 0 the caller should feed the result to closure_normal
    (THE gate); the (30, 0) acceptance test additionally needs the forward
    model with the reference T (validate_against_reference.py, out of this
    module's scope) via select_hypothesis."""
    man = load_manifest(manifest_path)
    runs = {(r["kind"], r["theta_deg"], r["phi_deg"]): r
            for r in man["runs"]}
    st = runs.get(("structure", float(theta_deg), float(phi_deg)))
    em = runs.get(("empty", float(theta_deg), 0.0))
    if st is None or em is None:
        raise DeembedError(f"manifest has no structure/empty pair at "
                           f"theta={theta_deg}, phi={phi_deg}")
    raw = read_project_sparams(st["project_path"], st["expected_s_tree"])
    empty = read_project_sparams(em["project_path"], em["expected_s_tree"])
    L_exp = man.get("L_expected_um")
    lm = map_cst_labels(theta_deg, phi_deg, empty_blocks=empty,
                        L_expected_um=L_exp)
    for a in (1, 2):
        chk = lm["empty_checks"][f"mode{a}"]
        if not chk["sign_ok"]:
            raise DeembedError(
                f"empty-cell phase-slope SIGN check FAILED for mode {a} at "
                f"theta={theta_deg} (slope {chk['slope_rad_per_THz']:+.4f} "
                f"rad/THz): conjugation direction or the 'inward' "
                f"convention is wrong -- do not de-embed")
    f, S11_de, S21_de = deembed_blocks(raw, empty)
    return f, S11_de, S21_de, lm, lm["empty_checks"]


# ===========================================================================
# LEGACY run_v3 / run_empty diagnostics (measurement, NOT a gate)
# ===========================================================================

def legacy_diagnostics(out_prefix="legacy_v3", verbose=True):
    """Apply the machinery to the pre-pinned-domain legacy runs.

    These runs PREDATE the par. 7 cellpad edit; the doc predicted a possible
    phase-reference systematic from differing auto-domains.  What the legacy
    data actually allows (discovered offline, 2026-08-06):

    * run_v3 (structure): solved; complex S available (csv = the offline
      cst.results tree's mode-2 pair, verified to 7e-10).
    * run_empty (empty): the solve FAILED -- Result/Model.log says
      'Could not read mesh'.  The project contains ZERO solids (history =
      Units/Freq/Boundary/Background/Ports/Samples only), so the un-pinned
      empty domain was degenerate and unmeshable.  There is NO empty-cell
      S data anywhere (no csv, no tree entries).  This is the doc's
      pinned-cellpad rationale confirmed in the strongest form: the
      un-pinned empty reference is not merely phase-shifted, it is
      UNSOLVABLE.

    The de-embedding therefore uses the ANALYTIC empty cell at run_v3's own
    measured z-extent (CADSurf bounding-box diagonal in Result/log.tet:
    L = sqrt(diag^2 - 2 p^2) = 5.814687 um = metal 0.1 um + 2 x
    lambda_center/4), clearly labeled as such.  Reported:

      1. the empty phase-slope check on the analytic empty (machinery
         self-check: slope sign + fitted L);
      2. the closure residual vs periodic_results.npz, magnitude AND phase
         separately (gate evaluated for information only);
      3. the residual phase slope in um of path length: the systematic
         length error left after removing the metadata L -- plus the
         directly fitted L from arg(conj(S21_csv)) - arg(S21_model);
      4. both projects' actual z-extents (run_v3 from the mesh log;
         run_empty: none exists -- mesh never built).
    """
    log = print if verbose else (lambda *a, **k: None)
    log("=" * 72)
    log("LEGACY run_v3 / run_empty diagnostics -- measurement, NOT a gate")
    log("(runs predate the par. 7 pinned-domain edit)")
    log("=" * 72)

    # ---- z-extents from project metadata --------------------------------
    L_v3 = parse_domain_z_extent(RUN_V3 / "saw_unitcell" / "Result"
                                 / "log.tet")
    lam_c = C_UM_THZ / (0.5 * (14.99 + 37.47))
    L_rule = 0.1 + 2.0 * lam_c / 4.0
    log(f"[z-extent] run_v3 (CADSurf diagonal): L = {L_v3:.6f} um")
    log(f"[z-extent] lambda_center/4 rule:      L = {L_rule:.6f} um "
        f"(|diff| = {abs(L_v3 - L_rule) * 1e3:.2e} nm)")
    empty_z_note = None
    try:
        parse_domain_z_extent(RUN_EMPTY / "empty_cell" / "Result"
                              / "log.tet")
    except DeembedError as e:
        empty_z_note = str(e)
        log(f"[z-extent] run_empty: NONE -- {e}")
    empty_status = None
    out_json = RUN_EMPTY / "empty_cell" / "Result" / "output.json"
    if out_json.exists():
        msgs = json.loads(out_json.read_text(encoding="utf-8"))
        empty_status = "; ".join(m["message"]
                                 for m in msgs.get("messages", []))
        log(f"[run_empty solver messages] {empty_status}")

    # ---- structure data --------------------------------------------------
    f_cst, s11_cst, s21_cst = read_sparams_csv(RUN_V3
                                               / "s_params_complex.csv")
    log(f"[run_v3] csv: {len(f_cst)} points, "
        f"{f_cst[0]:.4g}-{f_cst[-1]:.4g} THz (Floquet mode 2 pair)")

    # ---- offline cst.results extractor exercised on the real projects ----
    cst_reader_status = {}
    try:
        raw_v3 = read_project_sparams(RUN_V3 / "saw_unitcell.cst")
        s11_tree = raw_v3["1D Results\\S-Parameters\\SZmax(2),Zmax(2)"][1]
        s21_tree = raw_v3["1D Results\\S-Parameters\\SZmin(2),Zmax(2)"][1]
        dev = max(float(np.max(np.abs(s11_tree - s11_cst))),
                  float(np.max(np.abs(s21_tree - s21_cst))))
        xpol = max(float(np.max(np.abs(
            raw_v3[f"1D Results\\S-Parameters\\S{side}(1),Zmax(2)"][1])))
            for side in ("Zmax", "Zmin"))
        cst_reader_status["run_v3"] = (
            f"OK: 8 channel-dictionary entries read offline; mode-2 pair "
            f"matches csv to {dev:.1e}; raw cross-pol max {xpol:.1e}")
    except DeembedError as e:
        cst_reader_status["run_v3"] = f"FAILED: {e}"
    try:
        read_project_sparams(RUN_EMPTY / "empty_cell.cst")
        cst_reader_status["run_empty"] = ("unexpectedly OK -- legacy notes "
                                          "stale?")
    except DeembedError as e:
        cst_reader_status["run_empty"] = f"correctly refused: {e}"
    for k, v in cst_reader_status.items():
        log(f"[cst.results | {k}] {v}")

    # ---- empty data: none (see docstring); build the analytic empty ------
    s21_empty_cst = analytic_empty_s21(f_cst, L_v3, 0.0, convention="cst")
    log("[empty] no legacy empty-cell S data exists (solve failed); "
        "using the ANALYTIC empty at run_v3's own L")

    # ---- empty phase check (machinery self-check on the analytic empty) --
    chk = check_empty_phase(f_cst, conj_cst(s21_empty_cst), 0.0,
                            L_expected_um=L_v3)
    log(f"[empty phase check | ANALYTIC empty, self-check] "
        f"slope = {chk['slope_rad_per_THz']:+.6f} rad/THz "
        f"(expected {chk['slope_expected_rad_per_THz']:+.6f}), "
        f"L_fit = {chk['L_fit_um']:.6f} um, sign_ok = {chk['sign_ok']}, "
        f"passed = {chk['passed']}")

    # ---- de-embed + closure ---------------------------------------------
    s11_de, s21_de = deembed_spectra(s11_cst, s21_cst, s21_empty_cst)
    cl = closure_normal(f_cst, {"mode2": s11_de}, {"mode2": s21_de},
                        label=out_prefix + "_closure")
    log(f"[closure vs periodic_results.npz | LEGACY, informational] "
        f"worst complex residual = {cl['worst']:.3e} "
        f"(gate {cl['gate']:g}: {'PASS' if cl['passed'] else 'FAIL'})")
    log(f"  magnitude-only closure: "
        f"S11 {cl['mag_dev']['S11_mode2']:.3e}, "
        f"S21 {cl['mag_dev']['S21_mode2']:.3e} "
        f"(REPORT par. 4b record: 2.9e-3 / 1.1e-3)")
    log(f"  phase closure: S11 {cl['phase_dev_rad']['S11_mode2']:.3e} rad, "
        f"S21 {cl['phase_dev_rad']['S21_mode2']:.3e} rad")

    # ---- residual phase slope in um of path length ----------------------
    with np.load(REF_NPZ) as z:
        f_ref = z["freq"] / 1e12
        ref_s11, ref_s21 = z["S11"], z["S21"]
    s11_g = interp_to_grid(f_cst, s11_de, f_ref)
    s21_g = interp_to_grid(f_cst, s21_de, f_ref)
    k_ref = 2.0 * np.pi * f_ref / C_UM_THZ
    slopes = {}
    for name, de, ref in (("S21", s21_g, ref_s21), ("S11", s11_g, ref_s11)):
        dphi = np.unwrap(np.angle(de / ref))
        sl, off = np.polyfit(k_ref, dphi, 1)      # rad per (rad/um) = um
        slopes[name] = dict(dL_um=float(sl), offset_rad=float(off),
                            rms_rad=float(np.sqrt(np.mean(
                                (dphi - (sl * k_ref + off)) ** 2))))
        log(f"[residual phase slope | {name}] "
            f"dL = {sl * 1e3:+.2f} nm of path length "
            f"(offset {off:+.4f} rad, rms about fit "
            f"{slopes[name]['rms_rad']:.2e} rad)")
    # direct L fit: what L would a perfect empty have needed?
    dphi_raw = np.unwrap(np.angle(conj_cst(
        interp_to_grid(f_cst, s21_cst, f_ref)) / ref_s21))
    L_direct, _ = np.polyfit(k_ref, dphi_raw, 1)
    slopes["L_direct_um"] = float(L_direct)
    log(f"[direct fit] L from arg(conj(S21_raw)) - arg(S21_model): "
        f"{L_direct:.6f} um  (metadata L = {L_v3:.6f} um, "
        f"diff = {(L_direct - L_v3) * 1e3:+.2f} nm)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{out_prefix}_diagnostics.npz"
    np.savez(out, f_THz=f_cst, L_v3_um=L_v3, L_rule_um=L_rule,
             L_direct_um=slopes["L_direct_um"],
             s11_de=s11_de, s21_de=s21_de,
             closure_worst=cl["worst"],
             closure_sigma=cl["sigma"], closure_f_THz=cl["f_THz"],
             dL_S21_um=slopes["S21"]["dL_um"],
             dL_S11_um=slopes["S11"]["dL_um"],
             phase_offset_S21_rad=slopes["S21"]["offset_rad"],
             phase_offset_S11_rad=slopes["S11"]["offset_rad"],
             empty_status=str(empty_status),
             empty_phase_check=json.dumps(chk),
             cst_reader_status=json.dumps(cst_reader_status))
    log(f"[out] {out}")
    log(f"[out] {cl['out_npz']}")
    log(f"[out] {cl['out_fig']}")

    # ---- interpretation --------------------------------------------------
    log("-" * 72)
    log("INTERPRETATION (legacy-data diagnostics):")
    log(f"  * The un-pinned empty project failed outright "
        f"('{empty_status}'): it contains zero solids, so its auto domain "
        f"was degenerate.  The par. 7 pinned-cellpad edit is confirmed in "
        f"the strongest form -- without it there IS no empty reference.")
    log(f"  * De-embedding run_v3 against the ANALYTIC empty at its own "
        f"measured L = {L_v3:.4f} um leaves a path-length systematic of "
        f"{slopes['S21']['dL_um'] * 1e3:+.1f} nm (S21) / "
        f"{slopes['S11']['dL_um'] * 1e3:+.1f} nm (S11) and a constant "
        f"phase offset of {slopes['S21']['offset_rad']:+.4f} / "
        f"{slopes['S11']['offset_rad']:+.4f} rad.")
    log("  * Magnitudes close at the REPORT par. 4b level; the complex "
        "closure is limited by the phase terms above.  A REAL solved empty "
        "cell in the SAME pinned domain (the campaign's empty_th00) removes "
        "exactly these systematics -- the closure residual that remains "
        "after that division is the true fit sigma.")
    return dict(L_v3_um=L_v3, L_rule_um=L_rule, slopes=slopes,
                closure=cl, empty_phase_check=chk,
                empty_status=empty_status, empty_z_note=empty_z_note,
                cst_reader_status=cst_reader_status)


if __name__ == "__main__":
    legacy_diagnostics()

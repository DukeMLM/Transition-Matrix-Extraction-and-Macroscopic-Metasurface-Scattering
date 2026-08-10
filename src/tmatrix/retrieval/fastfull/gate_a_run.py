"""Run the M2 Gate A blind-recovery study over the candidate set.

Implements reviewer recommendation 3 (`review.md`, 2026-08-07): carry
`small@8` as the cost baseline, `pool-3@8` as the best optimistic-global-error
candidate and `generic@8` as the algebraic correctness reference, plus one
geometrically distinct encoding held out and never used in any recovery.

Designs are read from `results/fastfull/m1_study.json` so the study and the
recovery cannot drift apart.  The reference wheel T is used only as the
synthetic ground truth to be recovered blind, and the CONTROL run uses an
exactly-D4h passive draw so that the reference file's own symmetry violation
is separated from the recovery error.

Run:  python -m fastfull.gate_a_run [--trials 3] [--lam 8]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from tmatrix.aggregation.vswf import ModeBasis                                    # noqa: E402
from tmatrix.aggregation.tmat_io import TMatrixData                               # noqa: E402
from tmatrix.paths import FASTFULL_RESULTS

from . import design as dz                                    # noqa: E402
from . import symmetry as sym                                 # noqa: E402
from . import synthetic as sy                                 # noqa: E402
from . import m1_study as ms                                  # noqa: E402

STUDY_JSON = os.path.join(FASTFULL_RESULTS, "m1_study.json")
OUT_DIR = os.path.dirname(STUDY_JSON)

# A geometrically distinct holdout: different aspect ratio, different lattice
# angle, different Bloch point.  Never used to recover anything.
HOLDOUT = dz.Design(9.4, 12.7, 84.0, 61.0, 0.317, -0.204)


def _design(entry):
    if "design" in entry:
        q = entry["design"]
        return dz.Design(q["p1_um"], q["p2_um"], q["gamma_deg"],
                         q["alpha_deg"], q["f1"], q["f2"])
    return [dz.Design(q["p1_um"], q["p2_um"], q["gamma_deg"], q["alpha_deg"],
                      q["f1"], q["f2"]) for q in entry["designs"]]


def run(lam_um=8.0, n_trials=3, n_multistart=2, seed=12345, out_dir=OUT_DIR):
    modes = ModeBasis.standard(3)
    B, meta = sym.build_d4h_reciprocity_basis(modes)
    data = TMatrixData(ms.REF_TMAT)
    i = int(np.argmin(np.abs(data.wavelength_um - lam_um)))
    k, T_ref = data.k_at(i), data.T[i]
    sigma = dz.measured_sigma(float(data.wavelength_um[i]))
    cons = dz.Constraints(kz_min_frac=0.2, wood_margin=0.05, n_orders_min=1,
                          n_orders_max=40, area_max_um2=5000.0)

    with open(STUDY_JSON) as fh:
        study = json.load(fh)
    cands = []
    for tag in ("small@8", "generic@8", "pool-3@8"):
        if tag in study["configs"] and study["configs"][tag].get(
                "feasible", True) is not False:
            cands.append((tag, _design(study["configs"][tag])))

    print("lambda %.3f um, sigma %.4e, %d candidates + holdout"
          % (data.wavelength_um[i], sigma, len(cands)), flush=True)
    print("reference wheel D4h residual %.3e (its own C4v-violation noise) "
          "-- the control below separates it from recovery error"
          % sym.symmetry_residual(T_ref[None], B)[0], flush=True)

    # CONTROL: an exactly-D4h target.  Any residual noise-free error here
    # would be a defect in the recovery, not in the reference file.
    rng = np.random.default_rng(seed)
    T_ctrl = sym.random_passive_d4h(B, rng, n_draw=1,
                                    target_fro=float(
                                        np.linalg.norm(T_ref)))[0][0]
    print("\n=== control: exactly-D4h synthetic target ===", flush=True)
    ctrl = sy.gate_a_study(cands[:1], k, modes, B, T_ctrl, sigma, cons,
                           holdout=HOLDOUT, n_trials=1,
                           n_multistart=n_multistart, seed=seed)
    print(sy.format_gate_a(ctrl), flush=True)

    print("=== reference wheel as the blind target ===", flush=True)
    main_out = sy.gate_a_study(cands, k, modes, B, T_ref, sigma, cons,
                               holdout=HOLDOUT, n_trials=n_trials,
                               n_multistart=n_multistart, seed=seed)
    print(sy.format_gate_a(main_out), flush=True)

    rec = dict(lam_um=float(data.wavelength_um[i]), sigma=float(sigma),
               freq_index=i, holdout=HOLDOUT.to_dict(),
               reference_d4h_residual=float(
                   sym.symmetry_residual(T_ref[None], B)[0]),
               control=ctrl, reference=main_out)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "gate_a_%.0fum.json" % lam_um)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1, default=ms._json_default)
    print("wrote %s" % path, flush=True)

    md = os.path.join(out_dir, "GATE_A_STUDY.md")
    with open(md, "w") as fh:
        fh.write(_markdown(rec))
    print("wrote %s" % md, flush=True)
    return rec


def _markdown(rec):
    L = ["# Gate A — blind noisy synthetic recovery (M2)", "",
         "Generated by `retrieval/fastfull/gate_a_run.py`.  Blind recovery of "
         "a known T0 from synthetic coded-cell data, with the REAL Ewald "
         "lattice coupling, at the frequency-matched measured discrepancy "
         "sigma = %.4e (lambda = %.3f um)." % (rec["sigma"], rec["lam_um"]),
         "",
         "The point of this study is to close the M1 error bracket "
         "empirically.  M1 could only say the predicted T error lies between "
         "an iid posterior and a norm-bounded adversarial bound.  Here each "
         "error MODEL is injected and the resulting T error measured, so "
         "`bracket position` (0 = the iid end, 1 = the adversarial end) says "
         "where realistic structured discrepancies actually land.", "",
         "The recovery is blind throughout: a C = 0 linear seed, continuation "
         "in the lattice coupling, then Levenberg-Marquardt with the analytic "
         "Jacobian.  No bright mask, no oracle, no reference T.", "",
         "## Control — exactly-D4h target", "",
         "The reference wheel's own D4h residual is %.3e, so a recovery "
         "restricted to the 40-coefficient space can never reproduce it "
         "exactly.  This control uses a passive draw that lies exactly in "
         "the space, isolating recovery error from the file's asymmetry."
         % rec["reference_d4h_residual"], "", "```",
         sy.format_gate_a(rec["control"]), "```", "",
         "## Reference wheel as the blind target", "", "```",
         sy.format_gate_a(rec["reference"]), "```", ""]
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lam", type=float, default=8.0)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--multistart", type=int, default=2)
    p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args(argv)
    run(lam_um=a.lam, n_trials=a.trials, n_multistart=a.multistart,
        seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

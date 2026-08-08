"""CST cost proxy for the design objective (proposal par. 7.2, Gate speed).

The proposal's sharpest warning about the generic algebraic track is that
excellent angular conditioning can still LOSE: a cell large enough to open
eight diffraction orders has ~200x the area of the campaign cell, many more
degrees of freedom, and ~30 modal right-hand sides instead of 4.  "One CST
project" is not one solve (par. 1, par. 12), so the design objective has to
carry a cost term or it will happily pick an unaffordable cell.

What this model is
------------------
A RANKING proxy, not a predictor.  It is anchored on the campaign's MEASURED
numbers rather than on a first-principles mesh estimate, because the real
mesh is adaptive and refines around a 0.1 um metal film -- any absolute
a-priori dof count would be fiction.  Measured anchors (retrieval/HANDOFF.md,
campaign of 2026-08-07):

    cell            2.0 x 2.0 um, pinned domain height L = 11.714687 um
    structure solve 50-78 s        (used: 64 s)
    empty solve     16-23 s        (used: 20 s)
    excitations     4 port modes total (2 per port x 2 ports)

The model splits a solve into one mesh factorization plus one back
substitution per right-hand side,

    t_total = t_fac_ref (N/N_ref)^p_fac  +  n_rhs t_rhs_ref (N/N_ref)^p_rhs

with the relative dof count taken as area-proportional in the background and
constant in the metal,

    N/N_ref = f_bg (A_cell/A_ref) + (1 - f_bg).

Defaults p_fac = 1.5, p_rhs = 1.15, f_bg = 0.5 bracket a 3-D sparse direct
solver (factorization somewhere between O(N) and O(N^2), back substitution
near O(N^{4/3})).  EVERY one of these is a knob, and M3 must replace them
with a measured factorization/RHS split before any speed claim is made.
Peak memory is modelled as N^{4/3} for the same reason.

`n_rhs` deliberately counts EVERY modelled port mode on BOTH ports, including
the evanescent ones retained for port convergence, because CST's
`.Stimulation "All", "All"` excites all of them.
"""
import numpy as np

# ------------------------------------------------------- measured anchors
A_REF_UM2 = 4.0            # campaign cell area
T_FAC_REF_S = 64.0 - 4 * 5.0   # structure solve minus its four RHS solves
T_RHS_REF_S = 5.0          # per-excitation back substitution at the anchor
N_RHS_REF = 4
MEM_REF_GB = 1.0           # nominal; only ratios are used


class CostModel:
    """Relative CST cost of one Floquet project.  All exponents are knobs."""

    def __init__(self, a_ref=A_REF_UM2, t_fac_ref=T_FAC_REF_S,
                 t_rhs_ref=T_RHS_REF_S, p_fac=1.5, p_rhs=1.15, p_mem=4.0 / 3.0,
                 f_bg=0.5, mem_ref_gb=MEM_REF_GB):
        self.a_ref = float(a_ref)
        self.t_fac_ref = float(t_fac_ref)
        self.t_rhs_ref = float(t_rhs_ref)
        self.p_fac = float(p_fac)
        self.p_rhs = float(p_rhs)
        self.p_mem = float(p_mem)
        self.f_bg = float(f_bg)
        self.mem_ref_gb = float(mem_ref_gb)

    def dof_ratio(self, area):
        return self.f_bg * (float(area) / self.a_ref) + (1.0 - self.f_bg)

    def n_port_modes(self, n_orders, n_evanescent=2):
        """Modes modelled per physical port: 2 per propagating order plus a
        declared number of evanescent modes retained for port convergence
        (proposal par. 7.2; they are modelled but not used as excitations in
        version 1 -- CST still solves them)."""
        return 2 * int(n_orders) + int(n_evanescent)

    def n_rhs(self, n_orders, n_evanescent=2, n_ports=2):
        return n_ports * self.n_port_modes(n_orders, n_evanescent)

    def project(self, area, n_orders, n_evanescent=2, n_ports=2):
        """Wall time / memory / RHS count for one structured project."""
        r = self.dof_ratio(area)
        rhs = self.n_rhs(n_orders, n_evanescent, n_ports)
        t_fac = self.t_fac_ref * r ** self.p_fac
        t_rhs = rhs * self.t_rhs_ref * r ** self.p_rhs
        return dict(dof_ratio=float(r), n_rhs=int(rhs),
                    t_factor_s=float(t_fac), t_rhs_s=float(t_rhs),
                    t_total_s=float(t_fac + t_rhs),
                    mem_gb=float(self.mem_ref_gb * r ** self.p_mem))

    def campaign(self, area, n_orders, n_evanescent=2, n_ports=2,
                 n_encodings=1, empty_fraction=0.35):
        """Structured + matched-empty cost of a whole encoding set.

        The proposal requires the structured AND empty wall time to be
        reported together (par. 10, speed gate): an empty cell of the same
        size is a real cost, measured at ~0.35 of a structure solve in the
        campaign (16-23 s vs 50-78 s).
        """
        one = self.project(area, n_orders, n_evanescent, n_ports)
        total = n_encodings * one["t_total_s"] * (1.0 + float(empty_fraction))
        out = dict(one)
        out["n_encodings"] = int(n_encodings)
        out["t_campaign_s"] = float(total)
        out["t_campaign_min"] = float(total / 60.0)
        return out


def baseline_conventional(n_illuminations=46, t_per_run_s=64.0,
                          t_postprocess_s=0.0):
    """The benchmark the fast route must beat: conventional isolated-particle
    extraction with many plane-wave illuminations.

    Default 46 illuminations is the Fibonacci set the collaborator's
    `cst_tmatrix` actually used for the reference file (see the project
    memory), and 64 s reuses the campaign's measured structure-solve time so
    that the two sides of the comparison share one anchor.  This is a
    PLACEHOLDER benchmark: the real comparison must be run at matched
    accuracy (par. 10), not asserted here.
    """
    t = n_illuminations * t_per_run_s + t_postprocess_s
    return dict(n_runs=int(n_illuminations), t_total_s=float(t),
                t_total_min=float(t / 60.0))


def cost_penalty(cost_s, cost_ref_s, gamma=0.5):
    """Multiplicative design penalty (cost_ref / cost)^gamma, capped at 1.

    gamma trades conditioning against runtime; 0.5 means a 4x more expensive
    cell must double the worst singular value to be preferred.  Capped at 1
    so a cheap cell gets no bonus -- the objective is identifiability, with
    cost as a brake, not the other way round.
    """
    if cost_s <= 0:
        return 1.0
    return float(min(1.0, (float(cost_ref_s) / float(cost_s)) ** gamma))

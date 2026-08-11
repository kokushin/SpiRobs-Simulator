"""Auto-grasp tension schedule: the paper's Fig. 3A antagonistic sequence.

Faithful port of the sequence described in *SpiRobs* Fig. 3A, mapped onto
the 3-cable arm the way the paper's own 3-cable demo does ("a strategy
similar to the one for 2D grasping"): cable 0 plays the packing cable F1,
and cables 1+2 — whose planar resultant opposes cable 0 — play the
uncurling cable F2.

  packing    t < 2 s    ramp F1: the arm curls into a logarithmic spiral
  reaching   t < 5 s    ramp F2 with F1 held: capstan friction kills the
                        fresh tension toward the packed tip, so the base
                        uncurls first and the arm reaches out while the
                        tip spiral survives (the paper's sequential
                        deformation; needs a real cable friction mu)
  wrapping   t < 9 s    relax F1 with F2 held: the tip spiral unrolls
                        onto the object's surface (climbing)
  grasping   t < 11 s   raise F2 for a firm grasp
  holding    afterwards hold; at t = 12 s the presentation rig lets go
                        (string release / stage retract) and the wrap
                        alone carries the object

The paper gives no Newton values for the 3-cable arm (it was operated
manually), so the magnitudes here are this model's calibration. Whether
the grasp holds is up to the physics — "this object slips out of this
tension plan" is an observable outcome, not a scripted one.
"""

from dataclasses import dataclass


PHASES = ("packing", "reaching", "wrapping", "grasping", "holding")

# Phase boundaries [s], shared with the stage/string presentation timeline.
T_PACK = 2.0
T_REACH = 5.0
T_WRAP = 9.0
T_GRASP = 11.0
T_RELEASE = 12.0  # presentation rig lets go (string release / stage out)

# Calibrated tensions [N] for this model (see grasp.py docstring).
F_PACK = 6.0  # packed spiral
F_REACH = 3.0  # opposing resultant during reach/wrap
F_WRAP = 2.0  # relaxed packing tension at the end of wrapping
F_GRASP = 6.0  # opposing resultant for the firm grasp


def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


@dataclass(frozen=True)
class GraspCommand:
    phase: str
    cable_forces: tuple[float, float, float]


def auto_grasp_command(elapsed_seconds: float, primary_cable: int = 0) -> GraspCommand:
    t = max(0.0, elapsed_seconds)

    if t < T_PACK:
        phase = "packing"
        packing = F_PACK * (t / T_PACK)
        opposing = 0.0
    elif t < T_REACH:
        phase = "reaching"
        packing = F_PACK
        opposing = F_REACH * _smooth((t - T_PACK) / (T_REACH - T_PACK))
    elif t < T_WRAP:
        phase = "wrapping"
        packing = F_PACK + (F_WRAP - F_PACK) * _smooth((t - T_REACH) / (T_WRAP - T_REACH))
        opposing = F_REACH
    elif t < T_GRASP:
        phase = "grasping"
        packing = F_WRAP
        opposing = F_REACH + (F_GRASP - F_REACH) * _smooth((t - T_WRAP) / (T_GRASP - T_WRAP))
    else:
        phase = "holding"
        packing = F_WRAP
        opposing = F_GRASP

    forces = [opposing, opposing, opposing]
    forces[primary_cable % 3] = packing
    return GraspCommand(phase=phase, cable_forces=(forces[0], forces[1], forces[2]))

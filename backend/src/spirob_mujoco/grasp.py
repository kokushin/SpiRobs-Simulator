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


# ---------------------------------------------------------------- array mount
#
# The paper's multi-SpiRob array (Fig. 6B) grasps by ENTANGLEMENT with the
# same pack-first trick as Fig. 3A, rotated into the vertical: the arms
# REST packed into spirals near the base plate, the rigid arm APPROACHES
# from above so the packed spirals surround the object, and for the GRASP
# the packing tension is relaxed so every arm unrolls DOWNWARD along the
# object's surface — the Fig. 3A "climbing" in the gravity direction —
# draping around it before the cables re-tension for PICK UP. Because all
# arms drape at once, the radial push forces cancel and a free-standing
# object is not knocked over. The paper gives no tensions for the array
# either, so the magnitudes are this model's calibration.

ARRAY_PHASES = (
    "packing",
    "approaching",
    "draping",
    "squeezing",
    "lifting",
    "carrying",
    "lowering",
    "releasing",
    "retreating",
    "done",
)

# Phase boundaries [s].
TA_PACK = 2.5
TA_APPROACH = 5.5
TA_DRAPE = 9.0
# Quasi-static squeeze: a fast squeeze makes the three arms climb the rod
# with unsynchronized stick-slip and cants it; 4 s keeps the climb gentle
# enough that the cant stays inside the rod's self-righting cone by
# set-down (it also relaxes further as the carry accelerates the base).
TA_SQUEEZE = 13.0
TA_LIFT = 16.0
TA_CARRY = 21.0
TA_LOWER = 22.5
TA_RELEASE = 24.5
TA_RETREAT = 27.5

F_PACK_ARRAY = 6.0  # rest-state packing tension per arm [N]
# Residual tension after the drape: the capstan keeps the tip curled while
# the proximal arm unrolls down the object's flank (Fig. 3A wrapping).
F_DRAPE = 1.5
# Firm-grip tension for pick-up (the paper's "curl up to pick up"). The
# capstan law eats most of the base tension by the curled tip —
# exp(-0.4 * 2 pi) ~ 0.08 — so the distal units see ~1 N of it.
F_SQUEEZE = 15.0
APPROACH_DZ = 0.14  # rest/pack altitude above the grasp height [m]
LIFT_DZ = 0.05  # gantry lift height [m]
CARRY_DX = 0.20  # transport distance along +X [m]
LOWER_DZ = 0.02  # residual height while setting the object down [m]
# High enough that the straightened arms' tips clear the deposited rod's
# top even when it stands off the gantry axis (tips reach base_height +
# RETREAT_DZ - 0.231).
RETREAT_DZ = 0.25
# Opposing-pair tension while the cage leaves the deposited rod: pulling
# cables 1+2 (their planar resultant opposes cable 0) actively flares the
# arms radially OUTWARD, so the rising cage never brushes the rod — the
# 3-cable antagonism the paper's machines steer with, used as a claw-open.
F_OPEN = 2.5


@dataclass(frozen=True)
class ArrayGraspCommand:
    phase: str
    cable_forces: tuple[float, float, float]  # broadcast to every arm
    gantry_offset: tuple[float, float, float]


def array_grasp_command(elapsed_seconds: float) -> ArrayGraspCommand:
    t = max(0.0, elapsed_seconds)

    curl = F_SQUEEZE
    opening = 0.0
    dx, dz = CARRY_DX, LOWER_DZ
    if t < TA_PACK:
        phase = "packing"
        curl = F_PACK_ARRAY * _smooth(t / TA_PACK)
        dx, dz = 0.0, APPROACH_DZ
    elif t < TA_APPROACH:
        phase = "approaching"
        curl = F_PACK_ARRAY
        u = _smooth((t - TA_PACK) / (TA_APPROACH - TA_PACK))
        dx, dz = 0.0, APPROACH_DZ * (1.0 - u)
    elif t < TA_DRAPE:
        phase = "draping"
        curl = F_PACK_ARRAY + (F_DRAPE - F_PACK_ARRAY) * _smooth(
            (t - TA_APPROACH) / (TA_DRAPE - TA_APPROACH)
        )
        dx, dz = 0.0, 0.0
    elif t < TA_SQUEEZE:
        phase = "squeezing"
        curl = F_DRAPE + (F_SQUEEZE - F_DRAPE) * _smooth(
            (t - TA_DRAPE) / (TA_SQUEEZE - TA_DRAPE)
        )
        dx, dz = 0.0, 0.0
    elif t < TA_LIFT:
        phase = "lifting"
        dx, dz = 0.0, LIFT_DZ * _smooth((t - TA_SQUEEZE) / (TA_LIFT - TA_SQUEEZE))
    elif t < TA_CARRY:
        phase = "carrying"
        dx, dz = CARRY_DX * _smooth((t - TA_LIFT) / (TA_CARRY - TA_LIFT)), LIFT_DZ
    elif t < TA_LOWER:
        phase = "lowering"
        u = _smooth((t - TA_CARRY) / (TA_LOWER - TA_CARRY))
        dz = LIFT_DZ + (LOWER_DZ - LIFT_DZ) * u
    elif t < TA_RELEASE:
        # Trade the curl for the opposing pair: the grip fades while the
        # arms actively flare outward, off the grounded rod, instead of
        # springing back through it.
        phase = "releasing"
        u = _smooth((t - TA_LOWER) / (TA_RELEASE - TA_LOWER))
        curl = F_SQUEEZE * (1.0 - u)
        opening = F_OPEN * u
    elif t < TA_RETREAT:
        # Rise with the claw held open — no part of the cage brushes the
        # deposited rod on the way up.
        phase = "retreating"
        curl = 0.0
        opening = F_OPEN
        dz = LOWER_DZ + (RETREAT_DZ - LOWER_DZ) * _smooth(
            (t - TA_RELEASE) / (TA_RETREAT - TA_RELEASE)
        )
    else:
        phase, curl, opening, dz = "done", 0.0, 0.0, RETREAT_DZ

    return ArrayGraspCommand(
        phase=phase,
        cable_forces=(curl, opening, opening),
        gantry_offset=(dx, 0.0, dz),
    )

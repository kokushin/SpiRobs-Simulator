"""Auto-grasp tension schedule for the MuJoCo backend.

This is NOT the schedule of the web simulator: that one drove scripted
surface-following constraints, so its tensions never had to produce the
wrap physically. Here the wrap must emerge from tendon and contact
mechanics, which changes the working ranges completely (verified by
parameter sweeps):

- packing beyond ~2 N retracts the whole coil toward the base and peels
  it off the presented object, so the wrap forms at low tension;
- the presented object is released while a fading viscous damper absorbs
  the stored contact energy (instant release catapults rigid objects).

Phases:
  packing   0..4 s     settle for 1 s, then ramp cable 0 -> WRAP_TENSION
  wrapping  4..4.6 s   short hold at the wrap peak (longer holds let the
                       coil creep over the object surface and peel off)
  grasping  4.6..6.6 s fixture released; damper fades out
  holding   6.6 s..    constant tension, object fully dynamic

Known limitation: after release the wrap does not yet carry the object's
full weight — the distal joints are too soft under the r^3 similarity law
to close the coil mouth against a 45 g load, so the object settles out of
the coil. Wrap formation and the non-violent release are the validated
parts; sustained holding needs either more wrap turns (smaller objects),
capstan-graded tendons, or a hanging mount, and is future work.
"""

from dataclasses import dataclass


PHASES = ("packing", "wrapping", "grasping", "holding")

WRAP_TENSION = 1.8
SETTLE_END = 1.0
PACK_END = 4.0
WRAP_END = 4.6
RELEASE_END = 6.6


@dataclass(frozen=True)
class GraspCommand:
    phase: str
    cable_forces: tuple[float, float, float]
    release_target: bool
    # 1 -> full fixture assist, 0 -> object fully free. Fades during the
    # grasping phase so the load transfers to the wrap gradually.
    fixture_assist: float


def auto_grasp_command(elapsed_seconds: float, primary_cable: int = 0) -> GraspCommand:
    t = max(0.0, elapsed_seconds)

    if t < PACK_END:
        phase = "packing"
        ramp = max(0.0, t - SETTLE_END) / (PACK_END - SETTLE_END)
        packing = WRAP_TENSION * ramp
        assist = 1.0
    elif t < WRAP_END:
        phase = "wrapping"
        packing = WRAP_TENSION
        assist = 1.0
    elif t < RELEASE_END:
        phase = "grasping"
        packing = WRAP_TENSION
        assist = 1.0 - (t - WRAP_END) / (RELEASE_END - WRAP_END)
    else:
        phase = "holding"
        packing = WRAP_TENSION
        assist = 0.0

    forces = [0.0, 0.0, 0.0]
    forces[primary_cable % 3] = packing
    return GraspCommand(
        phase=phase,
        cable_forces=(forces[0], forces[1], forces[2]),
        release_target=t >= WRAP_END,
        fixture_assist=assist,
    )

"""Auto-grasp tension schedule: the paper's antagonistic sequence.

Faithful port of ``autoGraspCommand`` from ``src/physics.ts``, which encodes
the SpiRobs paper's Fig. 3A sequence with one packing cable against the
vector sum of the other two:

  packing    t < 1.2 s   ramp the packing cable to 6 N: the arm packs
                          into a logarithmic spiral from the tip
  reaching   t < 4.4 s   raise the opposing pair to 5.82 N: capstan
                          attenuation lets the base unwind while the
                          packed tip spiral survives, extending the arm
                          toward the object
  wrapping   t < 8.4 s   relax packing 6 -> 5.2 N with the opposing pair
                          held: the spiral climbs the object surface
  grasping   t < 10.2 s  raise the opposing pair to 9 N for friction
                          closure
  holding    afterwards  hold 5.2 / 9 N

Whether the grasp succeeds is up to the physics — the point of this
backend is that "this object is too heavy for this tension plan" is an
observable outcome, not a scripted one.
"""

from dataclasses import dataclass


PHASES = ("packing", "reaching", "wrapping", "grasping", "holding")


@dataclass(frozen=True)
class GraspCommand:
    phase: str
    cable_forces: tuple[float, float, float]


def auto_grasp_command(elapsed_seconds: float, primary_cable: int = 0) -> GraspCommand:
    t = max(0.0, elapsed_seconds)

    if t < 1.2:
        phase = "packing"
        packing = 6.0 * (t / 1.2)
        opposing = 0.0
    elif t < 4.4:
        phase = "reaching"
        progress = (t - 1.2) / 3.2
        packing = 6.0
        opposing = 5.82 * progress
    elif t < 8.4:
        phase = "wrapping"
        progress = (t - 4.4) / 4.0
        packing = 6.0 - 0.8 * progress
        opposing = 5.82
    elif t < 10.2:
        phase = "grasping"
        progress = (t - 8.4) / 1.8
        packing = 5.2
        opposing = 5.82 + 3.18 * progress
    else:
        phase = "holding"
        packing = 5.2
        opposing = 9.0

    forces = [opposing, opposing, opposing]
    forces[primary_cable % 3] = packing
    return GraspCommand(phase=phase, cable_forces=(forces[0], forces[1], forces[2]))

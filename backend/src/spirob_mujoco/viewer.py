"""Interactive MuJoCo viewer for model validation.

Usage:
    uv run spirob-viewer                       # idle robot, drag to inspect
    uv run spirob-viewer --grasp               # run the auto-grasp schedule
    uv run spirob-viewer --object soft_sphere  # elastic object (MuJoCo flex)
    uv run spirob-viewer --forces 6 0 0        # constant cable tensions [N]
"""

import argparse
import time

import mujoco.viewer

from .settings import OBJECT_KINDS, SimSettings
from .simulation import SpiRobSim


def main() -> None:
    parser = argparse.ArgumentParser(description="SpiRob MuJoCo viewer")
    parser.add_argument("--object", choices=OBJECT_KINDS, default="sphere")
    parser.add_argument("--size", type=float, default=52.0, help="object size [mm]")
    parser.add_argument("--mass", type=float, default=0.045, help="object mass [kg]")
    parser.add_argument("--grasp", action="store_true", help="run the auto-grasp schedule")
    parser.add_argument("--forces", type=float, nargs=3, default=None, metavar=("C0", "C1", "C2"))
    args = parser.parse_args()

    sim = SpiRobSim(
        SimSettings(object_kind=args.object, object_size_mm=args.size, object_mass=args.mass)
    )
    if args.grasp:
        sim.start_auto_grasp()
    elif args.forces is not None:
        sim.set_cable_forces(args.forces)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        previous = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            sim.step(min(now - previous, 0.05))
            previous = now
            viewer.sync()
            time.sleep(max(0.0, sim.model.opt.timestep - (time.monotonic() - now)))


if __name__ == "__main__":
    main()

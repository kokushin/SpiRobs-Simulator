"""Interactive MuJoCo viewer for model validation.

Usage:
    uv run spirob-viewer                       # idle robot, drag to inspect
    uv run spirob-viewer --grasp               # run the auto-grasp schedule
    uv run spirob-viewer --object soft_sphere  # elastic object (MuJoCo flex)
    uv run spirob-viewer --forces 6 0 0        # constant cable tensions [N]
    uv run spirob-viewer --mount array --grasp # 3-arm gantry transports the rod
"""

import argparse
import time

import mujoco.viewer

from .settings import MOUNTS, OBJECT_KINDS, SimSettings
from .simulation import SpiRobSim


def main() -> None:
    parser = argparse.ArgumentParser(description="SpiRob MuJoCo viewer")
    parser.add_argument("--object", choices=OBJECT_KINDS, default="cylinder")
    parser.add_argument("--size", type=float, default=30.0, help="object size (diameter) [mm]")
    parser.add_argument("--mass", type=float, default=0.07, help="object mass [kg]")
    parser.add_argument("--mount", choices=MOUNTS, default="planar")
    parser.add_argument(
        "--arms",
        type=int,
        default=None,
        help="arm count on the gantry ring (array mount only; default 3)",
    )
    parser.add_argument("--stiffness", type=float, default=0.7, help="base joint stiffness [Nm/rad]")
    parser.add_argument("--young", type=float, default=2.0e4, help="soft object Young's modulus [Pa]")
    parser.add_argument("--grasp", action="store_true", help="run the paper's auto-grasp sequence")
    parser.add_argument("--forces", type=float, nargs=3, default=None, metavar=("C0", "C1", "C2"))
    args = parser.parse_args()

    sim = SpiRobSim(
        SimSettings(
            object_kind=args.object,
            object_size_mm=args.size,
            object_mass=args.mass,
            mount=args.mount,
            arm_count=args.arms,
            stiffness=args.stiffness,
            object_young=args.young,
        )
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

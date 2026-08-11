"""Simulation settings mirroring the TypeScript ``SimSettings``.

Values keep the calibration semantics of the web simulator: ``stiffness`` is
the base-joint rotational stiffness in Nm/rad (distal joints follow the r^3
similarity law), masses are kilograms, sizes are millimeters.
"""

from dataclasses import dataclass, field


OBJECT_KINDS = ("sphere", "box", "cylinder", "soft_sphere", "none")
MOUNTS = ("hanging", "horizontal")


@dataclass
class SimSettings:
    # The paper's grasp sequence (Fig. 3A) operates a hanging arm reaching
    # down to an object, so that is the default here; 'horizontal' matches
    # the web simulator's default mount.
    mount: str = "hanging"
    cable_forces: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Base joint stiffness [Nm/rad]. The web simulator's 0.09 was tuned for
    # its scripted solver; with real dynamics the paper's 6 N pack tension
    # then coils the arm to the base, contradicting Fig. 3A. 0.7 reproduces
    # the paper's tip-first packing at the paper's tensions (calibrated to
    # the figure's kinematics, not to a physical measurement).
    stiffness: float = 0.7
    damping_ratio: float = 0.8
    # Capstan friction coefficient mu in T = T0 exp(-mu * sum |dtheta|),
    # same default as the web simulator.
    cable_friction: float = 0.08
    body_friction: float = 0.72
    gravity: float = 9.81
    total_mass: float = 0.0384
    object_kind: str = "sphere"
    object_size_mm: float = 52.0
    object_mass: float = 0.045
    # Soft object material (only used for object_kind == "soft_sphere").
    # 20 kPa is in the range of a soft silicone/foam ball.
    object_young: float = 2.0e4
    object_poisson: float = 0.3
    timestep: float = 0.002
    # Grasp-point position of the pedestal rail stage: the wrap-phase
    # spiral sweep runs x in [-0.02, +0.035] at z ~0.12-0.13, so the stage
    # delivers the object into its center. pedestal_height is the platform
    # top above the floor.
    object_x: float = 0.01
    pedestal_height: float = 0.075

    def __post_init__(self) -> None:
        if self.object_kind not in OBJECT_KINDS:
            raise ValueError(f"unknown object_kind: {self.object_kind!r}")
        if self.mount not in MOUNTS:
            raise ValueError(f"unknown mount: {self.mount!r}")

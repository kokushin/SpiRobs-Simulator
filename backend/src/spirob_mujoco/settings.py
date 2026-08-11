"""Simulation settings mirroring the TypeScript ``SimSettings``.

Values keep the calibration semantics of the web simulator: ``stiffness`` is
the base-joint rotational stiffness in Nm/rad (distal joints follow the r^3
similarity law), masses are kilograms, sizes are millimeters.
"""

from dataclasses import dataclass, field


OBJECT_KINDS = ("sphere", "box", "cylinder", "soft_sphere", "none")


@dataclass
class SimSettings:
    cable_forces: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    stiffness: float = 0.09
    damping_ratio: float = 0.8
    cable_friction: float = 0.1
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
    # AUTO GRASP presentation pose (MuJoCo Z-up meters). None uses the
    # default fixture pose in model.FIXTURE_POS.
    fixture_pos: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.object_kind not in OBJECT_KINDS:
            raise ValueError(f"unknown object_kind: {self.object_kind!r}")

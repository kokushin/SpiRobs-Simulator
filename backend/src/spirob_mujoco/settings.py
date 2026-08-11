"""Simulation settings mirroring the TypeScript ``SimSettings``.

Values keep the calibration semantics of the web simulator: ``stiffness`` is
the base-joint rotational stiffness in Nm/rad (distal joints follow the r^3
similarity law), masses are kilograms, sizes are millimeters.
"""

from dataclasses import dataclass, field


OBJECT_KINDS = ("sphere", "box", "cylinder", "soft_sphere", "none")
MOUNTS = ("planar", "hanging", "horizontal", "standing")


# Default object presentation pose per mount: (object_x, grasp_center_z).
# standing: legacy vertical-plane scene — the object is offered beside the
#   arm where the packed spiral's mouth opens (calibrated by sweep: the
#   wrapping phase reaches 4 simultaneous contact units there).
# hanging/horizontal: legacy scenes; 0.102 is the center height the 52 mm
#   pedestal calibration used.
# planar uses DEFAULT_PLANAR_POSE instead (the second value is a lateral
#   offset, not a height).
DEFAULT_GRASP_POSE = {
    "standing": (0.07, 0.14),
    "hanging": (0.01, 0.102),
    "horizontal": (0.01, 0.102),
}

# Planar (paper Fig. 3A, camera above the table): rod position (x, y) in the
# curl plane. The packing spiral curls toward +Y; the hand carries the rod
# to this pose during REACH so it meets the arm's flank just proximal of
# the packed tip spiral — the paper's Contact frame. Calibrated by sweep:
# here the tip spiral climbs over the rod during WRAP and coils around it
# with 3-4 simultaneous contact units during GRASP (positions further out,
# e.g. x=0.18, give 5 contacts but skip the climbing motion).
DEFAULT_PLANAR_POSE = (0.15, 0.02)


@dataclass
class SimSettings:
    # The paper's Fig. 3A photos are a TOP VIEW: the arm curls in a
    # horizontal plane on a smooth table (gravity perpendicular to the
    # bending plane) and a vertical wooden rod is held into the curl by
    # hand. 'planar' reproduces that arrangement and is the faithful
    # default. 'standing' / 'hanging' / 'horizontal' remain for exploration
    # ('horizontal' matches the web simulator's default mount).
    mount: str = "planar"
    cable_forces: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Base joint stiffness [Nm/rad]. The web simulator's 0.09 was tuned for
    # its scripted solver; with real dynamics the paper's 6 N pack tension
    # then coils the arm to the base, contradicting Fig. 3A. 0.7 reproduces
    # the paper's tip-first packing at the paper's tensions (calibrated to
    # the figure's kinematics, not to a physical measurement).
    stiffness: float = 0.7
    damping_ratio: float = 0.8
    # Capstan friction coefficient mu in T = T0 exp(-mu * sum |dtheta|).
    # The paper's grasp sequence relies on sequential deformation: freshly
    # applied opposing tension must die out toward the packed tip so the
    # base uncurls first (Reaching) and the tip unrolls onto the object
    # (Wrapping). The web simulator's 0.08 is far too weak for that — a
    # packed arm (~2 pi accumulated bend) still passes 60% of the tension
    # to the tip and the tip spiral unwinds mid-reach. 0.4 reproduces the
    # paper's phenomenology (calibrated by sweep, not measured).
    cable_friction: float = 0.4
    body_friction: float = 0.72
    gravity: float = 9.81
    total_mass: float = 0.0384
    # Default object: the paper's wooden rod (~30 mm dia, held vertically
    # into the planar curl). size_mm is the diameter; a beech rod of that
    # diameter and the modeled length weighs roughly 70 g.
    object_kind: str = "cylinder"
    object_size_mm: float = 30.0
    object_mass: float = 0.07
    # Soft object material (only used for object_kind == "soft_sphere").
    # 20 kPa is in the range of a soft silicone/foam ball.
    object_young: float = 2.0e4
    object_poisson: float = 0.3
    timestep: float = 0.002
    # Object presentation pose. planar: the object is held ("by hand", a
    # weld released during HOLDING) at (object_x, object_y) in the curl
    # plane. Other mounts: the object CENTER is held at
    # (object_x, grasp_center_z) regardless of size — by a hanging string
    # for the standing mount or by a size-compensated pedestal stage for
    # hanging/horizontal. None means "use the mount's calibrated default".
    object_x: float | None = None
    object_y: float | None = None
    grasp_center_z: float | None = None

    def __post_init__(self) -> None:
        if self.object_kind not in OBJECT_KINDS:
            raise ValueError(f"unknown object_kind: {self.object_kind!r}")
        if self.mount not in MOUNTS:
            raise ValueError(f"unknown mount: {self.mount!r}")
        if self.mount == "planar":
            default_x, default_y = DEFAULT_PLANAR_POSE
            if self.object_x is None:
                self.object_x = default_x
            if self.object_y is None:
                self.object_y = default_y
            if self.grasp_center_z is None:
                self.grasp_center_z = 0.0  # unused in the planar scene
            return
        default_x, default_z = DEFAULT_GRASP_POSE[self.mount]
        if self.object_x is None:
            self.object_x = default_x
        if self.object_y is None:
            self.object_y = 0.0
        if self.grasp_center_z is None:
            self.grasp_center_z = default_z

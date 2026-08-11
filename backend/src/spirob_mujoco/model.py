"""MJCF model builder for the 3-cable SpiRob.

The kinematic layout ports the discrete Cosserat rod of the web simulator:
20 tapered units chained along +X, each elastic joint modeled as a Y/Z
hinge pair located between adjacent units. Joint stiffness follows the
``EI / l`` similarity law of the STL taper (r^3 scaling), and the three
cables become spatial tendons routed through per-unit sites at 120 deg,
driven by force-controlled motors — so capstan-style routing loads and the
wrap around an object emerge from the tendon and contact models instead of
being scripted.

MuJoCo is Z-up here; the base sits 0.2 m above the floor like the web
simulator (root y=0.08, floor y=-0.12).
"""

import xml.etree.ElementTree as ET
from math import cos, radians, sin

from .settings import SimSettings
from .unit_data import MM, UNIT_COUNT, UNIT_DATA, along_m, cable_radius_m, unit_volumes

# Base pose per mount. Hanging: arm points straight down (+X body axis
# rotated onto -Z) from high enough that the straight arm tip clears the
# pedestal; the packing cable 0 then curls the spiral toward -X.
MOUNT_POSE = {
    "horizontal": {"pos": (-0.06, 0.0, 0.2), "quat": (1.0, 0.0, 0.0, 0.0)},
    "hanging": {"pos": (0.0, 0.0, 0.32), "quat": (0.70710678, 0.0, 0.70710678, 0.0)},
}
JOINT_RANGE_DEG = 28.8  # 30 deg paper limit x 0.96, same margin as the rod solver
CABLE_PHASES_DEG = (0.0, 120.0, 240.0)
MAX_CABLE_FORCE = 15.0
# The object rests on a kinematic pedestal stage (a physical test rig, not
# a magic fixture): it waits outside the packing sweep, slides in during
# the REACH phase carrying the object by friction, and retracts during
# HOLDING so the wrap alone must carry the weight.
PEDESTAL_RADIUS = 0.016
PEDESTAL_FAR_OFFSET = -0.13


def _fmt(*values: float) -> str:
    return " ".join(f"{v:.8g}" for v in values)


def _joint_gains(settings: SimSettings, i: int) -> tuple[float, float, float]:
    """Stiffness, damping and armature for the joint driving unit ``i``.

    The web solver tracks each joint with a second-order response of
    natural frequency 13..21 rad/s and a tunable damping ratio. Matching
    that here: armature = k / w^2 gives the same effective inertia, and
    damping = 2 zeta k / w the same decay.
    """
    r0 = (UNIT_DATA[0][2] + UNIT_DATA[0][3]) * 0.25
    r = (UNIT_DATA[i][2] + UNIT_DATA[i][3]) * 0.25
    scale = max(0.012, (r / r0) ** 3)
    stiffness = settings.stiffness * scale
    omega = 13.0 + 8.0 * (i / (UNIT_COUNT - 1))
    armature = stiffness / omega**2
    damping = 2.0 * settings.damping_ratio * stiffness / omega
    return stiffness, damping, armature


def build_mjcf(settings: SimSettings) -> str:
    root = ET.Element("mujoco", model="spirob")
    # Flex elasticity is integrated explicitly, so the soft object needs a
    # smaller step than the rigid scenes (0.5 ms was verified stable for
    # young 5e3..2e4 Pa; 2 ms blows up regardless of integrator).
    timestep = settings.timestep
    if settings.object_kind == "soft_sphere":
        # Lighter vertices raise the elastic eigenfrequencies, so the stable
        # step shrinks with sqrt(mass).
        soft_step = 5e-4 * (settings.object_mass / 0.045) ** 0.5
        timestep = min(timestep, max(2e-4, soft_step))
    ET.SubElement(
        root,
        "option",
        timestep=_fmt(timestep),
        integrator="implicitfast",
        gravity=_fmt(0, 0, -settings.gravity),
    )

    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        friction=_fmt(settings.body_friction, 0.005, 0.0001),
    )
    ET.SubElement(default, "site", size="0.0008")

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", pos="0 0 1.5", dir="0 0 -1")
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="1 1 0.05",
        pos="0 0 0",
        friction=_fmt(0.92, 0.005, 0.0001),
        rgba="0.35 0.37 0.4 1",
    )

    volumes = unit_volumes()
    volume_sum = sum(volumes)

    parent = worldbody
    for i, (_, length, h, w) in enumerate(UNIT_DATA):
        if i == 0:
            pose = MOUNT_POSE[settings.mount]
            body = ET.SubElement(
                parent,
                "body",
                name="unit00",
                pos=_fmt(*pose["pos"]),
                quat=_fmt(*pose["quat"]),
            )
        else:
            separation = along_m(i) - along_m(i - 1)
            body = ET.SubElement(parent, "body", name=f"unit{i:02d}", pos=_fmt(separation, 0, 0))
            gap = max(0.0, separation - (UNIT_DATA[i - 1][1] + length) * 0.5 * MM)
            joint_x = -(length * 0.5 * MM + gap * 0.5)
            stiffness, damping, armature = _joint_gains(settings, i)
            for axis_name, axis in (("y", "0 1 0"), ("z", "0 0 1")):
                ET.SubElement(
                    body,
                    "joint",
                    name=f"u{i:02d}_{axis_name}",
                    type="hinge",
                    axis=axis,
                    pos=_fmt(joint_x, 0, 0),
                    range=_fmt(-JOINT_RANGE_DEG, JOINT_RANGE_DEG),
                    stiffness=_fmt(stiffness),
                    damping=_fmt(damping),
                    armature=_fmt(armature),
                )

        mass = settings.total_mass * volumes[i] / volume_sum
        ET.SubElement(
            body,
            "geom",
            name=f"unit{i:02d}_geom",
            type="box",
            size=_fmt(length * MM * 0.47, h * MM * 0.43, w * MM * 0.43),
            mass=_fmt(mass),
            rgba="0.92 0.92 0.95 1",
        )

        # Cable 0 runs along the bottom of the cross-section (-Z): packing it
        # coils the arm downward toward the presented object, while cables 1/2
        # sit symmetrically on the upper half so their sum opposes cable 0 in
        # the vertical plane — the planar antagonism of the paper's sequence.
        radius = cable_radius_m(i)
        for c, phase in enumerate(CABLE_PHASES_DEG):
            ET.SubElement(
                body,
                "site",
                name=f"u{i:02d}_c{c}",
                pos=_fmt(
                    0,
                    -radius * sin(radians(phase)),
                    -radius * cos(radians(phase)),
                ),
            )
        parent = body

    _add_object(worldbody, root, settings)

    contact = ET.SubElement(root, "contact")
    for i in range(1, UNIT_COUNT):
        ET.SubElement(
            contact,
            "exclude",
            body1=f"unit{i - 1:02d}",
            body2=f"unit{i:02d}",
        )

    # Each cable is split into per-joint tendon segments so the controller
    # can impose the capstan law T_i = T0 * exp(-mu * sum |dtheta|): a single
    # continuous MuJoCo tendon equalizes tension along its whole path, which
    # erases exactly the pack/unwind asymmetry the paper's antagonistic
    # grasp sequence depends on. Actuator order: cable-major, then joint.
    tendon = ET.SubElement(root, "tendon")
    actuator = ET.SubElement(root, "actuator")
    for c in range(3):
        for i in range(1, UNIT_COUNT):
            spatial = ET.SubElement(
                tendon,
                "spatial",
                name=f"cable{c}_seg{i:02d}",
                width="0.0004",
                rgba="0.85 0.4 0.15 1",
            )
            ET.SubElement(spatial, "site", site=f"u{i - 1:02d}_c{c}")
            ET.SubElement(spatial, "site", site=f"u{i:02d}_c{c}")
            ET.SubElement(
                actuator,
                "motor",
                name=f"cable{c}_seg{i:02d}_motor",
                tendon=f"cable{c}_seg{i:02d}",
                gear="-1",
                ctrlrange=_fmt(0, MAX_CABLE_FORCE),
            )

    if settings.object_kind != "none":
        # Stage servo (always the LAST actuator; the capstan controller
        # relies on the cable motors occupying ctrl[0:57]).
        ET.SubElement(
            actuator,
            "position",
            name="stage_servo",
            joint="stage_x",
            kp="400",
            kv="60",
            ctrlrange=_fmt(-0.05, 0.2),
        )

    return ET.tostring(root, encoding="unicode")


def _add_object(worldbody: ET.Element, root: ET.Element, settings: SimSettings) -> None:
    if settings.object_kind == "none":
        return
    radius = settings.object_size_mm * MM * 0.5
    x = settings.object_x + PEDESTAL_FAR_OFFSET  # stage waits outside the pack sweep
    top = settings.pedestal_height

    # Kinematic rail stage: a slide joint with a position servo, so the
    # platform has a real velocity and friction carries the object with it
    # (a mocap body reports zero velocity and objects slip off).
    stage = ET.SubElement(
        worldbody,
        "body",
        name="pedestal",
        pos=_fmt(x, 0, top * 0.5),
    )
    ET.SubElement(stage, "joint", name="stage_x", type="slide", axis="1 0 0", damping="8")
    # condim 6 + rolling friction so a rigid ball is carried by the moving
    # platform instead of rolling off the back (soft balls are carried by
    # their deformed contact patch either way).
    ET.SubElement(
        stage,
        "geom",
        name="pedestal_geom",
        type="cylinder",
        size=_fmt(PEDESTAL_RADIUS, top * 0.5),
        mass="0.5",
        condim="6",
        friction=_fmt(0.92, 0.02, 0.02),
        rgba="0.5 0.48 0.45 1",
    )

    if settings.object_kind == "soft_sphere":
        # MuJoCo 3 flex deformable: a tetrahedral ellipsoid with linear
        # elasticity. count/spacing span the requested diameter.
        count = 6
        spacing = (2 * radius) / (count - 1)
        flexcomp = ET.SubElement(
            worldbody,
            "flexcomp",
            name="object",
            type="ellipsoid",
            count=_fmt(count, count, count),
            spacing=_fmt(spacing, spacing, spacing),
            pos=_fmt(x, 0, top + radius + 0.001),
            dim="3",
            mass=_fmt(settings.object_mass),
            rgba="0.3 0.62 0.9 1",
        )
        # NOTE: keep elasticity damping at 0 — nonzero values destabilize the
        # flex solver in MuJoCo 3.11 (verified by parameter sweep).
        ET.SubElement(
            flexcomp,
            "elasticity",
            young=_fmt(settings.object_young),
            poisson=_fmt(settings.object_poisson),
        )
        ET.SubElement(flexcomp, "contact", selfcollide="none", internal="false")
        return

    body = ET.SubElement(
        worldbody,
        "body",
        name="object",
        pos=_fmt(x, 0, top + radius + 0.001),
    )
    ET.SubElement(body, "freejoint", name="object_free")
    common = {
        "name": "object_geom",
        "mass": _fmt(settings.object_mass),
        "condim": "6",
        "friction": _fmt(0.92, 0.02, 0.02),
        "rgba": "0.3 0.62 0.9 1",
    }
    if settings.object_kind == "sphere":
        ET.SubElement(body, "geom", type="sphere", size=_fmt(radius), **common)
    elif settings.object_kind == "box":
        ET.SubElement(body, "geom", type="box", size=_fmt(radius, radius, radius), **common)
    elif settings.object_kind == "cylinder":
        ET.SubElement(body, "geom", type="cylinder", size=_fmt(radius, radius), **common)

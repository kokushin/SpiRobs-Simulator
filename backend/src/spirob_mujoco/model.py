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
from math import cos, pi, radians, sin

from .settings import SimSettings
from .unit_data import MM, UNIT_COUNT, UNIT_DATA, along_m, cable_radius_m, unit_volumes

# Base pose per mount. Planar is the paper's Fig. 3A arrangement seen in
# the figure's top-view photos: the arm lies on a smooth table and curls in
# the HORIZONTAL plane, so gravity acts perpendicular to the bending plane
# instead of fighting the spiral. The base quat rotates the body frame 90
# deg about +X, which maps the packing cable 0 (body -Z) onto world +Y: the
# spiral curls toward +Y where the rod is presented. Standing/hanging are
# legacy vertical-plane scenes (gravity in-plane), horizontal matches the
# web simulator's default cantilever.
MOUNT_POSE = {
    "planar": {"pos": (0.0, 0.0, 0.0125), "quat": (0.70710678, 0.70710678, 0.0, 0.0)},
    "horizontal": {"pos": (-0.06, 0.0, 0.2), "quat": (1.0, 0.0, 0.0, 0.0)},
    "hanging": {"pos": (0.0, 0.0, 0.32), "quat": (0.70710678, 0.0, 0.70710678, 0.0)},
    "standing": {"pos": (0.0, 0.0, 0.02), "quat": (0.70710678, 0.0, -0.70710678, 0.0)},
}
JOINT_RANGE_DEG = 28.8  # 30 deg paper limit x 0.96, same margin as the rod solver
CABLE_PHASES_DEG = (0.0, 120.0, 240.0)
MAX_CABLE_FORCE = 15.0
# The object rests on a kinematic pedestal stage (a physical test rig, not
# a magic fixture): it waits outside the packing sweep, slides in during
# the REACH phase carrying the object by friction, and retracts during
# HOLDING so the wrap alone must carry the weight.
PEDESTAL_RADIUS = 0.016


def pedestal_far_offset(mount: str) -> float:
    """Stage parking offset relative to the grasp point.

    The stage waits on the side the object is delivered from: -X for the
    hanging/horizontal scenes, +X for the standing scene whose packing
    spiral curls toward +X. Planar has no stage (the rod is hand-held).
    """
    if mount in ("planar", "array"):
        return 0.0
    return 0.15 if mount == "standing" else -0.13


def _fmt(*values: float) -> str:
    return " ".join(f"{v:.8g}" for v in values)


def _quat_mul(a: tuple, b: tuple) -> tuple:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _array_arm_pose(settings: SimSettings, k: int) -> tuple[tuple, tuple]:
    """Base pos (gantry-relative) and quat for arm ``k`` on the ring.

    Each arm hangs tip-down (local +X -> world -Z) with its packing
    cable 0 (local -Z of the cross-section) facing the gantry axis, so
    tensioning cable 0 curls the arm radially inward onto the object.
    quat = Rz(azimuth) * Ry(90 deg).
    """
    azimuth = 2.0 * pi * k / settings.arm_count
    pos = (
        settings.ring_radius * cos(azimuth),
        settings.ring_radius * sin(azimuth),
        0.0,
    )
    qz = (cos(azimuth * 0.5), 0.0, 0.0, sin(azimuth * 0.5))
    qy90 = (cos(pi / 4), 0.0, sin(pi / 4), 0.0)
    return pos, _quat_mul(qz, qy90)


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
    # Planar scene: the floor is the paper's smooth table the arm slides on
    # while curling, so its friction must be low — and because MuJoCo takes
    # the element-wise max of the two geom frictions, the table also needs
    # priority=1 so its low mu actually governs table contacts (the arm and
    # object keep their high mutual friction for the grasp itself).
    floor_attrs = {}
    if settings.mount == "planar":
        floor_attrs = {"friction": _fmt(0.25, 0.005, 0.0001), "priority": "1"}
    else:
        floor_attrs = {"friction": _fmt(0.92, 0.005, 0.0001)}
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="1 1 0.05",
        pos="0 0 0",
        rgba="0.35 0.37 0.4 1",
        **floor_attrs,
    )

    if settings.mount == "array":
        gantry = _add_gantry(worldbody, settings)
        prefixes = [f"a{k}_" for k in range(settings.arm_count)]
        for k, prefix in enumerate(prefixes):
            pos, quat = _array_arm_pose(settings, k)
            _add_arm(gantry, prefix, settings, pos, quat)
    else:
        prefixes = [""]
        pose = MOUNT_POSE[settings.mount]
        _add_arm(worldbody, "", settings, pose["pos"], pose["quat"])

    _add_object(worldbody, root, settings)

    contact = ET.SubElement(root, "contact")
    for prefix in prefixes:
        for i in range(1, UNIT_COUNT):
            ET.SubElement(
                contact,
                "exclude",
                body1=f"{prefix}unit{i - 1:02d}",
                body2=f"{prefix}unit{i:02d}",
            )

    # Each cable is split into per-joint tendon segments so the controller
    # can impose the capstan law T_i = T0 * exp(-mu * sum |dtheta|): a single
    # continuous MuJoCo tendon equalizes tension along its whole path, which
    # erases exactly the pack/unwind asymmetry the paper's antagonistic
    # grasp sequence depends on. Actuator order: arm-major, then cable,
    # then joint — the capstan controller relies on this layout.
    tendon = ET.SubElement(root, "tendon")
    actuator = ET.SubElement(root, "actuator")
    for prefix in prefixes:
        for c in range(3):
            for i in range(1, UNIT_COUNT):
                spatial = ET.SubElement(
                    tendon,
                    "spatial",
                    name=f"{prefix}cable{c}_seg{i:02d}",
                    width="0.0004",
                    rgba="0.85 0.4 0.15 1",
                )
                ET.SubElement(spatial, "site", site=f"{prefix}u{i - 1:02d}_c{c}")
                ET.SubElement(spatial, "site", site=f"{prefix}u{i:02d}_c{c}")
                ET.SubElement(
                    actuator,
                    "motor",
                    name=f"{prefix}cable{c}_seg{i:02d}_motor",
                    tendon=f"{prefix}cable{c}_seg{i:02d}",
                    gear="-1",
                    ctrlrange=_fmt(0, MAX_CABLE_FORCE),
                )

    if settings.mount == "array":
        # Gantry servos (always the LAST actuators, after every cable
        # motor): ctrl is the slide offset from the starting pose.
        for axis in ("x", "y", "z"):
            ET.SubElement(
                actuator,
                "position",
                name=f"gantry_{axis}",
                joint=f"gantry_{axis}",
                kp="1500",
                kv="200",
                ctrlrange=_fmt(-0.4, 0.4),
            )
    elif settings.object_kind != "none" and _uses_stage(settings):
        # Stage servo (always the LAST actuator; the capstan controller
        # relies on the cable motors occupying ctrl[0:57]).
        ET.SubElement(
            actuator,
            "position",
            name="stage_servo",
            joint="stage_x",
            kp="400",
            kv="60",
            ctrlrange=_fmt(-0.25, 0.25),
        )

    return ET.tostring(root, encoding="unicode")


def _add_gantry(worldbody: ET.Element, settings: SimSettings) -> ET.Element:
    """Transport gantry for the array mount: an XYZ-slide platform with
    position servos, so the ring has real velocities and accelerations
    while it lifts and carries the entangled object (paper Fig. 6 hangs
    the array from a rigid robot arm)."""
    gantry = ET.SubElement(
        worldbody,
        "body",
        name="gantry",
        pos=_fmt(0.0, 0.0, settings.base_height),
    )
    for axis, direction in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")):
        ET.SubElement(
            gantry,
            "joint",
            name=f"gantry_{axis}",
            type="slide",
            axis=direction,
            range="-0.5 0.5",
            damping="30",
        )
    # Visual-only ring plate; the arms provide the contact.
    ET.SubElement(
        gantry,
        "geom",
        name="gantry_plate",
        type="cylinder",
        size=_fmt(settings.ring_radius + 0.03, 0.004),
        mass="0.4",
        contype="0",
        conaffinity="0",
        rgba="0.45 0.47 0.52 1",
    )
    return gantry


def _add_arm(parent: ET.Element, prefix: str, settings: SimSettings, pos, quat) -> None:
    volumes = unit_volumes()
    volume_sum = sum(volumes)

    for i, (_, length, h, w) in enumerate(UNIT_DATA):
        if i == 0:
            body = ET.SubElement(
                parent,
                "body",
                name=f"{prefix}unit00",
                pos=_fmt(*pos),
                quat=_fmt(*quat),
            )
        else:
            separation = along_m(i) - along_m(i - 1)
            body = ET.SubElement(
                parent, "body", name=f"{prefix}unit{i:02d}", pos=_fmt(separation, 0, 0)
            )
            gap = max(0.0, separation - (UNIT_DATA[i - 1][1] + length) * 0.5 * MM)
            joint_x = -(length * 0.5 * MM + gap * 0.5)
            stiffness, damping, armature = _joint_gains(settings, i)
            for axis_name, axis in (("y", "0 1 0"), ("z", "0 0 1")):
                ET.SubElement(
                    body,
                    "joint",
                    name=f"{prefix}u{i:02d}_{axis_name}",
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
            name=f"{prefix}unit{i:02d}_geom",
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
                name=f"{prefix}u{i:02d}_c{c}",
                pos=_fmt(
                    0,
                    -radius * sin(radians(phase)),
                    -radius * cos(radians(phase)),
                ),
            )
        parent = body


STRING_ANCHOR_DROP = 0.25  # string length: anchor sits this far above the object


def _uses_stage(settings: SimSettings) -> bool:
    """Whether the object is presented on the rail stage.

    The planar scene never uses the stage: the rod is hand-held (a weld
    released during HOLDING). The standing scene never uses it either: a
    pedestal under the object blocks the arm from closing beneath it,
    making form closure geometrically impossible — rigid objects hang from
    a string instead, and soft objects rest on the floor (the paper's
    Fig. 5C table scenario — no string attachment point exists on a flex
    body). The array scene never uses it: the object stands free on the
    floor under the gantry.
    """
    return settings.mount not in ("standing", "planar", "array")


ROD_HALF_LENGTH = 0.08  # planar rod: 16 cm of vertical wooden rod
# The presenting hand parks this far ahead (+X) of the grasp pose while the
# arm packs, then carries the object in during REACH (paper Fig. 3A: the
# rod only meets the arm after the tip spiral has formed).
HAND_PARK_OFFSET = 0.12


def _add_object(worldbody: ET.Element, root: ET.Element, settings: SimSettings) -> None:
    if settings.object_kind == "none":
        return
    radius = settings.object_size_mm * MM * 0.5
    if settings.mount == "array":
        if settings.object_kind == "soft_sphere":
            _add_floor_soft_object(worldbody, settings, radius)
        else:
            _add_free_standing_object(worldbody, settings, radius)
        return
    if settings.mount == "planar":
        if settings.object_kind == "soft_sphere":
            _add_floor_soft_object(worldbody, settings, radius)
        else:
            _add_held_object(worldbody, root, settings, radius)
        return
    if not _uses_stage(settings):
        if settings.object_kind == "soft_sphere":
            _add_floor_soft_object(worldbody, settings, radius)
        else:
            _add_string_object(worldbody, root, settings, radius)
        return
    x = settings.object_x + pedestal_far_offset(settings.mount)  # parked clear of the arm
    # Platform height that puts the object center at the grasp height for
    # any size. Clamped so the pedestal stays a real cylinder; objects with
    # radius > ~0.09 m then present above grasp_center_z, a limit of the
    # rig rather than a tuning knob.
    top = max(0.01, settings.grasp_center_z - radius - 0.001)

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
        # elasticity. count/spacing span the requested diameter; the vertex
        # radius must stay below the spacing for small objects.
        count = 6
        spacing = (2 * radius) / (count - 1)
        flexcomp = ET.SubElement(
            worldbody,
            "flexcomp",
            name="object",
            type="ellipsoid",
            count=_fmt(count, count, count),
            spacing=_fmt(spacing, spacing, spacing),
            radius=_fmt(min(0.005, spacing * 0.45)),
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


def _add_free_standing_object(
    worldbody: ET.Element, settings: SimSettings, radius: float
) -> None:
    """Rigid object standing free on the floor at the gantry axis (array).

    Nothing holds it — no hand, no string, no stage. Whether the curling
    arms knock it over before the entanglement closes is exactly what the
    scene measures (the paper flags the uncurling push force as the open
    problem its multi-arm array alleviates).
    """
    if settings.object_kind == "cylinder":
        z = ROD_HALF_LENGTH + 0.002
    else:
        z = radius + 0.002
    body = ET.SubElement(
        worldbody,
        "body",
        name="object",
        pos=_fmt(settings.object_x, settings.object_y, z),
    )
    ET.SubElement(body, "freejoint", name="object_free")
    common = {
        "name": "object_geom",
        "mass": _fmt(settings.object_mass),
        "condim": "6",
        "friction": _fmt(0.92, 0.02, 0.02),
        "rgba": "0.78 0.62 0.42 1",
    }
    if settings.object_kind == "sphere":
        ET.SubElement(body, "geom", type="sphere", size=_fmt(radius), **common)
    elif settings.object_kind == "box":
        ET.SubElement(body, "geom", type="box", size=_fmt(radius, radius, radius), **common)
    elif settings.object_kind == "cylinder":
        ET.SubElement(body, "geom", type="cylinder", size=_fmt(radius, ROD_HALF_LENGTH), **common)


def _add_held_object(
    worldbody: ET.Element, root: ET.Element, settings: SimSettings, radius: float
) -> None:
    """Rigid object carried into the planar curl by "hand" (paper Fig. 3A).

    The hand is a mocap body welded to the object: it parks ahead of the
    arm (+X) while the tip spiral packs, then ``SpiRobSim`` slides it to
    (object_x, object_y) during REACH — exactly the experimenter's hand
    bringing the rod to the arm in the figure. ``SpiRobSim.release_object``
    disables the weld during HOLDING, so the wrap alone must carry the
    object's weight afterwards. A cylinder is the paper's vertical wooden
    rod; sphere/box are held just above the table at the curl height.
    """
    x, y = settings.object_x + HAND_PARK_OFFSET, settings.object_y
    if settings.object_kind == "cylinder":
        z = ROD_HALF_LENGTH + 0.002
    else:
        z = radius + 0.002
    hand = ET.SubElement(
        worldbody, "body", name="hand", mocap="true", pos=_fmt(x, y, z)
    )
    # Visual-only marker so the presenting hand is visible in the viewer.
    ET.SubElement(
        hand,
        "geom",
        type="sphere",
        size="0.008",
        contype="0",
        conaffinity="0",
        rgba="0.9 0.75 0.6 0.5",
    )
    body = ET.SubElement(worldbody, "body", name="object", pos=_fmt(x, y, z))
    ET.SubElement(body, "freejoint", name="object_free")
    common = {
        "name": "object_geom",
        "mass": _fmt(settings.object_mass),
        "condim": "6",
        "friction": _fmt(0.92, 0.02, 0.02),
        "rgba": "0.78 0.62 0.42 1",
    }
    if settings.object_kind == "sphere":
        ET.SubElement(body, "geom", type="sphere", size=_fmt(radius), **common)
    elif settings.object_kind == "box":
        ET.SubElement(body, "geom", type="box", size=_fmt(radius, radius, radius), **common)
    elif settings.object_kind == "cylinder":
        ET.SubElement(body, "geom", type="cylinder", size=_fmt(radius, ROD_HALF_LENGTH), **common)

    equality = ET.SubElement(root, "equality")
    # A zero relpose quat means "use the qpos0 relative pose": the object
    # rigidly follows the hand until the weld is released.
    ET.SubElement(equality, "weld", name="hand_weld", body1="hand", body2="object")


def _add_floor_soft_object(
    worldbody: ET.Element, settings: SimSettings, radius: float
) -> None:
    """Soft object resting on the floor beside the arm (Fig. 5C table)."""
    count = 6
    spacing = (2 * radius) / (count - 1)
    flexcomp = ET.SubElement(
        worldbody,
        "flexcomp",
        name="object",
        type="ellipsoid",
        count=_fmt(count, count, count),
        spacing=_fmt(spacing, spacing, spacing),
        radius=_fmt(min(0.005, spacing * 0.45)),
        pos=_fmt(settings.object_x, settings.object_y, radius + 0.002),
        dim="3",
        mass=_fmt(settings.object_mass),
        rgba="0.3 0.62 0.9 1",
    )
    ET.SubElement(
        flexcomp,
        "elasticity",
        young=_fmt(settings.object_young),
        poisson=_fmt(settings.object_poisson),
    )
    ET.SubElement(flexcomp, "contact", selfcollide="none", internal="false")


def _add_string_object(
    worldbody: ET.Element, root: ET.Element, settings: SimSettings, radius: float
) -> None:
    """Rigid object hanging from an inextensible string at the grasp point.

    A limited spatial tendon from a world anchor to the object's center acts
    as the presenting hand: it carries the weight while the arm closes, and
    ``SpiRobSim.release_string`` slackens it so the wrap alone must hold.
    MuJoCo tendons do not collide with geoms, so the string passing through
    the wrap region is force-free (a deliberate modeling artifact).
    """
    x, z = settings.object_x, settings.grasp_center_z
    ET.SubElement(
        worldbody,
        "site",
        name="string_anchor",
        pos=_fmt(x, 0, z + STRING_ANCHOR_DROP),
    )
    body = ET.SubElement(worldbody, "body", name="object", pos=_fmt(x, 0, z))
    ET.SubElement(body, "freejoint", name="object_free")
    ET.SubElement(body, "site", name="object_center")
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

    tendon = ET.SubElement(root, "tendon")
    string = ET.SubElement(
        tendon,
        "spatial",
        name="string",
        limited="true",
        range=_fmt(0, STRING_ANCHOR_DROP),
        width="0.0003",
        rgba="0.8 0.8 0.85 1",
    )
    ET.SubElement(string, "site", site="string_anchor")
    ET.SubElement(string, "site", site="object_center")

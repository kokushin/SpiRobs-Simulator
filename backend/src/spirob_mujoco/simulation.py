"""Stepping wrapper around the MJCF model.

Owns the MjModel/MjData pair and exposes the same knobs the web UI uses:
cable tensions, the paper's auto-grasp schedule, reset, and a
JSON-serializable state snapshot for the transport layer.

Cable mechanics: each cable is 19 tendon segments (one per joint), and the
controller distributes the commanded motor tension along them with the
capstan law ``T_i = T0 * exp(-mu * sum_{j<i} |dtheta_j|)`` recomputed every
substep from the current joint angles — the same attenuation model as the
web simulator, but acting inside a full contact simulation. No artificial
staging forces exist; the object is carried in by the presenting hand
(planar), rests on its pedestal stage (hanging/horizontal), or hangs from
the presentation string (standing) until the rig lets go.
"""

import mujoco
import numpy as np

from .grasp import (
    F_SQUEEZE,
    RETREAT_DZ,
    T_PACK,
    T_REACH,
    T_RELEASE,
    array_grasp_command,
    array_lift_end,
    auto_grasp_command,
)
from .model import build_mjcf, pedestal_far_offset
from .settings import SimSettings
from .unit_data import UNIT_COUNT, UNIT_DATA

# Pedestal stage timeline, relative to auto-grasp start [s]: slide in
# during REACH, hold through WRAP/GRASP, retract shortly into HOLDING so
# the wrap alone must carry the object. (String-presented scenes release
# the string at T_RELEASE instead.)
STAGE_IN_START = T_PACK
STAGE_IN_END = T_REACH
STAGE_OUT_START = T_RELEASE
STAGE_OUT_END = T_RELEASE + 2.0


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


class SpiRobSim:
    def __init__(self, settings: SimSettings | None = None):
        self.settings = settings or SimSettings()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.settings))
        self.data = mujoco.MjData(self.model)

        def body_id(name: str) -> int:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        # Arm naming: the single-arm mounts keep the historical unprefixed
        # names; the array mount prefixes each arm ("a0_", "a1_", ...).
        self.arm_count = self.settings.arm_count
        self._prefixes = (
            [""] if self.arm_count == 1 else [f"a{k}_" for k in range(self.arm_count)]
        )

        self._arm_unit_body_ids = [
            [body_id(f"{prefix}unit{i:02d}") for i in range(UNIT_COUNT)]
            for prefix in self._prefixes
        ]
        self._unit_body_ids = self._arm_unit_body_ids[0]
        # geom id -> (arm, unit)
        self._unit_geom_ids = {
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}unit{i:02d}_geom"
            ): (a, i)
            for a, prefix in enumerate(self._prefixes)
            for i in range(UNIT_COUNT)
        }

        # Joint hinge-pair qpos addresses per arm, ordered by unit (1..19).
        self._joint_qposadr = np.array(
            [
                [
                    self.model.jnt_qposadr[
                        mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}u{i:02d}_y"
                        )
                    ]
                    for i in range(1, UNIT_COUNT)
                ]
                for prefix in self._prefixes
            ]
        )

        # Gantry servos (array mount): the last three actuators.
        self._gantry_ctrl_index = {
            axis: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"gantry_{axis}"
            )
            for axis in ("x", "y", "z")
        }
        if any(v < 0 for v in self._gantry_ctrl_index.values()):
            self._gantry_ctrl_index = None

        self._object_body_id = body_id("object")
        self._object_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
        )
        # All geoms belonging to the object (the fragile sausage has two).
        self._object_geom_ids = {
            gid
            for gid in (
                self._object_geom_id,
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_top_geom"
                ),
            )
            if gid >= 0
        }
        self._object_top_body_id = body_id("object_top")
        self._flex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, "object")

        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        # Fragile-object state (sausage): the mid weld snaps and the skin
        # darkens once the summed grip force exceeds the crush threshold.
        self._sausage_weld_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "sausage_weld"
        )
        self._object_rgba0 = {
            gid: self.model.geom_rgba[gid].copy() for gid in self._object_geom_ids
        }
        self.object_broken = False
        self.peak_contact_force = 0.0
        self.break_event: dict | None = None
        self._array_squeeze = (
            self.settings.squeeze_force
            if self.settings.squeeze_force is not None
            else F_SQUEEZE
        )

        # Stage servo: last actuator; its ctrl is the slide-joint offset
        # relative to the far parking position.
        self._stage_ctrl_index = (
            self.model.nu - 1
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "stage_servo") >= 0
            else -1
        )
        self._stage_travel = -pedestal_far_offset(self.settings.mount)  # far -> grasp point

        # Presentation string (standing mount): released once the wrap closes.
        self._string_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_TENDON, "string"
        )
        self._string_range = (
            self.model.tendon_range[self._string_id].copy() if self._string_id >= 0 else None
        )
        # Presentation hand (planar mount): a mocap body welded to the
        # object. It parks ahead of the arm during PACK, carries the object
        # in during REACH, and the weld opens at release.
        self._hand_eq_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "hand_weld"
        )
        hand_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._hand_mocap_id = (
            self.model.body_mocapid[hand_body] if hand_body >= 0 else -1
        )
        if self._hand_mocap_id >= 0:
            self._hand_park = self.model.body_pos[hand_body].copy()
            self._hand_target = self._hand_park.copy()
            self._hand_target[0] = self.settings.object_x
            self._hand_target[1] = self.settings.object_y
        self.string_released = False

        self.grasp_start_time: float | None = None
        self.grasp_primary_cable = 0
        self.grasp_phase: str | None = None
        self._base_forces = np.zeros(3)  # broadcast to every arm
        # Contact-conditional deposit (array): set when the object leaves
        # the grasp early, so the rig retreats instead of uncurling around
        # the free-standing object.
        self._array_abort_time: float | None = None
        self._array_abort_state: tuple[float, float, float, float] | None = None
        self._contact_lost_since: float | None = None
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ control

    def set_cable_forces(self, forces) -> None:
        """Manual tension input: cancels any auto-grasp automation, like the
        web UI does."""
        self.grasp_start_time = None
        self.grasp_phase = None
        self._apply_forces(forces)

    def _apply_forces(self, forces) -> None:
        values = np.clip(np.asarray(forces, dtype=float), 0.0, None)
        if values.shape != (3,):
            raise ValueError("cable forces must have exactly 3 entries")
        self.settings.cable_forces = values.tolist()
        self._base_forces = values

    def _apply_capstan(self) -> None:
        """Distribute motor tensions along the segments with the capstan law.

        The accumulated turn for the joint driving unit i sums the bends of
        joints 2..i-1 (matching the web solver, which never attenuates a
        straight robot). Actuator layout is arm-major, then cable, then
        segment, so arm a / cable c occupies
        ctrl[(a*3 + c)*segments : (a*3 + c + 1)*segments].
        """
        segments = UNIT_COUNT - 1
        for a in range(self.arm_count):
            qy = self.data.qpos[self._joint_qposadr[a]]
            qz = self.data.qpos[self._joint_qposadr[a] + 1]
            bend = np.hypot(qy, qz)
            accumulated = np.concatenate(([0.0], np.cumsum(bend[:-1])))
            attenuation = np.exp(-self.settings.cable_friction * accumulated)
            for c in range(3):
                start = (a * 3 + c) * segments
                self.data.ctrl[start : start + segments] = (
                    self._base_forces[c] * attenuation
                )

    def start_auto_grasp(self, primary_cable: int = 0) -> None:
        self.grasp_start_time = self.data.time
        self.grasp_primary_cable = primary_cable % 3
        self.grasp_phase = "settling" if self.settings.mount == "array" else "packing"
        self._array_abort_time = None
        self._array_abort_state = None
        self._contact_lost_since = None

    def release_object(self) -> None:
        """The presentation rig lets go: slacken the string (standing) or
        open the hand weld (planar), so the wrap alone carries the object."""
        if self.string_released:
            return
        if self._string_id >= 0:
            self.model.tendon_range[self._string_id][1] = 2.0
        if self._hand_eq_id >= 0:
            self.data.eq_active[self._hand_eq_id] = 0
        self.string_released = True

    # Backwards-compatible alias (pre-planar API).
    release_string = release_object

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if self._string_id >= 0 and self._string_range is not None:
            self.model.tendon_range[self._string_id] = self._string_range
        self.string_released = False
        self.grasp_start_time = None
        self.grasp_phase = None
        self._array_abort_time = None
        self._array_abort_state = None
        self._contact_lost_since = None
        self.object_broken = False
        self.peak_contact_force = 0.0
        self.break_event = None
        for gid, rgba in self._object_rgba0.items():
            self.model.geom_rgba[gid] = rgba
        self._apply_forces([0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ stepping

    def _presentation_progress(self, elapsed: float) -> float:
        if elapsed < STAGE_IN_START:
            return 0.0
        if elapsed < STAGE_IN_END:
            return _smoothstep((elapsed - STAGE_IN_START) / (STAGE_IN_END - STAGE_IN_START))
        if elapsed < STAGE_OUT_START:
            return 1.0
        return 1.0 - _smoothstep(
            (elapsed - STAGE_OUT_START) / (STAGE_OUT_END - STAGE_OUT_START)
        )

    def _move_stage(self, elapsed: float) -> None:
        if self._stage_ctrl_index < 0:
            return
        u = self._presentation_progress(elapsed)
        self.data.ctrl[self._stage_ctrl_index] = self._stage_travel * u

    def _move_hand(self, elapsed: float) -> None:
        """Carry the object in during REACH; after release the empty hand
        retreats (the weld is already open, so this is purely visual)."""
        if self._hand_mocap_id < 0:
            return
        u = self._presentation_progress(elapsed)
        self.data.mocap_pos[self._hand_mocap_id] = (
            self._hand_park + (self._hand_target - self._hand_park) * u
        )

    def _move_gantry(self, offset) -> None:
        if self._gantry_ctrl_index is None:
            return
        for axis, value in zip(("x", "y", "z"), offset):
            self.data.ctrl[self._gantry_ctrl_index[axis]] = value

    def _array_step_command(self, elapsed: float):
        """Array schedule with the paper's contact-triggered switching.

        The paper's automatic strategy changes phase on sensed contact
        (motor current); here the contact solver is read directly. If the
        object leaves the grasp after the lift — a drop the open-loop
        timeline cannot know about — lowering and uncurling would sweep
        the still-curled arms through the free-standing object and knock
        it over. Instead the rig aborts the deposit: arms stay curled
        (compact, tips high), the gantry retreats vertically, and only
        then do the cables release.
        """
        command = array_grasp_command(
            elapsed,
            self._array_squeeze,
            self.settings.approach_dz,
            self.settings.approach_duration,
        )

        if self._array_abort_time is None:
            # Contact loss counts as a dropped object only while the
            # schedule still believes it is holding one — during the
            # releasing/retreating phases zero contact is the goal, and
            # aborting there would re-curl the arms onto the deposited
            # object and drag it away.
            if elapsed > array_lift_end(
                self.settings.approach_duration
            ) and command.phase in ("carrying", "lowering"):
                if any(self.contacting_units_per_arm()):
                    self._contact_lost_since = None
                else:
                    if self._contact_lost_since is None:
                        self._contact_lost_since = elapsed
                    elif elapsed - self._contact_lost_since > 0.3:
                        self._array_abort_time = elapsed
                        self._array_abort_state = (
                            command.cable_forces[0],
                            *command.gantry_offset,
                        )
            if self._array_abort_time is None:
                return command

        curl0, dx, dy, dz0 = self._array_abort_state
        u = elapsed - self._array_abort_time
        if u < 2.0:
            phase = "retreating"
            # hold the curl compact while rising
            curl = max(curl0, self._array_squeeze)
            dz = dz0 + (RETREAT_DZ - dz0) * _smoothstep(u / 2.0)
        elif u < 3.0:
            phase = "releasing"
            curl = self._array_squeeze * (1.0 - _smoothstep(u - 2.0))
            dz = RETREAT_DZ
        else:
            phase = "done"
            curl = 0.0
            dz = RETREAT_DZ
        return type(command)(
            phase=phase, cable_forces=(curl, 0.0, 0.0), gantry_offset=(dx, dy, dz)
        )

    def _monitor_fragility(self) -> None:
        """Crush detection for the fragile object (call after mj_step).

        The crush metric is the SUM of all contact normal forces on the
        object — the total radial compression of the cylinder, like a
        fist closing around it. (A single-contact maximum does not
        discriminate grip strength here: the drape spreads the squeeze
        over many contacts of ~1 N each.) Crossing ``object_crush_force``
        snaps the mid weld and darkens the skin — one-way until reset().
        """
        if self._sausage_weld_id < 0:
            return
        force = np.zeros(6)
        total = 0.0
        for i in range(self.data.ncon):
            geoms = (self.data.contact[i].geom[0], self.data.contact[i].geom[1])
            if geoms[0] not in self._object_geom_ids and (
                geoms[1] not in self._object_geom_ids
            ):
                continue
            # The floor is support, not grip: its reaction (mg at rest,
            # spiking on any bump) would pollute the crush metric.
            if self._floor_geom_id in geoms:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, force)
            total += force[0]
        self.peak_contact_force = max(self.peak_contact_force, total)
        if not self.object_broken and total > self.settings.object_crush_force:
            self.object_broken = True
            self.break_event = {
                "time": float(self.data.time),
                "force": float(total),
                "phase": self.grasp_phase,
            }
            self.data.eq_active[self._sausage_weld_id] = 0
            for gid in self._object_geom_ids:
                self.model.geom_rgba[gid] = (0.45, 0.24, 0.16, 1.0)

    def step(self, duration: float) -> None:
        steps = max(1, round(duration / self.model.opt.timestep))
        for _ in range(steps):
            if self.grasp_start_time is not None:
                elapsed = self.data.time - self.grasp_start_time
                if self.settings.mount == "array":
                    command = self._array_step_command(elapsed)
                    self.grasp_phase = command.phase
                    self._apply_forces(command.cable_forces)
                    self._move_gantry(command.gantry_offset)
                else:
                    command = auto_grasp_command(elapsed, self.grasp_primary_cable)
                    self.grasp_phase = command.phase
                    self._apply_forces(command.cable_forces)
                    self._move_stage(elapsed)
                    self._move_hand(elapsed)
                    if elapsed >= T_RELEASE:
                        self.release_object()
            self._apply_capstan()
            mujoco.mj_step(self.model, self.data)
            self._monitor_fragility()

    # ------------------------------------------------------------------ state

    def contacting_units_per_arm(self) -> list[list[int]]:
        """Units of each arm currently in contact with the object."""
        units: list[set[int]] = [set() for _ in range(self.arm_count)]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = (contact.geom[0], contact.geom[1])
            if self._object_geom_ids:
                if geoms[0] in self._object_geom_ids:
                    other = geoms[1]
                elif geoms[1] in self._object_geom_ids:
                    other = geoms[0]
                else:
                    continue
            elif self._flex_id >= 0:
                flexes = (contact.flex[0], contact.flex[1])
                if self._flex_id not in flexes:
                    continue
                other = geoms[0] if flexes[1] == self._flex_id else geoms[1]
            else:
                continue
            if other in self._unit_geom_ids:
                arm, unit = self._unit_geom_ids[other]
                units[arm].add(unit)
        return [sorted(s) for s in units]

    def contacting_units(self) -> list[int]:
        """Union of contacting unit indices across arms (single-arm shape)."""
        merged: set[int] = set()
        for arm_units in self.contacting_units_per_arm():
            merged.update(arm_units)
        return sorted(merged)

    def state(self) -> dict:
        units = []
        for bid in self._unit_body_ids:
            units.append(
                {
                    "pos": self.data.xpos[bid].tolist(),
                    "quat": self.data.xquat[bid].tolist(),
                }
            )
        segments = UNIT_COUNT - 1
        tip_tensions = [float(self.data.ctrl[c * segments + segments - 1]) for c in range(3)]

        state: dict = {
            "time": self.data.time,
            "units": units,
            "cableForces": list(self.settings.cable_forces),
            "cableTipTensions": tip_tensions,
            "graspPhase": self.grasp_phase,
            "contactingUnits": self.contacting_units(),
        }

        if self.arm_count > 1:
            # "units" above stays arm 0 for the single-arm clients; the
            # array's full pose set travels under "arms".
            state["arms"] = [
                [
                    {
                        "pos": self.data.xpos[bid].tolist(),
                        "quat": self.data.xquat[bid].tolist(),
                    }
                    for bid in arm_ids
                ]
                for arm_ids in self._arm_unit_body_ids
            ]
            state["contactingUnitsPerArm"] = self.contacting_units_per_arm()

        if self._object_body_id >= 0:
            state["object"] = {
                "kind": self.settings.object_kind,
                "pos": self.data.xpos[self._object_body_id].tolist(),
                "quat": self.data.xquat[self._object_body_id].tolist(),
            }
            if self._object_top_body_id >= 0:
                state["object"]["topPos"] = self.data.xpos[
                    self._object_top_body_id
                ].tolist()
                state["object"]["topQuat"] = self.data.xquat[
                    self._object_top_body_id
                ].tolist()
            if self._sausage_weld_id >= 0:
                state["object"]["broken"] = self.object_broken
                state["object"]["peakContactForce"] = self.peak_contact_force
                state["object"]["breakEvent"] = self.break_event
        elif self._flex_id >= 0:
            state["object"] = {
                "kind": self.settings.object_kind,
                "vertices": self.data.flexvert_xpos.reshape(-1).tolist(),
            }
        return state

    def meta(self) -> dict:
        meta: dict = {
            "armCount": self.arm_count,
            "unitCount": UNIT_COUNT,
            "unitData": [list(row) for row in UNIT_DATA],
            "timestep": self.model.opt.timestep,
            "objectKind": self.settings.object_kind,
            "objectSizeMm": self.settings.object_size_mm,
            "objectMassKg": self.settings.object_mass,
        }
        if self._flex_id >= 0:
            meta["flexVertexCount"] = int(self.model.nflexvert)
            meta["flexElements"] = self.model.flex_elem.reshape(-1).tolist()
        return meta

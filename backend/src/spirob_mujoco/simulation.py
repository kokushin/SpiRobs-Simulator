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

from .grasp import T_PACK, T_REACH, T_RELEASE, auto_grasp_command
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

        self._unit_body_ids = [body_id(f"unit{i:02d}") for i in range(UNIT_COUNT)]
        self._unit_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"unit{i:02d}_geom"): i
            for i in range(UNIT_COUNT)
        }

        # Joint hinge-pair qpos addresses, ordered by unit (i = 1..19).
        self._joint_qposadr = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"u{i:02d}_y")
                ]
                for i in range(1, UNIT_COUNT)
            ]
        )

        self._object_body_id = body_id("object")
        self._object_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
        )
        self._flex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, "object")

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
        self._base_forces = np.zeros(3)
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
        straight robot).
        """
        qy = self.data.qpos[self._joint_qposadr]
        qz = self.data.qpos[self._joint_qposadr + 1]
        bend = np.hypot(qy, qz)
        accumulated = np.concatenate(([0.0], np.cumsum(bend[:-1])))
        attenuation = np.exp(-self.settings.cable_friction * accumulated)
        segments = UNIT_COUNT - 1
        for c in range(3):
            self.data.ctrl[c * segments : (c + 1) * segments] = (
                self._base_forces[c] * attenuation
            )

    def start_auto_grasp(self, primary_cable: int = 0) -> None:
        self.grasp_start_time = self.data.time
        self.grasp_primary_cable = primary_cable % 3
        self.grasp_phase = "packing"

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

    def step(self, duration: float) -> None:
        steps = max(1, round(duration / self.model.opt.timestep))
        for _ in range(steps):
            if self.grasp_start_time is not None:
                elapsed = self.data.time - self.grasp_start_time
                command = auto_grasp_command(elapsed, self.grasp_primary_cable)
                self.grasp_phase = command.phase
                self._apply_forces(command.cable_forces)
                self._move_stage(elapsed)
                self._move_hand(elapsed)
                if elapsed >= T_RELEASE:
                    self.release_object()
            self._apply_capstan()
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ state

    def contacting_units(self) -> list[int]:
        """Robot units currently in contact with the object."""
        units: set[int] = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = (contact.geom[0], contact.geom[1])
            if self._object_geom_id >= 0:
                if self._object_geom_id not in geoms:
                    continue
                other = geoms[0] if geoms[1] == self._object_geom_id else geoms[1]
            elif self._flex_id >= 0:
                flexes = (contact.flex[0], contact.flex[1])
                if self._flex_id not in flexes:
                    continue
                other = geoms[0] if flexes[1] == self._flex_id else geoms[1]
            else:
                continue
            if other in self._unit_geom_ids:
                units.add(self._unit_geom_ids[other])
        return sorted(units)

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

        if self._object_body_id >= 0:
            state["object"] = {
                "kind": self.settings.object_kind,
                "pos": self.data.xpos[self._object_body_id].tolist(),
                "quat": self.data.xquat[self._object_body_id].tolist(),
            }
        elif self._flex_id >= 0:
            state["object"] = {
                "kind": self.settings.object_kind,
                "vertices": self.data.flexvert_xpos.reshape(-1).tolist(),
            }
        return state

    def meta(self) -> dict:
        meta: dict = {
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

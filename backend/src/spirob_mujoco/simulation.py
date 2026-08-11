"""Stepping wrapper around the MJCF model.

Owns the MjModel/MjData pair and exposes the same knobs the web UI uses:
cable tensions, the paper's auto-grasp schedule, reset, and a
JSON-serializable state snapshot for the transport layer.

Cable mechanics: each cable is 19 tendon segments (one per joint), and the
controller distributes the commanded motor tension along them with the
capstan law ``T_i = T0 * exp(-mu * sum_{j<i} |dtheta_j|)`` recomputed every
substep from the current joint angles — the same attenuation model as the
web simulator, but acting inside a full contact simulation. No artificial
staging forces exist; the object simply rests on its pedestal.
"""

import mujoco
import numpy as np

from .grasp import auto_grasp_command
from .model import PEDESTAL_FAR_OFFSET, build_mjcf
from .settings import SimSettings
from .unit_data import UNIT_COUNT, UNIT_DATA

# Pedestal stage timeline, relative to auto-grasp start [s]: slide in
# during REACH, hold through WRAP/GRASP, retract shortly into HOLDING so
# the wrap alone must carry the object.
STAGE_IN_START = 1.2
STAGE_IN_END = 4.4
STAGE_OUT_START = 11.2
STAGE_OUT_END = 13.2


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
        self._stage_travel = -PEDESTAL_FAR_OFFSET  # far -> grasp point distance

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

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.grasp_start_time = None
        self.grasp_phase = None
        self._apply_forces([0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ stepping

    def _move_stage(self, elapsed: float) -> None:
        if self._stage_ctrl_index < 0:
            return
        if elapsed < STAGE_IN_START:
            u = 0.0
        elif elapsed < STAGE_IN_END:
            u = _smoothstep((elapsed - STAGE_IN_START) / (STAGE_IN_END - STAGE_IN_START))
        elif elapsed < STAGE_OUT_START:
            u = 1.0
        else:
            u = 1.0 - _smoothstep(
                (elapsed - STAGE_OUT_START) / (STAGE_OUT_END - STAGE_OUT_START)
            )
        self.data.ctrl[self._stage_ctrl_index] = self._stage_travel * u

    def step(self, duration: float) -> None:
        steps = max(1, round(duration / self.model.opt.timestep))
        for _ in range(steps):
            if self.grasp_start_time is not None:
                elapsed = self.data.time - self.grasp_start_time
                command = auto_grasp_command(elapsed, self.grasp_primary_cable)
                self.grasp_phase = command.phase
                self._apply_forces(command.cable_forces)
                self._move_stage(elapsed)
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

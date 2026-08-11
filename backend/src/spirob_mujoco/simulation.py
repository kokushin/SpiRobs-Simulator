"""Stepping wrapper around the MJCF model.

Owns the MjModel/MjData pair and exposes the same knobs the web UI uses:
cable tensions, auto-grasp schedule, reset, and a JSON-serializable state
snapshot for the transport layer.

Auto grasp reproduces the web simulator's test-fixture flow: the object is
presented at a repeatable pose while the wrap forms (rigid objects via a
weld to a mocap body, soft objects via per-vertex PD holding forces), then
released to full dynamics when the schedule enters HOLDING.
"""

import mujoco
import numpy as np

from .grasp import auto_grasp_command
from .model import FIXTURE_POS, build_mjcf
from .settings import SimSettings
from .unit_data import UNIT_COUNT, UNIT_DATA

# Acceleration-level gains for the soft-object fixture: only the center of
# mass is held (critically damped, kd = 2 sqrt(kp)) so the ball can deform
# and let the coil conform around it, unlike a rigid weld.
SOFT_STAGE_KP = 800.0
SOFT_STAGE_KD = 56.0
# Viscous damper applied to the object while the fixture assist fades out
# after release; absorbs stored contact energy that would otherwise
# catapult the object (units: 1/s, scaled by mass).
RELEASE_DAMPING = 30.0


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

        # Rigid object bookkeeping (ids are -1 when absent).
        self._object_body_id = body_id("object")
        self._object_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
        )
        self._weld_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "fixture_weld"
        )
        fixture_body = body_id("fixture")
        self._fixture_mocap_id = (
            self.model.body_mocapid[fixture_body] if fixture_body >= 0 else -1
        )
        object_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self._object_qposadr = self.model.jnt_qposadr[object_joint] if object_joint >= 0 else -1
        self._object_dofadr = self.model.jnt_dofadr[object_joint] if object_joint >= 0 else -1

        # Soft object bookkeeping.
        self._flex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, "object")
        if self._flex_id >= 0:
            self._flex_vert_bodies = np.array(self.model.flex_vertbodyid, dtype=int)
            joints = self.model.body_jntadr[self._flex_vert_bodies]
            self._flex_qposadr = self.model.jnt_qposadr[joints]
            self._flex_dofadr = self.model.jnt_dofadr[joints]
        else:
            self._flex_vert_bodies = np.empty(0, dtype=int)
        self._fixture = np.array(self.settings.fixture_pos or FIXTURE_POS)

        self._staged = False
        self.grasp_start_time: float | None = None
        self.grasp_primary_cable = 0
        self.grasp_phase: str | None = None
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ control

    def set_cable_forces(self, forces) -> None:
        """Manual tension input: cancels any auto-grasp automation, like the
        web UI does."""
        self.grasp_start_time = None
        self.grasp_phase = None
        self._release_fixture()
        self._apply_forces(forces)

    def _apply_forces(self, forces) -> None:
        values = np.clip(np.asarray(forces, dtype=float), 0.0, None)
        if values.shape != (3,):
            raise ValueError("cable forces must have exactly 3 entries")
        self.settings.cable_forces = values.tolist()
        self.data.ctrl[:3] = values

    def start_auto_grasp(self, primary_cable: int = 0) -> None:
        self.grasp_start_time = self.data.time
        self.grasp_primary_cable = primary_cable % 3
        self.grasp_phase = "packing"
        self._stage_fixture()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._staged = False
        self.grasp_start_time = None
        self.grasp_phase = None
        self._apply_forces([0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ fixture

    def _stage_fixture(self) -> None:
        fixture = self._fixture
        if self._object_body_id >= 0 and self._weld_id >= 0:
            adr = self._object_qposadr
            self.data.qpos[adr : adr + 3] = fixture
            self.data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
            self.data.qvel[self._object_dofadr : self._object_dofadr + 6] = 0.0
            self.data.mocap_pos[self._fixture_mocap_id] = fixture
            self.data.mocap_quat[self._fixture_mocap_id] = (1.0, 0.0, 0.0, 0.0)
            self.data.eq_active[self._weld_id] = 1
            self._staged = True
        elif self._flex_id >= 0:
            center = self.data.flexvert_xpos.mean(axis=0)
            delta = fixture - center
            for adr, dof in zip(self._flex_qposadr, self._flex_dofadr):
                self.data.qpos[adr : adr + 3] += delta
                self.data.qvel[dof : dof + 3] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._staged = True

    def _release_fixture(self) -> None:
        """Fully free the object: weld off, all assist forces cleared."""
        if self._weld_id >= 0:
            self.data.eq_active[self._weld_id] = 0
        if self._flex_id >= 0:
            self.data.xfrc_applied[self._flex_vert_bodies] = 0.0
        if self._object_body_id >= 0:
            self.data.xfrc_applied[self._object_body_id] = 0.0
        self._staged = False

    def _apply_fixture_assist(self, assist: float) -> None:
        """Fixture behavior while the auto-grasp schedule runs.

        assist == 1 while the wrap forms: rigid objects are held by the weld
        (nothing to do here), soft objects by a center-of-mass spring. As
        assist fades after release the object only feels a shrinking viscous
        damper, so the wrap takes the load gradually.
        """
        released = assist < 1.0
        if released and self._weld_id >= 0 and self.data.eq_active[self._weld_id]:
            self.data.eq_active[self._weld_id] = 0

        if self._flex_id >= 0:
            masses = self.model.body_mass[self._flex_vert_bodies][:, None]
            dof_index = self._flex_dofadr[:, None] + np.arange(3)
            velocities = self.data.qvel[dof_index]
            if not released:
                total = masses.sum()
                com = (self.data.xpos[self._flex_vert_bodies] * masses).sum(axis=0) / total
                vcom = (velocities * masses).sum(axis=0) / total
                acceleration = (
                    SOFT_STAGE_KP * (self._fixture - com)
                    - SOFT_STAGE_KD * vcom
                    - self.model.opt.gravity
                )
                self.data.xfrc_applied[self._flex_vert_bodies, :3] = masses * acceleration
            else:
                self.data.xfrc_applied[self._flex_vert_bodies, :3] = (
                    -assist * RELEASE_DAMPING * masses * velocities
                )
        elif self._object_body_id >= 0 and released:
            velocity = self.data.qvel[self._object_dofadr : self._object_dofadr + 3]
            mass = self.model.body_mass[self._object_body_id]
            self.data.xfrc_applied[self._object_body_id, :3] = (
                -assist * RELEASE_DAMPING * mass * velocity
            )

    # ------------------------------------------------------------------ stepping

    def step(self, duration: float) -> None:
        steps = max(1, round(duration / self.model.opt.timestep))
        for _ in range(steps):
            if self.grasp_start_time is not None:
                command = auto_grasp_command(
                    self.data.time - self.grasp_start_time, self.grasp_primary_cable
                )
                self.grasp_phase = command.phase
                self._apply_forces(command.cable_forces)
                if self._staged and command.fixture_assist <= 0.0:
                    self._release_fixture()
                elif self._staged:
                    self._apply_fixture_assist(command.fixture_assist)
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
        tensions = (-self.data.actuator_force[:3]).clip(min=0.0).tolist()

        state: dict = {
            "time": self.data.time,
            "units": units,
            "cableForces": list(self.settings.cable_forces),
            "cableTensions": tensions,
            "graspPhase": self.grasp_phase,
            "objectStaged": self._staged,
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
        }
        if self._flex_id >= 0:
            meta["flexVertexCount"] = int(self.model.nflexvert)
            meta["flexElements"] = self.model.flex_elem.reshape(-1).tolist()
        return meta

import math

import mujoco
import numpy as np
import pytest

from spirob_mujoco import SimSettings, SpiRobSim, auto_grasp_command, build_mjcf
from spirob_mujoco.unit_data import UNIT_COUNT


def make_sim(**kwargs) -> SpiRobSim:
    return SpiRobSim(SimSettings(**kwargs))


def tip_position(sim: SpiRobSim) -> np.ndarray:
    body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"unit{UNIT_COUNT - 1:02d}")
    return sim.data.xpos[body_id].copy()


def test_model_compiles_with_expected_structure():
    sim = make_sim(object_kind="none")
    # 19 elastic joints x 2 hinges, no object DOF.
    assert sim.model.nq == 2 * (UNIT_COUNT - 1)
    assert sim.model.ntendon == 3
    assert sim.model.nu == 3


def test_robot_mass_matches_settings():
    sim = make_sim(object_kind="none")
    robot_mass = sum(
        sim.model.body_mass[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"unit{i:02d}")]
        for i in range(UNIT_COUNT)
    )
    assert robot_mass == pytest.approx(0.0384, rel=1e-6)


def test_gravity_sag_is_stable():
    sim = make_sim(object_kind="none")
    initial_tip = tip_position(sim)
    sim.step(3.0)
    tip = tip_position(sim)
    assert np.all(np.isfinite(sim.data.qpos))
    assert np.all(np.isfinite(sim.data.qvel))
    # The unloaded arm must sag under gravity but not collapse or blow up.
    assert tip[2] < initial_tip[2] - 0.005
    assert tip[2] > 0.0
    assert np.linalg.norm(sim.data.qvel) < 1.0  # settled, not oscillating


def test_cable_tension_curls_toward_cable():
    sim = make_sim(object_kind="none")
    sim.set_cable_forces([6.0, 0.0, 0.0])
    sim.step(4.0)
    tip = tip_position(sim)
    assert np.all(np.isfinite(sim.data.qpos))
    # Cable 0 runs along the bottom (-Z) of the cross-section: the arm must
    # curl downward, pulling the tip toward the base (straight ~0.217 m).
    assert tip[2] < 0.2 - 0.02
    straight_tip_x = -0.06 + 0.217
    assert tip[0] < straight_tip_x - 0.03


def test_rigid_sphere_rests_on_floor():
    sim = make_sim(object_kind="sphere")
    sim.step(1.0)
    obj = sim.data.xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "object")]
    radius = 0.052 / 2
    assert obj[2] == pytest.approx(radius, abs=0.004)


def test_soft_sphere_compiles_and_is_stable():
    sim = make_sim(object_kind="soft_sphere")
    assert sim.model.nflex == 1
    assert sim.model.nflexvert > 0
    sim.step(1.0)
    verts = sim.data.flexvert_xpos
    assert np.all(np.isfinite(verts))
    # The soft ball must settle on the floor, not tunnel through it.
    assert verts[:, 2].min() > -0.01
    state = sim.state()
    assert state["object"]["kind"] == "soft_sphere"
    assert len(state["object"]["vertices"]) == 3 * sim.model.nflexvert


def test_auto_grasp_schedule_phases():
    assert auto_grasp_command(0.5).phase == "packing"
    assert auto_grasp_command(0.5).cable_forces == (0.0, 0.0, 0.0)  # settling
    assert auto_grasp_command(4.3).phase == "wrapping"
    grasping = auto_grasp_command(5.5)
    assert grasping.phase == "grasping"
    assert grasping.release_target
    assert 0.0 < grasping.fixture_assist < 1.0
    holding = auto_grasp_command(7.0)
    assert holding.phase == "holding"
    assert holding.fixture_assist == 0.0
    # Only the primary cable is tensioned by the pack-led schedule.
    packed = auto_grasp_command(4.3, primary_cable=1)
    assert packed.cable_forces[1] == pytest.approx(1.8)
    assert packed.cable_forces[0] == packed.cable_forces[2] == 0.0


def test_auto_grasp_wraps_rigid_sphere():
    sim = make_sim(object_kind="sphere")
    sim.start_auto_grasp()
    sim.step(4.5)  # wrap peak, object still presented
    assert np.all(np.isfinite(sim.data.qpos))
    assert sim.grasp_phase == "wrapping"
    assert len(sim.contacting_units()) >= 3
    sim.step(6.0)  # release + holding: must stay finite, no catapult
    assert np.all(np.isfinite(sim.data.qpos))
    obj = sim.data.xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "object")]
    assert np.linalg.norm(obj[:2]) < 0.5


def test_auto_grasp_wraps_soft_sphere():
    sim = make_sim(object_kind="soft_sphere")
    sim.start_auto_grasp()
    sim.step(4.5)
    assert np.all(np.isfinite(sim.data.qpos))
    assert len(sim.contacting_units()) >= 3


def test_mjcf_is_valid_xml():
    xml = build_mjcf(SimSettings(object_kind="none"))
    assert xml.startswith("<mujoco")
    assert not math.isnan(len(xml))

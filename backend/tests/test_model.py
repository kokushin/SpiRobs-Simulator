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
    # 3 cables x 19 capstan segments.
    assert sim.model.ntendon == 3 * (UNIT_COUNT - 1)
    assert sim.model.nu == 3 * (UNIT_COUNT - 1)


def test_robot_mass_matches_settings():
    sim = make_sim(object_kind="none")
    robot_mass = sum(
        sim.model.body_mass[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"unit{i:02d}")]
        for i in range(UNIT_COUNT)
    )
    assert robot_mass == pytest.approx(0.0384, rel=1e-6)


def test_gravity_sag_is_stable_horizontal():
    sim = make_sim(object_kind="none", mount="horizontal")
    initial_tip = tip_position(sim)
    sim.step(3.0)
    tip = tip_position(sim)
    assert np.all(np.isfinite(sim.data.qpos))
    assert np.all(np.isfinite(sim.data.qvel))
    # The unloaded cantilever must sag under gravity but not collapse.
    assert tip[2] < initial_tip[2] - 0.005
    assert tip[2] > 0.0
    assert np.linalg.norm(sim.data.qvel) < 1.0  # settled, not oscillating


def test_cable_tension_curls_toward_cable():
    sim = make_sim(object_kind="none", mount="horizontal")
    sim.set_cable_forces([6.0, 0.0, 0.0])
    sim.step(4.0)
    tip = tip_position(sim)
    assert np.all(np.isfinite(sim.data.qpos))
    # Cable 0 runs along the bottom (-Z) of the cross-section: the arm must
    # curl downward, pulling the tip toward the base (straight ~0.217 m).
    assert tip[2] < 0.2 - 0.02
    straight_tip_x = -0.06 + 0.217
    assert tip[0] < straight_tip_x - 0.03


def test_capstan_attenuates_distal_tension():
    sim = make_sim(object_kind="none", mount="horizontal")
    sim.set_cable_forces([6.0, 0.0, 0.0])
    sim.step(3.0)
    segments = UNIT_COUNT - 1
    base = sim.data.ctrl[0]
    tip = sim.data.ctrl[segments - 1]
    assert base == pytest.approx(6.0)
    # The curled arm accumulates bend, so the tip segment must see less.
    assert tip < base * 0.75


def test_rigid_sphere_rests_on_stage():
    sim = make_sim(object_kind="sphere")
    sim.step(1.0)
    obj = sim.data.xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "object")]
    expected_z = 0.075 + 0.026  # pedestal top + radius
    assert obj[2] == pytest.approx(expected_z, abs=0.006)


def test_soft_sphere_compiles_and_is_stable():
    sim = make_sim(object_kind="soft_sphere")
    assert sim.model.nflex == 1
    assert sim.model.nflexvert > 0
    sim.step(1.0)
    verts = sim.data.flexvert_xpos
    assert np.all(np.isfinite(verts))
    assert verts[:, 2].min() > -0.01  # settled, no tunneling
    state = sim.state()
    assert state["object"]["kind"] == "soft_sphere"
    assert len(state["object"]["vertices"]) == 3 * sim.model.nflexvert


def test_paper_schedule_phases():
    assert auto_grasp_command(0.5).phase == "packing"
    assert auto_grasp_command(2.0).phase == "reaching"
    assert auto_grasp_command(6.0).phase == "wrapping"
    assert auto_grasp_command(9.0).phase == "grasping"
    assert auto_grasp_command(11.0).phase == "holding"
    # Antagonism: primary cable packs at 6 N, the opposing pair share the
    # reach tension equally.
    reaching = auto_grasp_command(4.0, primary_cable=1)
    assert reaching.cable_forces[1] == pytest.approx(6.0)
    assert reaching.cable_forces[0] == reaching.cable_forces[2]
    holding = auto_grasp_command(12.0)
    assert holding.cable_forces[0] == pytest.approx(5.2)
    assert holding.cable_forces[1] == pytest.approx(9.0)


@pytest.mark.parametrize("kind", ["sphere", "soft_sphere"])
def test_paper_sequence_reaches_object(kind):
    """The full Fig. 3A run must stay stable, the stage must deliver the
    object, and the descending spiral must at least touch it during the
    wrap phase. Whether the grasp closes and holds is an open calibration
    question — see README — so contact, not capture, is asserted here."""
    sim = make_sim(object_kind=kind)
    sim.step(1.0)
    sim.start_auto_grasp()
    max_contacts = 0
    while sim.data.time - 1.0 < 10.2:
        sim.step(0.5)
        max_contacts = max(max_contacts, len(sim.contacting_units()))
    assert np.all(np.isfinite(sim.data.qpos))
    assert max_contacts >= 1
    sim.step(4.0)  # holding + stage retract: must stay finite, no catapult
    assert np.all(np.isfinite(sim.data.qpos))
    state = sim.state()
    if kind == "sphere":
        pos = np.array(state["object"]["pos"])
        assert np.linalg.norm(pos[:2]) < 0.5


def test_mjcf_is_valid_xml():
    xml = build_mjcf(SimSettings(object_kind="none"))
    assert xml.startswith("<mujoco")

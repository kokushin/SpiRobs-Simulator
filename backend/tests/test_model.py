import mujoco
import numpy as np
import pytest

from spirob_mujoco import SimSettings, SpiRobSim, auto_grasp_command, build_mjcf
from spirob_mujoco import grasp
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


@pytest.mark.parametrize("size_mm", [35.0, 52.0, 80.0])
@pytest.mark.parametrize("mount", ["standing", "hanging"])
def test_rigid_sphere_presented_at_grasp_height(mount, size_mm):
    # The presentation rig (string for standing, size-compensated pedestal
    # for hanging) must put the object CENTER at grasp_center_z regardless
    # of size, like the web simulator's fixture and the paper's hand-off.
    sim = make_sim(object_kind="sphere", object_size_mm=size_mm, mount=mount)
    sim.step(1.0)
    obj = sim.data.xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "object")]
    assert obj[2] == pytest.approx(sim.settings.grasp_center_z, abs=0.006)


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
    assert auto_grasp_command(3.0).phase == "reaching"
    assert auto_grasp_command(6.0).phase == "wrapping"
    assert auto_grasp_command(10.0).phase == "grasping"
    assert auto_grasp_command(11.5).phase == "holding"
    # Antagonism: primary cable packs at F_PACK, the opposing pair share
    # the reach tension equally.
    reaching = auto_grasp_command(4.0, primary_cable=1)
    assert reaching.cable_forces[1] == pytest.approx(grasp.F_PACK)
    assert reaching.cable_forces[0] == reaching.cable_forces[2]
    holding = auto_grasp_command(12.0)
    assert holding.cable_forces[0] == pytest.approx(grasp.F_WRAP)
    assert holding.cable_forces[1] == pytest.approx(grasp.F_GRASP)


def test_planar_hand_carries_rod_in_during_reach():
    """Paper Fig. 3A presentation: the rod parks ahead of the arm while the
    tip spiral packs (no interference), then the hand carries it to the
    grasp pose during REACH."""
    sim = make_sim()  # planar cylinder default
    sim.step(0.5)
    obj_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "object")
    parked_x = sim.data.xpos[obj_id][0]
    assert parked_x > sim.settings.object_x + 0.05  # clear of the packing sweep
    sim.start_auto_grasp()
    sim.step(5.5)  # PACK + REACH complete
    assert sim.data.xpos[obj_id][0] == pytest.approx(sim.settings.object_x, abs=0.01)
    assert sim.data.xpos[obj_id][1] == pytest.approx(sim.settings.object_y, abs=0.01)


@pytest.mark.parametrize("kind", ["cylinder", "soft_sphere"])
def test_paper_sequence_reaches_object(kind):
    """The full Fig. 3A run must stay stable and (for the hand-presented
    rod) the climbing spiral must wrap the rod with several units during
    wrap/grasp. Whether the grasp still holds after the hand lets go is an
    open calibration question — see README — so contact, not capture, is
    asserted here."""
    sim = make_sim(object_kind=kind)
    sim.step(1.0)
    sim.start_auto_grasp()
    max_contacts = 0
    while sim.data.time - 1.0 < 11.0:
        sim.step(0.5)
        max_contacts = max(max_contacts, len(sim.contacting_units()))
    assert np.all(np.isfinite(sim.data.qpos))
    if kind == "cylinder":
        assert max_contacts >= 3
    sim.step(5.0)  # holding + release: must stay finite, no catapult
    assert np.all(np.isfinite(sim.data.qpos))
    state = sim.state()
    if kind == "cylinder":
        pos = np.array(state["object"]["pos"])
        assert np.linalg.norm(pos[:2]) < 0.5


def test_mjcf_is_valid_xml():
    xml = build_mjcf(SimSettings(object_kind="none"))
    assert xml.startswith("<mujoco")


@pytest.mark.parametrize("arms", [2, 3, 8])
def test_array_model_structure(arms):
    sim = make_sim(object_kind="none", mount="array", arm_count=arms)
    assert sim.arm_count == arms
    # N arms x 19 joint pairs + no object; N arms x 57 cable motors + 3
    # gantry slides/servos.
    assert sim.model.nq == arms * 2 * (UNIT_COUNT - 1) + 3
    assert sim.model.nu == arms * 3 * (UNIT_COUNT - 1) + 3


def test_array_defaults_to_three_arms():
    sim = make_sim(object_kind="none", mount="array")
    assert sim.arm_count == 3


def test_array_transports_free_standing_rod():
    """The paper's multi-arm claim, end to end with the Fig. 6B sequence:
    the arms pack, approach from above, drape down around a rod that
    nothing holds (their radial pushes must cancel or it topples),
    squeeze, lift, carry ~20 cm, set it down under grip control, and
    leave — flaring the claw open with the opposing cables so the rising
    cage never brushes the deposited rod."""
    sim = make_sim(mount="array")  # default 30 mm / 70 g rod
    obj_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "object")
    sim.step(0.3)
    sim.start_auto_grasp()

    lifted = 0.0
    t0 = sim.data.time
    while sim.data.time - t0 < 29.0:
        sim.step(0.5)
        assert np.all(np.isfinite(sim.data.qpos))
        lifted = max(lifted, sim.data.xpos[obj_id][2])

    assert lifted > 0.11  # the 160 mm rod's center started at 0.08
    pos = sim.data.xpos[obj_id]
    zaxis = sim.data.xmat[obj_id].reshape(3, 3)[:, 2]
    assert pos[0] > 0.15  # transported toward the +X target
    assert zaxis[2] > 0.94  # still standing (< ~20 deg tilt)

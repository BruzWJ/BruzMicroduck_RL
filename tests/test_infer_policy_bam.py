"""infer_policy.py drives the CPU MuJoCo rehearsal with the SAME XC330 BAM M6 actuator
the policies are trained against in warp (bam.mujoco.MujocoController on a
motor-converted model). These tests lock the two halves together:

* the script's hardcoded BAM constants mirror ``_BAM_ACTUATOR_KWARGS`` in
  microduck_constants (not imported there to keep the script torch/warp-free);
* the motor conversion matches what ``bam.mjlab.BamActuator.edit_spec`` does
  (torque motors, voltage-bounded forcerange, armature, zeroed XML friction,
  stiff friction constraint) and a step loop runs with a live friction budget.
"""

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ip():
    spec = importlib.util.spec_from_file_location(
        "infer_policy", REPO / "scripts" / "infer_policy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cpu_bam_constants_mirror_training_cfg(ip):
    from bam.mjlab import BamActuator
    from mjlab_microduck.robot.microduck_constants import (
        XC330_T181_7V_BAM_JSON,
        _BAM_ACTUATOR_KWARGS as k,
    )

    assert ip.BAM_JSON_PATH == XC330_T181_7V_BAM_JSON
    assert ip.BAM_JSON_PATH == Path(k["json_path"])
    assert ip.BAM_KP_FW == k["kp_fw"]
    assert ip.BAM_VIN_RANGE == k["vin_range"]
    assert ip.BAM_VIN_DROP_GAIN_RANGE == k["vin_drop_gain_range"]
    assert ip.BAM_VIN_MIN == k["vin_min"]
    assert ip.BAM_MAX_CURRENT == k.get("max_current")
    assert ip.BAM_STIFF_SOLREF_FRICTION == BamActuator._STIFF_SOLREF_FRICTION
    assert ip.BAM_STIFF_SOLIMP_FRICTION == BamActuator._STIFF_SOLIMP_FRICTION


def test_behavior_handoff_defaults_match_the_bruz_bundle(ip):
    from mjlab_microduck.publish import bundle

    params = inspect.signature(ip.PolicyInference).parameters

    assert params["bam_ctrl"].default is inspect.Parameter.empty
    assert "new_cmd_obs" not in params
    assert "sit_onnx_path" not in params
    assert "ground_pick_end_phase" not in params
    assert params["kick_duration"].default == ip.KICK_DURATION_S == 0.5
    assert (
        params["roulade_duration"].default
        == ip.ROULADE_DURATION_S
        == bundle.ROULADE_DURATION_S
    )


def test_rehearsal_uses_one_unified_command_contract(ip):
    policy = object.__new__(ip.PolicyInference)
    policy.behavior_mode = None
    policy.current_policy = "walking"
    policy.vel_cmd = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    policy.head_offset = np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    policy.body_cmd = np.arange(6, dtype=np.float32)
    policy.sit_mode = False

    policy._update_command()
    assert policy.command.shape == (13,)
    assert np.array_equal(policy.command[:3], policy.vel_cmd)
    assert np.array_equal(policy.command[3:7], policy.head_offset)
    assert np.array_equal(policy.command[7:], policy.body_cmd)

    policy.current_policy = "sit"
    policy.sit_mode = True
    policy._update_command()
    assert policy.command[0] == 1.0

    policy.behavior_mode = "roulade"
    policy._update_command()
    assert np.array_equal(policy.command, np.zeros(13, dtype=np.float32))

    policy.bam_ctrl = SimpleNamespace(q_target=np.zeros(14, dtype=np.float32))
    targets = np.arange(14, dtype=np.float32)
    policy.set_position_targets(targets)
    assert np.array_equal(policy.bam_ctrl.q_target, targets)


@pytest.fixture(scope="module")
def bam_sim(ip):
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.0, ip.BAM_MAX_CURRENT)
    model, data, ctrl, names = ip.load_mujoco_with_bam(
        str(REPO / ip.MICRODUCK_XML), bam_model, 0.005, 0.1, ip.BAM_VIN_MIN
    )
    return ip, bam_model, model, data, ctrl, names


def test_actuators_converted_like_warp(bam_sim):
    ip, bam_model, model, data, ctrl, names = bam_sim
    kt, R = bam_model.kt.value, bam_model.R.value
    assert len(names) == 14 and model.nu == 14
    assert not any(n.startswith("passive_") for n in names)
    # Torque motors: ctrl is the BAM torque, no MuJoCo PD left over.
    assert (model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED).all()
    assert (model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE).all()
    assert np.allclose(model.actuator_gainprm[:, 0], 1.0)
    # (set_to_motor leaves the old PD biasprm bytes behind; inert under BIAS_NONE,
    # exactly as in warp's edit_spec.)
    assert (model.actuator_forcelimited == 1).all()
    assert np.allclose(model.actuator_forcerange[:, 1], 7.0 * kt / R)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    assert np.allclose(model.dof_armature[dofs], bam_model.actuator.get_extra_inertia())
    assert np.allclose(model.dof_solref[dofs], ip.BAM_STIFF_SOLREF_FRICTION)
    assert np.allclose(model.dof_solimp[dofs], ip.BAM_STIFF_SOLIMP_FRICTION)
    assert bam_model.actuator.kp == ip.BAM_KP_FW
    assert bam_model.actuator.max_current is None


def test_xc330_profile_matches_robotis_9v_endpoints(ip):
    """The effective DC terms reproduce the published torque/speed endpoints.

    The friction coefficients remain the XL330 M6 prior until an XC330 bench
    identification is available, so include that M6 budget in the speed check.
    """
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 9.0, None)
    a = bam_model.kt.value / bam_model.R.value
    assert a * 9.0 == pytest.approx(0.65, rel=1e-9)

    load = bam_model.load_friction_motor.value
    base = bam_model.friction_base.value
    viscous = bam_model.friction_viscous.value
    electrical_damping = bam_model.kt.value**2 / bam_model.R.value
    speed_rad_s = ((1.0 - load) * a * 9.0 - base) / (
        (1.0 - load) * electrical_damping + viscous
    )
    assert speed_rad_s * 60.0 / (2.0 * np.pi) == pytest.approx(83.0, rel=1e-9)


def test_bam_step_loop_runs_with_live_friction(bam_sim):
    ip, bam_model, model, data, ctrl, names = bam_sim
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = model.jnt_qposadr[fj]
    mujoco.mj_resetData(model, data)
    data.qpos[qa + 2] = 0.125
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    jq = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    data.qpos[jq] = ip.DEFAULT_POSE
    ctrl.reset(data.qpos)
    ctrl.q_target[:] = ip.DEFAULT_POSE
    mujoco.mj_forward(model, data)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    for _ in range(100):
        ctrl.update()
        mujoco.mj_step(model, data)
    assert not np.isnan(data.qpos).any()
    limit = model.actuator_forcerange[0, 1]
    assert (np.abs(data.ctrl) <= limit + 1e-9).all()  # ctrl IS the motor torque
    assert (model.dof_frictionloss[dofs] > 0).all()  # BAM budget written every step
    assert np.allclose(model.dof_damping[dofs], bam_model.friction_viscous.value)


def test_ground_pick_rehearsal_does_not_interrupt_the_return_to_stand(ip):
    from mjlab_microduck.publish import bundle
    from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
        GP_PERIOD,
        RISE_END,
    )

    policy = object.__new__(ip.PolicyInference)
    policy.ground_pick_mode = True
    policy.ground_pick_phase = 0.70
    policy.ground_pick_period = ip.GROUND_PICK_PERIOD_S
    policy.command = np.arange(13, dtype=np.float32)
    ended = []
    policy._end_ground_pick = lambda: ended.append(True)

    assert ip.GROUND_PICK_PERIOD_S == bundle.GROUND_PICK_PERIOD_S == GP_PERIOD
    assert ip.GROUND_PICK_END_PHASE == bundle.GROUND_PICK_END_PHASE
    assert RISE_END < ip.GROUND_PICK_END_PHASE
    policy.update_ground_pick_phase(0.4)  # phase 0.80: rise has only just completed
    assert policy.ground_pick_phase == pytest.approx(0.80)
    assert np.all(policy.command[3:] == 0.0)
    assert not ended

    policy.update_ground_pick_phase(0.8)  # phase 1.00: upright rest is complete
    assert ended == [True]

from types import SimpleNamespace

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import NullCurriculumManager

from mjlab_microduck.evaluate import (
    EvaluationConfig,
    RunningStats,
    _checkpoint_step,
    _cyclic_phase_masks,
    _ground_pick_phase_masks,
    _validate_config,
    apply_checkpoint_curriculum_and_freeze,
    set_joint_position_action_scale,
)
from mjlab_microduck.tasks.distill import strip_training_only_config
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import HOLD_END, RISE_END


def test_running_stats_counts_nonfinite_without_poisoning_summary():
    stats = RunningStats()
    stats.update(torch.tensor([1.0, 2.0, float("nan"), float("inf")]))
    result = stats.as_dict()

    assert result == {
        "count": 4,
        "finite_count": 2,
        "nonfinite_count": 2,
        "mean": 1.5,
        "std": 0.5,
        "min": 1.0,
        "max": 2.0,
        "sum": 3.0,
    }


def test_curriculum_is_reapplied_at_checkpoint_step_then_frozen():
    calls: list[tuple[str, int, list[int]]] = []

    def staged(env, env_ids, *, target):
        calls.append(("staged", env.common_step_counter, env_ids.tolist()))
        env.target = target
        return {"target": torch.tensor(target)}

    def terrain_levels_vel(env, env_ids):
        calls.append(("adaptive", env.common_step_counter, env_ids.tolist()))

    staged_cfg = SimpleNamespace(func=staged, params={"target": 7})
    adaptive_cfg = SimpleNamespace(func=terrain_levels_vel, params={})
    manager = SimpleNamespace(
        _term_names=["stage", "terrain_levels"],
        _term_cfgs=[staged_cfg, adaptive_cfg],
        _curriculum_state={"stage": None, "terrain_levels": None},
    )
    env = SimpleNamespace(
        curriculum_manager=manager,
        common_step_counter=4321,
        num_envs=2,
        device="cpu",
    )

    report = apply_checkpoint_curriculum_and_freeze(env)

    assert calls == [("staged", 4321, [0, 1])]
    assert env.target == 7
    assert env.common_step_counter == 4321
    assert isinstance(env.curriculum_manager, NullCurriculumManager)
    assert report["applied_step_based_terms"] == ["stage"]
    assert report["skipped_performance_adaptive_terms"] == ["terrain_levels"]
    assert report["state_after_apply"]["stage"] == {"target": 7}


def test_checkpoint_step_requires_explicit_env_state_shape():
    assert _checkpoint_step({"env_state": {"common_step_counter": 12}}) == 12
    assert _checkpoint_step({}) is None
    assert _checkpoint_step(None) is None


def test_deployment_action_scale_changes_targets_not_raw_actions():
    term = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=1.0,
        use_default_offset=True,
    )
    env_cfg = SimpleNamespace(actions={"joint_pos": term})

    original = set_joint_position_action_scale(env_cfg, 0.8)

    assert original == 1.0
    assert term.scale == 0.8


def test_deployment_action_scale_requires_one_scalar_joint_position_term():
    try:
        set_joint_position_action_scale(SimpleNamespace(actions={}), 0.8)
    except RuntimeError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("missing joint-position action term was accepted")

    term = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale={".*": 1.0},
        use_default_offset=True,
    )
    try:
        set_joint_position_action_scale(
            SimpleNamespace(actions={"joint_pos": term}), 0.8
        )
    except TypeError as exc:
        assert "must be a scalar" in str(exc)
    else:
        raise AssertionError("pattern-valued action scale was accepted")


def test_evaluation_rejects_nonpositive_or_nonfinite_action_scale(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    for scale in (0.0, -0.1, float("nan"), float("inf")):
        try:
            _validate_config(
                EvaluationConfig(checkpoint_file=str(checkpoint), action_scale=scale)
            )
        except ValueError as exc:
            assert "action_scale must be finite and positive" in str(exc)
        else:
            raise AssertionError(f"invalid action scale {scale!r} was accepted")


def test_inference_disables_bc_teacher_resolution_without_other_mutation():
    cfg = {"algorithm": {"bc_cfg": {"wandb_run_path": "private/run"}, "gamma": 0.99}}

    disabled = strip_training_only_config(cfg)

    assert disabled == ["algorithm.bc_cfg"]
    assert cfg == {"algorithm": {"bc_cfg": None, "gamma": 0.99}}


def test_ground_pick_phase_masks_cover_approach_rise_and_upright_rest():
    phases = torch.tensor(
        [0.0, HOLD_END - 0.001, HOLD_END, RISE_END - 0.001, RISE_END, 0.999]
    )
    angles = phases * (2.0 * torch.pi)
    command = torch.stack(
        (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=1
    )

    decoded, masks = _ground_pick_phase_masks(
        command, hold_end=HOLD_END, rise_end=RISE_END
    )

    assert torch.allclose(decoded, phases, atol=1.0e-6)
    assert masks["approach"].tolist() == [True, True, False, False, False, False]
    assert masks["rise"].tolist() == [False, False, True, True, False, False]
    assert masks["upright_rest"].tolist() == [False, False, False, False, True, True]
    coverage = sum(mask.to(torch.int8) for mask in masks.values())
    assert coverage.tolist() == [1, 1, 1, 1, 1, 1]


def test_ground_pick_phase_masks_reject_invalid_boundaries_and_commands():
    command = torch.tensor([[1.0, 0.0, 0.0]])
    with torch.no_grad():
        try:
            _ground_pick_phase_masks(command, hold_end=0.8, rise_end=0.8)
        except ValueError as exc:
            assert "0 < hold < rise < 1" in str(exc)
        else:
            raise AssertionError("invalid boundaries were accepted")

    command[0, 0] = float("nan")
    try:
        _ground_pick_phase_masks(command, hold_end=HOLD_END, rise_end=RISE_END)
    except RuntimeError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite command was accepted")


def test_cyclic_phase_masks_cover_roller_crouch_segments_and_reject_gaps():
    boundaries = {
        "descent": (0.0, 0.1),
        "crouch_hold": (0.1, 0.5),
        "rise": (0.5, 0.6),
        "stand_rest": (0.6, 1.0),
    }
    phases = torch.tensor([0.0, 0.099, 0.1, 0.499, 0.5, 0.599, 0.6, 0.999])
    angles = phases * (2.0 * torch.pi)
    command = torch.stack(
        (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=1
    )

    decoded, masks = _cyclic_phase_masks(
        command, boundaries, task_label="Roller-crouch"
    )

    assert torch.allclose(decoded, phases, atol=1.0e-6)
    assert masks["descent"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert masks["crouch_hold"].tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert masks["rise"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    assert masks["stand_rest"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]

    with torch.no_grad():
        try:
            _cyclic_phase_masks(
                command,
                {"first": (0.0, 0.4), "second": (0.5, 1.0)},
                task_label="Roller-crouch",
            )
        except ValueError as exc:
            assert "contiguous" in str(exc)
        else:
            raise AssertionError("phase gap was accepted")

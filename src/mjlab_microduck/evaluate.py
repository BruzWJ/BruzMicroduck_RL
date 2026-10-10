"""Deterministic, headless checkpoint rollouts for Microduck policies.

This is deliberately a checkpoint evaluator rather than an ONNX evaluator.  The
checkpoint gives rsl-rl the actor architecture and observation normalizer while the
task registry supplies the simulation, commands, rewards, and terminations.  ONNX
artifacts should still be checked separately with ``publish.manifest.check_onnx`` and
``smoke_run_onnx``; those tests validate the deployment graph but cannot measure task
quality without reimplementing the environment-side observation contract.

One ordering detail matters.  ``RslRlVecEnvWrapper`` resets the environment before
``runner.load`` restores ``common_step_counter`` from the checkpoint.  Therefore the
first reset applies stage-zero curricula.  :func:`run_evaluation` reapplies every
step-based curriculum term at the restored checkpoint step, then replaces the
curriculum manager with a no-op for the measurement window.  Performance-adaptive
terrain curricula are intentionally not replayed: their per-environment terrain
levels are not stored in checkpoints, and evaluating a just-reset episode must not
pretend it earned an up/down transition.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import NullCurriculumManager
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from mjlab_microduck.publish.manifest import ACTION_LEN, OBS_LEN
from mjlab_microduck.tasks.distill import strip_training_only_config

GROUND_PICK_TASK_ID = "Mjlab-GroundPick-Flat-MicroDuck"
ROLLER_CROUCH_TASK_ID = "Mjlab-RollerCrouch-Flat-MicroDuck"


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for one fixed-seed, headless checkpoint rollout."""

    checkpoint_file: str
    output: str = "checkpoint-eval.json"
    num_envs: int = 64
    steps: int = 500
    seed: int = 0
    device: str | None = None
    env_cfg: Literal["play", "training"] = "training"
    curriculum_step: int | None = None
    action_scale: float = 1.0


@dataclass
class RunningStats:
    """Numerically stable-enough aggregate statistics without retaining rollouts."""

    count: int = 0
    finite_count: int = 0
    nonfinite_count: int = 0
    total: float = 0.0
    total_squares: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        self.count += flat.numel()
        finite = flat[torch.isfinite(flat)]
        self.finite_count += finite.numel()
        self.nonfinite_count += flat.numel() - finite.numel()
        if finite.numel() == 0:
            return
        self.total += finite.sum().item()
        self.total_squares += torch.square(finite).sum().item()
        self.minimum = min(self.minimum, finite.min().item())
        self.maximum = max(self.maximum, finite.max().item())

    def as_dict(self) -> dict[str, int | float | None]:
        if self.finite_count == 0:
            mean = std = minimum = maximum = None
        else:
            mean = self.total / self.finite_count
            variance = max(0.0, self.total_squares / self.finite_count - mean * mean)
            std = math.sqrt(variance)
            minimum = self.minimum
            maximum = self.maximum
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
            "sum": self.total if self.finite_count else None,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_performance_adaptive_curriculum(name: str, term_cfg: Any) -> bool:
    """Return true for curricula whose state cannot be reconstructed from a step.

    Today all registered performance-adaptive terms are terrain-level functions.
    Match both the stable registry name and function name so the rule also covers
    the Microduck slope variant without depending on function identity.
    """

    func = getattr(term_cfg, "func", None)
    func_name = getattr(func, "__name__", type(func).__name__)
    return name == "terrain_levels" or func_name.startswith("terrain_levels")


def apply_checkpoint_curriculum_and_freeze(
    env: ManagerBasedRlEnv,
) -> dict[str, Any]:
    """Apply restorable curricula at ``env.common_step_counter`` and freeze them.

    The checkpoint's common step must already have been restored by the runner.
    The function does not alter it.  It is kept separate so the ordering invariant
    can be unit tested without constructing MuJoCo.
    """

    manager = env.curriculum_manager
    checkpoint_step = int(env.common_step_counter)
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
    applied: list[str] = []
    skipped: list[str] = []

    names = list(getattr(manager, "_term_names", []))
    cfgs = list(getattr(manager, "_term_cfgs", []))
    state = getattr(manager, "_curriculum_state", {})
    for name, term_cfg in zip(names, cfgs, strict=True):
        if _is_performance_adaptive_curriculum(name, term_cfg):
            skipped.append(name)
            continue
        term_state = term_cfg.func(env, env_ids, **term_cfg.params)
        state[name] = term_state
        applied.append(name)

    report = {
        "checkpoint_common_step": checkpoint_step,
        "applied_step_based_terms": applied,
        "skipped_performance_adaptive_terms": skipped,
        "state_after_apply": _jsonable(state),
        "frozen_during_rollout": True,
    }
    if int(env.common_step_counter) != checkpoint_step:
        raise RuntimeError("A curriculum term unexpectedly changed common_step_counter")
    env.curriculum_manager = NullCurriculumManager()
    return report


def _actor_observation(observations: TensorDict) -> torch.Tensor:
    actor = observations.get("actor")
    if not isinstance(actor, torch.Tensor):
        raise TypeError(
            "Expected the registry task to expose a tensor 'actor' observation"
        )
    return actor


def _checkpoint_step(infos: Any) -> int | None:
    if not isinstance(infos, dict):
        return None
    env_state = infos.get("env_state")
    if not isinstance(env_state, dict) or "common_step_counter" not in env_state:
        return None
    return int(env_state["common_step_counter"])


def set_joint_position_action_scale(env_cfg: Any, action_scale: float) -> float:
    """Set the simulated target scale while preserving raw previous-action observations.

    ``robotd`` stores the raw network output in the observation history, then applies
    its deployment scale only when producing joint targets.  Multiplying the tensor
    passed to ``env.step`` would incorrectly put the scaled value in the task's
    previous-action observation.  Updating the action term instead reproduces the
    daemon ordering.

    Returns the task's configured scale before the deployment override.
    """

    actions = getattr(env_cfg, "actions", None)
    if not isinstance(actions, dict):
        raise TypeError("Task environment actions must be a mapping")
    terms = [
        term for term in actions.values() if isinstance(term, JointPositionActionCfg)
    ]
    if len(terms) != 1:
        raise RuntimeError(
            f"Expected exactly one JointPositionActionCfg, found {len(terms)}"
        )
    term = terms[0]
    configured_scale = term.scale
    if not isinstance(configured_scale, (int, float)) or isinstance(
        configured_scale, bool
    ):
        raise TypeError(
            "Task joint-position action scale must be a scalar before deployment override"
        )
    term.scale = action_scale
    return float(configured_scale)


def _command_tensors(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name in env.command_manager.active_terms:
        command = env.command_manager.get_command(name)
        if isinstance(command, torch.Tensor):
            values[name] = command
    return values


def _state_tensors(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    robot = env.scene["robot"]
    quat = robot.data.root_link_quat_w
    # Quaternion convention is [w, x, y, z].  R_zz = 1 - 2(x^2 + y^2), which
    # is 1 upright, 0 sideways, and -1 inverted.
    upright_cosine = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    origin_z = env.scene.env_origins[:, 2]
    tensors = {
        "robot_height_m": robot.data.root_link_pos_w[:, 2] - origin_z,
        "robot_upright_cosine": upright_cosine,
        "robot_xy_speed_m_s": torch.linalg.vector_norm(
            robot.data.root_link_lin_vel_w[:, :2], dim=-1
        ),
    }
    if "ball" in env.scene.entities:
        ball = env.scene["ball"]
        tensors["ball_xy_speed_m_s"] = torch.linalg.vector_norm(
            ball.data.root_link_lin_vel_w[:, :2], dim=-1
        )
    return tensors


def _ground_pick_phase_masks(
    command: torch.Tensor,
    *,
    hold_end: float,
    rise_end: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Decode the cyclic command and partition one full ground-pick cycle.

    ``command[:, :2]`` is ``[cos(2π phase), sin(2π phase)]``.  Approach includes
    both the descent and short low hold, rise covers the commanded return, and
    upright-rest is the deployment-critical tail before control is handed back.
    """

    if not 0.0 < hold_end < rise_end < 1.0:
        raise ValueError(
            f"ground-pick phase boundaries must satisfy 0 < hold < rise < 1; "
            f"got hold={hold_end}, rise={rise_end}"
        )
    return _cyclic_phase_masks(
        command,
        {
            "approach": (0.0, hold_end),
            "rise": (hold_end, rise_end),
            "upright_rest": (rise_end, 1.0),
        },
        task_label="Ground-pick",
    )


def _cyclic_phase_masks(
    command: torch.Tensor,
    boundaries: Mapping[str, Sequence[float]],
    *,
    task_label: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Decode a cosine/sine phase command and apply a contiguous partition."""
    if command.ndim != 2 or command.shape[1] < 2:
        raise RuntimeError(
            f"{task_label} twist command must have shape [num_envs, >=2], got "
            f"{tuple(command.shape)}"
        )
    if not torch.all(torch.isfinite(command[:, :2])):
        raise RuntimeError(f"{task_label} phase command contains non-finite values")

    previous_end = 0.0
    normalized: dict[str, tuple[float, float]] = {}
    for name, limits in boundaries.items():
        if len(limits) != 2:
            raise ValueError(f"{task_label} segment {name!r} needs [start, end]")
        start, end = float(limits[0]), float(limits[1])
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and math.isclose(start, previous_end, abs_tol=1.0e-9)
            and start < end <= 1.0
        ):
            raise ValueError(
                f"{task_label} phase segments must be finite, contiguous, and "
                f"cover [0, 1]; {name!r} is [{start}, {end}] after {previous_end}"
            )
        normalized[name] = (start, end)
        previous_end = end
    if not normalized or not math.isclose(previous_end, 1.0, abs_tol=1.0e-9):
        raise ValueError(f"{task_label} phase segments must end at 1.0")

    phase = torch.remainder(
        torch.atan2(command[:, 1], command[:, 0]) / (2.0 * math.pi), 1.0
    )
    masks = {
        name: (phase >= start) & (phase < end)
        for name, (start, end) in normalized.items()
    }
    coverage = sum(mask.to(torch.int8) for mask in masks.values())
    if not torch.all(coverage == 1):
        raise RuntimeError(f"{task_label} phase partition was not one-hot")
    return phase, masks


def _validate_config(cfg: EvaluationConfig) -> tuple[Path, Path]:
    if cfg.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if cfg.steps <= 0:
        raise ValueError("steps must be positive")
    if cfg.curriculum_step is not None and cfg.curriculum_step < 0:
        raise ValueError("curriculum_step must be non-negative")
    if not math.isfinite(cfg.action_scale) or cfg.action_scale <= 0.0:
        raise ValueError("action_scale must be finite and positive")
    checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    output = Path(cfg.output).expanduser().resolve()
    return checkpoint, output


def run_evaluation(task_id: str, cfg: EvaluationConfig) -> dict[str, Any]:
    """Run one task-aware checkpoint evaluation and write its JSON report."""

    checkpoint, output = _validate_config(cfg)
    if os.environ.get("MICRODUCK_WARM_START", "") not in ("", "0"):
        raise RuntimeError(
            "MICRODUCK_WARM_START must be unset for evaluation: it discards the "
            "checkpoint curriculum step"
        )

    configure_torch_backends()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=cfg.env_cfg == "play")
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    task_action_scale = set_joint_position_action_scale(env_cfg, cfg.action_scale)

    wrapped: RslRlVecEnvWrapper | None = None
    base_env: ManagerBasedRlEnv | None = None
    try:
        base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
        wrapped = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

        train_cfg = asdict(agent_cfg)
        disabled_training_components = strip_training_only_config(train_cfg)
        runner_cls = load_runner_cls(task_id) or OnPolicyRunner
        runner = runner_cls(wrapped, train_cfg, log_dir=None, device=device)
        infos = runner.load(
            str(checkpoint),
            load_cfg={"actor": True, "iteration": True},
            strict=True,
            map_location=device,
        )

        saved_step = _checkpoint_step(infos)
        if cfg.curriculum_step is None:
            if saved_step is None:
                raise RuntimeError(
                    "Checkpoint has no infos.env_state.common_step_counter; pass "
                    "--curriculum-step explicitly rather than evaluating stage zero by accident"
                )
            curriculum_step_source = "checkpoint"
        else:
            base_env.common_step_counter = cfg.curriculum_step
            curriculum_step_source = "cli_override"

        curriculum = apply_checkpoint_curriculum_and_freeze(base_env)
        curriculum["source"] = curriculum_step_source
        curriculum["saved_checkpoint_common_step"] = saved_step

        # Reset once after the final-stage config mutations.  The no-op manager keeps
        # those values fixed across auto-resets for a stationary evaluation protocol.
        observations, _ = wrapped.reset()
        policy = runner.get_inference_policy(device=device)

        actor_obs = _actor_observation(observations)
        if actor_obs.shape != (cfg.num_envs, OBS_LEN):
            raise RuntimeError(
                f"Actor observation shape is {tuple(actor_obs.shape)}, expected "
                f"({cfg.num_envs}, {OBS_LEN})"
            )
        if not torch.all(torch.isfinite(actor_obs)):
            raise RuntimeError("Initial actor observations contain non-finite values")
        with torch.inference_mode():
            action_a = policy(observations)
            action_b = policy(observations)
        if action_a.shape != (cfg.num_envs, ACTION_LEN):
            raise RuntimeError(
                f"Actor action shape is {tuple(action_a.shape)}, expected "
                f"({cfg.num_envs}, {ACTION_LEN})"
            )
        if not torch.all(torch.isfinite(action_a)):
            raise RuntimeError("Initial actor actions contain non-finite values")
        actor_max_abs_diff = (
            torch.max(torch.abs(action_a - action_b)).item()
            if action_a.numel()
            else 0.0
        )
        actor_bitwise_equal = torch.equal(action_a, action_b)
        if not actor_bitwise_equal:
            raise RuntimeError(
                "Inference actor is not deterministic for identical observations "
                f"(max abs difference {actor_max_abs_diff:g})"
            )

        observation_stats = RunningStats()
        action_stats = RunningStats()
        reward_stats = RunningStats()
        state_stats: dict[str, RunningStats] = {}
        command_stats: dict[str, RunningStats] = {}
        reward_term_stats = {
            name: RunningStats() for name in base_env.reward_manager.active_terms
        }
        phase_metric_name: str | None = None
        phase_task_label: str | None = None
        phase_report: dict[str, Any] | None = None
        phase_segments: tuple[str, ...] = ()
        phase_segment_reward_stats: dict[str, dict[str, RunningStats]] = {}
        phase_segment_state_stats: dict[str, dict[str, RunningStats]] = {}
        phase_segment_command_samples: dict[str, int] = {}
        phase_segment_state_samples: dict[str, int] = {}
        if task_id == GROUND_PICK_TASK_ID:
            from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
                HOLD_END,
                RISE_END,
            )

            phase_metric_name = "ground_pick_phase_segments"
            phase_task_label = "Ground-pick"
            phase_segments = ("approach", "rise", "upright_rest")
            phase_report = {
                "encoding": "[cos(2*pi*phase), sin(2*pi*phase), 0]",
                "boundaries": {
                    "approach": [0.0, HOLD_END],
                    "rise": [HOLD_END, RISE_END],
                    "upright_rest": [RISE_END, 1.0],
                },
            }
        elif task_id == ROLLER_CROUCH_TASK_ID:
            from mjlab_microduck.tasks.microduck_roller_crouch_env_cfg import (
                DESCENT_END,
                HOLD_END,
                RISE_END,
            )

            phase_metric_name = "roller_crouch_phase_segments"
            phase_task_label = "Roller-crouch"
            phase_segments = ("descent", "crouch_hold", "rise", "stand_rest")
            phase_report = {
                "encoding": "[cos(2*pi*phase), sin(2*pi*phase), 0]",
                "boundaries": {
                    "descent": [0.0, DESCENT_END],
                    "crouch_hold": [DESCENT_END, HOLD_END],
                    "rise": [HOLD_END, RISE_END],
                    "stand_rest": [RISE_END, 1.0],
                },
            }
        if phase_report is not None:
            for segment in phase_segments:
                phase_segment_reward_stats[segment] = {
                    name: RunningStats()
                    for name in base_env.reward_manager.active_terms
                }
                phase_segment_state_stats[segment] = {}
                phase_segment_command_samples[segment] = 0
                phase_segment_state_samples[segment] = 0
        termination_counts = {
            name: 0 for name in base_env.termination_manager.active_terms
        }
        terminated_count = 0
        truncated_count = 0
        done_count = 0
        action_abs_gt_one = 0
        action_value_count = 0
        per_env_episode_steps = torch.zeros(
            cfg.num_envs, dtype=torch.int64, device=base_env.device
        )
        completed_episode_lengths = RunningStats()

        for _ in range(cfg.steps):
            actor_obs = _actor_observation(observations)
            observation_stats.update(actor_obs)
            if observation_stats.nonfinite_count:
                raise RuntimeError("Non-finite actor observations encountered")

            with torch.inference_mode():
                actions = policy(observations)
            action_stats.update(actions)
            if action_stats.nonfinite_count:
                raise RuntimeError("Non-finite policy actions encountered")
            action_abs_gt_one += torch.count_nonzero(torch.abs(actions) > 1.0).item()
            action_value_count += actions.numel()

            phase_masks: dict[str, torch.Tensor] | None = None
            if phase_report is not None and phase_task_label is not None:
                twist = base_env.command_manager.get_command("twist")
                if task_id == GROUND_PICK_TASK_ID:
                    boundaries = phase_report["boundaries"]
                    _, phase_masks = _ground_pick_phase_masks(
                        twist,
                        hold_end=boundaries["approach"][1],
                        rise_end=boundaries["rise"][1],
                    )
                else:
                    _, phase_masks = _cyclic_phase_masks(
                        twist,
                        phase_report["boundaries"],
                        task_label=phase_task_label,
                    )

            observations, rewards, dones, _ = wrapped.step(actions)
            reward_stats.update(rewards)
            if reward_stats.nonfinite_count:
                raise RuntimeError("Non-finite total rewards encountered")

            per_env_episode_steps += 1
            done_mask = dones.to(dtype=torch.bool)
            if torch.any(done_mask):
                completed_episode_lengths.update(per_env_episode_steps[done_mask])
                per_env_episode_steps[done_mask] = 0

            terminated_count += torch.count_nonzero(base_env.reset_terminated).item()
            truncated_count += torch.count_nonzero(base_env.reset_time_outs).item()
            done_count += torch.count_nonzero(done_mask).item()
            for name in termination_counts:
                termination_counts[name] += torch.count_nonzero(
                    base_env.termination_manager.get_term(name)
                ).item()

            step_reward = base_env.reward_manager._step_reward
            for term_index, stats in enumerate(reward_term_stats.values()):
                values = step_reward[:, term_index]
                stats.update(values)

            if phase_masks is not None:
                for segment, mask in phase_masks.items():
                    phase_segment_command_samples[segment] += int(mask.sum())
                    for term_index, stats in enumerate(
                        phase_segment_reward_stats[segment].values()
                    ):
                        stats.update(step_reward[:, term_index][mask])

            state_values = _state_tensors(base_env)
            for name, values in state_values.items():
                state_stats.setdefault(name, RunningStats()).update(values)
            if phase_masks is not None:
                # Auto-reset replaces terminal states before they can be sampled.
                # Exclude those rows rather than assigning a reset standing pose to
                # the phase that produced the termination.
                for segment, mask in phase_masks.items():
                    state_mask = mask & ~done_mask
                    phase_segment_state_samples[segment] += int(state_mask.sum())
                    for name, values in state_values.items():
                        phase_segment_state_stats[segment].setdefault(
                            name, RunningStats()
                        ).update(values[state_mask])
            for name, values in _command_tensors(base_env).items():
                command_stats.setdefault(name, RunningStats()).update(values)

        env_samples = cfg.num_envs * cfg.steps
        final_common_step = int(base_env.common_step_counter)
        result: dict[str, Any] = {
            "schema_version": 1,
            "task_id": task_id,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "iteration": int(runner.current_learning_iteration),
            },
            "protocol": {
                "seed": cfg.seed,
                "device": device,
                "env_cfg": cfg.env_cfg,
                "num_envs": cfg.num_envs,
                "steps": cfg.steps,
                "env_samples": env_samples,
                "step_dt_s": base_env.step_dt,
                "simulated_seconds_per_env": cfg.steps * base_env.step_dt,
                "headless": True,
                "auto_reset": base_env.cfg.auto_reset,
                "disabled_training_only_components": disabled_training_components,
                "action_scale": cfg.action_scale,
                "task_action_scale_before_override": task_action_scale,
                "start_common_step": curriculum["checkpoint_common_step"],
                "end_common_step": final_common_step,
            },
            "curriculum": curriculum,
            "actor": {
                "observation_shape": list(actor_obs.shape),
                "action_shape": list(action_a.shape),
                "same_input_bitwise_equal": actor_bitwise_equal,
                "same_input_max_abs_difference": actor_max_abs_diff,
                "observations": observation_stats.as_dict(),
                "actions": action_stats.as_dict(),
                "action_abs_gt_one_fraction": (
                    action_abs_gt_one / action_value_count
                    if action_value_count
                    else None
                ),
            },
            "reward": {
                "total": reward_stats.as_dict(),
                "terms_weighted_rate": {
                    name: {
                        **stats.as_dict(),
                        "integrated_sum": (
                            stats.total * base_env.step_dt
                            if stats.finite_count
                            else None
                        ),
                        "weight": float(
                            base_env.reward_manager.get_term_cfg(name).weight
                        ),
                    }
                    for name, stats in reward_term_stats.items()
                },
            },
            "terminations": {
                "terminated_count": terminated_count,
                "truncated_count": truncated_count,
                "done_count": done_count,
                "done_fraction_per_env_step": done_count / env_samples,
                "by_term": termination_counts,
                "completed_episode_lengths_steps": completed_episode_lengths.as_dict(),
            },
            "state": {name: stats.as_dict() for name, stats in state_stats.items()},
            "commands": {
                name: stats.as_dict() for name, stats in command_stats.items()
            },
        }
        if phase_report is not None and phase_metric_name is not None:
            phase_report["segments"] = {
                segment: {
                    "command_samples": phase_segment_command_samples[segment],
                    "post_step_state_samples": phase_segment_state_samples[segment],
                    "state": {
                        name: stats.as_dict()
                        for name, stats in phase_segment_state_stats[segment].items()
                    },
                    "reward_terms_weighted_rate": {
                        name: stats.as_dict()
                        for name, stats in phase_segment_reward_stats[segment].items()
                    },
                }
                for segment in phase_segments
            }
            result["task_metrics"] = {phase_metric_name: phase_report}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
        )
        print(f"[INFO] Evaluation report written to {output}")
        return result
    finally:
        if wrapped is not None:
            wrapped.close()
        elif base_env is not None:
            base_env.close()


def main() -> None:
    # Import both registries: mjlab provides the base registry and the plugin import
    # makes this package's Microduck tasks available when invoked as a plain script.
    import mjlab.tasks  # noqa: F401

    import mjlab_microduck.tasks  # noqa: F401

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
    )
    args = tyro.cli(
        EvaluationConfig,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
    )
    run_evaluation(chosen_task, args)


if __name__ == "__main__":
    main()

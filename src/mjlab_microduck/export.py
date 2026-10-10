"""Export a trained checkpoint to ONNX, with the observation normalizer baked in.

This is the ONE path from a checkpoint to a deployable `.onnx`: `runner.export_policy_to_onnx`
emits `actor(normalizer(obs))`, so what the robot runs is what training saw. In-sim `play`
applies the normalizer itself and hides a hand-converted checkpoint that forgot it — never
convert by hand.

`scripts/export.py` is the command-line wrapper; `mjlab_microduck.publish` calls
:func:`run_export` directly so a published policy cannot skip this step.
"""

import hashlib
import os
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

from mjlab_microduck.export_parity import (
    ExportParityResult,
    verify_onnx_export_parity,
)
from mjlab_microduck.tasks.distill import strip_training_only_config


@dataclass(frozen=True)
class ExportConfig:
    onnx_file: str = "output.onnx"
    wandb_run_path: str | None = None
    checkpoint: int | None = None  # Select checkpoint by iteration number (e.g. 3000)
    checkpoint_file: str | None = None
    num_envs: int | None = None
    device: str | None = None


@dataclass(frozen=True)
class ExportResult:
    """What an export produced and where it came from, for the publisher's provenance block."""

    onnx_path: Path
    checkpoint_path: Path
    wandb_run_path: str | None
    checkpoint_iteration: int | None
    parity: ExportParityResult


def _iteration_of(checkpoint_path: Path) -> int | None:
    match = re.search(r"model_(\d+)\.pt$", checkpoint_path.name)
    return int(match.group(1)) if match else None


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _atomic_export_onnx(
    destination: Path,
    export: Callable[[Path], None],
    verify: Callable[[Path], ExportParityResult],
    annotate: Callable[[Path, ExportParityResult], None],
) -> ExportParityResult:
    """Publish an ONNX export only after verification and annotation succeed."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary_dir:
        staged = Path(temporary_dir) / destination.name
        export(staged)
        parity = verify(staged)
        annotate(staged, parity)
        final_parity = verify(staged)
        if final_parity != parity:
            raise RuntimeError(
                "ONNX parity evidence changed after metadata annotation"
            )
        os.replace(staged, destination)
        return final_parity


def run_export(task_id: str, cfg: ExportConfig) -> ExportResult:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    resume_path: Path
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    elif cfg.checkpoint is not None:
        # Select a specific checkpoint iteration, from wandb or local.
        checkpoint_filename = f"model_{cfg.checkpoint}.pt"
        if cfg.wandb_run_path is not None:
            import wandb

            api = wandb.Api()
            wandb_run = api.run(str(cfg.wandb_run_path))
            run_id = cfg.wandb_run_path.split("/")[-1]
            download_dir = log_root_path / "wandb_checkpoints" / run_id
            resume_path = download_dir / checkpoint_filename
            if resume_path.exists():
                print(
                    f"[INFO]: Loading checkpoint: {checkpoint_filename} (run: {run_id}, cached)"
                )
            else:
                available = [f.name for f in wandb_run.files() if "model" in f.name]
                if checkpoint_filename not in available:
                    raise FileNotFoundError(
                        f"Checkpoint '{checkpoint_filename}' not found in wandb run. "
                        f"Available: {sorted(available)}"
                    )
                wandb_run.file(checkpoint_filename).download(
                    str(download_dir), replace=True
                )
                print(
                    f"[INFO]: Loading checkpoint: {checkpoint_filename} (run: {run_id}, downloaded)"
                )
        else:
            resume_path = get_checkpoint_path(
                log_root_path, checkpoint=re.escape(checkpoint_filename)
            )
            print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
        if cfg.wandb_run_path is None:
            raise ValueError(
                "`wandb_run_path` is required when `checkpoint_file` is not provided."
            )
        resume_path, was_cached = get_wandb_checkpoint_path(
            log_root_path, Path(cfg.wandb_run_path)
        )
        # Extract run_id and checkpoint name from path for display.
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
            f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
        )
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    train_cfg = asdict(agent_cfg)
    strip_training_only_config(train_cfg)
    runner = runner_cls(env, train_cfg, device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)
    current_observations = runner.env.get_observations()
    observation_groups = tuple(
        (name, int(current_observations[name].shape[-1])) for name in policy.obs_groups
    )

    # mjlab 1.3.0: ONNX export + metadata moved to mjlab.rl.exporter_utils and
    # the runner's built-in export_policy_to_onnx. Observation normalization is
    # baked into the exported graph automatically — EmpiricalNormalization is a
    # submodule of the policy's MLPModel (obs_normalization=True in RslRlModelCfg),
    # so export_policy_to_onnx emits actor(normalizer(obs)). No manual normalizer
    # handling needed (the old export_velocity_policy_as_onnx path is gone).
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    try:
        onnx_path = Path(os.path.abspath(cfg.onnx_file))
        metadata = get_base_metadata(
            runner.env.unwrapped, run_path=cfg.checkpoint_file
        )
        metadata.update(
            {
                "task_id": task_id,
                "source_file": resume_path.name,
                "checkpoint_sha256": _sha256(resume_path),
            }
        )

        def export_staged(staged: Path) -> None:
            runner.export_policy_to_onnx(str(staged.parent), staged.name)

        def verify_staged(staged: Path) -> ExportParityResult:
            return verify_onnx_export_parity(
                policy,
                staged,
                observation_groups,
            )

        def annotate_staged(
            staged: Path, result: ExportParityResult
        ) -> None:
            metadata.update(result.metadata())
            attach_metadata_to_onnx(str(staged), metadata)

        parity = _atomic_export_onnx(
            onnx_path,
            export_staged,
            verify_staged,
            annotate_staged,
        )
    except Exception:
        env.close()
        raise

    print(
        f"[INFO]: ONNX parity passed ({parity.samples} samples, "
        f"max abs {parity.max_abs_error:.3g}, max rel {parity.max_rel_error:.3g})"
    )
    print(f"Written {onnx_path}")

    env.close()
    return ExportResult(
        onnx_path=onnx_path,
        checkpoint_path=resume_path,
        wandb_run_path=cfg.wandb_run_path,
        checkpoint_iteration=_iteration_of(resume_path),
        parity=parity,
    )


def main():
    # Parse first argument to choose the task.
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    args = tyro.cli(
        ExportConfig,
        args=remaining_args,
        default=ExportConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )
    del remaining_args

    run_export(chosen_task, args)


if __name__ == "__main__":
    main()

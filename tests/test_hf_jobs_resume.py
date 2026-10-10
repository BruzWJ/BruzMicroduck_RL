"""HF Jobs resume staging must bridge the dataset mount to mjlab's loader."""

from __future__ import annotations

import importlib.util
import shlex
from dataclasses import fields
from pathlib import Path
from types import ModuleType

import pytest

from mjlab_microduck import hf_jobs
from mjlab_microduck.export import ExportConfig


def test_resume_args_are_injected_and_seed_is_normalized():
    checkpoint, args = hf_jobs._prepare_resume_training_args(
        "checkpoints//walk.pt",
        ["--env.scene.num-envs", "4096"],
    )

    assert checkpoint == "checkpoints/walk.pt"
    assert args[-6:] == [
        "--agent.resume",
        "True",
        "--agent.load-run",
        "^hf_seed$",
        "--agent.load-checkpoint",
        r"^model\.pt$",
    ]


@pytest.mark.parametrize(
    "checkpoint",
    ["", ".", "/src/checkpoints/walk.pt", "../walk.pt", "checkpoints/../walk.pt"],
)
def test_resume_checkpoint_must_be_safe_and_relative(checkpoint):
    with pytest.raises(ValueError, match="relative file path below /src"):
        hf_jobs._safe_dataset_file(checkpoint, "--hf-resume-checkpoint")


@pytest.mark.parametrize(
    "checkpoint",
    ["", ".", "/src/checkpoints/stand.pt", "../stand.pt", "checkpoints/../stand.pt"],
)
@pytest.mark.parametrize(
    "option", ["--hf-bc-expert-checkpoint", "--hf-bc-anchor-checkpoint"]
)
def test_bc_checkpoint_must_be_safe_and_relative(checkpoint, option):
    with pytest.raises(ValueError, match="relative file path below /src"):
        hf_jobs._safe_dataset_file(checkpoint, option)


def test_bc_checkpoint_is_normalized():
    assert (
        hf_jobs._safe_dataset_file(
            "checkpoints//official_actor/stand.pt", "--hf-bc-expert-checkpoint"
        )
        == "checkpoints/official_actor/stand.pt"
    )


def test_checkpoint_path_rejects_a_run_name_without_a_safe_component():
    with pytest.raises(ValueError, match="letter or number"):
        hf_jobs._safe_run_path_component(" / ")


def test_long_checkpoint_run_names_keep_a_collision_resistant_suffix():
    first = hf_jobs._safe_run_path_component("x" * 200 + "-one")
    second = hf_jobs._safe_run_path_component("x" * 200 + "-two")

    assert len(first) == 96
    assert first != second


@pytest.mark.parametrize("option", hf_jobs._RESUME_CONFLICT_OPTIONS)
@pytest.mark.parametrize("joined", [False, True])
def test_resume_checkpoint_rejects_conflicting_load_options(option, joined):
    train_args = [f"{option}=old"] if joined else [option, "old"]
    with pytest.raises(ValueError, match="conflicting option"):
        hf_jobs._prepare_resume_training_args("checkpoints/walk.pt", train_args)


@pytest.mark.parametrize(
    "experiment_args",
    [
        ["--agent.experiment-name", "xc330_walk"],
        ["--agent.experiment-name=xc330_walk"],
    ],
)
def test_stage_honors_explicit_experiment_and_stays_out_of_output_globs(
    tmp_path: Path, experiment_args: list[str]
):
    dataset = tmp_path / "src"
    source = dataset / "checkpoints" / "walk.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")
    work = tmp_path / "work"
    train_argv = [
        "Mjlab-Velocity-Flat-MicroDuck",
        *experiment_args,
        *hf_jobs._RESUME_TRAIN_ARGS,
    ]

    destination = hf_jobs._stage_resume_checkpoint(
        checkpoint="checkpoints/walk.pt",
        train_args=shlex.join(train_argv),
        dataset_root=dataset,
        work_root=work,
    )

    assert destination == work / "logs/rsl_rl/xc330_walk/hf_seed/model.pt"
    assert destination.read_bytes() == b"checkpoint"
    # uploader.py and hf_jobs.py's final export both use model_*.pt.
    assert list((work / "logs/rsl_rl").glob("**/model_*.pt")) == []


def test_stage_uses_registered_task_experiment_by_default(tmp_path: Path):
    dataset = tmp_path / "src"
    source = dataset / "checkpoints" / "walk.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")

    destination = hf_jobs._stage_resume_checkpoint(
        checkpoint="checkpoints/walk.pt",
        train_args=shlex.join(
            ["Mjlab-Velocity-Flat-MicroDuck", *hf_jobs._RESUME_TRAIN_ARGS]
        ),
        dataset_root=dataset,
        work_root=tmp_path / "work",
    )

    assert destination.parent.parent.name == "velocity"
    assert destination.name == "model.pt"


def test_stage_rejects_a_dataset_symlink_escape(tmp_path: Path):
    dataset = tmp_path / "src"
    source_dir = dataset / "checkpoints"
    source_dir.mkdir(parents=True)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"checkpoint")
    (source_dir / "escape.pt").symlink_to(outside)

    with pytest.raises(ValueError, match="resolves outside"):
        hf_jobs._stage_resume_checkpoint(
            checkpoint="checkpoints/escape.pt",
            train_args="Task --agent.experiment-name exp",
            dataset_root=dataset,
            work_root=tmp_path / "work",
        )


@pytest.mark.parametrize(
    "experiment", ["", ".", "/tmp/escape", "../escape", "x/../../escape"]
)
def test_stage_rejects_an_experiment_path_escape(tmp_path: Path, experiment: str):
    dataset = tmp_path / "src"
    source = dataset / "checkpoints" / "walk.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="experiment-name must stay below"):
        hf_jobs._stage_resume_checkpoint(
            checkpoint="checkpoints/walk.pt",
            train_args=shlex.join(["Task", "--agent.experiment-name", experiment]),
            dataset_root=dataset,
            work_root=tmp_path / "work",
        )


def test_bootstrap_stages_before_starting_uploader():
    stage = hf_jobs.BOOTSTRAP.index("_stage_resume_checkpoint")
    uploader = hf_jobs.BOOTSTRAP.index("launching checkpoint uploader")
    assert stage < uploader


def test_bootstrap_uses_decoded_argv_for_train():
    assert 'uv run train "${TRAIN_ARGV[@]}"' in hf_jobs.BOOTSTRAP
    assert "uv run train $TRAIN_ARGS" not in hf_jobs.BOOTSTRAP
    assert "AUTO_EXPORT" not in hf_jobs.BOOTSTRAP


def test_bootstrap_reaps_watcher_and_requires_final_checkpoint_upload():
    bootstrap = hf_jobs.BOOTSTRAP
    assert "wait $UPLOADER_PID" in bootstrap
    assert "for attempt in 1 2 3" in bootstrap
    assert "final checkpoint upload failed permanently" in bootstrap
    assert '[ "$TRAIN_RC" -eq 0 ] && TRAIN_RC=1' in bootstrap
    assert (
        "CKPT_ONE_SHOT=1 uv run python scripts/hf/uploader.py || true" not in bootstrap
    )


def test_hf_training_has_no_model_repository_fallback():
    submitter = Path(hf_jobs.__file__).read_text()
    uploader = (
        Path(__file__).parents[1] / "scripts" / "hf" / "uploader.py"
    ).read_text()

    for source in (submitter, uploader):
        assert "CKPT_REPO_TYPE" not in source
        assert 'repo_type="model"' not in source
    assert "create_repo" not in uploader


class _FakeHubApi:
    def __init__(self, commit_results: list[Exception | None]):
        self.commit_results = commit_results
        self.commit_calls = 0
        self.commits: list[dict] = []

    def create_commit(self, *args, **kwargs):
        assert not args
        self.commits.append(kwargs)
        result = self.commit_results[self.commit_calls]
        self.commit_calls += 1
        if result is not None:
            raise result


def _load_uploader() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "hf" / "uploader.py"
    spec = importlib.util.spec_from_file_location("test_hf_uploader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_uploader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    api: _FakeHubApi,
    uploader: ModuleType,
) -> Path:
    root = tmp_path / "logs" / "rsl_rl"
    checkpoint = root / "velocity" / "run" / "model_100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(uploader, "HfApi", lambda: api)
    monkeypatch.setenv("CKPT_REPO", "owner/checkpoints")
    monkeypatch.setenv("CKPT_ROOT", str(root))
    monkeypatch.delenv("CKPT_ONE_SHOT", raising=False)
    monkeypatch.delenv("CKPT_POLL_INTERVAL", raising=False)
    monkeypatch.setenv("CKPT_PATH_PREFIX", "training/runs/walk/launch")
    return checkpoint


def test_one_shot_uploader_propagates_commit_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    uploader = _load_uploader()
    api = _FakeHubApi([RuntimeError("commit failed")])
    _configure_uploader(monkeypatch, tmp_path, api, uploader)
    monkeypatch.setenv("CKPT_ONE_SHOT", "1")

    assert uploader.main() == 1
    assert api.commit_calls == 1


def test_uploader_retries_a_file_after_failed_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    uploader = _load_uploader()
    api = _FakeHubApi([RuntimeError("transient"), None])
    _configure_uploader(monkeypatch, tmp_path, api, uploader)

    class StopUploader(Exception):
        pass

    sleeps = 0

    def stop_after_retry(interval: float):
        nonlocal sleeps
        assert interval == 60.0
        sleeps += 1
        if sleeps == 2:
            raise StopUploader

    monkeypatch.setattr(uploader.time, "sleep", stop_after_retry)

    with pytest.raises(StopUploader):
        uploader.main()

    assert api.commit_calls == 2


def test_one_shot_uploader_rejects_an_empty_checkpoint_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    uploader = _load_uploader()
    api = _FakeHubApi([])
    root = tmp_path / "logs" / "rsl_rl"
    root.mkdir(parents=True)
    monkeypatch.setattr(uploader, "HfApi", lambda: api)
    monkeypatch.setenv("CKPT_REPO", "owner/checkpoints")
    monkeypatch.setenv("CKPT_ROOT", str(root))
    monkeypatch.setenv("CKPT_ONE_SHOT", "1")
    monkeypatch.setenv("CKPT_PATH_PREFIX", "training/runs/empty")

    assert uploader.main() == 1
    assert api.commit_calls == 0


def test_uploader_namespaces_a_run_in_the_checkpoint_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    uploader = _load_uploader()
    api = _FakeHubApi([None])
    _configure_uploader(monkeypatch, tmp_path, api, uploader)
    monkeypatch.setenv("CKPT_ONE_SHOT", "1")
    monkeypatch.setenv("CKPT_PATH_PREFIX", "runs/walk")

    assert uploader.main() == 0
    assert len(api.commits) == 1
    commit = api.commits[0]
    assert commit["repo_type"] == "dataset"
    assert [operation.path_in_repo for operation in commit["operations"]] == [
        "runs/walk/velocity/run/model_100.pt"
    ]


def test_uploader_requires_a_path_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    uploader = _load_uploader()
    api = _FakeHubApi([])
    _configure_uploader(monkeypatch, tmp_path, api, uploader)
    monkeypatch.delenv("CKPT_PATH_PREFIX")
    monkeypatch.setenv("CKPT_ONE_SHOT", "1")

    assert uploader.main() == 1
    assert api.commit_calls == 0


def test_export_config_has_no_dummy_policy_mode():
    names = {field.name for field in fields(ExportConfig)}

    assert names.isdisjoint(
        {
            "agent",
            "registry_name",
            "_demo_mode",
            "motion_file",
            "video",
            "video_length",
            "video_height",
            "video_width",
            "camera",
            "viewer",
        }
    )

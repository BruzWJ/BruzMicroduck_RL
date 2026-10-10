"""Build and publish BruzMicroduck's one exact ten-policy model repository."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, RepoFile, hf_hub_download

from mjlab_microduck.publish import manifest as policy_manifest

TARGET_REPO = "BWJ2310/bruzmicroduck-xc330-custom-weight-policies"
REPOSITORY_LICENSE = Path(__file__).resolve().parents[3] / "LICENSE"
TAG_RE = re.compile(r"v[1-9][0-9]*")

POLICY_TASKS: dict[str, str] = {
    "walk.onnx": "Mjlab-Velocity-Flat-MicroDuck",
    "walk_stand.onnx": "Mjlab-VelStand-Rough-Backlash-MicroDuck",
    "stand.onnx": "Mjlab-StandUp-Flat-MicroDuck",
    "roller.onnx": "Mjlab-Velocity-Flat-MicroDuck-Rollers",
    "sit_stand.onnx": "Mjlab-SitStand-Flat-MicroDuck",
    "ground_pick.onnx": "Mjlab-GroundPick-Flat-MicroDuck",
    "roller_crouch.onnx": "Mjlab-RollerCrouch-Flat-MicroDuck",
    "forward_roll.onnx": "Mjlab-Roulade-Flat-MicroDuck",
    "kick_left.onnx": "Mjlab-BallKickLeft-Flat-MicroDuck",
    "kick_right.onnx": "Mjlab-BallKick-Flat-MicroDuck",
}
POLICY_FILES = tuple(POLICY_TASKS)

GROUND_PICK_PERIOD_S = 4.0
GROUND_PICK_END_PHASE = 1.0
GROUND_PICK_DURATION_S = GROUND_PICK_PERIOD_S * GROUND_PICK_END_PHASE
ROULADE_DURATION_S = 1.5


def _policy_entries() -> list[dict[str, Any]]:
    """The runtime contracts for the ten files, with no historical aliases."""
    return [
        {"file": "walk.onnx", "name": "walk", "kind": "perpetual"},
        {
            "file": "walk_stand.onnx",
            "name": "walk_stand",
            "kind": "perpetual",
            "slot": "walk",
            "entry_pose": "standing",
            "description": "Walks when commanded and stands at zero command.",
        },
        {"file": "stand.onnx", "name": "stand", "kind": "perpetual"},
        {
            "file": "roller.onnx",
            "name": "roller",
            "kind": "perpetual",
            "mode": "roller",
            "action_scale": 0.8,
        },
        {
            "file": "sit_stand.onnx",
            "name": "sitstand",
            "kind": "scripted",
            "command": {
                "encoding": "posture_flag",
                "slot": "twist.vx",
                "sit": 1.0,
                "stand": 0.0,
                "idle": [0.0, 0.0, 0.0],
            },
            "ramp_s": 2.0,
            "unwind_s": 1.0,
        },
        {
            "file": "ground_pick.onnx",
            "name": "ground_pick",
            "kind": "episodic",
            "duration_s": GROUND_PICK_DURATION_S,
            "command": {
                "encoding": "phase",
                "slots": "twist.vx,twist.vy",
                "period_s": GROUND_PICK_PERIOD_S,
                "end_phase": GROUND_PICK_END_PHASE,
            },
        },
        {
            "file": "roller_crouch.onnx",
            "name": "crouch",
            "kind": "episodic",
            "duration_s": 3.5,
            "mode": "roller",
            "action_scale": 0.8,
            "command": {
                "encoding": "phase",
                "slots": "twist.vx,twist.vy",
                "period_s": 5.0,
                "end_phase": 0.7,
            },
        },
        {
            "file": "forward_roll.onnx",
            "name": "roulade",
            "kind": "episodic",
            "duration_s": ROULADE_DURATION_S,
            "chain": True,
            "action_scale": 1.0,
        },
        {
            "file": "kick_left.onnx",
            "name": "kick_left",
            "kind": "episodic",
            "duration_s": 0.5,
            "action_scale": 1.0,
        },
        {
            "file": "kick_right.onnx",
            "name": "kick_right",
            "kind": "episodic",
            "duration_s": 0.5,
            "action_scale": 1.0,
        },
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sources(path: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON value {value}")

    raw = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(raw, dict) or set(raw) != set(POLICY_FILES):
        raise policy_manifest.ManifestError(
            f"sources must contain exactly {list(POLICY_FILES)!r}"
        )

    sources: dict[str, tuple[Path, dict[str, Any]]] = {}
    for filename, task_id in POLICY_TASKS.items():
        record = raw[filename]
        if not isinstance(record, dict) or not isinstance(record.get("training"), dict):
            raise policy_manifest.ManifestError(
                f"{filename}: expected an object with onnx and training"
            )
        onnx_value = record.get("onnx")
        if not isinstance(onnx_value, str) or not onnx_value:
            raise policy_manifest.ManifestError(
                f"{filename}: onnx must be a nonempty path string"
            )
        source = Path(onnx_value)
        if not source.is_absolute():
            source = path.parent / source
        if source.is_symlink() or not source.is_file():
            raise policy_manifest.ManifestError(
                f"{filename}: ONNX source must be a regular file: {source}"
            )
        training = copy.deepcopy(record["training"])
        if training.get("task_id") != task_id:
            raise policy_manifest.ManifestError(
                f"{filename}: task_id must be {task_id!r}"
            )
        actual_hash = _sha256(source)
        if training.get("onnx_sha256") != actual_hash:
            raise policy_manifest.ManifestError(
                f"{filename}: onnx_sha256 does not match {source}"
            )
        training.pop("checkpoint_path", None)
        if isinstance(training.get("evaluation"), dict):
            training["evaluation"].pop("report_files", None)
            training["evaluation"].pop("evidence_summary_file", None)
        sources[filename] = (source, training)
    return sources


def build_manifest(
    sources: Mapping[str, tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    entries = _policy_entries()
    for entry in entries:
        entry["training"] = copy.deepcopy(sources[entry["file"]][1])
    manifest = {
        "schema_version": policy_manifest.SCHEMA_VERSION,
        "model_api": policy_manifest.MODEL_API,
        "obs_len": policy_manifest.OBS_LEN,
        "action_len": policy_manifest.ACTION_LEN,
        "robot": dict(policy_manifest.ROBOT),
        "description": (
            "BruzMicroduck policies fine-tuned together for XC330-T181 servos "
            "and the Bruz mass distribution."
        ),
        "policies": entries,
    }
    policy_manifest.validate_manifest(manifest)
    policy_manifest.dump_manifest(manifest)
    return manifest


def render_readme(manifest: Mapping[str, Any]) -> str:
    rows = "\n".join(
        f"| `{entry['file']}` | `{entry['name']}` | {entry['kind']} |"
        for entry in manifest["policies"]
    )
    return f"""---
license: apache-2.0
library_name: microduck
pipeline_tag: robotics
tags:
- microduck
- robotics
- reinforcement-learning
- onnx
---

# BruzMicroduck XC330-T181 policies

One complete policy set for BruzMicroduck's XC330-T181 servos, 250 g battery,
52 g power support, and 7 V servo bus. The policies use 61 observations, 14
actions, and a 50 Hz control loop; observation normalization is baked into each
ONNX graph.

| File | Robot command | Kind |
| --- | --- | --- |
{rows}

Install an immutable numeric tag through BruzMicroduck's policy-set update
flow. The manifest installs all ten files together; do not load this repository
as a single-policy community model.
"""


def assemble_bundle(source_map: Path, output_dir: Path) -> dict[str, Any]:
    """Validate all ten exports, then create a new self-contained directory."""
    sources = _load_sources(source_map.resolve())
    for filename in POLICY_FILES:
        source = sources[filename][0]
        policy_manifest.check_onnx(source)
        policy_manifest.smoke_run_onnx(source)
    manifest = build_manifest(sources)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        for filename in POLICY_FILES:
            shutil.copyfile(sources[filename][0], output_dir / filename)
        (output_dir / "manifest.json").write_text(
            policy_manifest.dump_manifest(manifest)
        )
        (output_dir / "README.md").write_text(render_readme(manifest))
        shutil.copyfile(REPOSITORY_LICENSE, output_dir / "LICENSE")
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return manifest


def _bundle_hashes(directory: Path) -> dict[str, str]:
    expected = {*POLICY_FILES, "manifest.json", "README.md", "LICENSE"}
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != expected or any(path.is_dir() for path in directory.iterdir()):
        raise policy_manifest.ManifestError(
            f"bundle inventory is {sorted(actual)!r}, expected {sorted(expected)!r}"
        )
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON value {value}")

    manifest = json.loads(
        (directory / "manifest.json").read_text(), parse_constant=reject_constant
    )
    policy_manifest.validate_manifest(manifest)
    if tuple(entry.get("file") for entry in manifest["policies"]) != POLICY_FILES:
        raise policy_manifest.ManifestError("manifest is not the exact ten-policy set")
    for filename in POLICY_FILES:
        policy_manifest.check_onnx(directory / filename)
        policy_manifest.smoke_run_onnx(directory / filename)
    return {name: _sha256(directory / name) for name in expected}


def publish_bundle(
    directory: Path,
    tag: str,
    *,
    api: HfApi | None = None,
    download: Callable[..., str] = hf_hub_download,
) -> str:
    """Upload, verify, tag, and only then expose the one canonical repository."""
    if TAG_RE.fullmatch(tag) is None:
        raise policy_manifest.ManifestError("tag must be numeric, for example v1")
    directory = directory.resolve()
    hashes = _bundle_hashes(directory)
    api = api or HfApi()

    existed = api.repo_exists(TARGET_REPO, repo_type="model")
    parent_commit: str | None = None
    if existed:
        refs = api.list_repo_refs(TARGET_REPO, repo_type="model")
        if tag in {ref.name for ref in refs.tags}:
            raise policy_manifest.ManifestError(
                f"immutable tag {tag!r} already exists in {TARGET_REPO}"
            )
        remote_files = {
            item.path
            for item in api.list_repo_tree(
                TARGET_REPO, repo_type="model", recursive=True
            )
            if isinstance(item, RepoFile)
        }
        unexpected = remote_files - set(hashes) - {".gitattributes"}
        if unexpected:
            raise policy_manifest.ManifestError(
                f"refusing repository with unexpected files: {sorted(unexpected)!r}"
            )
        parent_commit = api.repo_info(TARGET_REPO, repo_type="model").sha
        api.update_repo_settings(TARGET_REPO, repo_type="model", private=True)
    else:
        api.create_repo(
            TARGET_REPO,
            repo_type="model",
            private=True,
            exist_ok=False,
        )

    commit = api.upload_folder(
        repo_id=TARGET_REPO,
        repo_type="model",
        folder_path=directory,
        commit_message=f"Publish BruzMicroduck policy set {tag}",
        parent_commit=parent_commit,
    )
    revision = commit.oid

    uploaded_files = {
        item.path
        for item in api.list_repo_tree(
            TARGET_REPO,
            repo_type="model",
            revision=revision,
            recursive=True,
        )
        if isinstance(item, RepoFile)
    }
    if uploaded_files - {".gitattributes"} != set(hashes):
        raise policy_manifest.ManifestError(
            "uploaded revision does not contain the exact bundle inventory"
        )

    with tempfile.TemporaryDirectory() as temporary:
        for filename, expected_hash in sorted(hashes.items()):
            downloaded = Path(
                download(
                    TARGET_REPO,
                    filename,
                    repo_type="model",
                    revision=revision,
                    local_dir=temporary,
                    force_download=True,
                )
            )
            if _sha256(downloaded) != expected_hash:
                raise policy_manifest.ManifestError(
                    f"uploaded {filename} failed SHA-256 verification"
                )

    api.create_tag(
        TARGET_REPO,
        repo_type="model",
        tag=tag,
        revision=revision,
        exist_ok=False,
    )
    api.update_repo_settings(TARGET_REPO, repo_type="model", private=False)
    return revision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble or publish the exact ten BruzMicroduck policies."
    )
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--publish",
        action="store_true",
        help=f"publish only to {TARGET_REPO} after local validation",
    )
    parser.add_argument("--tag", help="immutable numeric tag, for example v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if bool(args.tag) != args.publish:
        parser.error("--publish and --tag must be supplied together")
    try:
        assemble_bundle(args.sources, args.output_dir)
        if args.publish:
            revision = publish_bundle(args.output_dir, args.tag)
            print(f"published {TARGET_REPO}@{args.tag} ({revision})")
        else:
            print(f"wrote {args.output_dir} ({len(POLICY_FILES)} policies)")
    except (OSError, ValueError, policy_manifest.ManifestError) as exc:
        parser.error(str(exc))
    return 0

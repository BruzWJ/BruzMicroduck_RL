"""The publisher emits one complete, immutable XC330 policy set."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from huggingface_hub import RepoFile
from onnx import TensorProto, helper, numpy_helper

from mjlab_microduck.publish import bundle
from mjlab_microduck.publish import manifest as policy_manifest


def _tiny_policy(path: Path, seed: int) -> Path:
    weight = numpy_helper.from_array(
        np.random.default_rng(seed)
        .normal(0, 0.1, (policy_manifest.OBS_LEN, policy_manifest.ACTION_LEN))
        .astype(np.float32),
        "weight",
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "weight"], ["actions"])],
        "policy",
        [
            helper.make_tensor_value_info(
                "obs", TensorProto.FLOAT, [1, policy_manifest.OBS_LEN]
            )
        ],
        [
            helper.make_tensor_value_info(
                "actions", TensorProto.FLOAT, [1, policy_manifest.ACTION_LEN]
            )
        ],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_map(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_files: dict[str, Path] = {}
    source_map = {}
    for index, (filename, task_id) in enumerate(bundle.POLICY_TASKS.items()):
        source = _tiny_policy(tmp_path / f"source-{index}.onnx", index)
        source_files[filename] = source
        source_map[filename] = {
            "onnx": source.name,
            "training": {
                "task_id": task_id,
                "onnx_sha256": _sha256(source),
                "checkpoint": 1000 + index,
                "checkpoint_path": f"/private/{filename}.pt",
                "evaluation": {
                    "reward_mean": float(index),
                    "report_files": {"seed_0": f"/private/{filename}.json"},
                    "evidence_summary_file": "/private/summary.json",
                },
            },
        }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(source_map, indent=2) + "\n")
    return path, source_files


def _assemble(tmp_path: Path) -> tuple[Path, dict, dict[str, Path]]:
    source_map, sources = _write_source_map(tmp_path)
    output = tmp_path / "bundle"
    manifest = bundle.assemble_bundle(source_map, output)
    return output, manifest, sources


def test_assemble_emits_the_exact_ten_policy_set(tmp_path: Path) -> None:
    output, manifest, sources = _assemble(tmp_path)

    assert bundle.POLICY_FILES == (
        "walk.onnx",
        "walk_stand.onnx",
        "stand.onnx",
        "roller.onnx",
        "sit_stand.onnx",
        "ground_pick.onnx",
        "roller_crouch.onnx",
        "forward_roll.onnx",
        "kick_left.onnx",
        "kick_right.onnx",
    )
    assert {path.name for path in output.iterdir()} == {
        *bundle.POLICY_FILES,
        "manifest.json",
        "README.md",
        "LICENSE",
    }
    assert manifest == json.loads((output / "manifest.json").read_text())
    assert tuple(entry["file"] for entry in manifest["policies"]) == (
        bundle.POLICY_FILES
    )
    assert manifest["robot"]["servos"] == "xc330-t181"

    by_file = {entry["file"]: entry for entry in manifest["policies"]}
    assert by_file["ground_pick.onnx"]["duration_s"] == 4.0
    assert by_file["ground_pick.onnx"]["command"] == {
        "encoding": "phase",
        "slots": "twist.vx,twist.vy",
        "period_s": 4.0,
        "end_phase": 1.0,
    }
    assert by_file["forward_roll.onnx"]["name"] == "roulade"
    assert by_file["forward_roll.onnx"]["duration_s"] == 1.5
    assert by_file["kick_left.onnx"]["action_scale"] == 1.0
    assert by_file["kick_right.onnx"]["action_scale"] == 1.0
    for filename, source in sources.items():
        assert (output / filename).read_bytes() == source.read_bytes()
        training = by_file[filename]["training"]
        assert training["task_id"] == bundle.POLICY_TASKS[filename]
        assert "checkpoint_path" not in training
        assert "report_files" not in training["evaluation"]
        assert "evidence_summary_file" not in training["evaluation"]

    public_text = (output / "README.md").read_text() + json.dumps(manifest)
    for discarded_label in (
        "alpha_walking",
        "beta",
        "not-working",
        "clean_actor",
    ):
        assert discarded_label not in public_text


def test_source_map_requires_exact_files_task_and_digest(tmp_path: Path) -> None:
    path, _ = _write_source_map(tmp_path)
    source_map = json.loads(path.read_text())
    source_map.pop("roller.onnx")
    path.write_text(json.dumps(source_map))
    with pytest.raises(policy_manifest.ManifestError, match="exactly"):
        bundle.assemble_bundle(path, tmp_path / "missing")

    path, _ = _write_source_map(tmp_path / "second")
    source_map = json.loads(path.read_text())
    source_map["walk.onnx"]["training"]["task_id"] = "wrong-task"
    path.write_text(json.dumps(source_map))
    with pytest.raises(policy_manifest.ManifestError, match="task_id must be"):
        bundle.assemble_bundle(path, tmp_path / "wrong-task")

    source_map["walk.onnx"]["training"]["task_id"] = bundle.POLICY_TASKS[
        "walk.onnx"
    ]
    source_map["walk.onnx"]["training"]["onnx_sha256"] = "0" * 64
    path.write_text(json.dumps(source_map))
    with pytest.raises(policy_manifest.ManifestError, match="onnx_sha256"):
        bundle.assemble_bundle(path, tmp_path / "wrong-digest")


def test_assembly_never_overwrites_an_existing_destination(tmp_path: Path) -> None:
    source_map, _ = _write_source_map(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("mine")

    with pytest.raises(FileExistsError):
        bundle.assemble_bundle(source_map, output)

    assert marker.read_text() == "mine"


class _FakeApi:
    def __init__(
        self,
        expected_files: set[str],
        *,
        existed: bool = False,
        existing_tag: str | None = None,
        unexpected_file: str | None = None,
    ) -> None:
        self.expected_files = expected_files
        self.existed = existed
        self.existing_tag = existing_tag
        self.unexpected_file = unexpected_file
        self.events: list[tuple] = []
        self.folder: Path | None = None

    @staticmethod
    def _files(names: set[str]) -> list[RepoFile]:
        return [RepoFile(path=name, size=1, oid="blob") for name in names]

    def repo_exists(self, repo_id: str, **kwargs) -> bool:
        self.events.append(("repo_exists", repo_id))
        return self.existed

    def list_repo_refs(self, repo_id: str, **kwargs):
        self.events.append(("list_refs", repo_id))
        tags = []
        if self.existing_tag is not None:
            tags.append(SimpleNamespace(name=self.existing_tag))
        return SimpleNamespace(tags=tags)

    def list_repo_tree(self, repo_id: str, *, revision=None, **kwargs):
        self.events.append(("list_tree", revision))
        if revision is not None:
            return self._files(self.expected_files | {".gitattributes"})
        names = set()
        if self.unexpected_file is not None:
            names.add(self.unexpected_file)
        return self._files(names)

    def repo_info(self, repo_id: str, **kwargs):
        self.events.append(("repo_info", repo_id))
        return SimpleNamespace(sha="parent-commit")

    def create_repo(self, repo_id: str, **kwargs) -> None:
        self.events.append(("create_repo", repo_id, kwargs["private"]))

    def update_repo_settings(self, repo_id: str, **kwargs) -> None:
        self.events.append(("visibility", kwargs["private"]))

    def upload_folder(self, *, folder_path: Path, parent_commit=None, **kwargs):
        self.folder = Path(folder_path)
        self.events.append(("upload", parent_commit))
        return SimpleNamespace(oid="uploaded-commit")

    def create_tag(self, repo_id: str, **kwargs) -> None:
        self.events.append(("tag", kwargs["tag"], kwargs["revision"]))


def _download_from_fake(api: _FakeApi, *, corrupt: str | None = None):
    def download(repo_id: str, filename: str, *, local_dir: str, **kwargs) -> str:
        api.events.append(("verify", filename, kwargs["revision"]))
        destination = Path(local_dir) / filename
        assert api.folder is not None
        shutil.copyfile(api.folder / filename, destination)
        if filename == corrupt:
            destination.write_bytes(b"corrupt")
        return str(destination)

    return download


def test_publish_verifies_before_immutable_tag_and_public_visibility(
    tmp_path: Path,
) -> None:
    output, _, _ = _assemble(tmp_path)
    expected = {*bundle.POLICY_FILES, "manifest.json", "README.md", "LICENSE"}
    api = _FakeApi(expected)

    revision = bundle.publish_bundle(
        output,
        "v1",
        api=api,
        download=_download_from_fake(api),
    )

    assert revision == "uploaded-commit"
    assert ("create_repo", bundle.TARGET_REPO, True) in api.events
    tag_index = api.events.index(("tag", "v1", "uploaded-commit"))
    assert sum(event[0] == "verify" for event in api.events[:tag_index]) == len(expected)
    assert api.events[-1] == ("visibility", False)


def test_publish_refuses_existing_tag_or_unexpected_remote_file(
    tmp_path: Path,
) -> None:
    output, _, _ = _assemble(tmp_path)
    expected = {*bundle.POLICY_FILES, "manifest.json", "README.md", "LICENSE"}

    tagged = _FakeApi(expected, existed=True, existing_tag="v2")
    with pytest.raises(policy_manifest.ManifestError, match="already exists"):
        bundle.publish_bundle(output, "v2", api=tagged)
    assert not any(event[0] in {"visibility", "upload"} for event in tagged.events)

    dirty = _FakeApi(expected, existed=True, unexpected_file="old-beta.onnx")
    with pytest.raises(policy_manifest.ManifestError, match="unexpected files"):
        bundle.publish_bundle(output, "v3", api=dirty)
    assert not any(event[0] in {"visibility", "upload"} for event in dirty.events)


def test_publish_never_tags_or_exposes_a_corrupt_upload(tmp_path: Path) -> None:
    output, _, _ = _assemble(tmp_path)
    expected = {*bundle.POLICY_FILES, "manifest.json", "README.md", "LICENSE"}
    api = _FakeApi(expected)

    with pytest.raises(policy_manifest.ManifestError, match="SHA-256"):
        bundle.publish_bundle(
            output,
            "v4",
            api=api,
            download=_download_from_fake(api, corrupt="walk.onnx"),
        )

    assert not any(event[0] == "tag" for event in api.events)
    assert ("visibility", False) not in api.events


def test_publish_requires_a_numeric_immutable_tag(tmp_path: Path) -> None:
    with pytest.raises(policy_manifest.ManifestError, match="numeric"):
        bundle.publish_bundle(tmp_path, "latest")

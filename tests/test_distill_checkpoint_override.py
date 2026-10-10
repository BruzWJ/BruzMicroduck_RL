"""Behavior-cloning teachers can come from the mounted HF source dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mjlab_microduck.tasks import distill


@pytest.mark.parametrize(
    ("prefix", "env_name"),
    [
        ("", "MICRODUCK_BC_EXPERT_CHECKPOINT"),
        ("anchor_", "MICRODUCK_BC_ANCHOR_CHECKPOINT"),
    ],
)
def test_environment_checkpoint_overrides_configured_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str, env_name: str
):
    checkpoint = tmp_path / f"{prefix or 'expert'}checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv(env_name, str(checkpoint))

    resolved = distill._resolve_checkpoint(
        {
            f"{prefix}checkpoint_path": tmp_path / "configured.pt",
            f"{prefix}wandb_run_path": "owner/run",
        },
        prefix,
    )

    assert resolved == checkpoint


def test_environment_checkpoint_must_exist(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MICRODUCK_BC_EXPERT_CHECKPOINT", "/missing/stand.pt")

    with pytest.raises(FileNotFoundError, match="MICRODUCK_BC_EXPERT_CHECKPOINT"):
        distill._resolve_checkpoint({}, "")


def test_unknown_checkpoint_prefix_is_rejected():
    with pytest.raises(ValueError, match="unsupported behavior-cloning"):
        distill._resolve_checkpoint({}, "other_")


def test_anchor_all_selects_every_frame_without_an_expert():
    obs = torch.zeros(3, 61)
    obs[:, 5] = torch.tensor([-1.0, 0.0, 1.0])
    cfg = {
        **distill.default_bc_cfg(),
        "anchor_all": True,
        "body_slice": None,
        "unanchor_turn_in_place": False,
    }

    expert, anchor, body, turn = distill.teacher_masks_from_obs(
        obs, cfg, has_expert=False, has_anchor=True
    )

    assert not expert.any()
    assert anchor.all()
    assert not body.any()
    assert not turn.any()


def test_default_teacher_masks_keep_fallen_and_upright_regions_separate():
    obs = torch.zeros(3, 61)
    obs[:, 5] = torch.tensor([-1.0, -0.9, 0.0])
    cfg = {
        **distill.default_bc_cfg(),
        "body_slice": None,
        "unanchor_turn_in_place": False,
    }

    expert, anchor, _, _ = distill.teacher_masks_from_obs(
        obs, cfg, has_expert=True, has_anchor=True
    )

    assert expert.tolist() == [False, False, True]
    assert anchor.tolist() == [True, True, False]

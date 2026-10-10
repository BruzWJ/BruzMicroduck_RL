from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from mjlab_microduck.export_parity import (
    DEFAULT_PARITY_ATOL,
    DEFAULT_PARITY_RTOL,
    ExportParityError,
    verify_onnx_export_parity,
)


def _successful_parity_result():
    from mjlab_microduck.export_parity import ExportParityResult

    return ExportParityResult(
        samples=128,
        seed=0,
        atol=1.0e-5,
        rtol=1.0e-5,
        max_abs_error=0.0,
        max_rel_error=0.0,
        max_allclose_ratio=0.0,
        observation_nonfinite_count=0,
        pytorch_action_nonfinite_count=0,
        onnx_action_nonfinite_count=0,
    )


class _Normalizer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mean = torch.linspace(-0.3, 0.3, 61).reshape(1, 61)
        std = torch.linspace(0.02, 1.2, 61).reshape(1, 61)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)
        self.eps = 1.0e-2

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self._mean) / (self._std + self.eps)


class _Policy(torch.nn.Module):
    obs_groups = ("proprio", "command")

    def __init__(self) -> None:
        super().__init__()
        self.obs_normalizer = _Normalizer()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(61, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 14),
        )
        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            for parameter in self.mlp.parameters():
                parameter.copy_(
                    torch.rand(parameter.shape, generator=generator) * 0.2 - 0.1
                )

    def forward(self, observations: TensorDict) -> torch.Tensor:
        flat = torch.cat([observations[name] for name in self.obs_groups], dim=-1)
        return self.mlp(self.obs_normalizer(flat))


class _FlatExport(torch.nn.Module):
    def __init__(
        self,
        policy: _Policy,
        *,
        include_normalizer: bool = True,
        output_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self.normalizer = policy.obs_normalizer
        self.mlp = policy.mlp
        self.include_normalizer = include_normalizer
        self.output_delta = output_delta

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.include_normalizer:
            values = self.normalizer(values)
        return self.mlp(values) + self.output_delta


def _export(
    path: Path,
    policy: _Policy,
    *,
    include_normalizer: bool = True,
    output_delta: float = 0.0,
) -> None:
    torch.onnx.export(
        _FlatExport(
            policy,
            include_normalizer=include_normalizer,
            output_delta=output_delta,
        ).eval(),
        (torch.zeros(1, 61),),
        path,
        opset_version=18,
        dynamo=False,
        input_names=["obs"],
        output_names=["actions"],
    )


def test_parity_matches_loaded_policy_with_baked_normalizer(tmp_path: Path) -> None:
    policy = _Policy().eval()
    onnx_path = tmp_path / "policy.onnx"
    _export(onnx_path, policy)

    result = verify_onnx_export_parity(
        policy,
        onnx_path,
        (("proprio", 48), ("command", 13)),
    )

    assert result.samples == 128
    assert result.seed == 0
    assert result.atol == DEFAULT_PARITY_ATOL
    assert result.rtol == DEFAULT_PARITY_RTOL
    assert result.max_abs_error <= 1.0e-5
    assert result.max_allclose_ratio <= 1.0
    assert result.metadata()["export_parity_method"] == (
        "loaded-pytorch-policy-vs-onnxruntime-cpu"
    )


def test_parity_accepts_the_exact_captured_onnx_bytes(tmp_path: Path) -> None:
    policy = _Policy().eval()
    onnx_path = tmp_path / "policy.onnx"
    _export(onnx_path, policy)

    captured = onnx_path.read_bytes()
    onnx_path.write_bytes(b"the live path changed after capture")
    result = verify_onnx_export_parity(
        policy,
        captured,
        (("proprio", 48), ("command", 13)),
    )

    assert result.max_allclose_ratio <= 1.0


def test_parity_rejects_export_without_observation_normalizer(tmp_path: Path) -> None:
    policy = _Policy().eval()
    onnx_path = tmp_path / "missing-normalizer.onnx"
    _export(onnx_path, policy, include_normalizer=False)

    with pytest.raises(
        ExportParityError, match="does not match the loaded PyTorch policy"
    ):
        verify_onnx_export_parity(
            policy,
            onnx_path,
            (("proprio", 48), ("command", 13)),
        )


def test_allclose_ratio_supports_unbounded_actor_outputs(tmp_path: Path) -> None:
    policy = _Policy().eval()
    with torch.no_grad():
        policy.mlp[-1].bias.fill_(8.0)
    onnx_path = tmp_path / "within-scaled-tolerance.onnx"
    _export(onnx_path, policy, output_delta=5.0e-5)

    result = verify_onnx_export_parity(
        policy,
        onnx_path,
        (("proprio", 48), ("command", 13)),
    )

    assert result.max_abs_error > result.atol + result.rtol
    assert 0.0 < result.max_allclose_ratio <= 1.0


def test_parity_rejects_wrong_observation_group_flattening(tmp_path: Path) -> None:
    policy = _Policy().eval()
    onnx_path = tmp_path / "policy.onnx"
    _export(onnx_path, policy)

    with pytest.raises(
        ExportParityError, match="does not match the loaded PyTorch policy"
    ):
        verify_onnx_export_parity(
            policy,
            onnx_path,
            (("command", 13), ("proprio", 48)),
        )


def test_parity_rejects_nonfinite_normalizer_before_inference(tmp_path: Path) -> None:
    policy = _Policy().eval()
    onnx_path = tmp_path / "policy.onnx"
    _export(onnx_path, policy)
    policy.obs_normalizer._mean[0, 7] = float("nan")

    with pytest.raises(ExportParityError, match="normalizer contains NaN"):
        verify_onnx_export_parity(
            policy,
            onnx_path,
            (("proprio", 48), ("command", 13)),
        )


@pytest.mark.parametrize("failure_stage", ("parity", "metadata"))
def test_atomic_export_preserves_prior_output_on_gate_failure(
    tmp_path: Path, failure_stage: str
) -> None:
    from mjlab_microduck.export import _atomic_export_onnx

    destination = tmp_path / "policy.onnx"
    destination.write_bytes(b"prior valid export")

    def export(staged: Path) -> None:
        staged.write_bytes(b"unverified replacement")

    def verify(staged: Path):
        assert staged.read_bytes() == b"unverified replacement"
        if failure_stage == "parity":
            raise RuntimeError("parity failed")
        return _successful_parity_result()

    def annotate(staged: Path, _result) -> None:
        staged.write_bytes(b"annotated replacement")
        if failure_stage == "metadata":
            raise RuntimeError("metadata failed")

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        _atomic_export_onnx(destination, export, verify, annotate)

    assert destination.read_bytes() == b"prior valid export"
    assert not list(tmp_path.glob(".policy.onnx.*"))


def test_atomic_export_replaces_after_all_gates(tmp_path: Path) -> None:
    from mjlab_microduck.export import _atomic_export_onnx

    destination = tmp_path / "policy.onnx"
    destination.write_bytes(b"prior valid export")

    def export(staged: Path) -> None:
        staged.write_bytes(b"replacement")

    def annotate(staged: Path, _result) -> None:
        staged.write_bytes(staged.read_bytes() + b" with metadata")

    result = _atomic_export_onnx(
        destination,
        export,
        lambda _staged: _successful_parity_result(),
        annotate,
    )

    assert result.max_allclose_ratio == 0.0
    assert destination.read_bytes() == b"replacement with metadata"
    assert not list(tmp_path.glob(".policy.onnx.*"))


def test_atomic_export_rejects_annotation_corruption(tmp_path: Path) -> None:
    from mjlab_microduck.export import _atomic_export_onnx

    destination = tmp_path / "policy.onnx"
    destination.write_bytes(b"prior valid export")

    def export(staged: Path) -> None:
        staged.write_bytes(b"valid graph")

    def verify(staged: Path):
        if staged.read_bytes() != b"valid graph":
            raise RuntimeError("annotated graph failed parity")
        return _successful_parity_result()

    def corrupt(staged: Path, _result) -> None:
        staged.write_bytes(b"corrupt graph")

    with pytest.raises(RuntimeError, match="annotated graph failed parity"):
        _atomic_export_onnx(destination, export, verify, corrupt)

    assert destination.read_bytes() == b"prior valid export"
    assert not list(tmp_path.glob(".policy.onnx.*"))



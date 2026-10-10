"""Numerically prove that an exported ONNX actor matches its PyTorch policy.

The exporter flattens the actor's observation groups and bakes the empirical
normalizer into the ONNX graph.  Shape checks cannot detect a missing or stale
normalizer, so every checkpoint export is compared with the loaded inference
policy before it is considered deployable.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict

DEFAULT_PARITY_SAMPLES = 128
DEFAULT_PARITY_SEED = 0
DEFAULT_PARITY_ATOL = 1.0e-5
DEFAULT_PARITY_RTOL = 1.0e-5


class ExportParityError(RuntimeError):
    """The ONNX graph is not numerically equivalent to its source policy."""


@dataclass(frozen=True)
class ExportParityResult:
    """Reproducible evidence from a successful PyTorch-to-ONNX comparison."""

    samples: int
    seed: int
    atol: float
    rtol: float
    max_abs_error: float
    max_rel_error: float
    max_allclose_ratio: float
    observation_nonfinite_count: int
    pytorch_action_nonfinite_count: int
    onnx_action_nonfinite_count: int

    def metadata(self) -> dict[str, str | float]:
        """Return flat values suitable for ONNX metadata or a provenance block."""

        return {
            "export_parity_method": "loaded-pytorch-policy-vs-onnxruntime-cpu",
            "export_parity_samples": str(self.samples),
            "export_parity_seed": str(self.seed),
            "export_parity_atol": self.atol,
            "export_parity_rtol": self.rtol,
            "export_parity_max_abs_error": self.max_abs_error,
            "export_parity_max_rel_error": self.max_rel_error,
            "export_parity_max_allclose_ratio": self.max_allclose_ratio,
            "export_parity_observation_nonfinite_count": str(
                self.observation_nonfinite_count
            ),
            "export_parity_pytorch_action_nonfinite_count": str(
                self.pytorch_action_nonfinite_count
            ),
            "export_parity_onnx_action_nonfinite_count": str(
                self.onnx_action_nonfinite_count
            ),
        }


def _normalizer_sampling_frame(
    policy: Any, observation_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite raw-observation mean and scale for useful parity samples."""

    normalizer = getattr(policy, "obs_normalizer", None)
    mean_tensor = getattr(normalizer, "_mean", None)
    std_tensor = getattr(normalizer, "_std", None)
    if mean_tensor is None or std_tensor is None:
        return (
            np.zeros((observation_dim,), dtype=np.float32),
            np.ones((observation_dim,), dtype=np.float32),
        )

    mean = mean_tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    std = std_tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    if mean.shape != (observation_dim,) or std.shape != (observation_dim,):
        raise ExportParityError(
            "policy normalizer shape does not match the flattened observations: "
            f"mean={mean.shape}, std={std.shape}, observations={observation_dim}"
        )
    eps = float(getattr(normalizer, "eps", 0.0))
    scale = std + np.float32(eps)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ExportParityError("policy normalizer contains NaN or infinity")
    if np.any(scale <= 0.0):
        raise ExportParityError("policy normalizer has a non-positive effective scale")
    return mean, scale


def _parity_observations(
    policy: Any,
    observation_dim: int,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    if samples < 4:
        raise ExportParityError("export parity requires at least four samples")
    if observation_dim < 1:
        raise ExportParityError("export parity requires a non-empty observation")

    mean, scale = _normalizer_sampling_frame(policy, observation_dim)
    rng = np.random.default_rng(seed)
    normalized = rng.normal(0.0, 2.0, size=(samples, observation_dim)).astype(
        np.float32
    )
    normalized = np.clip(normalized, -6.0, 6.0)
    observations = mean + scale * normalized

    # Stable anchors exercise the policy at its normalization centre, raw zero,
    # and one effective standard deviation on both sides.
    observations[0] = mean
    observations[1] = 0.0
    observations[2] = mean + scale
    observations[3] = mean - scale
    if not np.isfinite(observations).all():
        raise ExportParityError("generated parity observations contain NaN or infinity")
    return np.ascontiguousarray(observations, dtype=np.float32)


def _pytorch_actions(
    policy: Any,
    observations: np.ndarray,
    observation_groups: Sequence[tuple[str, int]],
) -> np.ndarray:
    try:
        reference_policy = copy.deepcopy(policy).to("cpu").eval()
    except Exception as exc:  # pragma: no cover - depends on third-party policy types
        raise ExportParityError(
            f"could not prepare the loaded policy on CPU: {exc}"
        ) from exc

    offset = 0
    values: dict[str, torch.Tensor] = {}
    for name, width in observation_groups:
        values[name] = torch.from_numpy(observations[:, offset : offset + width])
        offset += width
    obs = TensorDict(values, batch_size=[observations.shape[0]])
    try:
        with torch.inference_mode():
            actions = reference_policy(obs)
    except Exception as exc:
        raise ExportParityError(
            f"loaded PyTorch policy inference failed: {exc}"
        ) from exc
    if not isinstance(actions, torch.Tensor) or actions.ndim != 2:
        shape = getattr(actions, "shape", None)
        raise ExportParityError(
            f"loaded PyTorch policy returned {shape!r}; expected a 2-D action tensor"
        )
    return actions.detach().cpu().numpy().astype(np.float32, copy=False)


def _onnx_actions(onnx_source: Path | bytes, observations: np.ndarray) -> np.ndarray:
    source = onnx_source if isinstance(onnx_source, bytes) else str(onnx_source)
    label = (
        "captured ONNX bytes" if isinstance(onnx_source, bytes) else str(onnx_source)
    )
    try:
        session = ort.InferenceSession(source, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ExportParityError(f"ONNX Runtime could not load {label}: {exc}") from exc
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ExportParityError(
            "export parity requires exactly one ONNX input and one ONNX output"
        )
    try:
        rows = [
            session.run(
                [outputs[0].name],
                {inputs[0].name: observation.reshape(1, -1)},
            )[0]
            for observation in observations
        ]
    except Exception as exc:
        raise ExportParityError(f"ONNX Runtime inference failed: {exc}") from exc
    try:
        return np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    except Exception as exc:
        raise ExportParityError(
            f"ONNX outputs could not be concatenated: {exc}"
        ) from exc


def verify_onnx_export_parity(
    policy: Any,
    onnx_path: Path | str | bytes,
    observation_groups: Sequence[tuple[str, int]],
    *,
    samples: int = DEFAULT_PARITY_SAMPLES,
    seed: int = DEFAULT_PARITY_SEED,
    atol: float = DEFAULT_PARITY_ATOL,
    rtol: float = DEFAULT_PARITY_RTOL,
) -> ExportParityResult:
    """Compare a loaded deterministic policy with its just-exported ONNX graph.

    Samples are deterministic raw observations centred on the policy's own
    empirical-normalization statistics.  The PyTorch side calls the loaded
    policy itself with its original TensorDict group layout, rather than its
    ONNX wrapper, so an omitted normalizer or wrong flattening order is caught.
    """

    groups = tuple(observation_groups)
    if not groups or any(not name or width < 1 for name, width in groups):
        raise ExportParityError(
            "observation groups must have names and positive widths"
        )
    if len({name for name, _ in groups}) != len(groups):
        raise ExportParityError("observation group names must be unique")
    if samples < 4:
        raise ExportParityError("export parity requires at least four samples")
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0.0 or rtol < 0.0:
        raise ExportParityError(
            "export parity tolerances must be finite and non-negative"
        )

    observation_dim = sum(width for _, width in groups)
    observations = _parity_observations(
        policy, observation_dim, samples=samples, seed=seed
    )
    expected = _pytorch_actions(policy, observations, groups)
    onnx_source = onnx_path if isinstance(onnx_path, bytes) else Path(onnx_path)
    actual = _onnx_actions(onnx_source, observations)
    if expected.shape != actual.shape:
        raise ExportParityError(
            f"action shape mismatch: PyTorch={expected.shape}, ONNX={actual.shape}"
        )
    if expected.shape[0] != samples:
        raise ExportParityError(
            f"policy returned {expected.shape[0]} samples; expected {samples}"
        )
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ExportParityError("export parity produced NaN or infinity")

    absolute = np.abs(expected - actual)
    denominator = np.maximum(
        np.maximum(np.abs(expected), np.abs(actual)), np.finfo(np.float32).eps
    )
    relative = absolute / denominator
    max_abs_error = float(absolute.max(initial=0.0))
    max_rel_error = float(relative.max(initial=0.0))
    allowance = atol + rtol * np.abs(actual)
    allclose_ratio = np.divide(
        absolute,
        allowance,
        out=np.full_like(absolute, np.inf),
        where=allowance > 0.0,
    )
    allclose_ratio[(allowance == 0.0) & (absolute == 0.0)] = 0.0
    max_allclose_ratio = float(allclose_ratio.max(initial=0.0))
    if not np.allclose(expected, actual, atol=atol, rtol=rtol):
        worst = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
        raise ExportParityError(
            "ONNX export does not match the loaded PyTorch policy: "
            f"max_abs_error={max_abs_error:.9g} at {worst}, "
            f"pytorch={expected[worst]:.9g}, onnx={actual[worst]:.9g}, "
            f"atol={atol:.9g}, rtol={rtol:.9g}"
        )
    return ExportParityResult(
        samples=samples,
        seed=seed,
        atol=atol,
        rtol=rtol,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        max_allclose_ratio=max_allclose_ratio,
        observation_nonfinite_count=0,
        pytorch_action_nonfinite_count=0,
        onnx_action_nonfinite_count=0,
    )

#!/usr/bin/env python3
"""Apply the measured Bruz torso masses to an onshape-to-robot MJCF export.

The exporter merges both rigid components into ``trunk_base``, so their
individual masses cannot be changed on the mesh geoms.  Decomposing the source
spatial inertia gives 12.991728 g for ``power_support`` and 108.004710 g for
``np_f970``.  Replacing those with the measured values below yields the target
aggregate mass, center of mass, and full inertia recorded here.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

COMPONENT_MASSES_KG = {"power_support": 0.052, "np_f970": 0.250}

SOURCE = (
    0.199224,
    (-0.0226332, -2.70828e-05, 0.00280064),
    (
        0.000123975,
        0.000145931,
        0.000115351,
        7.1651e-08,
        -2.06915e-05,
        1.32429e-07,
    ),
)
TARGET = (
    0.380227561175,
    (-0.0276032800259, -1.1983515297e-05, 3.82828145214e-06),
    (
        0.000232698557612,
        0.000250188205672,
        0.00015991322052,
        9.03001782426e-08,
        -2.92603533184e-05,
        1.10193048873e-07,
    ),
)

MARKER = "<!-- Custom masses: power_support=0.052 kg, np_f970=0.250 kg -->"
BODY_RE = re.compile(r'<body\b[^>]*\bname="trunk_base"[^>]*>')
INERTIAL_RE = re.compile(r"<inertial\b[^>]*/>")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def _properties(tag: str) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    attrs = dict(ATTR_RE.findall(tag))
    return (
        float(attrs["mass"]),
        tuple(float(value) for value in attrs["pos"].split()),
        tuple(float(value) for value in attrs["fullinertia"].split()),
    )


def _matches(
    actual: tuple[float, tuple[float, ...], tuple[float, ...]],
    expected: tuple[float, tuple[float, ...], tuple[float, ...]],
) -> bool:
    return all(
        len(actual_values) == len(expected_values)
        and all(
            math.isclose(a, e, rel_tol=5e-10, abs_tol=5e-12)
            for a, e in zip(actual_values, expected_values, strict=True)
        )
        for actual_values, expected_values in (
            ((actual[0],), (expected[0],)),
            (actual[1], expected[1]),
            (actual[2], expected[2]),
        )
    )


def apply_override(path: Path) -> bool:
    """Update ``path`` in place; return whether its contents changed."""
    text = path.read_text()
    body = BODY_RE.search(text)
    if body is None:
        raise ValueError(f"{path}: body named trunk_base was not found")
    inertial = INERTIAL_RE.search(text, body.end())
    if inertial is None:
        raise ValueError(f"{path}: trunk_base has no inertial")

    actual = _properties(inertial.group(0))
    if not _matches(actual, SOURCE) and not _matches(actual, TARGET):
        raise ValueError(
            f"{path}: unexpected trunk inertia; re-derive the override for the "
            "changed CAD export"
        )

    inertial_line = text.rfind("\n", 0, inertial.start()) + 1
    indent = text[inertial_line : inertial.start()]
    target_tag = (
        f'{indent}<inertial pos="{" ".join(map(str, TARGET[1]))}" '
        f'mass="{TARGET[0]}" fullinertia="{" ".join(map(str, TARGET[2]))}"/>'
    )
    replacement = f"{indent}{MARKER}\n{target_tag}"

    marker = text.rfind(MARKER, body.end(), inertial.start())
    replace_start = text.rfind("\n", 0, marker) + 1 if marker >= 0 else inertial_line
    if text[replace_start : inertial.end()] == replacement:
        return False

    path.write_text(text[:replace_start] + replacement + text[inertial.end() :])
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path, help="generated MJCF to update in place")
    args = parser.parse_args()
    action = "updated" if apply_override(args.xml) else "already current"
    print(f"[mass-overrides] {args.xml}: {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

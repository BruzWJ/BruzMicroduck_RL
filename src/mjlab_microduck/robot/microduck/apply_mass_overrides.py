#!/usr/bin/env python3
"""Apply measured Bruz masses to an onshape-to-robot MJCF export.

The exporter merges fixed components into aggregate body inertials.  This
post-processor keeps those aggregate inertials physical after replacing:

* the 12.991728 g CAD power support with the measured 52 g part;
* the 108.004710 g CAD battery with the measured 250 g battery; and
* every 18 g XL330 servo with a 23 g XC330-T181 (15 servos total).

The 5 g servo delta is distributed at each servo mesh pose and folded into the
owning body's mass, center of mass, and full inertia.  The constants below are
the mass properties obtained by compiling ``assets/xl330.stl`` as a single
MuJoCo mesh geom with ``mass=0.005``.  XC330-T181 and XL330 share the same
20 x 34 x 26 mm case, so this preserves the exported housing distribution
without adding runtime-only bodies or geoms.
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

COMPONENT_MASSES_KG = {"power_support": 0.052, "np_f970": 0.250}
SERVO_MASSES_KG = {"xl330": 0.018, "xc330_t181": 0.023}
SERVO_MASS_DELTA_KG = SERVO_MASSES_KG["xc330_t181"] - SERVO_MASSES_KG["xl330"]
SERVO_COUNT = 15
SERVO_MESH_NAME = "xl330"

# Uniform-mesh mass properties of assets/xl330.stl at 5 g, expressed in the
# raw mesh frame.  MuJoCo/XML quaternions use (w, x, y, z); each visual geom's
# pose rotates/translates these properties into its owning body frame.
SERVO_DELTA_COM = (-1.32549428e-04, -2.30514671e-20, -6.90997620e-03)
SERVO_DELTA_INERTIA = (
    6.532062828540525e-07,
    7.454344493755343e-07,
    4.046077631698406e-07,
    4.1379924452274626e-23,
    -6.619269440493474e-10,
    0.0,
)

# Torso inertial in a fresh CAD export.
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

# Torso inertial after only the support/battery replacement.  Servo deltas are
# applied to this value (including the two servos owned by trunk_base).
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

LEGACY_MARKER = "<!-- Custom masses: power_support=0.052 kg, np_f970=0.250 kg -->"
MARKER = (
    "<!-- Custom masses: power_support=0.052 kg, np_f970=0.250 kg, "
    "15x xc330_t181=0.023 kg -->"
)
BODY_RE = re.compile(r'<body\b[^>]*\bname="trunk_base"[^>]*>')
INERTIAL_RE = re.compile(r"<inertial\b[^>]*/>")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

Vec3 = tuple[float, float, float]
Mat3 = tuple[tuple[float, float, float], ...]
Properties = tuple[float, tuple[float, ...], tuple[float, ...]]


def _properties(tag: str) -> Properties:
    attrs = dict(ATTR_RE.findall(tag))
    return (
        float(attrs["mass"]),
        tuple(float(value) for value in attrs["pos"].split()),
        tuple(float(value) for value in attrs["fullinertia"].split()),
    )


def _matches(actual: Properties, expected: Properties) -> bool:
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


def _vec(text: str | None, default: Vec3) -> Vec3:
    values = default if text is None else tuple(float(value) for value in text.split())
    if len(values) != 3:
        raise ValueError(f"expected three values, got {values}")
    return values  # type: ignore[return-value]


def _quat_matrix(text: str | None) -> Mat3:
    """Return the active local-to-parent rotation for an XML w-x-y-z quat."""
    values = (1.0, 0.0, 0.0, 0.0) if text is None else tuple(
        float(value) for value in text.split()
    )
    if len(values) != 4:
        raise ValueError(f"expected four quaternion values, got {values}")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("zero-length quaternion")
    w, x, y, z = (value / norm for value in values)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _mat_vec(matrix: Mat3, vector: Vec3) -> Vec3:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _mat_mul(left: Mat3, right: Mat3) -> Mat3:
    return tuple(
        tuple(
            sum(left[row][k] * right[k][column] for k in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _transpose(matrix: Mat3) -> Mat3:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def _full_matrix(values: tuple[float, ...]) -> Mat3:
    if len(values) != 6:
        raise ValueError(f"expected six full-inertia values, got {values}")
    ixx, iyy, izz, ixy, ixz, iyz = values
    return ((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz))


def _matrix_full(matrix: Mat3) -> tuple[float, ...]:
    return (
        matrix[0][0],
        matrix[1][1],
        matrix[2][2],
        matrix[0][1],
        matrix[0][2],
        matrix[1][2],
    )


def _parallel_axis(mass: float, com: Vec3) -> Mat3:
    squared = sum(value * value for value in com)
    return tuple(
        tuple(
            mass * ((squared if row == column else 0.0) - com[row] * com[column])
            for column in range(3)
        )
        for row in range(3)
    )


def _add_matrix(*matrices: Mat3) -> Mat3:
    return tuple(
        tuple(sum(matrix[row][column] for matrix in matrices) for column in range(3))
        for row in range(3)
    )


def _sub_matrix(left: Mat3, right: Mat3) -> Mat3:
    return tuple(
        tuple(left[row][column] - right[row][column] for column in range(3))
        for row in range(3)
    )


def _add_servo_deltas(base: Properties, geoms: list[ET.Element]) -> Properties:
    base_mass, base_com_values, base_inertia_values = base
    base_com: Vec3 = tuple(base_com_values)  # type: ignore[assignment]
    inertia_at_origin = _add_matrix(
        _full_matrix(base_inertia_values), _parallel_axis(base_mass, base_com)
    )
    first_moment = [base_mass * value for value in base_com]

    servo_inertia = _full_matrix(SERVO_DELTA_INERTIA)
    for geom in geoms:
        pos = _vec(geom.get("pos"), (0.0, 0.0, 0.0))
        rotation = _quat_matrix(geom.get("quat"))
        rotated_com = _mat_vec(rotation, SERVO_DELTA_COM)
        com = tuple(pos[i] + rotated_com[i] for i in range(3))
        rotated_inertia = _mat_mul(_mat_mul(rotation, servo_inertia), _transpose(rotation))
        inertia_at_origin = _add_matrix(
            inertia_at_origin,
            rotated_inertia,
            _parallel_axis(SERVO_MASS_DELTA_KG, com),
        )
        for axis in range(3):
            first_moment[axis] += SERVO_MASS_DELTA_KG * com[axis]

    mass = base_mass + len(geoms) * SERVO_MASS_DELTA_KG
    com = tuple(moment / mass for moment in first_moment)
    inertia_at_com = _sub_matrix(inertia_at_origin, _parallel_axis(mass, com))
    return mass, com, _matrix_full(inertia_at_com)


def _servo_geom_count(text: str) -> int:
    root = ET.fromstring(text)
    return sum(
        1
        for body in root.iter("body")
        for geom in body.findall("geom")
        if geom.get("mesh") == SERVO_MESH_NAME and geom.get("class") == "visual"
    )


def _servo_inertial_updates(text: str) -> dict[str, Properties]:
    root = ET.fromstring(text)
    updates: dict[str, Properties] = {}
    count = 0
    for body in root.iter("body"):
        geoms = [
            geom
            for geom in body.findall("geom")
            if geom.get("mesh") == SERVO_MESH_NAME and geom.get("class") == "visual"
        ]
        if not geoms:
            continue
        name = body.get("name")
        inertial = body.find("inertial")
        if name is None or inertial is None:
            raise ValueError("every servo-owning body must be named and have an inertial")
        if "fullinertia" not in inertial.attrib:
            raise ValueError(f"{name}: expected a fullinertia attribute")
        if name in updates:
            raise ValueError(f"duplicate servo-owning body name: {name}")
        base = (
            float(inertial.attrib["mass"]),
            tuple(float(value) for value in inertial.attrib["pos"].split()),
            tuple(float(value) for value in inertial.attrib["fullinertia"].split()),
        )
        updates[name] = _add_servo_deltas(base, geoms)
        count += len(geoms)
    if count != SERVO_COUNT:
        raise ValueError(f"expected {SERVO_COUNT} visual servo meshes, found {count}")
    return updates


def _format_values(values: tuple[float, ...]) -> str:
    return " ".join(format(value, ".17g") for value in values)


def _replace_body_inertial(text: str, body_name: str, target: Properties) -> str:
    body_re = re.compile(rf'<body\b[^>]*\bname="{re.escape(body_name)}"[^>]*>')
    body = body_re.search(text)
    if body is None:
        raise ValueError(f"body named {body_name!r} was not found")
    inertial = INERTIAL_RE.search(text, body.end())
    next_body = text.find("<body", body.end())
    if inertial is None or (next_body >= 0 and inertial.start() > next_body):
        raise ValueError(f"{body_name}: direct inertial was not found")
    mass, com, inertia = target
    tag = (
        f'<inertial pos="{_format_values(com)}" mass="{format(mass, ".17g")}" '
        f'fullinertia="{_format_values(inertia)}"/>'
    )
    return text[: inertial.start()] + tag + text[inertial.end() :]


def _apply_torso_override(text: str, path: Path) -> str:
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

    text = _replace_body_inertial(text, "trunk_base", TARGET)
    if LEGACY_MARKER not in text:
        body = BODY_RE.search(text)
        assert body is not None
        inertial = INERTIAL_RE.search(text, body.end())
        assert inertial is not None
        inertial_line = text.rfind("\n", 0, inertial.start()) + 1
        prefix = text[inertial_line : inertial.start()]
        # Generated MJCF puts the inertial on its own indented line.  Keeping
        # the inline fallback makes the post-processor safe for tiny fixtures
        # and hand-authored XML without duplicating the preceding tags.
        insertion = inertial_line if not prefix.strip() else inertial.start()
        indent = prefix if not prefix.strip() else ""
        text = text[:insertion] + f"{indent}{LEGACY_MARKER}\n" + text[insertion:]
    return text


def apply_override(path: Path) -> bool:
    """Update ``path`` in place; return whether its contents changed."""
    original = path.read_text()
    if MARKER in original:
        count = _servo_geom_count(original)
        if count != SERVO_COUNT:
            raise ValueError(
                f"{path}: marker says XC330 masses are applied but found "
                f"{count}/{SERVO_COUNT} visual servo meshes"
            )
        return False

    text = _apply_torso_override(original, path)
    for body_name, target in _servo_inertial_updates(text).items():
        text = _replace_body_inertial(text, body_name, target)
    text = text.replace(LEGACY_MARKER, MARKER, 1)

    if text == original:
        return False
    path.write_text(text)
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

"""Regression tests for the measured Bruz component mass distribution."""

from __future__ import annotations

import re
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROBOT_DIR = (
    Path(__file__).parents[1] / "src" / "mjlab_microduck" / "robot" / "microduck"
)
OVERRIDE_SCRIPT = ROBOT_DIR / "apply_mass_overrides.py"
OVERRIDES = runpy.run_path(str(OVERRIDE_SCRIPT))
TARGET_MASS, TARGET_COM, TARGET_INERTIA = OVERRIDES["TARGET"]
SERVO_COUNT = OVERRIDES["SERVO_COUNT"]
SERVO_MASS_DELTA_KG = OVERRIDES["SERVO_MASS_DELTA_KG"]


def _trunk_inertial(
    path: Path,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    root = ET.parse(path).getroot()
    inertial = root.find(".//body[@name='trunk_base']/inertial")
    assert inertial is not None, path.name
    return (
        float(inertial.attrib["mass"]),
        tuple(float(value) for value in inertial.attrib["pos"].split()),
        tuple(float(value) for value in inertial.attrib["fullinertia"].split()),
    )


def _total_body_mass(path: Path) -> float:
    return sum(
        float(inertial.attrib["mass"])
        for inertial in ET.parse(path).getroot().iter("inertial")
    )


def _visual_servo_count(path: Path) -> int:
    return sum(
        1
        for geom in ET.parse(path).getroot().iter("geom")
        if geom.get("mesh") == "xl330" and geom.get("class") == "visual"
    )


def test_all_robot_variants_use_measured_component_masses():
    assert OVERRIDES["COMPONENT_MASSES_KG"] == {
        "power_support": 0.052,
        "np_f970": 0.250,
    }
    assert OVERRIDES["SERVO_MASSES_KG"] == {
        "xl330": 0.018,
        "xc330_t181": 0.023,
    }

    robots = sorted(ROBOT_DIR.glob("robot_*.xml"))
    assert len(robots) == 8
    for path in robots:
        assert _visual_servo_count(path) == SERVO_COUNT, path.name
        assert path.read_text().count(OVERRIDES["MARKER"]) == 1, path.name

        # The roller CAD export is 0.4135599999999 g lighter; both families gain
        # exactly 15 * 5 g over their measured-support/battery XL330 baseline.
        baseline = 0.917833181175 if "rollers" in path.name else 0.918246741175
        assert _total_body_mass(path) == pytest.approx(
            baseline + SERVO_COUNT * SERVO_MASS_DELTA_KG, abs=1e-13
        ), path.name

        # Compilation also checks that every rewritten full inertia is physical.
        model = mujoco.MjModel.from_xml_path(str(path))
        assert float(model.body_mass.sum()) == pytest.approx(
            baseline + 0.075, abs=1e-13
        ), path.name
        assert np.all(model.body_inertia[1:] > 0.0), path.name


def test_servo_delta_properties_are_derived_from_the_exported_housing_mesh():
    mesh_dir = ROBOT_DIR / "assets"
    xml = f"""
    <mujoco>
      <compiler meshdir="{mesh_dir}"/>
      <asset><mesh name="servo" file="xl330.stl"/></asset>
      <worldbody><body><geom type="mesh" mesh="servo" mass="0.005"/></body></worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[1])
    rotation = rotation.reshape(3, 3)
    inertia = rotation @ np.diag(model.body_inertia[1]) @ rotation.T
    full = (
        inertia[0, 0],
        inertia[1, 1],
        inertia[2, 2],
        inertia[0, 1],
        inertia[0, 2],
        inertia[1, 2],
    )

    assert model.body_mass[1] == pytest.approx(SERVO_MASS_DELTA_KG)
    assert model.body_ipos[1] == pytest.approx(OVERRIDES["SERVO_DELTA_COM"], abs=5e-12)
    assert full == pytest.approx(OVERRIDES["SERVO_DELTA_INERTIA"], abs=1e-15)


def test_override_adds_each_visual_servo_once_and_is_idempotent(tmp_path: Path):
    source = OVERRIDES["SOURCE"]
    source_tag = (
        f'<inertial pos="{" ".join(map(str, source[1]))}" mass="{source[0]}" '
        f'fullinertia="{" ".join(map(str, source[2]))}"/>'
    )
    visual = '<geom class="visual" mesh="xl330"/>'
    collision = '<geom class="collision" mesh="xl330"/>'
    path = tmp_path / "robot.xml"
    path.write_text(
        "<mujoco><worldbody><body name=\"trunk_base\">"
        + source_tag
        + visual * SERVO_COUNT
        + collision * SERVO_COUNT
        + "</body></worldbody></mujoco>"
    )

    assert OVERRIDES["apply_override"](path)
    mass, _, _ = _trunk_inertial(path)
    assert mass == pytest.approx(
        TARGET_MASS + SERVO_COUNT * SERVO_MASS_DELTA_KG, abs=1e-13
    )
    once = path.read_text()
    assert not OVERRIDES["apply_override"](path)
    assert path.read_text() == once


def test_export_recipes_reapply_mass_override_before_backlash():
    configs = sorted(ROBOT_DIR.glob("config_mjcf_*.json"))
    assert len(configs) == 7
    for path in configs:
        text = path.read_text()
        output = re.search(r'"output_filename":\s*"([^"]+)"', text)
        assert output is not None, path.name
        command = f"python3 apply_mass_overrides.py {output.group(1)}.xml"
        assert text.count(command) == 1, path.name
        if "python3 add_backlash.py" in text:
            assert text.index(command) < text.index("python3 add_backlash.py"), path.name

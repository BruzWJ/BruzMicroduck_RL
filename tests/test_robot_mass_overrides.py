"""Regression tests for the measured Bruz torso mass distribution."""

from __future__ import annotations

import re
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROBOT_DIR = (
    Path(__file__).parents[1] / "src" / "mjlab_microduck" / "robot" / "microduck"
)
OVERRIDE_SCRIPT = ROBOT_DIR / "apply_mass_overrides.py"
OVERRIDES = runpy.run_path(str(OVERRIDE_SCRIPT))
TARGET_MASS, TARGET_COM, TARGET_INERTIA = OVERRIDES["TARGET"]


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


def test_all_robot_variants_use_measured_torso_masses():
    assert OVERRIDES["COMPONENT_MASSES_KG"] == {
        "power_support": 0.052,
        "np_f970": 0.250,
    }

    robots = sorted(ROBOT_DIR.glob("robot_*.xml"))
    assert len(robots) == 8
    for path in robots:
        mass, com, inertia = _trunk_inertial(path)
        assert mass == pytest.approx(TARGET_MASS, rel=5e-11, abs=5e-13), path.name
        assert com == pytest.approx(TARGET_COM, rel=5e-11, abs=5e-13), path.name
        assert inertia == pytest.approx(
            TARGET_INERTIA, rel=5e-11, abs=5e-13
        ), path.name
        assert OVERRIDES["MARKER"] in path.read_text(), path.name


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

"""The two ball-kick tasks stay mirrored and use distinct run roots."""

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    BALL_OFFSET_ABS_Y,
    BALL_OFFSET_X,
    make_microduck_ball_kick_env_cfg,
)


def _support_foot_pattern(cfg) -> str:
    sensor = next(
        sensor
        for sensor in cfg.scene.sensors
        if sensor.name == "support_foot_ground_contact"
    )
    return sensor.primary.pattern


def test_ball_kick_factories_are_physically_mirrored():
    right = make_microduck_ball_kick_env_cfg(kick_foot="right")
    left = make_microduck_ball_kick_env_cfg(kick_foot="left")

    assert right.events["reset_ball"].params["offset"] == (
        BALL_OFFSET_X,
        -BALL_OFFSET_ABS_Y,
    )
    assert left.events["reset_ball"].params["offset"] == (
        BALL_OFFSET_X,
        BALL_OFFSET_ABS_Y,
    )
    assert _support_foot_pattern(right) == r"^left_foot_collision$"
    assert _support_foot_pattern(left) == r"^right_foot_collision$"


def test_ball_kick_registrations_keep_side_specific_configs_and_run_roots():
    right_task = "Mjlab-BallKick-Flat-MicroDuck"
    left_task = "Mjlab-BallKickLeft-Flat-MicroDuck"

    right = load_env_cfg(right_task)
    left = load_env_cfg(left_task)
    assert right.events["reset_ball"].params["offset"][1] == -BALL_OFFSET_ABS_Y
    assert left.events["reset_ball"].params["offset"][1] == BALL_OFFSET_ABS_Y
    assert _support_foot_pattern(right) == r"^left_foot_collision$"
    assert _support_foot_pattern(left) == r"^right_foot_collision$"

    right_runner = load_rl_cfg(right_task)
    left_runner = load_rl_cfg(left_task)
    assert right_runner.experiment_name == "ball_kick_right"
    assert right_runner.run_name == "ball_kick_right"
    assert left_runner.experiment_name == "ball_kick_left"
    assert left_runner.run_name == "ball_kick_left"

"""Tests for the 4-region motion-file config: correct assignment, TripleStep excluded."""
from simple_goalkeeper.tasks.goalkeeper_multidisc_amp_cfg import (
    REGION_MOTION_FILES,
    goalkeeper_multidisc_amp_runner_cfg,
)


def test_region_motion_files_assignment():
    assert REGION_MOTION_FILES["left_near"][0].endswith("LeftStep_own_booster_t1.npz")
    assert REGION_MOTION_FILES["right_near"][0].endswith("Rightstep_own_booster_t1.npz")
    left_far = [p.split("/")[-1] for p in REGION_MOTION_FILES["left_far"]]
    assert left_far == [
        "new_doublestepleft_short_booster_t1.npz",
        "new_doublestepleft_booster_t1.npz",
        "new_doublestepleft_wide_booster_t1.npz",
    ]
    right_far = [p.split("/")[-1] for p in REGION_MOTION_FILES["right_far"]]
    assert right_far == [
        "new_doublestepright_short_booster_t1.npz",
        "new_doublestepright_booster_t1.npz",
        "new_doublestepright_wide_booster_t1.npz",
    ]


def test_no_triple_step_anywhere_in_region_motion_files():
    for paths in REGION_MOTION_FILES.values():
        for path in paths:
            assert "TripleStep" not in path


def test_runner_cfg_amp_data_matches_region_motion_files_and_no_triple_step():
    cfg = goalkeeper_multidisc_amp_runner_cfg()
    assert set(cfg["amp_data"].keys()) == set(REGION_MOTION_FILES.keys())
    for name, motion_cfg in cfg["amp_data"].items():
        assert motion_cfg.motion_files == REGION_MOTION_FILES[name]
        for path in motion_cfg.motion_files:
            assert "TripleStep" not in path


def test_far_region_motion_files_equally_weighted():
    cfg = goalkeeper_multidisc_amp_runner_cfg()
    for name in ("left_far", "right_far"):
        motion_cfg = cfg["amp_data"][name]
        assert motion_cfg.motion_weights == [1.0, 1.0, 1.0]
    for name in ("left_near", "right_near"):
        assert cfg["amp_data"][name].motion_weights is None

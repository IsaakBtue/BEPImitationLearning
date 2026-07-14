"""Tests for the 4-region motion-file config: correct assignment, TripleStep excluded."""
from simple_goalkeeper.tasks.goalkeeper_multidisc_amp_cfg import (
    REGION_MOTION_FILES,
    goalkeeper_multidisc_amp_runner_cfg,
)


def test_region_motion_files_assignment():
    assert REGION_MOTION_FILES["left_near"][0].endswith("LeftStep_own_booster_t1.npz")
    assert REGION_MOTION_FILES["left_far"][0].endswith("LeftDoubleStep_own_booster_t1_2p5x.npz")
    assert REGION_MOTION_FILES["right_near"][0].endswith("Rightstep_own_booster_t1.npz")
    assert REGION_MOTION_FILES["right_far"][0].endswith("RightDoubleStep_own_booster_t1_2p5x.npz")


def test_no_triple_step_anywhere_in_region_motion_files():
    for paths in REGION_MOTION_FILES.values():
        for path in paths:
            assert "TripleStep" not in path


def test_runner_cfg_amp_data_has_one_file_per_region_and_no_triple_step():
    cfg = goalkeeper_multidisc_amp_runner_cfg()
    assert set(cfg["amp_data"].keys()) == set(REGION_MOTION_FILES.keys())
    for name, motion_cfg in cfg["amp_data"].items():
        assert motion_cfg.motion_files == REGION_MOTION_FILES[name]
        for path in motion_cfg.motion_files:
            assert "TripleStep" not in path

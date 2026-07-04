"""Tests for the 4-region motion-file config: correct assignment, TripleStep excluded."""
from simple_goalkeeper.tasks.goalkeeper_multidisc_amp_cfg import (
    REGION_MOTION_FILES,
    goalkeeper_multidisc_amp_runner_cfg,
)


def test_region_motion_files_assignment():
    assert REGION_MOTION_FILES["left_near"].endswith("LeftStep_own_booster_t1.npz")
    assert REGION_MOTION_FILES["left_far"].endswith("LeftDoubleStep_own_booster_t1.npz")
    assert REGION_MOTION_FILES["right_near"].endswith("Rightstep_own_booster_t1.npz")
    assert REGION_MOTION_FILES["right_far"].endswith("RightDoubleStep_own_booster_t1.npz")


def test_no_triple_step_anywhere_in_region_motion_files():
    for path in REGION_MOTION_FILES.values():
        assert "TripleStep" not in path


def test_runner_cfg_amp_data_has_one_file_per_region_and_no_triple_step():
    cfg = goalkeeper_multidisc_amp_runner_cfg()
    assert set(cfg["amp_data"].keys()) == set(REGION_MOTION_FILES.keys())
    for name, motion_cfg in cfg["amp_data"].items():
        assert len(motion_cfg.motion_files) == 1
        assert motion_cfg.motion_files[0] == REGION_MOTION_FILES[name]
        assert "TripleStep" not in motion_cfg.motion_files[0]

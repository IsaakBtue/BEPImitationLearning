"""Tests for the 4-region motion-file config: correct assignment, TripleStep excluded.

2026-07-12: left_far/right_far now carry TWO files each (original pace + a 1.5x-retimed
variant, see retime_motion.py and docs/BugFixes.md) -- REGION_MOTION_FILES values became
list[str] instead of str. near regions stay single-file.
"""
from simple_goalkeeper.tasks.goalkeeper_multidisc_amp_cfg import (
    REGION_MOTION_FILES,
    goalkeeper_multidisc_amp_runner_cfg,
)


def test_region_motion_files_assignment():
    assert len(REGION_MOTION_FILES["left_near"]) == 1
    assert REGION_MOTION_FILES["left_near"][0].endswith("LeftStep_own_booster_t1.npz")
    assert any(p.endswith("LeftDoubleStep_own_booster_t1.npz") for p in REGION_MOTION_FILES["left_far"])
    assert any(p.endswith("LeftDoubleStep_own_booster_t1_1p5x.npz") for p in REGION_MOTION_FILES["left_far"])
    assert len(REGION_MOTION_FILES["right_near"]) == 1
    assert REGION_MOTION_FILES["right_near"][0].endswith("Rightstep_own_booster_t1.npz")
    assert any(p.endswith("RightDoubleStep_own_booster_t1.npz") for p in REGION_MOTION_FILES["right_far"])
    assert any(p.endswith("RightDoubleStep_own_booster_t1_1p5x.npz") for p in REGION_MOTION_FILES["right_far"])


def test_no_triple_step_anywhere_in_region_motion_files():
    for paths in REGION_MOTION_FILES.values():
        for path in paths:
            assert "TripleStep" not in path


def test_near_regions_single_file_far_regions_two_files_no_triple_step():
    cfg = goalkeeper_multidisc_amp_runner_cfg()
    assert set(cfg["amp_data"].keys()) == set(REGION_MOTION_FILES.keys())
    for name, motion_cfg in cfg["amp_data"].items():
        assert motion_cfg.motion_files == REGION_MOTION_FILES[name]
        expected_count = 1 if name.endswith("_near") else 2
        assert len(motion_cfg.motion_files) == expected_count
        for f in motion_cfg.motion_files:
            assert "TripleStep" not in f

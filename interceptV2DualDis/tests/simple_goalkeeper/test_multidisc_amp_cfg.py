"""Tests for the 4-region motion-file config: correct assignment, TripleStep excluded.

2026-07-12 (revised, same day): left_far/right_far now carry ONLY the 2x-retimed pace
(retime_motion.py) -- the original 1.0x and 1.5x paces were dropped after research found
blending multiple paces into one AMP discriminator exhibits a "mixes incompatible motion
statistics" failure mode (docs/BugFixes.md). All regions are single-file again, but
REGION_MOTION_FILES values stay list[str] (not reverted to str) since the mechanism
should still support multiple files per region if a future experiment wants that.

2026-07-12 (later same day): upgraded 2x -> 2.5x. The 2x clip (0.72s) was still slower
than the fastest observed wide-crossing ball-flight window (0.58s at full difficulty) --
user reported the 2x pace still looked too slow watching play. 2.5x compresses the clip
to 0.576s, matching that tightest window almost exactly.
"""
from simple_goalkeeper.tasks.goalkeeper_multidisc_amp_cfg import (
    REGION_MOTION_FILES,
    goalkeeper_multidisc_amp_runner_cfg,
)


def test_region_motion_files_assignment():
    assert len(REGION_MOTION_FILES["left_near"]) == 1
    assert REGION_MOTION_FILES["left_near"][0].endswith("LeftStep_own_booster_t1.npz")
    assert len(REGION_MOTION_FILES["left_far"]) == 1
    assert REGION_MOTION_FILES["left_far"][0].endswith("LeftDoubleStep_own_booster_t1_2p5x.npz")
    assert len(REGION_MOTION_FILES["right_near"]) == 1
    assert REGION_MOTION_FILES["right_near"][0].endswith("Rightstep_own_booster_t1.npz")
    assert len(REGION_MOTION_FILES["right_far"]) == 1
    assert REGION_MOTION_FILES["right_far"][0].endswith("RightDoubleStep_own_booster_t1_2p5x.npz")


def test_no_triple_step_anywhere_in_region_motion_files():
    for paths in REGION_MOTION_FILES.values():
        for path in paths:
            assert "TripleStep" not in path


def test_all_regions_single_file_no_triple_step():
    cfg = goalkeeper_multidisc_amp_runner_cfg()
    assert set(cfg["amp_data"].keys()) == set(REGION_MOTION_FILES.keys())
    for name, motion_cfg in cfg["amp_data"].items():
        assert motion_cfg.motion_files == REGION_MOTION_FILES[name]
        assert len(motion_cfg.motion_files) == 1
        for f in motion_cfg.motion_files:
            assert "TripleStep" not in f

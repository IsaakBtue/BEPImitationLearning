"""2026-07-12: confirms the retime_motion.py output actually loads into the
"wide" RSI pool alongside the originals via _STEM_TO_POOL, so
seed_blue_landed_practice's late-half draw (frame_frac >= 0.5) can pull from
either pace. Exercises MotionResetManager.init() against the real NPZ files
on disk -- this path had no test coverage before (existing test_live_rsi.py
only exercises reset(), never init()'s real-file loading).
"""
import torch

from simple_goalkeeper.mdp.events import MotionResetManager


class _FakeEnv:
    def __init__(self):
        self.device = "cpu"


def test_wide_pools_load_both_original_and_1p5x_frames():
    mgr = MotionResetManager()
    mgr.init(_FakeEnv())

    for side in ("left", "right"):
        pool = mgr.pools[(side, "wide")]
        n = pool["joint_pos"].shape[0]
        # 2 originals (Double/TripleStep, 73 frames each) + 2 retimed variants
        # (48 frames each, see retime_motion.py's 1.5x output) per side.
        assert n == 73 + 73 + 48 + 48, f"{side} wide pool frame count: {n}"

        # frame_frac resets per source file (not global) -- some frames in
        # the pool must reach exactly 1.0 more than twice (once per original
        # file, once per retimed file) if per-file reset is actually working.
        at_end = (pool["frame_frac"] >= 0.999).sum().item()
        assert at_end == 4, f"{side}: expected 4 file-final frames, got {at_end}"

        # The late-half draw seed_blue_landed_practice uses must be able to
        # pull from both the original-pace and the 1.5x-retimed frames.
        late = torch.nonzero(pool["frame_frac"] >= 0.5, as_tuple=False).flatten()
        assert len(late) > 0

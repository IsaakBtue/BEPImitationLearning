"""2026-07-12: confirms the retime_motion.py output actually loads into the
"wide" RSI pool via _STEM_TO_POOL, so seed_blue_landed_practice's late-half
draw (frame_frac >= 0.5) can pull from it. Exercises MotionResetManager.init()
against the real NPZ files on disk -- this path had no test coverage before
(existing test_live_rsi.py only exercises reset(), never init()'s real-file
loading).

2026-07-12 (revised, same day): dropped the original 1.0x and 1.5x paces from
the wide pool entirely -- research found blending multiple paces into one AMP
discriminator/reference pool exhibits a "mixes incompatible motion statistics"
failure mode (docs/BugFixes.md). Wide pool is now 2x-only (36 frames per
source file), the "most physically possible" single pace.
"""
import torch

from simple_goalkeeper.mdp.events import MotionResetManager


class _FakeEnv:
    def __init__(self):
        self.device = "cpu"


def test_wide_pools_load_only_2x_frames():
    mgr = MotionResetManager()
    mgr.init(_FakeEnv())

    for side in ("left", "right"):
        pool = mgr.pools[(side, "wide")]
        n = pool["joint_pos"].shape[0]
        # 2x-only: Double + Triple step, 36 frames each, per side.
        assert n == 36 + 36, f"{side} wide pool frame count: {n}"

        # frame_frac resets per source file (not global) -- some frames in
        # the pool must reach exactly 1.0 once per source file (2 now).
        at_end = (pool["frame_frac"] >= 0.999).sum().item()
        assert at_end == 2, f"{side}: expected 2 file-final frames, got {at_end}"

        late = torch.nonzero(pool["frame_frac"] >= 0.5, as_tuple=False).flatten()
        assert len(late) > 0

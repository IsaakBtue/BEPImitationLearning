"""Two-stage reach target for wide crossings (2026-07-03, user-directed design).

_get_reach_target_y (mdp.rewards) targets the MIDPOINT between the robot's stance
and the true crossing point for the first half of the ball's flight time on wide
(|crossing_y - start_y| > 0.6) crossings, then switches to the full crossing point
for the second half. Narrow crossings always target the full point.

Uses fake env objects (same approach as test_live_rsi.py / test_post_save_ball_
release.py) — no mjlab env build. env._ball_crossing_y is set directly to the
already-frozen value a real reset would have produced (_get_ball_crossing_y only
(re)computes it on the just_reset step, so priming it explicitly here decouples
these tests from that separate caching mechanism and tests _get_reach_target_y's
own switching logic in isolation).
"""
import torch

from simple_goalkeeper.mdp.rewards import _get_reach_target_y, _get_ball_crossing_y


class _Scene(dict):
    def __init__(self, env_origins):
        super().__init__()
        self.env_origins = env_origins


class _FakeEnv:
    def __init__(self, env_origins, rel_cross_y, t_flight, episode_step, step_dt=0.02):
        n = env_origins.shape[0]
        self.num_envs = n
        self.device = "cpu"
        self.scene = _Scene(env_origins)
        self._rsi_cross_y = rel_cross_y
        self._ball_t_flight = t_flight
        # Prime the frozen crossing-Y cache directly (bypasses the separate
        # just_reset-gated caching mechanism in _get_ball_crossing_y, which
        # this test file is not exercising).
        self._ball_crossing_y = env_origins[:, 1] + rel_cross_y
        # Not a reset step by default — most tests want the frozen cache above
        # to be returned as-is, not recomputed.
        self.episode_length_buf = torch.full((n,), episode_step, dtype=torch.long)
        self.step_dt = step_dt


def _make_env(rel_cross_y: float, t_flight: float, episode_step: int, step_dt: float = 0.02, n: int = 1):
    env_origins = torch.zeros(n, 3)
    env_origins[:, 1] = 10.0  # nonzero origin Y to catch any world/relative-frame bugs
    rel = torch.full((n,), rel_cross_y)
    t_flight_t = torch.full((n,), t_flight)
    return _FakeEnv(env_origins, rel, t_flight_t, episode_step, step_dt)


def test_narrow_crossing_always_targets_full_point_early():
    # rel = 0.5 <= wide_threshold (0.6) -> narrow, full target regardless of time.
    env = _make_env(rel_cross_y=0.5, t_flight=1.0, episode_step=2)
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball")
    assert torch.allclose(target, full_y)


def test_narrow_crossing_always_targets_full_point_late():
    env = _make_env(rel_cross_y=0.5, t_flight=1.0, episode_step=49)  # near end of flight
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball")
    assert torch.allclose(target, full_y)


def test_wide_crossing_first_half_targets_midpoint():
    # rel = 0.8 > 0.6 -> wide. t_flight=1.0s, step_dt=0.02 -> half = 0.5s = step 25.
    # episode_step=10 -> elapsed=0.2s < 0.5s -> first half -> midpoint.
    env = _make_env(rel_cross_y=0.8, t_flight=1.0, episode_step=10)
    start_y = env.scene.env_origins[0, 1].item()
    full_y = _get_ball_crossing_y(env, "ball")[0].item()
    expected_mid = start_y + (full_y - start_y) / 2.0
    target = _get_reach_target_y(env, "ball")
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)
    # Midpoint must be strictly between start and full target, not equal to either.
    assert abs(expected_mid - start_y) > 1e-6
    assert abs(expected_mid - full_y) > 1e-6


def test_wide_crossing_stays_at_midpoint_past_former_half_flight_boundary():
    # Under the 2026-07-04 hard gate there is no more time-based fallback:
    # elapsed = 30*0.02 = 0.6s (past the OLD t_flight/2 = 0.5s boundary) must
    # STILL target the midpoint, since this fake env has no "robot"/
    # "feet_contact" in scene and can therefore never satisfy the landing gate.
    env = _make_env(rel_cross_y=0.8, t_flight=1.0, episode_step=30)
    start_y = env.scene.env_origins[0, 1].item()
    full_y = _get_ball_crossing_y(env, "ball")[0].item()
    expected_mid = start_y + (full_y - start_y) / 2.0
    target = _get_reach_target_y(env, "ball")
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)


def test_wide_crossing_negative_side_midpoint_direction():
    # Ball crossing to the LEFT (negative rel) — midpoint must move the same
    # direction as the full target, not just match its magnitude.
    env = _make_env(rel_cross_y=-0.9, t_flight=1.0, episode_step=2)
    start_y = env.scene.env_origins[0, 1].item()
    full_y = _get_ball_crossing_y(env, "ball")[0].item()
    target = _get_reach_target_y(env, "ball")[0].item()
    assert full_y < start_y  # sanity: crossing is indeed to the left
    assert target < start_y  # midpoint also left of start
    assert target > full_y  # midpoint is closer to start than the full point


def test_wide_threshold_boundary_exactly_0_6_is_narrow():
    # |rel| == wide_threshold exactly -> NOT wide (strict >), always full target.
    env = _make_env(rel_cross_y=0.6, t_flight=1.0, episode_step=2)
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball")
    assert torch.allclose(target, full_y)


def test_wide_threshold_just_over_0_6_is_wide():
    env = _make_env(rel_cross_y=0.601, t_flight=1.0, episode_step=2)
    start_y = env.scene.env_origins[0, 1].item()
    full_y = _get_ball_crossing_y(env, "ball")[0].item()
    target = _get_reach_target_y(env, "ball")[0].item()
    assert target != full_y
    assert abs(target - (start_y + (full_y - start_y) / 2.0)) < 1e-5


def test_elapsed_time_no_longer_affects_phase_selection():
    # Under the hard gate, elapsed time plays no role at all -- exactly
    # t_flight/2 (the OLD boundary, step 25 at step_dt=0.02) must still be
    # midpoint, not full target.
    env = _make_env(rel_cross_y=0.8, t_flight=1.0, episode_step=25, step_dt=0.02)
    start_y = env.scene.env_origins[0, 1].item()
    full_y = _get_ball_crossing_y(env, "ball")[0].item()
    expected_mid = start_y + (full_y - start_y) / 2.0
    target = _get_reach_target_y(env, "ball")
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)


def test_missing_rsi_cross_y_falls_back_to_crossing_minus_origin():
    # If _rsi_cross_y isn't available, "wide" must still be computed correctly
    # from (full_y - start_y) rather than crashing or silently treating
    # everything as narrow.
    env = _make_env(rel_cross_y=0.8, t_flight=1.0, episode_step=2)
    del env._rsi_cross_y
    start_y = env.scene.env_origins[:, 1].clone()
    target = _get_reach_target_y(env, "ball")
    expected_mid = start_y[0].item() + 0.4  # (0.8)/2
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)


def test_mixed_batch_wide_narrow_independent_of_elapsed_time():
    # Four envs in one call: narrow / wide (early step) / wide (late step, past
    # the OLD t_flight/2 boundary) / wide-negative (early step). Under the hard
    # gate elapsed time is irrelevant -- both wide envs must behave identically
    # regardless of episode_step, since neither can ever land (no "robot"/
    # "feet_contact" in this fake env's scene).
    n = 4
    env_origins = torch.zeros(n, 3)
    env_origins[:, 1] = 5.0
    rel_cross_y = torch.tensor([0.3, 0.9, 0.9, -0.9])
    t_flight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    env = _FakeEnv(env_origins, rel_cross_y, t_flight, episode_step=2)
    env.episode_length_buf = torch.tensor([10, 10, 30, 10], dtype=torch.long)

    start_y = env.scene.env_origins[:, 1]
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball")

    # env 0: narrow -> full target.
    assert torch.allclose(target[0], full_y[0])
    # env 1: wide, early step -> midpoint.
    assert torch.allclose(target[1], start_y[1] + (full_y[1] - start_y[1]) / 2.0, atol=1e-5)
    # env 2: wide, late step (past the OLD t_flight/2 boundary) -> STILL
    # midpoint, since elapsed time no longer matters and this env can never land.
    assert torch.allclose(target[2], start_y[2] + (full_y[2] - start_y[2]) / 2.0, atol=1e-5)
    # env 3: wide negative side, early step -> midpoint, correct direction.
    assert torch.allclose(target[3], start_y[3] + (full_y[3] - start_y[3]) / 2.0, atol=1e-5)
    assert target[3] < start_y[3]

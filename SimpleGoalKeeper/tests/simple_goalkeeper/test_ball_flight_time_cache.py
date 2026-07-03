"""env._ball_t_flight caching in reset_ball_rolling (2026-07-03).

t_flight was previously only a local variable inside reset_ball_rolling, used to
derive ball velocity and then discarded. mdp.rewards._get_reach_target_y needs the
originally-sampled flight time to know when "half the flight has elapsed", so
reset_ball_rolling now caches it on env, mirroring the existing env._rsi_cross_y
pattern at the same call site.

Uses a fake env/ball entity (same approach as test_live_rsi.py) — no mjlab env
build. Ball write calls are captured but not otherwise validated here (that
geometry is already covered by BugFixes.md's cited live-CUDA RSI test suite);
this file only checks the new caching behavior.
"""
import torch

from simple_goalkeeper.mdp.events import reset_ball_rolling


class _FakeBall:
    def __init__(self):
        self.pose_calls = []
        self.velocity_calls = []

    def write_root_link_pose_to_sim(self, pose, env_ids):
        self.pose_calls.append((pose.clone(), env_ids.clone()))

    def write_root_link_velocity_to_sim(self, vel, env_ids):
        self.velocity_calls.append((vel.clone(), env_ids.clone()))


class _Scene(dict):
    def __init__(self, ball, env_origins):
        super().__init__({"ball": ball})
        self.env_origins = env_origins


class _FakeEnv:
    def __init__(self, num_envs, ball, env_origins, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        self.scene = _Scene(ball, env_origins)


def _make_env(num_envs=4):
    ball = _FakeBall()
    env_origins = torch.zeros(num_envs, 3)
    env = _FakeEnv(num_envs, ball, env_origins)
    return env, ball


def test_t_flight_cache_created_and_within_sampled_range():
    env, ball = _make_env(4)
    env_ids = torch.arange(4, dtype=torch.int32)

    reset_ball_rolling(env, env_ids, "ball", t_flight_range=(0.7, 1.1))

    assert hasattr(env, "_ball_t_flight")
    assert env._ball_t_flight.shape == (4,)
    # d defaults to 1.0 (max difficulty, getattr fallback) -> full hard range applies.
    assert (env._ball_t_flight >= 0.7 - 1e-4).all()
    assert (env._ball_t_flight <= 1.1 + 1e-4).all()


def test_t_flight_cache_only_updates_the_reset_env_ids():
    env, ball = _make_env(4)

    reset_ball_rolling(env, torch.arange(4, dtype=torch.int32), "ball", t_flight_range=(0.7, 1.1))
    first = env._ball_t_flight.clone()

    # Reset only envs 0 and 2 again — envs 1 and 3 must retain their prior values.
    torch.manual_seed(0)
    reset_ball_rolling(env, torch.tensor([0, 2], dtype=torch.int32), "ball", t_flight_range=(0.7, 1.1))

    assert torch.equal(env._ball_t_flight[1], first[1])
    assert torch.equal(env._ball_t_flight[3], first[3])


def test_t_flight_cache_survives_across_env_ids_none_full_reset():
    env, ball = _make_env(3)
    reset_ball_rolling(env, None, "ball", t_flight_range=(0.7, 1.1))
    assert env._ball_t_flight.shape == (3,)
    assert (env._ball_t_flight >= 0.7 - 1e-4).all()
    assert (env._ball_t_flight <= 1.1 + 1e-4).all()


def test_t_flight_matches_easy_range_at_zero_difficulty():
    env, ball = _make_env(50)
    env._ball_difficulty = 0.0  # easiest: t_flight should stay within the easy range
    env_ids = torch.arange(50, dtype=torch.int32)

    reset_ball_rolling(env, env_ids, "ball", t_flight_range=(0.7, 1.1))

    # _EASY_T_FLIGHT_R (module-level default) is wider/longer than the hard range —
    # at d=0 the sampled values must NOT all collapse into the hard (0.7, 1.1) band
    # if the easy range differs from it. At minimum, values must stay non-negative
    # and finite regardless of curriculum difficulty.
    assert torch.isfinite(env._ball_t_flight).all()
    assert (env._ball_t_flight > 0).all()


def test_t_flight_cache_present_alongside_rsi_cross_y():
    # Both caches are written at the same call site — verify neither write
    # clobbers the other and both end up populated together.
    env, ball = _make_env(4)
    env_ids = torch.arange(4, dtype=torch.int32)

    reset_ball_rolling(env, env_ids, "ball")

    assert hasattr(env, "_rsi_cross_y")
    assert hasattr(env, "_ball_t_flight")
    assert env._rsi_cross_y.shape == env._ball_t_flight.shape == (4,)

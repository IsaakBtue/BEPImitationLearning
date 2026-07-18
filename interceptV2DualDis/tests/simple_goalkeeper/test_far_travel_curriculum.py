"""Tests for far_travel_curriculum (research-doc recommendation B, redesigned
2026-07-18 to mirror G1's STEP region -- both edges move, not just the outer
one) and its wiring into reset_ball_rolling / reset_ball_rolling_by_region."""
import types

import torch

from simple_goalkeeper.mdp.events import far_travel_curriculum, reset_ball_rolling


class _FakeCfg:
    def __init__(self, params: dict):
        self.params = params


class _FakeEnv:
    def __init__(self, num_envs: int = 8):
        self.num_envs = num_envs
        self.device = "cpu"
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(num_envs)


# G1 ranges_4 (step) width: init=[0.2,1.2], max=[0.0,1.8].
_G1_STEP_INNER_FRAC = 0.2 / 1.8
_G1_STEP_OUTER_FRAC = 1.2 / 1.8


def test_far_travel_curriculum_starts_at_g1_step_proportions():
    env = _FakeEnv()
    env._ball_difficulty = 1.0  # ball_difficulty already saturated, independent
    cfg = _FakeCfg({})
    far_travel_curriculum(cfg, env)
    lo, hi = 0.5, 1.3
    span = hi - lo
    assert abs(env._far_inner - (lo + _G1_STEP_INNER_FRAC * span)) < 1e-9
    assert abs(env._far_outer - (lo + _G1_STEP_OUTER_FRAC * span)) < 1e-9
    # inner starts closer to lo, outer starts closer to hi -- real separation
    # from both the eventual floor/ceiling AND from the near region's own max
    # (which sits at lo, 0.5) from iteration 0, not a degenerate point.
    assert lo < env._far_inner < env._far_outer < hi


def test_far_travel_curriculum_configurable_lo_hi():
    env = _FakeEnv()
    cfg = _FakeCfg({"lo": 0.0, "hi": 1.0})
    far_travel_curriculum(cfg, env)
    assert abs(env._far_inner - _G1_STEP_INNER_FRAC) < 1e-9
    assert abs(env._far_outer - _G1_STEP_OUTER_FRAC) < 1e-9


def test_far_travel_curriculum_only_updates_every_update_interval():
    env = _FakeEnv()
    env.episode_length_buf = torch.full((8,), 150.0)  # cu=3 every window
    cfg = _FakeCfg({"update_interval": 500, "ep_len_divisor": 50, "step_size": 0.004})
    term = far_travel_curriculum(cfg, env)

    env.common_step_counter = 1
    out = term(env, env_ids=torch.arange(8))
    first_inner, first_outer = out["far_inner"].item(), out["far_outer"].item()

    env.common_step_counter = 2  # still inside the same window -> no further update
    out = term(env, env_ids=torch.arange(8))
    assert out["far_inner"].item() == first_inner
    assert out["far_outer"].item() == first_outer

    env.common_step_counter = 501  # a full window has now elapsed since step 1
    out = term(env, env_ids=torch.arange(8))
    assert out["far_inner"].item() < first_inner   # shrinks toward lo
    assert out["far_outer"].item() > first_outer   # grows toward hi


def test_far_travel_curriculum_inner_shrinks_and_outer_grows_together():
    env = _FakeEnv()
    env.episode_length_buf = torch.full((8,), 150.0)  # cu=3 every window
    cfg = _FakeCfg({"update_interval": 500, "ep_len_divisor": 50, "step_size": 0.004})
    term = far_travel_curriculum(cfg, env)

    prev_inner, prev_outer = None, None
    for i in range(1, 6):
        env.common_step_counter = i * 500
        out = term(env, env_ids=torch.arange(8))
        inner, outer = out["far_inner"].item(), out["far_outer"].item()
        if prev_inner is not None:
            assert inner <= prev_inner
            assert outer >= prev_outer
        prev_inner, prev_outer = inner, outer

    assert 0.5 <= prev_inner < prev_outer <= 1.3


def test_far_travel_curriculum_clips_at_lo_and_hi():
    env = _FakeEnv()
    env.episode_length_buf = torch.full((8,), 150.0)
    cfg = _FakeCfg({"update_interval": 1, "ep_len_divisor": 50, "step_size": 0.004})
    term = far_travel_curriculum(cfg, env)
    for i in range(1, 400):
        env.common_step_counter = i
        term(env, env_ids=torch.arange(8))
    assert env._far_inner == 0.5
    assert env._far_outer == 1.3


def test_reset_ball_rolling_far_travel_flag_uses_far_inner_outer_not_ball_difficulty():
    env = _FakeEnv(num_envs=4)
    env._ball_difficulty = 1.0     # full difficulty everywhere else

    import simple_goalkeeper.mdp.events as events_mod  # noqa: F401

    class _Scene(dict):
        env_origins = torch.zeros(4, 3)

    scene = _Scene()

    class _Ball:
        class data:
            root_link_pos_w = torch.zeros(4, 3) + torch.tensor([2.0, 0.0, 0.0])
            root_link_lin_vel_w = torch.zeros(4, 3)

        def write_root_link_pose_to_sim(self, *a, **k):
            pass

        def write_root_link_velocity_to_sim(self, *a, **k):
            pass

    scene["ball"] = _Ball()
    env.scene = scene

    # far_inner/outer NOT at their default full-span values -- a mid-curriculum
    # snapshot, e.g. inner shrunk to 0.55, outer grown to 1.1.
    env._far_inner = 0.55
    env._far_outer = 1.10

    torch.manual_seed(0)
    env_ids = torch.arange(4)
    reset_ball_rolling(
        env,
        env_ids,
        "ball",
        dist_range=(2.0, 2.0),
        y_start_range=(0.0, 0.0),
        y_end_range=(0.5, 1.3),
        t_flight_range=(0.7, 0.7),
        use_far_travel_curriculum=True,
    )
    assert hasattr(env, "_rsi_cross_y")
    # crossing-fraction for x_start=2.0, aim point 0.3m behind goal line: f = 2.0/2.3
    f = 2.0 / 2.3
    cross_y = env._rsi_cross_y.abs()
    # y_end must have been sampled from [0.55, 1.10] (env._far_inner/_far_outer),
    # NOT from [0.5, 1.3] (the full lo/hi span ball_difficulty=1.0 would give).
    assert (cross_y >= 0.55 * f - 1e-6).all()
    assert (cross_y <= 1.10 * f + 1e-6).all()


def test_reset_ball_rolling_far_travel_flag_defaults_to_full_span_when_curriculum_absent():
    """Matches env._ball_difficulty's own play/eval default (1.0 = full
    difficulty) when the CurriculumManager hasn't run (e.g. play mode)."""
    env = _FakeEnv(num_envs=4)
    env._ball_difficulty = 1.0

    class _Scene(dict):
        env_origins = torch.zeros(4, 3)

    scene = _Scene()

    class _Ball:
        class data:
            root_link_pos_w = torch.zeros(4, 3) + torch.tensor([2.0, 0.0, 0.0])
            root_link_lin_vel_w = torch.zeros(4, 3)

        def write_root_link_pose_to_sim(self, *a, **k):
            pass

        def write_root_link_velocity_to_sim(self, *a, **k):
            pass

    scene["ball"] = _Ball()
    env.scene = scene
    # env._far_inner/_far_outer deliberately never set.

    torch.manual_seed(0)
    env_ids = torch.arange(4)
    reset_ball_rolling(
        env,
        env_ids,
        "ball",
        dist_range=(2.0, 2.0),
        y_start_range=(0.0, 0.0),
        y_end_range=(0.5, 1.3),
        t_flight_range=(0.7, 0.7),
        use_far_travel_curriculum=True,
    )
    f = 2.0 / 2.3
    cross_y = env._rsi_cross_y.abs()
    assert (cross_y >= 0.5 * f - 1e-6).all()
    assert (cross_y <= 1.3 * f + 1e-6).all()

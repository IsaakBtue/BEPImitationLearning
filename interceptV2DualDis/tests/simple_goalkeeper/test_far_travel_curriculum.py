"""Tests for far_travel_curriculum (research-doc recommendation B) and its
wiring into reset_ball_rolling / reset_ball_rolling_by_region."""
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


def test_far_travel_curriculum_starts_at_zero_and_is_independent_of_ball_difficulty():
    env = _FakeEnv()
    env._ball_difficulty = 1.0  # ball_difficulty already saturated
    cfg = _FakeCfg({})
    far_travel_curriculum(cfg, env)
    assert env._far_travel_frac == 0.0


def test_far_travel_curriculum_only_updates_every_update_interval():
    # _last_update starts at -(update_interval), matching reward_curriculum_ep_len /
    # ball_difficulty_curriculum's own "fire immediately on first call" convention.
    env = _FakeEnv()
    env.episode_length_buf = torch.full((8,), 150.0)  # cu=3 every window
    cfg = _FakeCfg({"update_interval": 500, "ep_len_divisor": 50, "step_size": 0.004})
    term = far_travel_curriculum(cfg, env)

    env.common_step_counter = 1
    out = term(env, env_ids=torch.arange(8))
    first_value = out["far_travel_frac"].item()
    assert first_value > 0.0  # first call always fires

    env.common_step_counter = 2  # still inside the same window -> no further update
    out = term(env, env_ids=torch.arange(8))
    assert out["far_travel_frac"].item() == first_value

    env.common_step_counter = 501  # a full window has now elapsed since step 1
    out = term(env, env_ids=torch.arange(8))
    assert out["far_travel_frac"].item() > first_value


def test_far_travel_curriculum_ramps_slower_than_ball_difficulty_default():
    """Same cu history, same number of updates: far_travel_frac (step_size
    0.004) must lag ball_difficulty (step_size 0.01, tested separately in
    events.py's own docstring/history) by construction."""
    env = _FakeEnv()
    env.episode_length_buf = torch.full((8,), 150.0)  # cu=3 every window
    cfg = _FakeCfg({"update_interval": 500, "ep_len_divisor": 50, "step_size": 0.004})
    term = far_travel_curriculum(cfg, env)

    for i in range(1, 6):
        env.common_step_counter = i * 500
        term(env, env_ids=torch.arange(8))

    # 5 updates * step_size(0.004) * cu(3) = 0.06
    assert abs(env._far_travel_frac - 0.06) < 1e-9
    # Same cu history under ball_difficulty's own default step_size (0.01)
    # would reach 5*0.01*3 = 0.15 -- far_travel_frac must stay below that.
    assert env._far_travel_frac < 0.15


def test_far_travel_curriculum_clips_at_one():
    env = _FakeEnv()
    env.episode_length_buf = torch.full((8,), 150.0)
    cfg = _FakeCfg({"update_interval": 1, "ep_len_divisor": 50, "step_size": 0.004})
    term = far_travel_curriculum(cfg, env)
    for i in range(1, 400):
        env.common_step_counter = i
        term(env, env_ids=torch.arange(8))
    assert env._far_travel_frac == 1.0


def test_reset_ball_rolling_far_travel_flag_uses_far_travel_frac_not_ball_difficulty():
    env = _FakeEnv(num_envs=4)
    env.scene = types.SimpleNamespace(
        env_origins=torch.zeros(4, 3),
        __getitem__=lambda self, name: types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_link_pos_w=torch.zeros(4, 3),
                root_link_lin_vel_w=torch.zeros(4, 3),
            ),
            write_root_link_pose_to_sim=lambda *a, **k: None,
            write_root_link_velocity_to_sim=lambda *a, **k: None,
        ),
    )
    env._ball_difficulty = 1.0     # full difficulty everywhere else
    env._far_travel_frac = 0.0     # but far-travel curriculum hasn't started

    # left_far-style one-sided range: (0.5, 1.3). At ball_difficulty=1.0 the
    # old behavior would sample the FULL [0.5, 1.3] band; with the curriculum
    # flag it must instead collapse to the inner edge (0.5) since
    # far_travel_frac=0.0 -> outer = lo + (hi-lo)*0 = lo.
    env.scene.__setitem__ = None
    import simple_goalkeeper.mdp.events as events_mod

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
    y_end = env._y_end_cache if hasattr(env, "_y_end_cache") else None
    # reset_ball_rolling doesn't expose y_end directly; assert indirectly via
    # the cached crossing-y state it does set, which is monotonic in y_end.
    assert hasattr(env, "_rsi_cross_y")
    # With far_travel_frac=0.0, the sampled |y_end| must never exceed the
    # region's own inner edge (0.5) -- i.e. the outer band never opens up
    # despite ball_difficulty=1.0.
    assert env._rsi_cross_y.abs().max().item() <= 0.5 + 1e-6

"""Post-save release gate: actor's ball obs zeroes once _ball_is_behind fires.

Uses fake env objects (same approach as test_live_rsi.py) — no mjlab env build.
Identity root quaternion (w,x,y,z)=(1,0,0,0) so body frame == world frame and
expected values are just (ball - robot) XY offsets.
"""
import torch

from simple_goalkeeper.mdp.observations import ball_pos_xy_b


class _FakeEntityData:
    def __init__(self, pos, quat):
        self.root_link_pos_w = pos
        self.root_link_quat_w = quat


class _FakeEntity:
    def __init__(self, pos, quat=None):
        n = pos.shape[0]
        if quat is None:
            quat = torch.zeros(n, 4)
            quat[:, 0] = 1.0  # identity (w, x, y, z)
        self.data = _FakeEntityData(pos, quat)


class _FakeScene(dict):
    def __init__(self, entities, env_origins):
        super().__init__(entities)
        self.env_origins = env_origins


class _FakeEnv:
    def __init__(self, robot_pos, ball_pos, env_origins):
        n = robot_pos.shape[0]
        self.num_envs = n
        self.device = "cpu"
        self.scene = _FakeScene(
            {"robot": _FakeEntity(robot_pos), "ball": _FakeEntity(ball_pos)},
            env_origins,
        )


def _make_env(ball_x_local=1.5):
    """Robot at origin of each env; ball ball_x_local ahead of the goal line."""
    n = 4
    env_origins = torch.zeros(n, 3)
    robot_pos = env_origins.clone()
    ball_pos = env_origins.clone()
    ball_pos[:, 0] = ball_x_local
    ball_pos[:, 1] = 0.3
    return _FakeEnv(robot_pos, ball_pos, env_origins)


def test_visible_before_save_even_with_gate_enabled():
    env = _make_env(ball_x_local=1.5)  # in front, no flags set
    out = ball_pos_xy_b(env, "ball", always_visible=True, hide_when_behind=True)
    expected = torch.tensor([[1.5, 0.3]] * 4)
    assert torch.allclose(out, expected, atol=1e-5)


def test_zeroed_when_sb_flag_set():
    env = _make_env(ball_x_local=1.5)
    env._sb_flag = torch.ones(4, dtype=torch.bool)
    out = ball_pos_xy_b(env, "ball", always_visible=True, hide_when_behind=True)
    assert torch.equal(out, torch.zeros(4, 2))


def test_zeroed_when_softstop_flag_set():
    env = _make_env(ball_x_local=1.5)
    env._softstop_flag = torch.ones(4, dtype=torch.bool)
    out = ball_pos_xy_b(env, "ball", always_visible=True, hide_when_behind=True)
    assert torch.equal(out, torch.zeros(4, 2))


def test_zeroed_when_ball_crossed_goal_line():
    env = _make_env(ball_x_local=-0.2)  # behind goal line, no flags
    out = ball_pos_xy_b(env, "ball", always_visible=True, hide_when_behind=True)
    assert torch.equal(out, torch.zeros(4, 2))


def test_mixed_batch_only_saved_envs_zeroed():
    env = _make_env(ball_x_local=1.5)
    env._sb_flag = torch.tensor([True, False, True, False])
    out = ball_pos_xy_b(env, "ball", always_visible=True, hide_when_behind=True)
    assert torch.equal(out[0], torch.zeros(2))
    assert torch.equal(out[2], torch.zeros(2))
    assert torch.allclose(out[1], torch.tensor([1.5, 0.3]), atol=1e-5)
    assert torch.allclose(out[3], torch.tensor([1.5, 0.3]), atol=1e-5)


def test_default_hide_when_behind_false_is_backward_compatible():
    env = _make_env(ball_x_local=1.5)
    env._sb_flag = torch.ones(4, dtype=torch.bool)  # saved — but gate off
    out = ball_pos_xy_b(env, "ball", always_visible=True)
    expected = torch.tensor([[1.5, 0.3]] * 4)
    assert torch.allclose(out, expected, atol=1e-5)


def test_actor_ball_term_has_release_gate_in_train_and_play():
    from simple_goalkeeper.tasks.goalkeeper_env_cfg import goalkeeper_env_cfg

    for play in (False, True):
        cfg = goalkeeper_env_cfg(play=play)
        actor_term = cfg.observations["actor"].terms["ball_pos_b"]
        assert actor_term.params["always_visible"] is True
        assert actor_term.params["hide_when_behind"] is True


def test_critic_ball_terms_stay_ungated():
    from simple_goalkeeper.tasks.goalkeeper_env_cfg import goalkeeper_env_cfg

    cfg = goalkeeper_env_cfg(play=False)
    critic = cfg.observations["critic"].terms
    assert critic["ball_pos_b"].params["always_visible"] is True
    assert "hide_when_behind" not in critic["ball_pos_b"].params
    assert critic["ball_vel_b"].params["always_visible"] is True
    assert "hide_when_behind" not in critic["ball_vel_b"].params

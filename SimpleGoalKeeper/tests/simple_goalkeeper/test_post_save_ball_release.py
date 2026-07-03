"""Post-save release gate v2 (G1-faithful): actor's ball obs zeroes when the
ball is behind the torso (x_body < 0.05, G1 legged_robot.py:401 front edge) or
once the visibility window since ball launch closes (G1 catchstep > 0 analog,
window sized to outlast SGK's max 1.3 s flight time). Noise is applied BEFORE
the mask (G1 legged_robot.py:425-426 ordering), so a hidden ball is exact
zeros even in noise mode. Neither condition can fire while the ball is still
approaching — full visibility during the save is preserved by construction.

Uses fake env objects (same approach as test_live_rsi.py) — no mjlab env build.
Identity root quaternion (w,x,y,z)=(1,0,0,0) so body frame == world frame and
expected values are just (ball - robot) XY offsets.
"""
import torch

from simple_goalkeeper.mdp.observations import ball_pos_xy_b

_GATE = {"always_visible": True, "hide_behind_torso": True, "hide_after_steps": 75}


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
    def __init__(self, robot_pos, ball_pos, env_origins, episode_step=10):
        n = robot_pos.shape[0]
        self.num_envs = n
        self.device = "cpu"
        self.episode_length_buf = torch.full((n,), episode_step, dtype=torch.long)
        self.scene = _FakeScene(
            {"robot": _FakeEntity(robot_pos), "ball": _FakeEntity(ball_pos)},
            env_origins,
        )


def _make_env(ball_x_local=1.5, episode_step=10):
    """Robot at origin of each env; ball ball_x_local ahead of the torso."""
    n = 4
    env_origins = torch.zeros(n, 3)
    robot_pos = env_origins.clone()
    ball_pos = env_origins.clone()
    ball_pos[:, 0] = ball_x_local
    ball_pos[:, 1] = 0.3
    return _FakeEnv(robot_pos, ball_pos, env_origins, episode_step=episode_step)


def test_visible_in_front_within_window():
    env = _make_env(ball_x_local=1.5, episode_step=10)
    out = ball_pos_xy_b(env, "ball", **_GATE)
    expected = torch.tensor([[1.5, 0.3]] * 4)
    assert torch.allclose(out, expected, atol=1e-5)


def test_hidden_when_behind_torso():
    env = _make_env(ball_x_local=-0.2, episode_step=10)
    out = ball_pos_xy_b(env, "ball", **_GATE)
    assert torch.equal(out, torch.zeros(4, 2))


def test_behind_torso_edge_is_0_05():
    # G1 front edge: visible requires x_body > 0.05 (legged_robot.py:401).
    env = _make_env(ball_x_local=0.04, episode_step=10)
    assert torch.equal(ball_pos_xy_b(env, "ball", **_GATE), torch.zeros(4, 2))
    env = _make_env(ball_x_local=0.06, episode_step=10)
    assert not torch.equal(ball_pos_xy_b(env, "ball", **_GATE), torch.zeros(4, 2))


def test_hidden_after_window_closes():
    env = _make_env(ball_x_local=1.5, episode_step=76)
    out = ball_pos_xy_b(env, "ball", **_GATE)
    assert torch.equal(out, torch.zeros(4, 2))


def test_visible_at_window_boundary():
    # Visible while step <= hide_after_steps (hidden strictly after).
    env = _make_env(ball_x_local=1.5, episode_step=75)
    out = ball_pos_xy_b(env, "ball", **_GATE)
    expected = torch.tensor([[1.5, 0.3]] * 4)
    assert torch.allclose(out, expected, atol=1e-5)


def test_save_flags_do_not_hide():
    # v2 gate is torso-edge + window only; the v1 latched-flag trigger is gone.
    env = _make_env(ball_x_local=1.5, episode_step=10)
    env._sb_flag = torch.ones(4, dtype=torch.bool)
    env._softstop_flag = torch.ones(4, dtype=torch.bool)
    out = ball_pos_xy_b(env, "ball", **_GATE)
    expected = torch.tensor([[1.5, 0.3]] * 4)
    assert torch.allclose(out, expected, atol=1e-5)


def test_mixed_batch_per_env_window():
    env = _make_env(ball_x_local=1.5)
    env.episode_length_buf = torch.tensor([10, 80, 10, 80], dtype=torch.long)
    out = ball_pos_xy_b(env, "ball", **_GATE)
    expected = torch.tensor([[1.5, 0.3], [0.0, 0.0], [1.5, 0.3], [0.0, 0.0]])
    assert torch.allclose(out, expected, atol=1e-5)


def test_noise_applied_before_mask_hidden_is_exact_zero():
    # G1 ordering (legged_robot.py:425-426): noise first, then mask — a hidden
    # ball must be exactly 0.0, never noise around zero.
    env = _make_env(ball_x_local=-0.2, episode_step=10)
    out = ball_pos_xy_b(env, "ball", noise_scale=0.05, **_GATE)
    assert torch.equal(out, torch.zeros(4, 2))


def test_noise_applied_to_visible_ball():
    torch.manual_seed(0)
    env = _make_env(ball_x_local=1.5, episode_step=10)
    out = ball_pos_xy_b(env, "ball", noise_scale=0.05, **_GATE)
    expected = torch.tensor([[1.5, 0.3]] * 4)
    assert torch.all((out - expected).abs() <= 0.05 + 1e-6)
    assert not torch.allclose(out, expected, atol=1e-7)  # noise actually added


def test_actor_ball_term_has_v2_gate_in_train_and_play():
    from simple_goalkeeper.tasks.goalkeeper_env_cfg import goalkeeper_env_cfg

    for play in (False, True):
        cfg = goalkeeper_env_cfg(play=play)
        actor_term = cfg.observations["actor"].terms["ball_pos_b"]
        assert actor_term.params["always_visible"] is True
        assert actor_term.params["hide_behind_torso"] is True
        assert actor_term.params["hide_after_steps"] == 75
        # In-term noise replaces manager noise (G1 noise-before-mask ordering);
        # play mode disables it, mirroring enable_corruption=False.
        assert actor_term.noise is None
        expected_noise = 0.0 if play else 0.05
        assert actor_term.params["noise_scale"] == expected_noise


def test_critic_ball_terms_stay_ungated():
    from simple_goalkeeper.tasks.goalkeeper_env_cfg import goalkeeper_env_cfg

    cfg = goalkeeper_env_cfg(play=False)
    critic = cfg.observations["critic"].terms
    assert critic["ball_pos_b"].params["always_visible"] is True
    assert "hide_behind_torso" not in critic["ball_pos_b"].params
    assert critic["ball_vel_b"].params["always_visible"] is True
    assert "hide_behind_torso" not in critic["ball_vel_b"].params

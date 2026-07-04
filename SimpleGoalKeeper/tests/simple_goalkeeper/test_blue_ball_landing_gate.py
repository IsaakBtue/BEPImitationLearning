"""Landing gate on the two-stage footreach schedule (2026-07-04, user-directed design).

_get_reach_target_y now requires the assigned foot to physically land near the
blue (midpoint) target before advancing early to the green (full) target on wide
crossings, rather than advancing on elapsed time alone. See rewards._get_reach_target_y
and docs/superpowers/specs/2026-07-04-blue-ball-landing-gate-design.md.

Uses the same fake env/robot/ball approach as test_footreach_two_stage_wiring.py,
extended with a fake "feet_contact" ContactSensor (.data.found, shape (N, 8)).
"""
import types

import torch

from simple_goalkeeper.mdp.rewards import _get_reach_target_y, _get_ball_crossing_y


class _EntityData:
    pass


class _Entity:
    def __init__(self, **kwargs):
        self.data = _EntityData()
        for k, v in kwargs.items():
            setattr(self.data, k, v)


class _Scene(dict):
    def __init__(self, entities, env_origins):
        super().__init__(entities)
        self.env_origins = env_origins


class _FakeEnv:
    def __init__(self, robot, ball, feet_contact, env_origins, episode_step, rel_cross_y, t_flight, step_dt=0.02):
        n = env_origins.shape[0]
        self.num_envs = n
        self.device = "cpu"
        self.scene = _Scene({"robot": robot, "ball": ball, "feet_contact": feet_contact}, env_origins)
        self.episode_length_buf = torch.full((n,), episode_step, dtype=torch.long)
        self.step_dt = step_dt
        self._rsi_cross_y = rel_cross_y
        self._ball_t_flight = t_flight
        self._ball_crossing_y = env_origins[:, 1] + rel_cross_y


def _feet_cfg():
    cfg = types.SimpleNamespace()
    cfg.name = "robot"
    cfg.body_ids = [0, 1]
    return cfg


def _make_contact(found_left: bool, found_right: bool):
    # 8 geoms: 0-3 left, 4-7 right (feet_contact geom layout; see feet_slippage/
    # penalize_sharpcontact in rewards.py for the same convention).
    found = torch.zeros(1, 8)
    found[:, :4] = 1.0 if found_left else 0.0
    found[:, 4:] = 1.0 if found_right else 0.0
    return _Entity(found=found)


def _make_env(foot_y: float, rel_cross_y: float, episode_step: int,
              found_left: bool, found_right: bool, t_flight: float = 1.0):
    n = 1
    env_origins = torch.zeros(n, 3)  # start_y=0, goal_x_w=0, floor_z=0

    robot = _Entity(
        body_link_pos_w=torch.tensor([[[0.0, foot_y, 0.0], [0.0, foot_y, 0.0]]]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.8]]),
        root_link_lin_vel_w=torch.zeros(n, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
    )
    ball = _Entity(root_link_pos_w=torch.tensor([[2.0, rel_cross_y, 0.11]]))
    feet_contact = _make_contact(found_left, found_right)

    rel = torch.tensor([rel_cross_y])
    t_flight_t = torch.tensor([t_flight])
    return _FakeEnv(robot, ball, feet_contact, env_origins, episode_step, rel, t_flight_t)


def test_landing_never_fires_without_prior_airborne():
    # rel_cross_y=0.9 (wide) -> crossing_y(0.9) > origin_y(0.0) -> assigned foot = left (idx 0).
    # Foot planted exactly at the midpoint (0.45) AND in contact from the very
    # first check -- it was never airborne, so landing must not fire.
    env = _make_env(foot_y=0.45, rel_cross_y=0.9, episode_step=2, found_left=True, found_right=False)
    target = _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    full_y = _get_ball_crossing_y(env, "ball")[0].item()
    expected_mid = (0.0 + full_y) / 2.0
    assert not env._blue_landed[0].item()
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)


def test_landing_fires_after_airborne_then_contact_within_radius():
    # Step A: foot airborne (found=False), far from target.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=5, found_left=False, found_right=False)
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert env._blue_was_airborne[0].item()
    assert not env._blue_landed[0].item()

    # Step B (same env, later step): foot now in contact, planted at the midpoint.
    env.episode_length_buf[:] = 6  # elapsed = 0.12s, still < t_flight/2 = 0.5s
    env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.45, 0.0], [0.0, 0.45, 0.0]]])
    env.scene["feet_contact"].data.found[:, :4] = 1.0  # left foot now in contact

    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert env._blue_landed[0].item()
    # Landing switches to the full target EARLY, even though elapsed < t_flight/2.
    assert torch.allclose(target, full_y)


def test_landing_gate_fallback_advances_at_half_flight_time_when_never_landed():
    # Foot stays airborne the whole time (never lands near the target) -- the
    # schedule must still fall back to the full target once elapsed >= t_flight/2,
    # exactly like the pre-landing-gate behavior.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=30, found_left=False, found_right=False)
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert not env._blue_landed[0].item()
    assert torch.allclose(target, full_y)


def test_landing_far_from_target_does_not_count():
    # Foot goes airborne then lands, but far from the blue target -- must not count.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=5, found_left=False, found_right=False)
    _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    env.episode_length_buf[:] = 6
    env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]])  # far away
    env.scene["feet_contact"].data.found[:, :4] = 1.0
    target = _get_reach_target_y(env, "ball", asset_cfg=_feet_cfg())
    assert not env._blue_landed[0].item()
    full_y = _get_ball_crossing_y(env, "ball")[0].item()
    expected_mid = (0.0 + full_y) / 2.0
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)


def test_reach_target_y_without_robot_or_sensor_in_scene_is_unaffected():
    # Backward compatibility: envs with no "robot"/"feet_contact" in scene (the
    # existing test_reach_target_two_stage.py fake envs) must behave exactly as
    # before -- landing detection silently no-ops via the KeyError guard.
    class _BareScene(dict):
        def __init__(self, env_origins):
            super().__init__()
            self.env_origins = env_origins

    class _BareEnv:
        def __init__(self):
            n = 1
            env_origins = torch.zeros(n, 3)
            env_origins[:, 1] = 10.0
            self.num_envs = n
            self.device = "cpu"
            self.scene = _BareScene(env_origins)
            self._rsi_cross_y = torch.tensor([0.9])
            self._ball_t_flight = torch.tensor([1.0])
            self._ball_crossing_y = env_origins[:, 1] + 0.9
            self.episode_length_buf = torch.full((n,), 10, dtype=torch.long)
            self.step_dt = 0.02

    env = _BareEnv()
    full_y = _get_ball_crossing_y(env, "ball")
    target = _get_reach_target_y(env, "ball")
    expected_mid = 10.0 + (full_y[0].item() - 10.0) / 2.0
    assert torch.allclose(target, torch.tensor([expected_mid]), atol=1e-5)


def test_blue_ball_landed_fires_once_per_episode():
    from simple_goalkeeper.mdp.rewards import blue_ball_landed

    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=5, found_left=False, found_right=False)
    r0 = blue_ball_landed(env, "ball", asset_cfg=_feet_cfg())
    assert r0.item() == 0.0

    env.episode_length_buf[:] = 6
    env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.45, 0.0], [0.0, 0.45, 0.0]]])
    env.scene["feet_contact"].data.found[:, :4] = 1.0
    r1 = blue_ball_landed(env, "ball", asset_cfg=_feet_cfg())
    assert r1.item() == 1.0  # fires exactly on the landing step

    env.episode_length_buf[:] = 7
    r2 = blue_ball_landed(env, "ball", asset_cfg=_feet_cfg())
    assert r2.item() == 0.0  # already paid, does not fire again


def test_blue_ball_landed_resets_on_new_episode():
    from simple_goalkeeper.mdp.rewards import blue_ball_landed

    env = _make_env(foot_y=0.45, rel_cross_y=0.9, episode_step=6, found_left=True, found_right=False)
    env._blue_landed = torch.ones(1, dtype=torch.bool)
    env._blue_was_airborne = torch.ones(1, dtype=torch.bool)
    env._blue_landed_bonus_flag = torch.ones(1, dtype=torch.bool)  # already paid last episode

    env.episode_length_buf[:] = 1  # new episode (reset step)
    r = blue_ball_landed(env, "ball", asset_cfg=_feet_cfg())
    assert not env._blue_landed_bonus_flag[0].item()
    assert not env._blue_landed[0].item()  # never airborne THIS episode -- must not land for free
    assert r.item() == 0.0

"""stopball/softstop are gated on the blue-ball landing latch for wide crossings
(2026-07-05), closing the gap where the two-stage waypoint (_get_reach_target_y,
env._blue_landed) could be skipped entirely and the policy still collected the full
save reward by beelining straight for the true crossing point -- stopball/softstop
never referenced the two-stage state at all before this fix, only the much smaller
footreach/blue_ball_landed shaping rewards did. See rewards.stopball/softstop
docstrings and SimpleGoalKeeper/docs/BugFixes.md 2026-07-05.

Uses the same fake env/robot/ball/feet_contact approach as
test_blue_ball_landing_gate.py.
"""
import types

import torch

from simple_goalkeeper.mdp.rewards import stopball, softstop


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
    found = torch.zeros(1, 8)
    found[:, :4] = 1.0 if found_left else 0.0
    found[:, 4:] = 1.0 if found_right else 0.0
    return _Entity(found=found)


def _make_env(foot_y: float, rel_cross_y: float, episode_step: int,
              found_left: bool, found_right: bool, ball_x: float, ball_x_vel: float,
              t_flight: float = 1.0):
    n = 1
    env_origins = torch.zeros(n, 3)  # start_y=0, goal_x_w=0, floor_z=0

    robot = _Entity(
        body_link_pos_w=torch.tensor([[[0.0, foot_y, 0.0], [0.0, foot_y, 0.0]]]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.8]]),
        root_link_lin_vel_w=torch.zeros(n, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
    )
    ball = _Entity(
        root_link_pos_w=torch.tensor([[ball_x, rel_cross_y, 0.11]]),
        root_link_lin_vel_w=torch.tensor([[ball_x_vel, 0.0, 0.0]]),
    )
    feet_contact = _make_contact(found_left, found_right)

    rel = torch.tensor([rel_cross_y])
    t_flight_t = torch.tensor([t_flight])
    return _FakeEnv(robot, ball, feet_contact, env_origins, episode_step, rel, t_flight_t)


# --- stopball ---

def test_stopball_blocked_on_wide_crossing_without_landing():
    # Wide crossing (rel_cross_y=0.9), assigned foot never airborne/landed at the
    # blue midpoint. Ball still deflects hard (delta_vx > threshold next step) --
    # stopball must NOT fire even though the physical save condition is met.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=2,
                     found_left=True, found_right=False, ball_x=0.0, ball_x_vel=-2.0)
    stopball(env, "ball")  # establishes _sb_init_vx baseline

    env.episode_length_buf[:] = 3
    env.scene["ball"].data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.0]])  # delta_vx=2.0
    reward = stopball(env, "ball")
    assert reward.item() == 0.0
    assert env._sb_flag.item() is False


def test_stopball_fires_on_wide_crossing_after_landing():
    # Same wide crossing, but the assigned foot goes airborne then lands at the
    # blue midpoint (0.45) before the deflection -- landing_ok becomes True, so
    # stopball fires normally once delta_vx crosses the threshold.
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=2,
                     found_left=True, found_right=False, ball_x=0.0, ball_x_vel=-2.0)
    stopball(env, "ball")  # establishes _sb_init_vx baseline

    env.episode_length_buf[:] = 3
    env.scene["feet_contact"].data.found[:, :4] = 0.0  # left foot airborne
    stopball(env, "ball")

    env.episode_length_buf[:] = 4
    env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.45, 0.0], [0.0, 0.45, 0.0]]])
    env.scene["feet_contact"].data.found[:, :4] = 1.0  # left foot lands at the midpoint
    env.scene["ball"].data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.0]])  # delta_vx=2.0
    reward = stopball(env, "ball")
    assert reward.item() == 1.0
    assert env._sb_flag.item() is True


def test_stopball_unaffected_on_narrow_crossing():
    # Narrow crossing (rel_cross_y=0.2 <= wide_threshold 0.5) -- no waypoint to
    # skip, so stopball fires as before with no landing ever recorded.
    env = _make_env(foot_y=0.0, rel_cross_y=0.2, episode_step=2,
                     found_left=False, found_right=False, ball_x=0.0, ball_x_vel=-2.0)
    stopball(env, "ball")  # establishes _sb_init_vx baseline

    env.episode_length_buf[:] = 3
    env.scene["ball"].data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.0]])  # delta_vx=2.0
    reward = stopball(env, "ball")
    assert reward.item() == 1.0
    assert env._sb_flag.item() is True


# --- softstop ---

def test_softstop_blocked_on_wide_crossing_without_landing():
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=2,
                     found_left=True, found_right=False, ball_x=0.0, ball_x_vel=-2.0)

    env.episode_length_buf[:] = 3
    env.scene["ball"].data.root_link_lin_vel_w = torch.tensor([[0.5, 0.0, 0.0]])  # reversed, > threshold
    reward = softstop(env, "ball")
    assert reward.item() == 0.0
    assert env._softstop_flag.item() is False


def test_softstop_fires_on_wide_crossing_after_landing():
    env = _make_env(foot_y=0.0, rel_cross_y=0.9, episode_step=2,
                     found_left=True, found_right=False, ball_x=0.0, ball_x_vel=-2.0)

    env.episode_length_buf[:] = 3
    env.scene["feet_contact"].data.found[:, :4] = 0.0  # left foot airborne
    softstop(env, "ball")

    env.episode_length_buf[:] = 4
    env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.45, 0.0], [0.0, 0.45, 0.0]]])
    env.scene["feet_contact"].data.found[:, :4] = 1.0  # left foot lands at the midpoint
    env.scene["ball"].data.root_link_lin_vel_w = torch.tensor([[0.5, 0.0, 0.0]])
    reward = softstop(env, "ball")
    assert reward.item() == 1.0
    assert env._softstop_flag.item() is True


def test_softstop_unaffected_on_narrow_crossing():
    env = _make_env(foot_y=0.0, rel_cross_y=0.2, episode_step=2,
                     found_left=False, found_right=False, ball_x=0.0, ball_x_vel=-2.0)

    env.episode_length_buf[:] = 3
    env.scene["ball"].data.root_link_lin_vel_w = torch.tensor([[0.5, 0.0, 0.0]])
    reward = softstop(env, "ball")
    assert reward.item() == 1.0
    assert env._softstop_flag.item() is True

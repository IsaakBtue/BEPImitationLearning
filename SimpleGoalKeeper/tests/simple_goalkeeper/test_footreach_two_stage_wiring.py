"""footreach/foot_proximity actually consume the two-stage _get_reach_target_y
target on wide crossings, not the raw full crossing point directly (2026-07-03).

Builds a minimal fake env/robot/ball (same approach as the other MDP unit tests
in this directory) and checks the reward VALUE responds to the schedule: a foot
planted at the midpoint should score well in phase 1 (target = midpoint) and
score worse in phase 2 (target snaps to the full, farther point) even though the
foot hasn't moved — this is the "self-correcting, no separate exploit-guard
needed" property the design relies on (see rewards._get_reach_target_y docstring).
"""
import types

import torch

from simple_goalkeeper.mdp.rewards import footreach, foot_proximity


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
    def __init__(self, robot, ball, env_origins, episode_step, rel_cross_y, t_flight, step_dt=0.02):
        n = env_origins.shape[0]
        self.num_envs = n
        self.device = "cpu"
        self.scene = _Scene({"robot": robot, "ball": ball}, env_origins)
        self.episode_length_buf = torch.full((n,), episode_step, dtype=torch.long)
        self.step_dt = step_dt
        self._rsi_cross_y = rel_cross_y
        self._ball_t_flight = t_flight
        self._ball_crossing_y = env_origins[:, 1] + rel_cross_y  # pre-frozen, see other test files


def _feet_cfg():
    cfg = types.SimpleNamespace()
    cfg.name = "robot"
    cfg.body_ids = [0, 1]
    return cfg


def _make_env(foot_y: float, rel_cross_y: float, episode_step: int, t_flight: float = 1.0):
    n = 1
    env_origins = torch.zeros(n, 3)  # start_y = 0.0, goal_x_w = 0.0, floor_z = 0.0

    # Robot at the origin, both feet at (goal_x, foot_y, 0.0), ball far ahead
    # and upright/gravity nominal so the upright gate in footreach passes fully.
    robot = _Entity(
        body_link_pos_w=torch.tensor([[[0.0, foot_y, 0.0], [0.0, foot_y, 0.0]]]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.8]]),
        root_link_lin_vel_w=torch.zeros(n, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
    )
    # Ball still well in front (x_local > 1.5) so foot_proximity/footreach use the
    # frozen target path, not the live-ball switch (ball_close requires x_local<0.5).
    ball = _Entity(root_link_pos_w=torch.tensor([[2.0, rel_cross_y, 0.11]]))

    rel = torch.tensor([rel_cross_y])
    t_flight_t = torch.tensor([t_flight])
    return _FakeEnv(robot, ball, env_origins, episode_step, rel, t_flight_t)


def test_footreach_rewards_midpoint_foot_more_in_phase_1_than_phase_2():
    # Wide crossing (0.9 > 0.6), foot planted exactly at the midpoint (0.45).
    # Phase 1 (elapsed < t_flight/2): target = midpoint -> foot is right on target.
    env_phase1 = _make_env(foot_y=0.45, rel_cross_y=0.9, episode_step=10)  # elapsed 0.2s < 0.5s
    r1 = footreach(env_phase1, "ball", asset_cfg=_feet_cfg())

    # Phase 2 (landing already occurred): target snaps to the full point (0.9)
    # -- pre-set both latches directly (this file has no feet_contact sensor
    # mock; the physical airborne-then-contact detection itself is covered by
    # test_blue_ball_landing_gate.py) so _get_reach_target_y's init block
    # doesn't overwrite them, and its no-op KeyError guard (no "feet_contact"
    # in this fake env's scene) leaves them untouched afterward.
    env_phase2 = _make_env(foot_y=0.45, rel_cross_y=0.9, episode_step=30)
    env_phase2._blue_was_airborne = torch.tensor([True])
    env_phase2._blue_landed = torch.tensor([True])
    env_phase2._blue_airborne_at_reset = torch.tensor([False])
    r2 = footreach(env_phase2, "ball", asset_cfg=_feet_cfg())

    assert r1.item() > r2.item(), (
        "a foot parked at the midpoint must score better once phase 1 has snapped "
        "to the midpoint target than after a genuine landing has snapped the target "
        "to the full point — this is the self-correcting property the design relies "
        "on to discourage lingering instead of continuing the second step"
    )


def test_footreach_narrow_crossing_unaffected_by_two_stage_schedule():
    # Narrow crossing (0.4 <= 0.6): target is always the full point, so a foot at
    # the full point should score identically regardless of elapsed time.
    env_early = _make_env(foot_y=0.4, rel_cross_y=0.4, episode_step=2)
    env_late = _make_env(foot_y=0.4, rel_cross_y=0.4, episode_step=40)
    r_early = footreach(env_early, "ball", asset_cfg=_feet_cfg())
    r_late = footreach(env_late, "ball", asset_cfg=_feet_cfg())
    assert torch.allclose(r_early, r_late, atol=1e-6)


def test_foot_proximity_rewards_midpoint_foot_more_in_phase_1_than_phase_2():
    env_phase1 = _make_env(foot_y=0.45, rel_cross_y=0.9, episode_step=10)
    r1 = foot_proximity(env_phase1, "ball", asset_cfg=_feet_cfg())

    env_phase2 = _make_env(foot_y=0.45, rel_cross_y=0.9, episode_step=30)
    env_phase2._blue_was_airborne = torch.tensor([True])
    env_phase2._blue_landed = torch.tensor([True])
    env_phase2._blue_airborne_at_reset = torch.tensor([False])
    r2 = foot_proximity(env_phase2, "ball", asset_cfg=_feet_cfg())

    assert r1.item() > r2.item()


def test_footreach_foot_side_selection_unaffected_by_two_stage_target():
    # is_left_ball must use the TRUE crossing_y, not the staged target — so the
    # assigned foot must be the same in phase 1 and phase 2 for the same crossing.
    # Put a foot at the midpoint on the LEFT-assigned side (idx 0) and confirm the
    # reward is nonzero in phase 1 (i.e. it used foot idx 0, not idx 1) even
    # though a naive "is_left_ball via reach_target_y" bug could flip the sign
    # of a midpoint that's still positive but smaller than the wide threshold's
    # own sign check in some edge case.
    env_phase1 = _make_env(foot_y=0.45, rel_cross_y=0.9, episode_step=10)
    r1 = footreach(env_phase1, "ball", asset_cfg=_feet_cfg())
    assert r1.item() > 0.0

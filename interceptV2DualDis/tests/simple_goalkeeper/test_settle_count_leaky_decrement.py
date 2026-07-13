"""2026-07-13: regression test for the settle-window leaky-decrement fix.

Root cause (see docs/BugFixes.md): the real AMP reference clips' own "plant"
phase between the two double-step swings is only ~11 frames (0.22s) wide at
1.0x pace, and retiming to 2.5x (for AMP-timing-budget reasons) compresses
that proportionally to ~4.4 real sim steps. The old settle-window counter
reset to 0 on ANY single miss, so 3 consecutive hits had to happen inside
that ~4-frame window with zero interruption -- a single dropped frame
(contact-sensor bounce, minor policy/demo mismatch) threw away all progress.
Fixed by decrementing by 1 (floored at 0) on a miss instead of hard-resetting,
so brief isolated dropouts don't erase everything already accumulated.

This test drives env._blue_settle_count through _get_reach_target_y directly,
with a fake robot/feet_contact sensor on a SINGLE persistent env object
(mutated between steps, matching real usage -- _get_reach_target_y caches
several attributes, like _ball_crossing_y, directly on the env object across
calls, so a fresh env per step would silently break that caching).
"""
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from simple_goalkeeper.mdp.rewards import _get_reach_target_y

_ASSET_CFG = SceneEntityCfg(
    "robot", body_names=("left_foot_link", "right_foot_link"), body_ids=[0, 1]
)


class _FakeData:
    def __init__(self, pos, vel):
        self.body_link_pos_w = pos
        self.body_link_lin_vel_w = vel


class _FakeRobot:
    def __init__(self, pos, vel):
        self.data = _FakeData(pos, vel)


class _FakeContactData:
    def __init__(self, found):
        self.found = found


class _FakeContactSensor:
    def __init__(self, found):
        self.data = _FakeContactData(found)


class _FakeScene:
    def __init__(self, num_envs):
        self.env_origins = torch.zeros(num_envs, 3)
        foot_pos = torch.zeros(num_envs, 2, 3)
        foot_vel = torch.zeros(num_envs, 2, 3)
        found = torch.zeros(num_envs, 8)
        self._robot = _FakeRobot(foot_pos, foot_vel)
        self._contact = _FakeContactSensor(found)
        self._entities = {"robot": self._robot, "feet_contact": self._contact}

    def set_left_foot(self, y, speed, in_contact: bool):
        self._robot.data.body_link_pos_w[0, 0, 1] = y
        self._robot.data.body_link_pos_w[0, 1, 1] = -5.0  # right foot, far away, irrelevant
        self._robot.data.body_link_lin_vel_w[0, 0, 0] = speed
        found = self._contact.data.found
        found.zero_()
        if in_contact:
            found[0, 0] = 1.0  # left foot contact channels are found[:, :4]

    def __getitem__(self, key):
        return self._entities[key]


class _FakeEnv:
    """Single persistent env, wide LEFT crossing, assigned foot = index 0 (left)."""

    def __init__(self):
        self.num_envs = 1
        self.device = "cpu"
        self._ball_difficulty = 1.0  # strict landing_radius=0.08, speed_threshold=1.0
        self._rsi_cross_y = torch.tensor([0.9])  # wide crossing, left side (positive)
        self.episode_length_buf = torch.tensor([1], dtype=torch.int64)
        self.scene = _FakeScene(1)

    def set_step(self, ep_len, foot_y, foot_speed, in_contact):
        self.episode_length_buf = torch.tensor([ep_len], dtype=torch.int64)
        self.scene.set_left_foot(foot_y, foot_speed, in_contact)


def _half_y():
    # start_y=0, full_y=_rsi_cross_y=0.9 -> half_y = 0.45
    return 0.45


def _step(env, ep_len, foot_y, foot_speed, in_contact):
    env.set_step(ep_len=ep_len, foot_y=foot_y, foot_speed=foot_speed, in_contact=in_contact)
    _get_reach_target_y(env, "ball", asset_cfg=_ASSET_CFG)
    return env


def _fresh_env_with_airborne_established():
    """ep_len=1 (genuine reset) lets _get_reach_target_y's own hasattr-init
    block run AND lets _get_ball_crossing_y compute crossing_y from
    _rsi_cross_y (only happens when episode_length_buf<=1). Then one
    in_contact=False step flips _blue_was_airborne True, required before
    `candidate` can ever be True on a later step (a foot must leave the
    ground before it can be judged to have landed)."""
    env = _FakeEnv()
    _step(env, ep_len=1, foot_y=_half_y(), foot_speed=2.0, in_contact=False)
    assert bool(env._blue_wide.item()) is True
    assert bool(env._blue_was_airborne.item()) is True
    return env


def test_single_miss_decrements_instead_of_resetting():
    y = _half_y()
    slow_speed = 0.1  # well under landing_speed_threshold=1.0
    env = _fresh_env_with_airborne_established()
    assert env._blue_settle_count.item() == 0

    # Step 1: hit (in contact, within radius).
    _step(env, ep_len=10, foot_y=y, foot_speed=slow_speed, in_contact=True)
    assert env._blue_settle_count.item() == 1

    # Step 2: hit.
    _step(env, ep_len=11, foot_y=y, foot_speed=slow_speed, in_contact=True)
    assert env._blue_settle_count.item() == 2

    # Step 3: MISS (contact drops for one frame -- e.g. sensor bounce).
    _step(env, ep_len=12, foot_y=y, foot_speed=slow_speed, in_contact=False)
    # Old behavior would reset to 0. New behavior: decrement by 1.
    assert env._blue_settle_count.item() == 1, (
        "a single miss must decrement, not hard-reset, the settle counter"
    )

    # Step 4: hit again -- recovers back to 2.
    _step(env, ep_len=13, foot_y=y, foot_speed=slow_speed, in_contact=True)
    assert env._blue_settle_count.item() == 2

    # Step 5: hit -- reaches 3, and with speed already under threshold, lands.
    _step(env, ep_len=14, foot_y=y, foot_speed=slow_speed, in_contact=True)
    assert env._blue_settle_count.item() == 3
    assert bool(env._blue_landed.item()) is True


def test_decrement_floors_at_zero_not_negative():
    y = _half_y()
    env = _fresh_env_with_airborne_established()
    assert env._blue_settle_count.item() == 0
    # Further misses from an already-zero counter must stay at 0, not go negative.
    for ep_len in [10, 11, 12]:
        _step(env, ep_len=ep_len, foot_y=y, foot_speed=0.1, in_contact=False)
        assert env._blue_settle_count.item() == 0


def test_sustained_misses_never_land_even_with_decrement():
    """A policy that's never actually close must still never land -- the
    decrement doesn't turn this into a free pass, it only forgives brief
    isolated dropouts."""
    y = _half_y()
    env = _fresh_env_with_airborne_established()
    ep_len = 10
    for _ in range(20):
        # alternate hit/miss forever -- net progress hovers, never sustains 3 hits in a row
        _step(env, ep_len=ep_len, foot_y=y, foot_speed=0.1, in_contact=(ep_len % 2 == 0))
        ep_len += 1
    assert bool(env._blue_landed.item()) is False

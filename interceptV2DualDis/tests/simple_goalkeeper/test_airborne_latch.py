"""Tests for `_leading_foot_airborne_latched` (rewards.py), the shared,
LATCHED "is the leading foot still airborne since the save" flag.

FIX 2026-08-06 (user request, "spikes twice" bug): `postlegdofpos`,
`postleadfootorientation`, and `postsave_foot_airtime` each used to compute
`airborne = ~leading_in_contact` FRESH, every step, straight from the raw
`feet_contact` sensor. A contact-sensor dropout or small bounce right after
the foot's genuine FIRST landing could flip `airborne` back to True later in
the same post-save window, causing `postleadfootorientation` to visibly fire
a second time well after the real landing. This test reproduces exactly that
flicker sequence (airborne -> land -> sensor dropout/bounce -> replant) and
asserts the fixed, latched flag never re-opens once it has genuinely landed.
"""
import torch

from simple_goalkeeper.mdp.rewards import _leading_foot_airborne_latched


class _FakeBallData:
    def __init__(self, num_envs: int):
        # x_local = -1.0 (< 0) for every env -> _ball_is_behind is True the
        # whole test, matching "the save has already happened, we're now in
        # the post-save recovery window" -- the scenario this latch exists
        # to track correctly.
        self.root_link_pos_w = torch.tensor([[-1.0, 0.0, 0.0]] * num_envs)
        # Only read by _get_ball_crossing_y's just_reset fallback branch
        # (no _rsi_cross_y in this fake harness) -- a small negative x
        # velocity keeps t_cross finite/well-defined.
        self.root_link_lin_vel_w = torch.tensor([[-1.0, 0.0, 0.0]] * num_envs)


class _FakeBall:
    def __init__(self, num_envs: int):
        self.data = _FakeBallData(num_envs)


class _FakeContactSensorData:
    def __init__(self, num_envs: int):
        self.found = torch.zeros(num_envs, 8)


class _FakeContactSensor:
    def __init__(self, num_envs: int):
        self.data = _FakeContactSensorData(num_envs)

    def set_right_foot_contact(self, in_contact: bool) -> None:
        # geoms 4:8 = right foot -- _get_correct_foot_idx always resolves to
        # the right foot (1) in this harness, same convention
        # test_post_save_stance.py already established (see _FakeEnv below).
        self.data.found[:, 4:] = 1.0 if in_contact else 0.0


class _Scene:
    def __init__(self, num_envs: int):
        self.env_origins = torch.zeros(num_envs, 3)
        self._ball = _FakeBall(num_envs)
        self.feet_contact = _FakeContactSensor(num_envs)
        self.ball_contact = _FakeContactSensor(num_envs)  # never touches the ball in this test

    def __getitem__(self, name: str):
        if name == "feet_contact":
            return self.feet_contact
        if name == "ball_contact":
            return self.ball_contact
        return self._ball


class _FakeEnv:
    def __init__(self, num_envs: int = 1):
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = _Scene(num_envs)
        # > 1 ("not just reset") so _get_correct_foot_idx's crossing_y
        # fallback (env_origins[:, 1] = 0.0) resolves to the RIGHT foot
        # (0.0 > 0.0 is False) -- same trick test_post_save_stance.py uses.
        self.episode_length_buf = torch.full((num_envs,), 10, dtype=torch.long)


def test_airborne_latch_stays_true_until_first_landing():
    env = _FakeEnv()
    # Step 0: save just happened, foot still in the air.
    assert _leading_foot_airborne_latched(env, "ball")[0].item() is True
    # Step 1: still airborne.
    assert _leading_foot_airborne_latched(env, "ball")[0].item() is True


def test_airborne_latch_stays_false_after_a_bounce_or_sensor_dropout():
    """The core regression test for the reported "spikes twice" bug: once
    the foot has genuinely landed, a later contact-sensor dropout (or a
    real small bounce) must NOT reopen the airborne flag."""
    env = _FakeEnv()

    # Step 0-1: airborne.
    _leading_foot_airborne_latched(env, "ball")
    _leading_foot_airborne_latched(env, "ball")

    # Step 2: FIRST genuine landing.
    env.scene.feet_contact.set_right_foot_contact(True)
    airborne_at_landing = _leading_foot_airborne_latched(env, "ball")
    assert airborne_at_landing[0].item() is False

    # Step 3: bounce / sensor dropout -- contact reads "not touching" again.
    # This is EXACTLY the flicker that caused the old raw, unlatched
    # computation to report airborne=True a second time.
    env.scene.feet_contact.set_right_foot_contact(False)
    airborne_after_dropout = _leading_foot_airborne_latched(env, "ball")
    assert airborne_after_dropout[0].item() is False, (
        "airborne flag re-opened after a bounce/sensor dropout post-landing "
        "-- this is the exact 'spikes twice' bug the latch exists to fix"
    )

    # Step 4: contact reads true again (genuine replant) -- still latched.
    env.scene.feet_contact.set_right_foot_contact(True)
    airborne_after_replant = _leading_foot_airborne_latched(env, "ball")
    assert airborne_after_replant[0].item() is False


def test_airborne_latch_resets_on_episode_boundary():
    env = _FakeEnv()
    env.scene.feet_contact.set_right_foot_contact(True)
    _leading_foot_airborne_latched(env, "ball")  # landed, latch set True

    # New episode: episode_length_buf <= 1 signals a fresh reset.
    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    env.scene.feet_contact.set_right_foot_contact(False)  # airborne again in the new episode
    airborne_new_episode = _leading_foot_airborne_latched(env, "ball")
    assert airborne_new_episode[0].item() is True

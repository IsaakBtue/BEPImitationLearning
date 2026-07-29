"""Tests for the shared post-save stance target used by postupperdofpos,
postlegdofpos, and postwaistdofpos.

FIX 2026-07-27: these three functions previously compared live joint position
against robot.data.default_joint_pos (the crouched HOME_KEYFRAME pose) --
confirmed via training logs to be a poor target for legs/waist (mean episode
reward ~0.07/~0.31 respectively, vs. postupperdofpos's ~2.0), consistent with
a bent-knee crouch requiring continuous active torque to hold. Retargeted to
_POST_SAVE_STANCE_MAP's straight-leg, centered-waist stance, resolved once
into env._post_save_stance_target and shared across all three functions (the
first lazy cache in this file owned by more than one reward function).
postwaistdofpos's weight was also raised 1.0 -> 3.0 in the same change
(goalkeeper_env_cfg.py) -- not covered by these tests, which only exercise
the reward functions directly.

FIX 2026-07-27 (same day, user request, "look more soccer-like"): the arm
portion of _POST_SAVE_STANCE_MAP was originally a flat 45-deg pure-roll
T-pose (Shoulder_Pitch/Elbow_Pitch/Elbow_Yaw all 0.0); replaced with
HOME_KEYFRAME's exact arm values at the time (Shoulder_Pitch=-0.21,
Shoulder_Roll=-0.41/0.41, Elbow_Pitch=-0.13, Elbow_Yaw=-0.21/0.21) -- legs
stay straight (the fix above is unaffected).

FIX 2026-07-27 (iterated several times same day, THEN CORRECTED, THEN
matched to Booster's official pose): Shoulder_Roll was reduced twice
toward 0.0 (0.41 -> 0.1745 -> 0.05 rad), each time believing this brought
the arm closer to hanging down. Confirmed via an offscreen mujoco.Renderer
image (not just joint_pos numbers, which read "correct" throughout but
never revealed the actual rendered pose) that 0.0 is this joint's T-POSE
reference (arms horizontal), the OPPOSITE of what was intended. Corrected
toward the joint's range limit (1.5 rad, ~86deg, user visually confirmed),
hand-tuned to 20deg of flare (1.2217 rad) for more arm-torso gap, then
finally matched exactly to Booster Robotics' own official T1 walk-policy
default pose (booster_deploy repo, tasks/locomotion/locomotion.py:205-214,
T1WalkControllerCfg): Shoulder_Pitch=0.2, Shoulder_Roll=∓1.3,
Elbow_Pitch=0.0, Elbow_Yaw=∓0.5 -- all four arm joint types now differ
from the pre-session default, not just Shoulder_Roll. Both HOME_KEYFRAME
(t1_constants.py) and this map updated together at each step.

old_row below is a FROZEN historical fixture (the very first
default_joint_pos, from before any of today's fixes) -- it deliberately
does NOT track live constants. Since the target now differs from old_row
on ALL FOUR arm joint types (not just Shoulder_Roll), old_row's arm
portion scores substantially worse against the corrected target than in
any earlier iteration of this test.

See docs/superpowers/specs/2026-07-25-post-save-stance-design.md and
docs/BugFixes.md.
"""
import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from simple_goalkeeper.mdp.rewards import (
    _POST_SAVE_STANCE_MAP,
    postlegdofpos,
    postupperdofpos,
    postwaistdofpos,
)

# Mirrors goalkeeper_env_cfg.py's _RECOVERY_ARM_CFG/_RECOVERY_LEG_CFG/
# _RECOVERY_WAIST_CFG joint-name tuples (not imported directly since those
# live in a task-config module with heavier import requirements).
_ARM_JOINT_NAMES = (
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
)
_LEG_JOINT_NAMES = (
    "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Hip_Pitch", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Hip_Pitch", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
_WAIST_JOINT_NAMES = ("Waist",)


class _FakeRobotData:
    def __init__(self, joint_pos: torch.Tensor):
        self.joint_pos = joint_pos
        # Deliberately NOT the new stance target, so a test that accidentally
        # still reads default_joint_pos fails loudly instead of coincidentally
        # passing.
        self.default_joint_pos = torch.full_like(joint_pos, 99.0)


class _FakeRobot:
    def __init__(self, joint_names: list[str], joint_pos: torch.Tensor):
        self.joint_names = joint_names
        self.data = _FakeRobotData(joint_pos)


class _FakeBallData:
    def __init__(self, x_local: torch.Tensor):
        zeros = torch.zeros_like(x_local)
        self.root_link_pos_w = torch.stack([x_local, zeros, zeros], dim=-1)


class _FakeBall:
    def __init__(self, x_local: torch.Tensor):
        self.data = _FakeBallData(x_local)


class _Scene:
    def __init__(self, robot: _FakeRobot, ball: _FakeBall, num_envs: int):
        self._robot = robot
        self._ball = ball
        self.env_origins = torch.zeros(num_envs, 3)

    def __getitem__(self, name: str):
        return self._robot if name == "robot" else self._ball


class _FakeEnv:
    def __init__(self, joint_names: list[str], joint_pos: torch.Tensor, ball_behind: bool = True):
        num_envs = joint_pos.shape[0]
        self.num_envs = num_envs
        self.device = "cpu"
        x_local = torch.full((num_envs,), -1.0 if ball_behind else 1.0)
        self.scene = _Scene(_FakeRobot(joint_names, joint_pos), _FakeBall(x_local), num_envs)


def _cfg(joint_names: list[str], names: tuple[str, ...]) -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_ids=[joint_names.index(n) for n in names])


def _target_row(joint_names: list[str]) -> torch.Tensor:
    row = torch.zeros(len(joint_names))
    for i, name in enumerate(joint_names):
        if name in _POST_SAVE_STANCE_MAP:
            row[i] = _POST_SAVE_STANCE_MAP[name]
    return row


def test_stance_target_resolves_in_declared_joint_order():
    joint_names = list(_LEG_JOINT_NAMES + _ARM_JOINT_NAMES + _WAIST_JOINT_NAMES)
    target_row = _target_row(joint_names)
    joint_pos = target_row.unsqueeze(0)  # (1, J), exactly at target -> err=0

    env = _FakeEnv(joint_names, joint_pos)
    arm_cfg = _cfg(joint_names, _ARM_JOINT_NAMES)
    leg_cfg = _cfg(joint_names, _LEG_JOINT_NAMES)
    waist_cfg = _cfg(joint_names, _WAIST_JOINT_NAMES)

    assert torch.allclose(postupperdofpos(env, "ball", asset_cfg=arm_cfg), torch.ones(1))
    assert torch.allclose(postlegdofpos(env, "ball", asset_cfg=leg_cfg), torch.ones(1))
    assert torch.allclose(postwaistdofpos(env, "ball", asset_cfg=waist_cfg), torch.ones(1))

    cached = env._post_save_stance_target
    assert cached[joint_names.index("Left_Shoulder_Roll")].item() == pytest.approx(-1.3)
    assert cached[joint_names.index("Right_Shoulder_Roll")].item() == pytest.approx(1.3)
    assert cached[joint_names.index("Left_Knee_Pitch")].item() == 0.0
    assert cached[joint_names.index("Waist")].item() == 0.0


def test_stance_target_resolves_correctly_under_shuffled_joint_order():
    # Deliberately non-declaration-order, with extra joints not in any
    # recovery cfg (Head, Waist appears twice conceptually via a stand-in
    # name) to prove unmapped-but-present joints fall back to 0.0 cleanly.
    joint_names = [
        "Right_Ankle_Roll", "Left_Elbow_Yaw", "Waist", "Right_Hip_Pitch",
        "Left_Shoulder_Roll", "Right_Knee_Pitch", "Left_Hip_Roll",
        "Right_Shoulder_Roll", "Left_Ankle_Pitch", "Right_Elbow_Pitch",
        "Left_Hip_Yaw", "Right_Hip_Roll", "Left_Knee_Pitch", "Right_Hip_Yaw",
        "Left_Ankle_Roll", "Right_Ankle_Pitch", "Left_Shoulder_Pitch",
        "Right_Shoulder_Pitch", "Left_Elbow_Pitch", "Right_Elbow_Yaw",
        "Left_Hip_Pitch",
    ]
    assert set(joint_names) == set(_LEG_JOINT_NAMES + _ARM_JOINT_NAMES + _WAIST_JOINT_NAMES)

    target_row = _target_row(joint_names)
    joint_pos = target_row.unsqueeze(0)
    env = _FakeEnv(joint_names, joint_pos)
    arm_cfg = _cfg(joint_names, _ARM_JOINT_NAMES)

    assert torch.allclose(postupperdofpos(env, "ball", asset_cfg=arm_cfg), torch.ones(1))
    cached = env._post_save_stance_target
    assert cached[joint_names.index("Left_Shoulder_Roll")].item() == pytest.approx(-1.3)
    assert cached[joint_names.index("Right_Shoulder_Roll")].item() == pytest.approx(1.3)
    assert cached[joint_names.index("Waist")].item() == 0.0


def test_stance_cache_is_shared_and_only_built_once():
    joint_names = list(_LEG_JOINT_NAMES + _ARM_JOINT_NAMES + _WAIST_JOINT_NAMES)
    target_row = _target_row(joint_names)
    joint_pos = target_row.unsqueeze(0)
    env = _FakeEnv(joint_names, joint_pos)
    arm_cfg = _cfg(joint_names, _ARM_JOINT_NAMES)
    leg_cfg = _cfg(joint_names, _LEG_JOINT_NAMES)
    waist_cfg = _cfg(joint_names, _WAIST_JOINT_NAMES)

    postupperdofpos(env, "ball", asset_cfg=arm_cfg)  # builds the cache
    sentinel_idx = joint_names.index("Left_Knee_Pitch")
    env._post_save_stance_target[sentinel_idx] = 42.0

    postlegdofpos(env, "ball", asset_cfg=leg_cfg)
    postwaistdofpos(env, "ball", asset_cfg=waist_cfg)

    # Neither call rebuilt the cache -- the mutated sentinel survives.
    assert env._post_save_stance_target[sentinel_idx].item() == 42.0


def test_reward_lower_at_old_default_pose_than_at_new_target():
    joint_names = list(_LEG_JOINT_NAMES + _ARM_JOINT_NAMES + _WAIST_JOINT_NAMES)
    old_defaults = {
        "Left_Hip_Pitch": -0.3, "Right_Hip_Pitch": -0.3,
        "Left_Knee_Pitch": 0.6, "Right_Knee_Pitch": 0.6,
        "Left_Ankle_Pitch": -0.3, "Right_Ankle_Pitch": -0.3,
        "Left_Hip_Roll": 0.0, "Right_Hip_Roll": 0.0,
        "Left_Hip_Yaw": 0.0, "Right_Hip_Yaw": 0.0,
        "Left_Ankle_Roll": 0.0, "Right_Ankle_Roll": 0.0,
        "Left_Shoulder_Pitch": -0.21, "Right_Shoulder_Pitch": -0.21,
        "Left_Shoulder_Roll": -0.41, "Right_Shoulder_Roll": 0.41,
        "Left_Elbow_Pitch": -0.13, "Right_Elbow_Pitch": -0.13,
        "Left_Elbow_Yaw": -0.21, "Right_Elbow_Yaw": 0.21,
        "Waist": 0.0,
    }
    old_row = torch.tensor([old_defaults[n] for n in joint_names]).unsqueeze(0)
    new_row = _target_row(joint_names).unsqueeze(0)

    arm_cfg = _cfg(joint_names, _ARM_JOINT_NAMES)
    leg_cfg = _cfg(joint_names, _LEG_JOINT_NAMES)
    waist_cfg = _cfg(joint_names, _WAIST_JOINT_NAMES)

    env_old = _FakeEnv(joint_names, old_row)
    env_new = _FakeEnv(joint_names, new_row)

    def expected(cfg: SceneEntityCfg, row: torch.Tensor, exponent: float) -> float:
        target = _target_row(joint_names)
        delta = row[0, cfg.joint_ids] - target[cfg.joint_ids]
        return torch.exp(-exponent * torch.sum(delta**2)).item()

    arm_old = postupperdofpos(env_old, "ball", asset_cfg=arm_cfg).item()
    leg_old = postlegdofpos(env_old, "ball", asset_cfg=leg_cfg).item()
    waist_old = postwaistdofpos(env_old, "ball", asset_cfg=waist_cfg).item()

    assert arm_old == expected(arm_cfg, old_row, 1.0)
    assert leg_old == expected(leg_cfg, old_row, 1.0)
    assert waist_old == expected(waist_cfg, old_row, 3.0)

    # Legs: old (crouched) pose no longer scores near 1.0 -- the retarget bites.
    assert leg_old < 0.5
    # Waist: target value didn't move (0.0 -> 0.0), old pose still scores ~1.0.
    assert waist_old > 0.99
    # Arms: target's Shoulder_Roll was reduced a second time, same day
    # (0.41 -> 0.1745 rad, matching HOME_KEYFRAME's own second reduction) --
    # old_row's arm portion (frozen at the ORIGINAL 0.41 default, see this
    # dict above) no longer matches the current target, so arm_old is
    # measurably below 1.0 again (Shoulder_Pitch/Elbow_Pitch/Elbow_Yaw are
    # unchanged between old and target, only Shoulder_Roll differs).
    assert 0.09 < arm_old < 0.15

    arm_new = postupperdofpos(env_new, "ball", asset_cfg=arm_cfg).item()
    leg_new = postlegdofpos(env_new, "ball", asset_cfg=leg_cfg).item()
    # Legs: new (straight) target scores strictly better than the old pose.
    assert leg_new > leg_old
    assert torch.allclose(torch.tensor(leg_new), torch.tensor(1.0), atol=1e-5)
    # Arms: new row is placed exactly at the current target by construction,
    # so it always scores the ceiling regardless of what that target is.
    assert arm_new > arm_old
    assert torch.allclose(torch.tensor(arm_new), torch.tensor(1.0), atol=1e-5)


def test_gated_off_when_ball_not_behind():
    joint_names = list(_LEG_JOINT_NAMES + _ARM_JOINT_NAMES + _WAIST_JOINT_NAMES)
    target_row = _target_row(joint_names)
    joint_pos = target_row.unsqueeze(0)
    env = _FakeEnv(joint_names, joint_pos, ball_behind=False)
    arm_cfg = _cfg(joint_names, _ARM_JOINT_NAMES)
    leg_cfg = _cfg(joint_names, _LEG_JOINT_NAMES)
    waist_cfg = _cfg(joint_names, _WAIST_JOINT_NAMES)

    # FIX 2026-07-28 (user request): postupperdofpos is no longer fully gated
    # off pre-save -- it stays active at during_scale (default, tuned 0.3->0.5
    # on 2026-07-29 alongside reverting arm_torque_limits/arm_action_rate_l2/
    # arm_action_acc_l2, see rewards.py) so it provides some pull toward the
    # arm stance throughout, not only post-save. At the target pose (err=0,
    # exp(-err)=1.0), that's during_scale exactly.
    assert postupperdofpos(env, "ball", asset_cfg=arm_cfg).item() == pytest.approx(0.5)
    assert postlegdofpos(env, "ball", asset_cfg=leg_cfg).item() == 0.0
    assert postwaistdofpos(env, "ball", asset_cfg=waist_cfg).item() == 0.0

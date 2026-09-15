"""Regression tests for the AMP-observation masking functions in
observations.py (joint_pos_abs_arms_masked_by_region /
joint_vel_abs_arms_masked_by_region) -- no direct unit test existed for
these before, despite them being the exact functions the 2026-09-15
investigation's softstop-freeze finding centers on. Only live-probe-verified
previously (real env, real checkpoint); these pin the masking LOGIC down
with synthetic, fully-controlled data so a future edit can't silently change
which columns/envs get frozen without a test catching it.
"""
import torch

from simple_goalkeeper.mdp.observations import (
    _ARM_JOINT_NAMES,
    joint_pos_abs_arms_masked_by_region,
    joint_vel_abs_arms_masked_by_region,
)

_ALL_JOINT_NAMES = list(_ARM_JOINT_NAMES) + [
    "Waist", "Left_Hip_Pitch", "Right_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Right_Hip_Roll", "Right_Hip_Yaw", "Left_Knee_Pitch", "Right_Knee_Pitch",
    "Left_Ankle_Pitch", "Right_Ankle_Pitch", "Left_Ankle_Roll", "Right_Ankle_Roll",
]
_NDOF = len(_ALL_JOINT_NAMES)
_ARM_COL_IDX = list(range(len(_ARM_JOINT_NAMES)))  # arms placed first, by construction above
_NON_ARM_COL_IDX = list(range(len(_ARM_JOINT_NAMES), _NDOF))


class _FakeRobotData:
    def __init__(self, joint_pos, joint_vel, default_joint_pos):
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.default_joint_pos = default_joint_pos


class _FakeRobot:
    def __init__(self, num_envs, device="cpu"):
        torch.manual_seed(0)
        # Real (non-default) values are large/distinctive so a masked
        # column is unambiguous: default=100s, "live" values are small
        # random floats near 0.
        self.data = _FakeRobotData(
            joint_pos=torch.randn(num_envs, _NDOF, device=device),
            joint_vel=torch.randn(num_envs, _NDOF, device=device) * 5,
            default_joint_pos=torch.full((num_envs, _NDOF), 100.0, device=device),
        )

    def find_joints(self, names, preserve_order=True):
        indices = [_ALL_JOINT_NAMES.index(n) for n in names]
        return indices, list(names)


class _FakeScene:
    def __init__(self, robot):
        self._robot = robot

    def __getitem__(self, name):
        return self._robot


class _FakeEnv:
    def __init__(self, num_envs, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        self.scene = _FakeScene(_FakeRobot(num_envs, device))


def test_far_region_envs_get_arm_columns_frozen_non_arm_untouched():
    env = _FakeEnv(num_envs=4)
    env._region_id = torch.tensor([0, 1, 2, 3])  # left_near, left_far, right_near, right_far
    live_joint_pos = env.scene["robot"].data.joint_pos.clone()

    out = joint_pos_abs_arms_masked_by_region(env, far_region_ids=(1, 3))

    # Regions 1 and 3 (far) -> arm columns frozen to default (100.0).
    assert torch.allclose(out[1][_ARM_COL_IDX], torch.full((len(_ARM_COL_IDX),), 100.0))
    assert torch.allclose(out[3][_ARM_COL_IDX], torch.full((len(_ARM_COL_IDX),), 100.0))
    # Regions 0 and 2 (near) -> arm columns keep the real, live value.
    assert torch.allclose(out[0][_ARM_COL_IDX], live_joint_pos[0][_ARM_COL_IDX])
    assert torch.allclose(out[2][_ARM_COL_IDX], live_joint_pos[2][_ARM_COL_IDX])
    # Non-arm columns are NEVER touched by this function, for any region.
    for env_idx in range(4):
        assert torch.allclose(out[env_idx][_NON_ARM_COL_IDX], live_joint_pos[env_idx][_NON_ARM_COL_IDX])


def test_far_region_ids_empty_tuple_means_no_masking_anywhere():
    """The historical default (2026-08-04, since reverted) -- far_region_ids=()
    means every region keeps live arm data. Regression guard: an empty tuple
    must genuinely mean "mask nothing", not accidentally match every region
    (e.g. via a stray `if not far_region_ids: is_far[:] = True` typo)."""
    env = _FakeEnv(num_envs=4)
    env._region_id = torch.tensor([0, 1, 2, 3])
    live_joint_pos = env.scene["robot"].data.joint_pos.clone()

    out = joint_pos_abs_arms_masked_by_region(env, far_region_ids=())

    assert torch.allclose(out, live_joint_pos)


def test_softstop_freeze_applies_before_and_independently_of_arm_masking():
    """softstop_fired freezes the WHOLE 21-dof vector (not just arms) --
    verify it composes correctly with arm masking rather than one silently
    overriding the other."""
    env = _FakeEnv(num_envs=2)
    env._region_id = torch.tensor([0, 0])  # both near -> arms would stay live if not for softstop
    env._softstop_flag = torch.tensor([True, False])
    live_joint_pos = env.scene["robot"].data.joint_pos.clone()

    out = joint_pos_abs_arms_masked_by_region(env, far_region_ids=())

    # env0: softstop fired -> ENTIRE vector (arm AND non-arm columns) frozen to default.
    assert torch.allclose(out[0], torch.full((_NDOF,), 100.0))
    # env1: no softstop, near region, no masking -> untouched live value.
    assert torch.allclose(out[1], live_joint_pos[1])


def test_joint_vel_masked_regions_freeze_to_zero_not_default_pose():
    """Unlike joint_pos (frozen to default_joint_pos), joint_vel has no
    'default' velocity concept -- masked/frozen columns must go to exactly
    zero, per the function's own docstring ('Zero is the natural "frozen"
    velocity')."""
    env = _FakeEnv(num_envs=2)
    env._region_id = torch.tensor([1, 0])  # env0 far, env1 near
    live_joint_vel = env.scene["robot"].data.joint_vel.clone()

    out = joint_vel_abs_arms_masked_by_region(env, far_region_ids=(1,))

    assert torch.allclose(out[0][_ARM_COL_IDX], torch.zeros(len(_ARM_COL_IDX)))
    assert torch.allclose(out[0][_NON_ARM_COL_IDX], live_joint_vel[0][_NON_ARM_COL_IDX])
    assert torch.allclose(out[1], live_joint_vel[1])


def test_no_region_id_attribute_means_no_arm_masking_applied():
    """If env._region_id doesn't exist yet (e.g. very first call before
    region assignment runs), the function must degrade to "no masking"
    rather than crash or mask everything."""
    env = _FakeEnv(num_envs=3)
    live_joint_pos = env.scene["robot"].data.joint_pos.clone()

    out = joint_pos_abs_arms_masked_by_region(env, far_region_ids=(0, 1, 2, 3))

    assert torch.allclose(out, live_joint_pos)

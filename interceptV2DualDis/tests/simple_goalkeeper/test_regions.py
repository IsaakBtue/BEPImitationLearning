"""Tests for static region assignment and region-conditioned ball spawn."""
import types

import torch

from simple_goalkeeper.mdp.regions import (
    REGION_NAMES,
    assign_static_regions,
    reset_ball_rolling_by_region,
)


class _FakeEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.device = "cpu"


def test_region_names_are_four_in_order():
    assert REGION_NAMES == ("left_near", "left_far", "right_near", "right_far")


def test_assign_static_regions_splits_into_four_equal_contiguous_blocks():
    env = _FakeEnv(num_envs=12)
    assign_static_regions(env, env_ids=None)
    assert env._region_id.shape == (12,)
    assert env._region_id.dtype == torch.int64
    expected = torch.tensor([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3, dtype=torch.int64)
    assert torch.equal(env._region_id, expected)


def test_assign_static_regions_handles_non_multiple_of_four():
    # 10 envs: quarter=2, remainder 2 envs go to the last block (right_far).
    env = _FakeEnv(num_envs=10)
    assign_static_regions(env, env_ids=None)
    assert env._region_id.shape == (10,)
    counts = torch.bincount(env._region_id, minlength=4)
    assert counts[0].item() == 2
    assert counts[1].item() == 2
    assert counts[2].item() == 2
    assert counts[3].item() == 4  # 2 base + 2 remainder


def test_reset_ball_rolling_by_region_calls_reset_ball_rolling_per_region(monkeypatch):
    calls = []

    def fake_reset_ball_rolling(env, env_ids, ball_name, **kwargs):
        calls.append((tuple(env_ids.tolist()), kwargs["y_start_range"], kwargs["y_end_range"]))

    import simple_goalkeeper.mdp.regions as regions_mod
    monkeypatch.setattr(regions_mod, "reset_ball_rolling", fake_reset_ball_rolling)

    env = _FakeEnv(num_envs=8)
    assign_static_regions(env, env_ids=None)
    reset_ball_rolling_by_region(env, env_ids=None, ball_name="ball")

    # 4 regions, 2 envs each (8 // 4 = 2) -> 4 calls, one per region.
    assert len(calls) == 4
    called_env_ids = {c[0] for c in calls}
    assert called_env_ids == {(0, 1), (2, 3), (4, 5), (6, 7)}


def test_reset_ball_rolling_by_region_only_flags_far_regions_for_far_travel_curriculum(monkeypatch):
    calls = {}

    def fake_reset_ball_rolling(env, env_ids, ball_name, **kwargs):
        calls[kwargs["y_end_range"]] = kwargs["use_far_travel_curriculum"]

    import simple_goalkeeper.mdp.regions as regions_mod
    monkeypatch.setattr(regions_mod, "reset_ball_rolling", fake_reset_ball_rolling)

    env = _FakeEnv(num_envs=8)
    assign_static_regions(env, env_ids=None)
    reset_ball_rolling_by_region(env, env_ids=None, ball_name="ball")

    from simple_goalkeeper.mdp.regions import _REGION_Y_END_RANGE

    assert calls[_REGION_Y_END_RANGE[0]] is False   # left_near
    assert calls[_REGION_Y_END_RANGE[1]] is True    # left_far
    assert calls[_REGION_Y_END_RANGE[2]] is False   # right_near
    assert calls[_REGION_Y_END_RANGE[3]] is True    # right_far


def test_region_id_gt_returns_float_column_vector():
    from simple_goalkeeper.mdp.regions import region_id_gt

    env = _FakeEnv(num_envs=8)
    assign_static_regions(env, env_ids=None)
    out = region_id_gt(env)
    assert out.shape == (8, 1)
    assert out.dtype == torch.float32
    assert torch.equal(out.squeeze(-1), env._region_id.float())


def test_pin_region_on_reset_pins_every_env_to_the_given_region():
    from simple_goalkeeper.mdp.regions import pin_region_on_reset

    env = _FakeEnv(num_envs=8)
    pin_region_on_reset(env, env_ids=None, region_id=3)
    assert torch.equal(env._region_id, torch.full((8,), 3, dtype=torch.int64))


def test_pin_region_on_reset_only_touches_the_given_env_ids():
    from simple_goalkeeper.mdp.regions import pin_region_on_reset

    env = _FakeEnv(num_envs=8)
    assign_static_regions(env, env_ids=None)  # env 0,1 -> region 0; env 2,3 -> region 1; ...
    pin_region_on_reset(env, env_ids=torch.tensor([0, 1]), region_id=2)
    assert env._region_id[0].item() == 2
    assert env._region_id[1].item() == 2
    assert env._region_id[2].item() == 1  # untouched, still assign_static_regions' original value

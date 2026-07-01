import torch

from simple_goalkeeper.mdp.events import _POOL_KEYS, _POOL_ID, _select_live_donors


def test_pool_id_covers_six_side_tier_combinations():
    assert len(_POOL_KEYS) == 6
    assert set(_POOL_KEYS) == {
        ("left", "double"), ("left", "triple"), ("left", "wide"),
        ("right", "double"), ("right", "triple"), ("right", "wide"),
    }
    assert set(_POOL_ID.values()) == {0, 1, 2, 3, 4, 5}
    assert set(_POOL_ID.keys()) == set(_POOL_KEYS)


def test_select_live_donors_matches_pool_and_maturity():
    # 6 envs: pool assignment per env, episode age per env.
    pool_id = torch.tensor([0, 0, 1, 0, -1, 0])
    episode_steps = torch.tensor([50, 3, 50, 50, 50, 50])
    exclude_ids = torch.tensor([], dtype=torch.long)

    donors = _select_live_donors(
        pool_id, episode_steps, exclude_ids,
        target_pool=0, min_maturity_steps=10,
    )
    # env 0: pool matches, mature -> eligible
    # env 1: pool matches, but only 3 steps old -> excluded (immature)
    # env 2: wrong pool -> excluded
    # env 3: pool matches, mature -> eligible
    # env 4: pool -1 (standing) -> excluded
    # env 5: pool matches, mature -> eligible
    assert sorted(donors.tolist()) == [0, 3, 5]


def test_select_live_donors_excludes_current_reset_batch():
    pool_id = torch.tensor([0, 0, 0])
    episode_steps = torch.tensor([50, 50, 50])
    exclude_ids = torch.tensor([1], dtype=torch.long)

    donors = _select_live_donors(
        pool_id, episode_steps, exclude_ids,
        target_pool=0, min_maturity_steps=10,
    )
    assert sorted(donors.tolist()) == [0, 2]


def test_select_live_donors_returns_empty_when_no_match():
    pool_id = torch.tensor([1, 2, 3])
    episode_steps = torch.tensor([50, 50, 50])
    exclude_ids = torch.tensor([], dtype=torch.long)

    donors = _select_live_donors(
        pool_id, episode_steps, exclude_ids,
        target_pool=0, min_maturity_steps=10,
    )
    assert donors.numel() == 0

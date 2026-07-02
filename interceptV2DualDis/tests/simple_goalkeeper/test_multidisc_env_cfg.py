"""Tests for goalkeeper_multidisc_env_cfg: history group, critic gt terms, region events."""
from simple_goalkeeper.tasks.goalkeeper_multidisc_amp_cfg import goalkeeper_multidisc_env_cfg


def test_actor_history_group_has_history_length_ten():
    cfg = goalkeeper_multidisc_env_cfg(play=False)
    assert "actor_history" in cfg.observations
    assert cfg.observations["actor_history"].history_length == 10
    # Same terms as the plain "actor" group (same dict of ObservationTermCfg).
    assert set(cfg.observations["actor_history"].terms.keys()) == set(cfg.observations["actor"].terms.keys())


def test_critic_group_has_ball_and_region_ground_truth_terms():
    cfg = goalkeeper_multidisc_env_cfg(play=False)
    assert "ball_gt" in cfg.observations["critic"].terms
    assert "region_gt" in cfg.observations["critic"].terms


def test_region_events_registered():
    cfg = goalkeeper_multidisc_env_cfg(play=False)
    assert "assign_static_regions" in cfg.events
    assert cfg.events["assign_static_regions"].mode == "startup"
    assert cfg.events["reset_ball"].func.__name__ == "reset_ball_rolling_by_region"

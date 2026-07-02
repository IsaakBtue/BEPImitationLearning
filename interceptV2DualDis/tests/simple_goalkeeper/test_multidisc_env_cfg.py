"""Tests for goalkeeper_multidisc_env_cfg: actor_current group, actor history, critic gt terms, region events."""
from simple_goalkeeper.tasks.goalkeeper_multidisc_amp_cfg import goalkeeper_multidisc_env_cfg


def test_actor_current_group_is_single_step_and_matches_actor_term_names():
    cfg = goalkeeper_multidisc_env_cfg(play=False)
    assert "actor_current" in cfg.observations
    # Same term names as the plain "actor" group, but every term is single-step.
    assert set(cfg.observations["actor_current"].terms.keys()) == set(cfg.observations["actor"].terms.keys())
    for term_cfg in cfg.observations["actor_current"].terms.values():
        assert term_cfg.history_length == 0


def test_actor_group_still_has_history_length_ten():
    cfg = goalkeeper_multidisc_env_cfg(play=False)
    assert cfg.observations["actor"].history_length == 10


def test_actor_history_group_no_longer_exists():
    cfg = goalkeeper_multidisc_env_cfg(play=False)
    assert "actor_history" not in cfg.observations


def test_critic_group_has_ball_and_region_ground_truth_terms():
    cfg = goalkeeper_multidisc_env_cfg(play=False)
    assert "ball_gt" in cfg.observations["critic"].terms
    assert "region_gt" in cfg.observations["critic"].terms


def test_region_events_registered():
    cfg = goalkeeper_multidisc_env_cfg(play=False)
    assert "assign_static_regions" in cfg.events
    assert cfg.events["assign_static_regions"].mode == "startup"
    assert cfg.events["reset_ball"].func.__name__ == "reset_ball_rolling_by_region"

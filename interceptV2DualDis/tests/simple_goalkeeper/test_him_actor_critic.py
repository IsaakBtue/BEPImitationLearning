"""Tests for HimActorCritic: shapes, estimator heads, act/evaluate contract."""
import torch

from simple_goalkeeper.rsl_rl_multi.him_actor_critic import HimActorCritic


def _make_model(num_one_step_obs=20, history_length=10, num_critic_obs=40, num_actions=21):
    return HimActorCritic(
        num_one_step_obs=num_one_step_obs,
        actor_history_length=history_length,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
    )


def test_estimator_head_output_shapes():
    model = _make_model()
    assert model.history_encoder[-1].out_features == 16
    assert model.ball_estimator[-1].out_features == 4
    assert model.region_estimator[-1].out_features == 4


def test_actor_input_dim_matches_composition():
    num_one_step_obs = 20
    model = _make_model(num_one_step_obs=num_one_step_obs)
    # actor input = last raw one-step obs (20) + history_latent (16) + ball (4) + region argmax (1)
    expected = num_one_step_obs + 16 + 4 + 1
    assert model.num_actor_input == expected
    assert model.actor[0].in_features == expected


def test_act_sets_estimate_ball_and_estimate_region_with_correct_shapes():
    num_envs = 5
    num_one_step_obs = 20
    history_length = 10
    model = _make_model(num_one_step_obs=num_one_step_obs, history_length=history_length)
    obs_current = torch.randn(num_envs, num_one_step_obs)
    obs_history = torch.randn(num_envs, num_one_step_obs * history_length)
    actions = model.act(obs_current, obs_history)
    assert actions.shape == (num_envs, 21)
    assert model.estimate_ball.shape == (num_envs, 4)
    assert model.estimate_region.shape == (num_envs, 4)


def test_act_inference_returns_deterministic_action_mean():
    num_envs = 3
    num_one_step_obs = 20
    history_length = 10
    model = _make_model(num_one_step_obs=num_one_step_obs, history_length=history_length)
    obs_current = torch.randn(num_envs, num_one_step_obs)
    obs_history = torch.randn(num_envs, num_one_step_obs * history_length)
    mean1 = model.act_inference(obs_current, obs_history)
    mean2 = model.act_inference(obs_current, obs_history)
    assert torch.equal(mean1, mean2)
    assert mean1.shape == (num_envs, 21)


def test_evaluate_returns_scalar_value_per_env():
    num_envs = 4
    model = _make_model(num_critic_obs=40)
    critic_obs = torch.randn(num_envs, 40)
    value = model.evaluate(critic_obs)
    assert value.shape == (num_envs, 1)

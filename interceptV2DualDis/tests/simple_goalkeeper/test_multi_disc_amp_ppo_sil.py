"""Tests for MultiDiscAMPPPO's self-imitation (SIL) loss integration --
see success_buffer.py for the full rationale. No env, synthetic data,
mirrors test_multi_disc_amp_ppo.py's setup pattern.
"""
import torch

from simple_goalkeeper.rsl_rl_multi.him_actor_critic import HimActorCritic
from simple_goalkeeper.rsl_rl_multi.multi_disc_amp_ppo import MultiDiscAMPPPO
from simple_goalkeeper.rsl_rl_multi.success_buffer import SuccessReplayBuffer
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator
from rsl_rl_amp.utils.utils import Normalizer


NUM_ONE_STEP_OBS = 10
HISTORY_LEN = 10
NUM_CRITIC_OBS = 25
NUM_ACTIONS = 6
AMP_OBS_DIM = 8
REGION_NAMES = ("left_near", "left_far", "right_near", "right_far")


class _FakeMotionDataset:
    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            yield (torch.randn(mini_batch_size, self.obs_dim), torch.randn(mini_batch_size, self.obs_dim))


def _make_alg(num_envs=8, num_transitions=4, success_buffer=None, sil_batch_size=8):
    actor_critic = HimActorCritic(
        num_one_step_obs=NUM_ONE_STEP_OBS,
        actor_history_length=HISTORY_LEN,
        num_critic_obs=NUM_CRITIC_OBS,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
    )
    discriminators = {
        name: AMPDiscriminator(AMP_OBS_DIM * 2, amp_reward_coef=1.0,
                                hidden_layer_sizes=[16, 8], device="cpu", task_reward_lerp=0.5)
        for name in REGION_NAMES
    }
    amp_datasets = {name: _FakeMotionDataset(AMP_OBS_DIM) for name in REGION_NAMES}
    normalizer = Normalizer(AMP_OBS_DIM)
    alg = MultiDiscAMPPPO(
        actor_critic=actor_critic,
        discriminators=discriminators,
        amp_datasets=amp_datasets,
        amp_normalizer=normalizer,
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
        region_id_critic_obs_index=-1,
        ball_gt_critic_obs_slice=slice(-5, -1),
        amp_obs_dim=AMP_OBS_DIM,
        amp_replay_buffer_size=64,
        success_buffer=success_buffer,
        sil_batch_size=sil_batch_size,
    )
    alg.init_storage(num_envs, num_transitions, [NUM_ONE_STEP_OBS], [NUM_ONE_STEP_OBS * HISTORY_LEN],
                      [NUM_CRITIC_OBS], [NUM_ACTIONS])
    return alg, actor_critic


def _run_one_update(alg, num_envs=4, num_transitions=2):
    region_ids = torch.tensor([0.0, 0.0, 2.0, 2.0])
    for _ in range(num_transitions):
        obs_current = torch.randn(num_envs, NUM_ONE_STEP_OBS)
        obs_history = torch.randn(num_envs, NUM_ONE_STEP_OBS * HISTORY_LEN)
        critic_obs = torch.randn(num_envs, NUM_CRITIC_OBS)
        critic_obs[:, -1] = region_ids
        amp_obs = torch.randn(num_envs, AMP_OBS_DIM)
        alg.act(obs_current, obs_history, critic_obs, amp_obs)
        alg.process_env_step(
            rewards=torch.zeros(num_envs), dones=torch.zeros(num_envs, dtype=torch.bool),
            infos={}, amp_obs=torch.randn(num_envs, AMP_OBS_DIM),
        )
    alg.compute_returns(torch.randn(num_envs, NUM_CRITIC_OBS))
    return alg.update()


def test_sil_loss_is_zero_when_no_success_buffer_provided():
    alg, _ = _make_alg(num_envs=4, num_transitions=2, success_buffer=None)
    result = _run_one_update(alg, num_envs=4, num_transitions=2)
    mean_sil_loss = result[-1]
    assert mean_sil_loss == 0.0


def test_sil_loss_is_zero_when_buffer_below_sil_batch_size():
    buf = SuccessReplayBuffer(
        obs_current_dim=NUM_ONE_STEP_OBS, obs_history_dim=NUM_ONE_STEP_OBS * HISTORY_LEN,
        action_dim=NUM_ACTIONS, num_envs=4, device="cpu", capacity=256, lookback=10,
    )
    # Commit fewer transitions than sil_batch_size=8.
    reset_mask = torch.zeros(4, dtype=torch.bool)
    buf.record_step(torch.randn(4, NUM_ONE_STEP_OBS), torch.randn(4, NUM_ONE_STEP_OBS * HISTORY_LEN),
                     torch.randn(4, NUM_ACTIONS), reset_mask)
    buf.commit_success(torch.tensor([0]))
    assert len(buf) < 8

    alg, _ = _make_alg(num_envs=4, num_transitions=2, success_buffer=buf, sil_batch_size=8)
    result = _run_one_update(alg, num_envs=4, num_transitions=2)
    mean_sil_loss = result[-1]
    assert mean_sil_loss == 0.0


def test_sil_loss_is_nonzero_and_updates_actor_when_buffer_has_enough_samples():
    buf = SuccessReplayBuffer(
        obs_current_dim=NUM_ONE_STEP_OBS, obs_history_dim=NUM_ONE_STEP_OBS * HISTORY_LEN,
        action_dim=NUM_ACTIONS, num_envs=4, device="cpu", capacity=256, lookback=20,
    )
    reset_mask = torch.zeros(4, dtype=torch.bool)
    for _ in range(20):
        buf.record_step(torch.randn(4, NUM_ONE_STEP_OBS), torch.randn(4, NUM_ONE_STEP_OBS * HISTORY_LEN),
                         torch.randn(4, NUM_ACTIONS), reset_mask)
    buf.commit_success(torch.tensor([0, 1, 2, 3]))  # 4 envs x 20 steps = 80 transitions
    assert len(buf) >= 8

    alg, actor_critic = _make_alg(num_envs=4, num_transitions=2, success_buffer=buf, sil_batch_size=8)
    actor_params_before = [p.clone() for p in actor_critic.parameters()]

    result = _run_one_update(alg, num_envs=4, num_transitions=2)
    mean_sil_loss = result[-1]

    assert mean_sil_loss != 0.0
    actor_params_after = list(actor_critic.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(actor_params_before, actor_params_after))
    assert changed, "actor_critic parameters should have been updated by the combined loss"

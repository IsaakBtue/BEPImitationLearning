"""Tests for MultiDiscAMPPPO region-routed loss/reward — no env, synthetic data."""
import numpy as np
import torch

from simple_goalkeeper.rsl_rl_multi.him_actor_critic import HimActorCritic
from simple_goalkeeper.rsl_rl_multi.multi_disc_amp_ppo import MultiDiscAMPPPO
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator
from rsl_rl_amp.utils.utils import Normalizer


NUM_ONE_STEP_OBS = 10
HISTORY_LEN = 10
NUM_CRITIC_OBS = 25  # includes 4 (ball_gt) + 1 (region_gt) appended at the end
NUM_ACTIONS = 6
AMP_OBS_DIM = 8
REGION_NAMES = ("left_near", "left_far", "right_near", "right_far")


class _FakeMotionDataset:
    """Minimal stand-in for beyondAMP.motion.motion_dataset.MotionDataset."""
    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            yield (
                torch.randn(mini_batch_size, self.obs_dim),
                torch.randn(mini_batch_size, self.obs_dim),
            )


def _make_alg(num_envs=8, num_transitions=4):
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
    )
    alg.init_storage(num_envs, num_transitions, [NUM_ONE_STEP_OBS], [NUM_ONE_STEP_OBS * HISTORY_LEN],
                      [NUM_CRITIC_OBS], [NUM_ACTIONS])
    return alg, actor_critic, discriminators


def test_region_routing_only_updates_the_matching_discriminator():
    alg, actor_critic, discriminators = _make_alg(num_envs=4, num_transitions=2)

    # Build a rollout where env 0-1 are region 0 (left_near) and env 2-3 are region 2 (right_near).
    region_ids = torch.tensor([0.0, 0.0, 2.0, 2.0])
    before = {name: [p.clone() for p in d.trunk.parameters()] for name, d in discriminators.items()}

    for _ in range(2):  # num_transitions
        obs_current = torch.randn(4, NUM_ONE_STEP_OBS)
        obs_history = torch.randn(4, NUM_ONE_STEP_OBS * HISTORY_LEN)
        critic_obs = torch.randn(4, NUM_CRITIC_OBS)
        critic_obs[:, -1] = region_ids
        amp_obs = torch.randn(4, AMP_OBS_DIM)
        alg.act(obs_current, obs_history, critic_obs, amp_obs)
        alg.process_env_step(
            rewards=torch.zeros(4), dones=torch.zeros(4, dtype=torch.bool),
            infos={}, amp_obs=torch.randn(4, AMP_OBS_DIM),
        )
    alg.compute_returns(torch.randn(4, NUM_CRITIC_OBS))
    alg.update()

    after = {name: [p.clone() for p in d.trunk.parameters()] for name, d in discriminators.items()}

    for name in REGION_NAMES:
        changed = any(not torch.equal(b, a) for b, a in zip(before[name], after[name]))
        if name in ("left_near", "right_near"):
            assert changed, f"{name} discriminator should have been updated (region present in batch)"
        else:
            assert not changed, f"{name} discriminator should NOT have been touched (region absent)"


def test_amp_loss_uses_policy_replay_buffer_not_proprioceptive_obs():
    alg, actor_critic, discriminators = _make_alg(num_envs=4, num_transitions=2)
    region_ids = torch.tensor([0.0, 0.0, 2.0, 2.0])
    for _ in range(2):
        obs_current = torch.randn(4, NUM_ONE_STEP_OBS)
        obs_history = torch.randn(4, NUM_ONE_STEP_OBS * HISTORY_LEN)
        critic_obs = torch.randn(4, NUM_CRITIC_OBS)
        critic_obs[:, -1] = region_ids
        amp_obs = torch.randn(4, AMP_OBS_DIM)
        alg.act(obs_current, obs_history, critic_obs, amp_obs)
        alg.process_env_step(
            rewards=torch.zeros(4), dones=torch.zeros(4, dtype=torch.bool),
            infos={}, amp_obs=torch.randn(4, AMP_OBS_DIM),
        )
    assert alg.amp_storages["left_near"].step > 0 or alg.amp_storages["left_near"].num_samples > 0
    alg.compute_returns(torch.randn(4, NUM_CRITIC_OBS))
    result = alg.update()
    mean_amp_loss = result[2]
    assert mean_amp_loss > 0.0  # both expert and policy terms now contribute


def test_update_refreshes_amp_normalizer_running_stats():
    """The normalizer must not stay frozen at init (zero mean, unit var,
    epsilon count) — update() should feed it every region's policy/expert
    samples, mirroring amp_ppo.py:227-229's per-minibatch normalizer.update()."""
    alg, actor_critic, discriminators = _make_alg(num_envs=4, num_transitions=2)
    region_ids = torch.tensor([0.0, 0.0, 2.0, 2.0])

    initial_count = alg.amp_normalizer.count
    assert initial_count == 1e-4  # RunningMeanStd epsilon default, untouched

    for _ in range(2):
        obs_current = torch.randn(4, NUM_ONE_STEP_OBS)
        obs_history = torch.randn(4, NUM_ONE_STEP_OBS * HISTORY_LEN)
        critic_obs = torch.randn(4, NUM_CRITIC_OBS)
        critic_obs[:, -1] = region_ids
        amp_obs = torch.randn(4, AMP_OBS_DIM)
        alg.act(obs_current, obs_history, critic_obs, amp_obs)
        alg.process_env_step(
            rewards=torch.zeros(4), dones=torch.zeros(4, dtype=torch.bool),
            infos={}, amp_obs=torch.randn(4, AMP_OBS_DIM),
        )
    alg.compute_returns(torch.randn(4, NUM_CRITIC_OBS))
    alg.update()

    # Two regions were present in the batch (left_near, right_near), so the
    # normalizer should have been fed at least 2 x 2 (policy + expert) batches.
    assert alg.amp_normalizer.count > initial_count
    assert not (alg.amp_normalizer.mean == 0.0).all()
    assert not (alg.amp_normalizer.var == 1.0).all()


def test_process_env_step_excludes_done_and_softstop_frozen_transitions_from_amp_storage():
    """FIX 2026-09-15: terminal (dones=True) transitions and post-softstop
    -frozen transitions are known-bad discriminator training data (a
    mislabeled reset "teleport", and a massively overrepresented constant
    cluster respectively -- see multi_disc_amp_ppo.py's process_env_step)
    and must never reach amp_storages, even though they still get a normal
    reward via predict_region_routed_amp_reward (untouched, called earlier
    in the real rollout loop, not exercised by this unit test)."""
    alg, actor_critic, discriminators = _make_alg(num_envs=4, num_transitions=1)
    region_ids = torch.tensor([0.0, 0.0, 0.0, 0.0])  # all left_near, one region to keep this simple

    obs_current = torch.randn(4, NUM_ONE_STEP_OBS)
    obs_history = torch.randn(4, NUM_ONE_STEP_OBS * HISTORY_LEN)
    critic_obs = torch.randn(4, NUM_CRITIC_OBS)
    critic_obs[:, -1] = region_ids
    amp_obs = torch.randn(4, AMP_OBS_DIM)
    alg.act(obs_current, obs_history, critic_obs, amp_obs)

    # env0: normal (kept). env1: done (excluded). env2: softstop-frozen
    # (excluded). env3: both done AND frozen (excluded, still just once).
    dones = torch.tensor([0, 1, 0, 1], dtype=torch.bool)
    softstop_flag = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
    alg.process_env_step(
        rewards=torch.zeros(4), dones=dones,
        infos={}, amp_obs=torch.randn(4, AMP_OBS_DIM),
        softstop_flag=softstop_flag,
    )

    assert alg.amp_storages["left_near"].num_samples == 1, (
        "only env0 (not done, not frozen) should have been inserted"
    )


def test_many_consecutive_rollout_update_cycles_stay_finite():
    """Integration/stress test: no existing test ran more than ~4 rollout+
    update() cycles. Runs 30 full cycles (all 4 regions active, mixed
    dones/softstop_flag patterns every step, matching real usage of the
    2026-09-15 exclusion fixes together with the persistent replay buffer)
    and asserts every discriminator's parameters, every returned loss
    component, and the shared normalizer's running stats stay finite the
    entire time -- would catch a slow NaN/Inf drift that a short test can't,
    e.g. from the persistent (never-cleared) replay buffer compounding a
    numerical issue over many updates, or the disc_logit_reg/grad_pen
    combination blowing up under sustained training."""
    torch.manual_seed(0)
    num_envs = 16
    alg, actor_critic, discriminators = _make_alg(num_envs=num_envs, num_transitions=4)
    region_ids = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                                2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])

    for cycle in range(30):
        for _ in range(4):  # num_transitions
            obs_current = torch.randn(num_envs, NUM_ONE_STEP_OBS)
            obs_history = torch.randn(num_envs, NUM_ONE_STEP_OBS * HISTORY_LEN)
            critic_obs = torch.randn(num_envs, NUM_CRITIC_OBS)
            critic_obs[:, -1] = region_ids
            amp_obs = torch.randn(num_envs, AMP_OBS_DIM)
            alg.act(obs_current, obs_history, critic_obs, amp_obs)

            dones = torch.rand(num_envs) < 0.1
            softstop_flag = torch.rand(num_envs) < 0.3
            alg.process_env_step(
                rewards=torch.randn(num_envs), dones=dones,
                infos={}, amp_obs=torch.randn(num_envs, AMP_OBS_DIM),
                softstop_flag=softstop_flag,
            )
        alg.compute_returns(torch.randn(num_envs, NUM_CRITIC_OBS))
        result = alg.update()

        for name, value in zip(
            ["value_loss", "surrogate_loss", "amp_loss", "grad_pen_loss",
             "est_loss", "region_loss", "policy_pred", "expert_pred"],
            result,
        ):
            assert np.isfinite(value), f"cycle {cycle}: {name}={value} is not finite"

        for region_name, discr in discriminators.items():
            for p in list(discr.trunk.parameters()) + list(discr.amp_linear.parameters()):
                assert torch.isfinite(p).all(), (
                    f"cycle {cycle}: {region_name} discriminator has non-finite "
                    f"parameters after update()"
                )

        assert np.isfinite(alg.amp_normalizer.mean).all()
        assert np.isfinite(alg.amp_normalizer.var).all()
        assert (alg.amp_normalizer.var >= 0).all()

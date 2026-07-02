# Revert beyondAMP Smoothing — Restore Training Stability

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip the policy/value smoothing loss that was added to beyondAMP after commit `4afc861`, restore the original upstream algorithm, and delete all dead code introduced alongside it.

**Architecture:** The smoothing loss added in `5aab674` required structural changes to `RolloutStorage` (extra buffers), `AMPPPO`/`PPO` (extra `__init__` params + new `process_env_step` signature), the mini-batch generator (extra yields), and the runners (dead `obs_prev` clones). All of those changes are reverted here. The discriminator, reward computation, and task-level code are untouched.

**Tech Stack:** Python 3.11, PyTorch, beyondAMP (`rsl_rl_amp` package at `SimpleGoalKeeper/beyondAMP/source/rsl_rl_amp/`)

## Global Constraints

- Working directory: `/home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper/`
- beyondAMP source lives at `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/`
- **Do NOT touch the discriminator, AMP reward computation, or any task/env code** — only the 6 files listed below
- `amp_replay_buffer_size=500_000` in `goalkeeper_amp_cfg.py` is a **keeper** — do not remove it
- Verify import sanity after each file: `cd SimpleGoalKeeper && uv run python -c "from rsl_rl_amp.<module> import <Class>"`

---

## File Map

| File | Change |
|---|---|
| `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/storage/rollout_storage.py` | Remove `next_obs`/`next_critic_obs` from `Transition`, storage buffers, `add_transitions`, and `mini_batch_generator` |
| `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/algorithms/amp_ppo/amp_ppo.py` | Remove 3 smoothing params, restore `process_env_step` signature, restore mini-batch unpack, delete smooth loss block |
| `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/algorithms/ppo.py` | Same as `amp_ppo.py` |
| `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/runners/amp_on_policy_runner.py` | Delete dead `obs_prev`/`critic_obs_prev` clones; restore `process_env_step` call to 4-arg form |
| `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/runners/on_policy_runner.py` | Restore `process_env_step` call to 3-arg form |
| `src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py` | Remove the 3 smoothing kwargs that are no longer accepted |

---

## Task 1: Revert `rollout_storage.py`

**Files:**
- Modify: `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/storage/rollout_storage.py`

**What to do:** Remove `next_observations` and `next_critic_observations` from `Transition.__init__`. Remove the two extra buffers from `RolloutStorage.__init__`. Remove the two extra `copy_` calls in `add_transitions`. Restore `mini_batch_generator` to yield 11 items (remove `next_obs_batch`, `next_critic_observations_batch`, `cont_batch`, and the `dones` flatten).

- [ ] **Step 1: Edit `Transition.__init__` — remove next obs fields**

In `rollout_storage.py` lines 38–50, change:
```python
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.next_observations = None
            self.next_critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()
```
to:
```python
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
        
        def clear(self):
            self.__init__()
```

- [ ] **Step 2: Edit `RolloutStorage.__init__` — remove next_observations buffers**

Lines 64–72, change:
```python
        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
            self.next_privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
            self.next_privileged_observations = None
```
to:
```python
        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
```

- [ ] **Step 3: Edit `add_transitions` — remove next obs copy**

Lines 97–101, change:
```python
        self.observations[self.step].copy_(transition.observations)
        self.next_observations[self.step].copy_(transition.next_observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
            self.next_privileged_observations[self.step].copy_(transition.next_critic_observations)
```
to:
```python
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
```

- [ ] **Step 4: Edit `mini_batch_generator` — restore original 11-item yield**

Lines 160–199, change the entire generator body to:
```python
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None
```

- [ ] **Step 5: Verify import**

```bash
cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run python -c "from rsl_rl_amp.storage import RolloutStorage; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add SimpleGoalKeeper/beyondAMP/source/rsl_rl_amp/rsl_rl_amp/storage/rollout_storage.py
git commit -m "revert(beyondAMP): remove next_obs buffers from RolloutStorage — smoothing reverted"
```

---

## Task 2: Revert `amp_ppo.py`

**Files:**
- Modify: `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/algorithms/amp_ppo/amp_ppo.py`

**What to do:** Remove 3 smoothing `__init__` params and their `self.` assignments. Restore `process_env_step` to 4-arg form (drop `next_obs`, `next_critic_obs`). Restore mini-batch unpack to 11-item form. Delete the smooth loss block. Restore total loss to `surrogate + value - entropy + amp + grad_pen`.

- [ ] **Step 1: Remove smoothing params from `__init__`**

Lines 33–35, remove these three lines entirely:
```python
                 value_smoothness_coef=0.0,
                 smoothness_upper_bound=1.0,
                 smoothness_lower_bound=0.01,
```

Also remove the 5-line comment block and the three `self.` assignments that follow (lines ~82–87):
```python
        # Policy & Value smoothing parameters (from G1 HIM-PPO)
        self.value_smoothness_coef = value_smoothness_coef
        self.smoothness_upper_bound = smoothness_upper_bound
        self.smoothness_lower_bound = smoothness_lower_bound
```

- [ ] **Step 2: Restore `process_env_step` signature**

Change:
```python
    def process_env_step(self, rewards, dones, infos, amp_obs, next_obs, next_critic_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_observations = next_obs
        self.transition.next_critic_observations = next_critic_obs
```
to:
```python
    def process_env_step(self, rewards, dones, infos, amp_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
```

- [ ] **Step 3: Restore mini-batch unpack (11 items)**

Change:
```python
                obs_batch, next_obs_batch, critic_obs_batch, next_critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch, cont_batch, hid_states_batch, masks_batch = sample
```
to:
```python
                obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
```

- [ ] **Step 4: Delete smooth loss block**

Remove the entire block (currently lines ~202–212):
```python
                # Policy & Value smoothing loss (from G1 HIM-PPO)
                epsilon = self.smoothness_lower_bound / (self.smoothness_upper_bound - self.smoothness_lower_bound)
                policy_smooth_coef = self.smoothness_upper_bound * epsilon
                value_smooth_coef = self.value_smoothness_coef * policy_smooth_coef

                mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0
                mix_obs_batch = obs_batch + mix_weights * (next_obs_batch - obs_batch)
                mix_critic_obs_batch = critic_obs_batch + mix_weights * (next_critic_obs_batch - critic_obs_batch)
                policy_smooth_loss = torch.square(torch.norm(self.actor_critic.act_inference(mix_obs_batch) - mu_batch, dim=-1)).mean()
                value_smooth_loss = torch.square(torch.norm(self.actor_critic.evaluate(mix_critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]) - value_batch, dim=-1)).mean()
                smooth_loss = policy_smooth_coef * policy_smooth_loss + value_smooth_coef * value_smooth_loss
```

- [ ] **Step 5: Restore total loss (remove `smooth_loss`)**

Change:
```python
                loss = (
                    surrogate_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy_batch.mean() +
                    amp_loss + grad_pen_loss +
                    smooth_loss)
```
to:
```python
                loss = (
                    surrogate_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy_batch.mean() +
                    amp_loss + grad_pen_loss)
```

- [ ] **Step 6: Verify import**

```bash
cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run python -c "from rsl_rl_amp.algorithms import AMPPPO; print('OK')"
```
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add SimpleGoalKeeper/beyondAMP/source/rsl_rl_amp/rsl_rl_amp/algorithms/amp_ppo/amp_ppo.py
git commit -m "revert(beyondAMP): remove smooth loss from AMPPPO — restore original AMP update"
```

---

## Task 3: Revert `ppo.py`

**Files:**
- Modify: `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/algorithms/ppo.py`

**What to do:** Same pattern as Task 2 but for the plain PPO class. Remove smoothing params, restore `process_env_step`, restore mini-batch unpack, delete smooth loss.

- [ ] **Step 1: Remove smoothing params from PPO `__init__`**

Remove these three param lines (around line 54–56):
```python
                 value_smoothness_coef=0.1,
                 smoothness_upper_bound=1.0,
                 smoothness_lower_bound=0.1,
```

Remove the comment block and three `self.` assignments (around line 85–88):
```python
        # Policy & Value smoothing parameters (from G1 HIM-PPO)
        self.value_smoothness_coef = value_smoothness_coef
        self.smoothness_upper_bound = smoothness_upper_bound
        self.smoothness_lower_bound = smoothness_lower_bound
```

- [ ] **Step 2: Restore PPO `process_env_step` signature**

Change:
```python
    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_observations = next_obs
        self.transition.next_critic_observations = next_critic_obs
```
to:
```python
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
```

- [ ] **Step 3: Restore PPO mini-batch unpack (11 items)**

Change:
```python
        for obs_batch, next_obs_batch, critic_obs_batch, next_critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, cont_batch, hid_states_batch, masks_batch in generator:
```
to:
```python
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
```

- [ ] **Step 4: Delete PPO smooth loss block and revert `loss`**

Change this entire section:
```python
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Policy & Value smoothing loss (from G1 HIM-PPO, lines 231-242)
                epsilon = self.smoothness_lower_bound / (self.smoothness_upper_bound - self.smoothness_lower_bound)
                policy_smooth_coef = self.smoothness_upper_bound * epsilon
                value_smooth_coef = self.value_smoothness_coef * policy_smooth_coef

                mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0
                mix_obs_batch = obs_batch + mix_weights * (next_obs_batch - obs_batch)
                mix_critic_obs_batch = critic_obs_batch + mix_weights * (next_critic_obs_batch - critic_obs_batch)
                policy_smooth_loss = torch.square(torch.norm(mu_batch - self.actor_critic.act_inference(mix_obs_batch), dim=-1)).mean()
                value_smooth_loss = torch.square(torch.norm(value_batch - self.actor_critic.evaluate(mix_critic_obs_batch), dim=-1)).mean()
                smooth_loss = policy_smooth_coef * policy_smooth_loss + value_smooth_coef * value_smooth_loss

                loss += smooth_loss
```
to:
```python
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
```

- [ ] **Step 5: Verify import**

```bash
cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run python -c "from rsl_rl_amp.algorithms import PPO; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add SimpleGoalKeeper/beyondAMP/source/rsl_rl_amp/rsl_rl_amp/algorithms/ppo.py
git commit -m "revert(beyondAMP): remove smooth loss from PPO — restore original update"
```

---

## Task 4: Clean `amp_on_policy_runner.py` — delete dead code + restore call

**Files:**
- Modify: `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/runners/amp_on_policy_runner.py`

**What to do:** Delete the two dead `.clone()` lines that were never read. Change the `process_env_step` call back to 4 arguments.

- [ ] **Step 1: Delete dead clone lines**

Lines 118–119 — remove both of these:
```python
                    obs_prev = obs.clone()
                    critic_obs_prev = critic_obs.clone()
```

The loop body should go directly from the `actions = self.alg.act(...)` call to the `obs, privileged_obs, rewards, dones, ... = self.env.step(...)` call.

- [ ] **Step 2: Restore `process_env_step` call to 4-arg form**

Change:
```python
                    self.alg.process_env_step(lerp_rewards, dones, infos, next_amp_obs_with_term, obs, critic_obs)
```
to:
```python
                    self.alg.process_env_step(lerp_rewards, dones, infos, next_amp_obs_with_term)
```

- [ ] **Step 3: Verify import**

```bash
cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run python -c "from rsl_rl_amp.runners import AMPOnPolicyRunner; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add SimpleGoalKeeper/beyondAMP/source/rsl_rl_amp/rsl_rl_amp/runners/amp_on_policy_runner.py
git commit -m "fix(beyondAMP): delete dead obs_prev clones; restore 4-arg process_env_step"
```

---

## Task 5: Restore `on_policy_runner.py`

**Files:**
- Modify: `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/runners/on_policy_runner.py`

**What to do:** The plain PPO runner's `process_env_step` call gained 2 extra args (`obs, critic_obs`). Restore it to 3 args.

- [ ] **Step 1: Restore call**

Change:
```python
                    self.alg.process_env_step(rewards, dones, infos, obs, critic_obs)
```
to:
```python
                    self.alg.process_env_step(rewards, dones, infos)
```

- [ ] **Step 2: Verify import**

```bash
cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run python -c "from rsl_rl_amp.runners import AMPOnPolicyRunner; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add SimpleGoalKeeper/beyondAMP/source/rsl_rl_amp/rsl_rl_amp/runners/on_policy_runner.py
git commit -m "revert(beyondAMP): restore on_policy_runner process_env_step to 3-arg form"
```

---

## Task 6: Clean `goalkeeper_amp_cfg.py` — remove orphaned smoothing kwargs

**Files:**
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py`

**What to do:** AMPPPO no longer accepts `value_smoothness_coef`, `smoothness_upper_bound`, or `smoothness_lower_bound`. Remove those three kwargs and the comment above them. Keep `amp_replay_buffer_size=500_000`.

- [ ] **Step 1: Remove the 4 lines (comment + 3 kwargs)**

Lines 74–79, remove:
```python
            # Value smooth loss disabled: scales as V² and blows up with
            # goalkeeper reward magnitudes (100-250). Policy smooth coef
            # reduced 10× vs G1 HIM-PPO default for the same reason.
            value_smoothness_coef=0.0,
            smoothness_upper_bound=1.0,
            smoothness_lower_bound=0.01,
```

The `algorithm` block should end at `amp_replay_buffer_size=500_000,` followed by the closing `),`.

- [ ] **Step 2: Verify the full training config loads without errors**

```bash
cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run python -c "
from simple_goalkeeper.tasks.goalkeeper_amp_cfg import goalkeeper_amp_runner_cfg
cfg = goalkeeper_amp_runner_cfg()
print('algorithm class:', cfg.algorithm.class_name)
print('amp_replay_buffer_size:', cfg.algorithm.amp_replay_buffer_size)
print('OK')
"
```
Expected output:
```
algorithm class: AMPPPO
amp_replay_buffer_size: 500000
OK
```

- [ ] **Step 3: Verify training can be instantiated end-to-end (dry run)**

```bash
cd /home/ibouwmeest/BEPImitationLearning/SimpleGoalKeeper && uv run python -c "
import torch
from rsl_rl_amp.storage import RolloutStorage
s = RolloutStorage(4, 10, (8,), (None,), (3,))
from rsl_rl_amp.storage.rollout_storage import RolloutStorage
gen = s.mini_batch_generator(2, 1)
batch = next(gen)
assert len(batch) == 11, f'expected 11 items, got {len(batch)}'
print('mini_batch_generator yields 11 items: OK')
"
```
Expected: `mini_batch_generator yields 11 items: OK`

- [ ] **Step 4: Commit**

```bash
git add SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py
git commit -m "chore(sgk): remove orphaned smoothing kwargs from AMPPPO config"
```

---

## Self-Review

**Spec coverage:**
- ✅ Dead code removed (`obs_prev`, `critic_obs_prev`)
- ✅ Smooth loss reverted from `amp_ppo.py`
- ✅ Smooth loss reverted from `ppo.py`
- ✅ `process_env_step` signature restored in both algorithms and both runners
- ✅ `RolloutStorage` buffers and `Transition` fields cleaned up
- ✅ `mini_batch_generator` restored to 11-item yield (matching original AMPPPO unpack)
- ✅ `goalkeeper_amp_cfg.py` no longer passes kwargs that don't exist
- ✅ `amp_replay_buffer_size=500_000` is explicitly preserved

**No placeholders:** All steps contain exact file paths and complete code.

**Type consistency:** `mini_batch_generator` yields 11 items → `AMPPPO.update()` unpacks 11 items → ✅

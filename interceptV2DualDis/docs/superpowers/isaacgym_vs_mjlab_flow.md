# Isaac Gym (G1 reference) vs. mjlab (interceptV2DualDis) — phase-by-phase information flow

**Purpose.** A low-level, line-referenced trace of how state flows through one RL step in each
framework, so that porting bugs (especially ones that make AMP policy transitions systematically
separable from expert transitions) can be spotted by comparison rather than by guesswork.

**Sources of truth** (all read directly for this document; docs were used only to cross-check):

| Side | Files |
|---|---|
| Isaac Gym | `/home/robocup/IsaakB/BEPImitationLearning/Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py`, `.../envs/g1/g1_utils.py`, `.../envs/g1/g1_29_config.py`, `Humanoid-Goalkeeper/rsl_rl/rsl_rl/algorithms/him_ppo.py`, `.../runners/him_on_policy_runner.py`, `.../modules/amp.py`, `.../utils/utils.py` |
| mjlab | `/home/robocup/IsaakB/BEPImitationLearning/interceptV2DualDis/.venv/lib/python3.11/site-packages/mjlab/envs/manager_based_rl_env.py`, `.../managers/{observation,termination,reward,action,event}_manager.py`, `.../utils/buffers/circular_buffer.py`, `.../actuator/builtin_group.py`, `.../entity/entity.py` |
| Our port | `interceptV2DualDis/beyondAMP/source/beyondAMP/beyondAMP/mjlab/rsl_rl/{amp_wrapper,vecenv_wrapper}.py`, `interceptV2DualDis/src/simple_goalkeeper/rsl_rl_multi/{him_amp_on_policy_runner,multi_disc_amp_ppo}.py`, `.../beyondAMP/motion/motion_dataset.py`, `.../rsl_rl_amp/{modules/amp_discriminator,storage/replay_buffer,utils/utils}.py`, `interceptV2DualDis/src/simple_goalkeeper/tasks/goalkeeper_multidisc_amp_cfg.py`, `interceptV2DualDis/src/simple_goalkeeper/mdp/{observations,rewards,events}.py` |

Timing constants: both sides run a **50 Hz control loop**. G1: `decimation` substeps of the PhysX dt.
mjlab: `timestep=0.005`, `decimation=4` (inherited from `mjlab/tasks/velocity/velocity_env_cfg.py:446,451`
via `make_velocity_env_cfg()`), so `step_dt = 0.02 s`, `physics_dt = 0.005 s`.

---

## Contents

1. [Phase 0 — one `step()` call, top-level order](#phase-0)
2. [Phase 1 — action ingestion, clipping, delay, domain randomization](#phase-1)
3. [Phase 2 — the physics substep loop and what gets refreshed inside it](#phase-2)
4. [Phase 3 — derived-quantity freshness at reward/termination time](#phase-3)
5. [Phase 4 — termination check](#phase-4)
6. [Phase 5 — reward computation](#phase-5)
7. [Phase 6 — reset (`reset_idx` / `_reset_idx`)](#phase-6)
8. [Phase 7 — observation computation: pre- vs post-reset](#phase-7)
9. [Phase 8 — terminal-observation capture and the 7-tuple](#phase-8)
10. [Phase 9 — extras, `time_outs`, episode logging](#phase-9)
11. [Phase 10 — per-env indexing / ordering consistency](#phase-10)
12. [Phase 11 — AMP policy transition assembly](#phase-11)
13. [Phase 12 — AMP expert transition sampling](#phase-12)
14. [Phase 13 — normalizer](#phase-13)
15. [Phase 14 — discriminator batching, replay, loss, reward mixing](#phase-14)
16. [Ranked divergences](#ranked)

---

<a name="phase-0"></a>
## Phase 0 — one `step()` call, top-level order

### Isaac Gym

`legged_robot.py:121-157` (`step`) → `legged_robot.py:167-248` (`post_physics_step`):

```
step(actions):
  1. actions = clip(actions, ±cfg.normalization.clip_actions)            # :127-128
  2. build delayed_actions[decimation, N, A]                             # :130-134
  3. randomize joint_injection (once per control step)                   # :137-139
  4. render()                                                            # :141
  5. for i in range(decimation):                                         # :142-148
        torques = _compute_torques(delayed_actions[i])
        gym.set_dof_actuation_force_tensor(torques)
        gym.simulate()
        [cpu only] gym.fetch_results()
        gym.refresh_dof_state_tensor()            <-- ONLY dof state
  6. termination_ids, termination_privileged_obs = post_physics_step()   # :149
  7. obs_buf = clip(obs_buf, ±clip_observations)                         # :152-155
     privileged_obs_buf = clip(..., ±clip_observations)
  8. return (obs, privileged_obs, rew, reset_buf, extras,
             termination_ids, termination_privileged_obs)                # :157

post_physics_step():
  a. refresh actor_root_state / net_contact_force / rigid_body_state     # :172-174
  b. episode_length_buf += 1; common_step_counter += 1; catchstep -= 1   # :176-178
  c. recompute base_quat, rpy, base_lin/ang_vel, torso_pos,
     projected_gravity, base_lin_acc, joint_powers                       # :180-198
  d. _post_physics_step_callback()  -> _randomize_balls, _push_robots    # :200, :619-625
  e. end_target / per-region hand distance bookkeeping                   # :203-223
  f. compute_reward()                          <-- REWARD FIRST          # :227
  g. check_termination()                       <-- TERMINATION SECOND    # :229
  h. env_ids = reset_buf.nonzero()                                       # :231
  i. termination_privileged_obs = compute_termination_observations(env_ids)
                                               <-- PRE-RESET CAPTURE     # :233
  j. reset_idx(env_ids)                                                  # :235
  k. compute_observations()                    <-- POST-RESET            # :237
  l. last_last_actions/last_actions/last_dof_vel/last_torques/
     last_root_vel updated                                               # :239-243
  m. return env_ids, termination_privileged_obs                          # :248
```

### mjlab

`manager_based_rl_env.py:378-479`:

```
step(action):
  1. guard: auto_reset=False + pending manual reset -> RuntimeError      # :411-416
  2. extras["log"] = {}                                                  # :418
  3. action_manager.process_action(action)   (clip/scale/offset, once)   # :419
  4. for _ in range(decimation):                                         # :421-427
        _sim_step_counter += 1
        action_manager.apply_action()
        scene.write_data_to_sim()             <-- actuator ctrl + delay
        sim.step()
        scene.update(dt=physics_dt)
        metrics_manager.compute_substep()
  5. episode_length_buf += 1; common_step_counter += 1                   # :430-431
  6. reset_buf = termination_manager.compute()  <-- TERMINATION FIRST    # :436-438
  7. reward_buf = reward_manager.compute(dt=step_dt)  <-- REWARD SECOND  # :440
  8. metrics_manager.compute()                                           # :441
  9. if auto_reset and any done:                                         # :444-448
        recorder.record_pre_reset(ids); _reset_idx(ids); scene.write_data_to_sim()
 10. sim.forward()               <-- single refresh of derived qtys      # :454
 11. command_manager.compute(dt=step_dt)                                 # :456
 12. event_manager.apply("step"); event_manager.apply("interval")        # :458-461
 13. sim.sense()                                                         # :463
 14. obs_buf = observation_manager.compute(update_history=True)          # :464
 15. return (obs_buf, reward_buf, terminated, truncated, extras)         # :473-479
```

**Verdict: DIVERGES** in three structural ways — (a) reward/termination order is swapped,
(b) the terminal-observation hook (step i above) has no mjlab equivalent, (c) mjlab's
`"step"`/`"interval"` events run *after* the reset block (mjlab's own published docs show them
*before*; the **installed source is authoritative and puts them after**). Each is treated
separately below.

---

<a name="phase-1"></a>
## Phase 1 — action ingestion, clipping, delay, domain randomization

### Isaac Gym

- Clip: `torch.clip(actions, ±cfg.normalization.clip_actions)` once per control step (`:127-128`).
- **Action delay** (`:130-134`): `delayed_actions` starts as `actions` repeated `decimation` times.
  If `domain_rand.delay`, one `delay_steps ~ randint(0, decimation)` is drawn **per env, per control
  step**, and substep `i` gets `last_actions + (actions - last_actions) * (i >= delay_steps)`.
  This is a **step function**, not a ramp — the actuator target jumps from the previous control
  step's action to this one's at a random substep boundary. Lag ∈ {0 … decimation-1} substeps,
  and **0 lag is possible**.
- **Joint injection** (`:137-139`): `joint_injection ~ U(range) * torque_limits`, sampled once per
  control step, held for all substeps, zeroed on `curriculum_dof_indices`.
- **Torque law** (`_compute_torques`, `:627-655`):
  `joint_pos_target = default_dof_poses + action*action_scale`; then
  **overridden to `init_dof_pos` for envs with `catchstep > startstep`** (`:643`) — i.e. during the
  first few steps of an episode the policy's action is *discarded* and the robot is PD-held at its
  reset pose. Then `τ = p_gains*Kp_factors*(target - dof_pos) - d_gains*Kd_factors*dof_vel`,
  `+ actuation_offset + joint_injection`, clipped to `±torque_limits`.
  `dof_pos`/`dof_vel` here are **fresh each substep** (see Phase 2).

### mjlab

- Clip: `AMPEnvWrapper.step` → `RslRlVecEnvWrapper.step` clamps to `self.clip_actions` if set
  (`vecenv_wrapper.py:65-66`, `amp_wrapper.py:72-73`); then `action_manager.process_action` applies
  per-actuator `scale`/`offset`/`clip`.
- **Action delay**: not in the action manager. It lives in the actuator layer —
  `BuiltinPositionActuatorCfg(delay_min_lag=1, delay_max_lag=3)` for every T1 actuator group
  (`t1_constants.py:47` `DELAY_MIN, DELAY_MAX = 1, 3`, applied at `:53,60,66,72,78`).
  `BuiltinActuatorGroup.apply_controls` (`builtin_group.py:186-192`) **appends to the DelayBuffer
  once per call**, and `apply_controls` is reached via `scene.write_data_to_sim()` **inside the
  decimation loop** → the buffer advances at **physics rate**, so the lag is **1–3 physics substeps
  (5–15 ms)** and **never zero**.
- No joint-injection / actuation-offset equivalent registered.
- No `catchstep`-style PD hold: SGK's `_catchstep` (`events.py:593-635`) only gates **ball
  visibility** in observations (`observations.py:48-57,176-182`). The policy commands the robot from
  step 0 of every episode.

**Verdict: DIVERGES (port does NOT match, deliberately in part).**
- Delay semantics differ: G1 = 0…(dec-1) substeps, resampled per control step, applied as a
  last→current step change; mjlab = 1…3 substeps, resampled at physics rate, applied as a FIFO
  lookup that carries targets across control-step boundaries. Same order of magnitude, different
  distribution, and mjlab's never has zero lag. Documented as "SGK's action-delay equivalent" in
  CLAUDE.md; the substep-rate resampling is **not** documented.
- `joint_injection` and `actuation_offset` (G1's two additive torque DR terms) have **no port**.
- G1's warm-up PD hold has no port. Consequence for AMP: G1's first few AMP frames per episode are
  a *physically settling* robot near its init pose; SGK's are a *policy-driven* robot from a
  randomized pose. See divergence #4.

---

<a name="phase-2"></a>
## Phase 2 — the physics substep loop and what gets refreshed inside it

### Isaac Gym

Inside the loop (`:142-148`) **only `refresh_dof_state_tensor` is called**. Root states, net contact
forces and rigid-body states are refreshed **once**, at the top of `post_physics_step` (`:172-174`).
Practical consequences:

- `_compute_torques` sees `dof_pos`/`dof_vel` from the immediately preceding substep — correct PD.
- Anything reading `root_states`, `contact_forces`, `rigid_body_states` *inside* the loop would see
  pre-loop values. Nothing does.
- After `:172-174`, **every** tensor is consistent with the final `gym.simulate()`.
- `self.torques` retained after the loop is the **last substep's** torque, and that is what
  `joint_powers` (`:197`) and `_reward_torques` consume — not an average.

### mjlab

Inside the loop: `apply_action → write_data_to_sim → sim.step() → scene.update(dt)` each substep.
`sim.step()` is MuJoCo's `mj_step`, which runs forward kinematics *before* integration. mjlab's own
`step()` docstring (`manager_based_rl_env.py:386-401`) states this explicitly: after the loop,
`xpos`, `xquat`, `site_xpos`, `cvel`, `sensordata` **lag `qpos`/`qvel` by one physics substep**.
mjlab resolves this with **one** `sim.forward()` (`:454`) placed **after** the reset block and
**before** observation computation, deliberately accepting staleness at reward/termination time.

`sim.sense()` is likewise called **once**, at `:463` — after the reset block. Sensor-derived
quantities read by termination/reward terms therefore come from whatever the last in-loop
`scene.update()` left, while sensor-derived *observation* terms read the post-reset `sense()`.

**Verdict: DIVERGES (port does NOT compensate).** G1 guarantees "all tensors consistent with the
final physics step" before any reward/termination/observation code runs. mjlab guarantees it only
for observations. See Phase 3.

---

<a name="phase-3"></a>
## Phase 3 — derived-quantity freshness at reward/termination time

### Isaac Gym

Fully fresh (Phase 2). `post_physics_step` additionally *derives* and caches, in this order
(`:180-198`): `base_quat`, `roll/pitch/yaw`, `base_lin_vel`/`base_ang_vel` (note: taken from the
**upper-body link's** rigid-body state, not the root — `:184-185`), `torso_pos`,
`projected_gravity` (also from the upper-body quaternion, `:190`), `base_lin_acc` (finite difference
against `last_root_vel`, `:191`), and the `joint_powers` ring buffer.

### mjlab

At `termination_manager.compute()` / `reward_manager.compute()` time, `xpos`/`xquat`/`site_xpos`/
`cvel`/`sensordata` are **one physics substep (5 ms) stale**; `qpos`/`qvel` are current.
Every SGK reward/termination that reads body pose or velocity in the world frame
(`root_link_pos_w`, `root_link_lin_vel_w`, foot positions, contact forces) is reading a 5 ms-old
snapshot. mjlab argues the staleness is *uniform*, so the MDP stays well-defined.

**Verdict: DIVERGES; port does not compensate; low-to-medium impact.** It is uniform, so it does
not create a policy-vs-expert tell. It *does* shift threshold-crossing events by one substep, which
matters for `sharpforce` (force thresholds), `stopball`/`softstop` (`delta_vx` thresholds) and
`_ball_is_behind` — all of which are latched booleans that then gate other machinery.

---

<a name="phase-4"></a>
## Phase 4 — termination check

### Isaac Gym

`check_termination` (`:250-262`), run **after** `compute_reward`:

```python
self.reset_buf = min(rigid_body_states[:, knee_indices, 2], dim=-1).values < 0.10
self.time_out_buf = self.episode_length_buf > self.max_episode_length        # strict >
self.gravity_termination_buf = any(norm(projected_gravity[:, 0:2]) > 0.8)
sharpforce_buf = mean(norm(contact_forces[:, contact_feet_indices, :])) > 1.5 * max_contact_force
self.reset_buf |= time_out_buf | gravity_termination_buf | sharpforce_buf
```

Note `self.reset_buf` is **rebound to a new tensor** every step; `reset_idx` then sets
`reset_buf[env_ids] = 1` (`:300`, a no-op given they are already 1). The returned `dones` is this
same tensor. Timeouts are folded into `dones`, and separately exposed through `extras["time_outs"]`.

### mjlab

`TerminationManager.compute()` (`termination_manager.py:102-113`) clears `_truncated_buf` and
`_terminated_buf`, evaluates each term, ORs into truncated (if `time_out=True`) or terminated,
records per-term dones for logging, and returns `truncated | terminated`.
`ManagerBasedRlEnv.step` stores `reset_terminated`/`reset_time_outs` separately and returns both;
`vecenv_wrapper.py:68` collapses them: `dones = (terminated | truncated).long()`.

SGK terms: `time_out`, `bad_orientation` (G1-exact XY-norm formula), `base_height`, `ball_exit`,
`sharpforce`.

**Verdict: MATCHES in substance, DIVERGES in ordering (see Phase 5) and in one off-by-one:**
mjlab's stock `mdp.time_out` is `episode_length_buf >= max_episode_length`, G1's is
`episode_length_buf > max_episode_length`. One extra step per episode in G1. Cosmetic.

---

<a name="phase-5"></a>
## Phase 5 — reward computation

### Isaac Gym

`compute_reward` (`:350-388`) runs **before** `check_termination`.

- `rew_buf[:] = 0`.
- A handful of scales are **rewritten in place every step** from the curriculum counter
  (`eereach`, `success`, `stopball` × `(1 + 0.5*curriculumupdate)`; `dof_pos_limits`/`torque_limits`
  stepped at `curriculumupdate > 1.0 / > 2.0`).
- Loop over `reward_functions`, accumulate into `rew_buf` and `episode_sums`.
- `only_positive_rewards` clamp at `min=0` **before** the termination reward is added.
- `_reward_termination()` reads `self.reset_buf` — which at that moment still holds **the previous
  step's** value, because `check_termination` has not run yet. This is a genuine upstream quirk.
- Per-step dt scaling is *pre-baked*: `_prepare_reward_function` multiplies every
  `reward_scales[key]` by `self.dt` at startup.

### mjlab

`RewardManager.compute(dt=step_dt)` (`reward_manager.py:116-133`) runs **after** the termination
manager. Per term: skip if `weight == 0.0`; `value = func(env) * weight * dt`;
`nan_to_num(value, 0.0)`; accumulate into `_reward_buf` and `_episode_sums`; store the *unscaled*
rate in `_step_reward`. No global positive clamp, no termination-reward special case.
`Episode_Reward/<term>` is logged at reset as `mean(episode_sum) / max_episode_length_s`.

**Verdict: DIVERGES on ordering; MATCHES on dt scaling.**
Because mjlab evaluates terminations first, an SGK reward term that reads
`env.termination_manager.terminated` (or any flag a termination term latched) sees **this step's**
value, whereas the equivalent G1 term would see the previous step's. Conversely G1's
`only_positive_rewards` clamp and its "termination reward added after clipping" structure have no
mjlab analog — SGK does not use either, so this is inert today.
mjlab's `nan_to_num` silently zeroes a NaN reward term; G1 lets it propagate (and `HIMPPO.act`
zeroes the *observation* batch instead, `him_ppo.py:138-140` — ported at
`multi_disc_amp_ppo.py:156-159`).

---

<a name="phase-6"></a>
## Phase 6 — reset

### Isaac Gym — `reset_idx(env_ids)` (`:266-348`)

Order, exactly:

1. Early return if `len(env_ids) == 0`.
2. Success-rate bookkeeping (`:280-285`).
3. `refresh_actor_rigid_shape_props(env_ids)` — resamples friction/restitution **per reset**,
   then writes them through the (slow, per-env Python loop) Gym API (`:514-531`).
4. `_reset_dofs(env_ids)` (`:657-689`):
   - `continue_keep` branch, taken when `torch.rand(1).item() > 0.2` — i.e. **one coin flip for the
     whole batch**, ~80% of the time: `dof_pos[env_ids] = dof_pos[randint(0, num_envs, len(env_ids))]`
     — a **donor copy from any currently-running env**, unscoped by region, un-clamped.
   - else branch (~20%): `standpos * U(0.5, 1.5) + U(-0.1, 0.1)`, clipped to `dof_pos_limits`
     (which `_process_dof_props` has already overwritten to the **soft** 0.9× limits, `:545-563`).
   - `init_dof_pos[env_ids] = dof_pos[env_ids]` (used by the warm-up PD hold).
   - `dof_vel[env_ids] = 0.` unconditionally.
   - `set_dof_state_tensor_indexed` with actor index `2*env_ids` (robot is actor 0 of 2 per env).
5. `_reset_root_states(env_ids)` (`:693-735`): root ← `base_init_state + env_origins`; root velocity
   ← `U(-0.3, 0.3)` on **all 6 components**, every reset; ball state assigned; `vanish_step` ←
   `randint(0, 30)`; `set_actor_root_state_tensor_indexed` over `[2*ids, 2*ids+1]`.
6. Zero `last_actions`, `last_last_actions`, `last_dof_vel`, `last_torques`, `joint_powers`;
   `reset_buf[env_ids] = 1`.
7. Resample `Kp_factors`, `Kd_factors`, `actuation_offset` per reset.
8. Fill `extras["episode"]` from `episode_sums` **then zero them**; `extras["time_outs"] = time_out_buf`.
9. Every ~500 global steps: recompute `startstep = 50 - randint(3,10)`, `curriculumupdate`, and the
   command ranges.
10. `episode_length_buf[env_ids] = 0`  ← **last**, after the extras have used it as a divisor.

Critically, **writes take effect in the CPU-side tensors immediately** (`self.dof_pos` is a view of
`dof_state`), so any code after `reset_idx` that reads `dof_pos`/`root_states` sees post-reset values
even before the next `gym.simulate()`.

### mjlab — `_reset_idx(env_ids)` (`manager_based_rl_env.py:553-591`)

1. `curriculum_manager.compute(env_ids)`
2. `sim.reset(env_ids)`; `scene.reset(env_ids)`
3. `event_manager.apply(mode="reset", env_ids=..., global_env_step_count=...)`
   — this is where SGK's `assign_static_regions` → `reset_ball` → `reset_from_motion_data` →
   `tick_catchstep` ordering lives (dict insertion order, re-established explicitly in
   `goalkeeper_multidisc_amp_cfg.py:186-221`).
4. `observation_manager.reset(env_ids)` — **invalidates the obs cache and resets the per-term
   `DelayBuffer`s and history `CircularBuffer`s for those envs**.
5. `action_manager.reset` → `reward_manager.reset` (emits `Episode_Reward/*`, zeroes sums) →
   `metrics` → `curriculum` → `command` → `event` → `termination` (emits `Episode_Termination/*`).
6. `episode_length_buf[env_ids] = 0`; `_manual_reset_pending[env_ids] = False`.

Back in `step()`, `scene.write_data_to_sim()` (`:448`) flushes the written state, and `sim.forward()`
(`:454`) recomputes derived quantities for **all** envs from current `qpos`/`qvel`.

SGK's own reset content (`events.py:230-358`, `reset_from_motion_data`) is a documented literal port
of `_reset_dofs`, **but `rsi_fraction` is currently `0.0`** (CLAUDE.md "RSI split" row), so the
`continue_keep`/donor branch is dead code and **100% of resets take the
`default_joint_pos * U(0.5,1.5) + U(-0.1,0.1)` branch** clipped to `soft_joint_pos_limits`.

**Verdict: DIVERGES.**
- Ordering and content are faithful; the **branch probability is inverted**: G1 takes the donor-copy
  branch ~80% of the time, SGK takes the randomized-default branch 100% of the time. See
  divergence #4.
- mjlab clears observation history/delay buffers at reset; G1 does not (see Phase 7).
- G1 randomizes root linear+angular velocity ±0.3 on every reset; SGK's `reset_base` does not
  (already noted in `events.py:266-270`).
- G1 resamples `Kp_factors`/`Kd_factors`/`actuation_offset`/friction/restitution per reset; SGK ports
  only the foot-ball restitution randomization.

---

<a name="phase-7"></a>
## Phase 7 — observation computation: pre- vs post-reset

### Isaac Gym — `compute_observations()` (`:390-432`), called **after** `reset_idx`

```python
current_obs = cat(end_target_local, base_ang_vel*s, projected_gravity,
                  (dof_pos - default_dof_pos)*s, dof_vel*s, actions,
                  base_lin_vel*s, end_regions/3, end_target_local_b,
                  ball_vel_b*s, hand_pos_r, hand_pos_l, dist)
current_actor_obs = clone(current_obs[:, :num_one_step_obs])
if add_noise:
    current_actor_obs += (2*rand-1) * noise_scale_vec[:...]
    current_actor_obs[:, :num_ballobs] *= flying * random_vanish
else:
    current_actor_obs[:, :num_ballobs] *= flying
self.obs_buf = cat(self.obs_buf[:, num_one_step_obs:actor_obs_length], current_actor_obs)
self.privileged_obs_buf = current_obs
```

Key properties:
- The actor observation is a **rolling window that is never cleared on reset**. A just-reset env's
  `obs_buf` still contains 9 frames of pre-reset history plus 1 post-reset frame.
- Noise is uniform, additive, applied **only to the actor slice**, and only to the first
  `num_ballobs + 6 + 2*num_dof + num_actions` entries.
- The ball columns are multiplied by the visibility masks **after** noise, so an invisible ball is
  exactly zero, not "zero plus noise".
- `privileged_obs_buf` is a **single frame, un-noised**.
- Both are clipped to `±clip_observations` back in `step()` (`:152-155`).
- `self.actions` in the obs is the **current** step's action — `last_actions` is only updated at
  `:239-243`, *after* `compute_observations()`.

### mjlab — `ObservationManager.compute(update_history=True)` (`observation_manager.py:305-387`)

Per term: `func(env) → noise → clip → scale → delay → history`
(`ObservationTermCfg` docstring, `:17-24`; implementation `:329-362`).
Group-level `enable_corruption=False` **deletes** every term's noise config at prepare time
(`:436-437`); `group_cfg.history_length` overrides each term's (`:438-440`).

- **Caching**: `compute()` returns `self._obs_buffer` unchanged when `update_history=False` and the
  cache is populated (`:311-312`). `observation_manager.reset()` sets `_obs_buffer = None` (`:241`).
  This is why the runner can call `observation_manager.compute()` three extra times per step
  (`_get_actor_current_obs`, `_get_actor_history_obs`, `get_amp_observations`) **without** resampling
  noise or double-pushing delay buffers — all three read the same cached tensors produced at
  `manager_based_rl_env.py:464`. ✔
- **History on reset**: `CircularBuffer.reset(batch_ids)` zeroes those rows and sets
  `current_length = 0`; the **next append backfills all `max_len` slots with that single new value**
  (`circular_buffer.py` module docstring, "Per-Batch Reset"). So a just-reset env's 10-frame actor
  history is **10 identical copies of the post-reset frame**.
- Concatenation order is `cfg.terms` dict order; flattened history is **term-major**
  (`[A_t0..A_tH-1, B_t0..B_tH-1, ...]`, `:60-65`) — *not* the frame-major layout G1's
  `cat(obs_buf[num_one_step:], current)` produces. This is why the port needs a separate
  `actor_current` group rather than slicing the newest frame out of the history tensor (documented
  at `him_amp_on_policy_runner.py:38-52`).
- Clipping to `±500` happens in `amp_wrapper.py:77-79` for actor/critic/amp groups.

SGK groups: `actor` (`history_length=10`, `enable_corruption=True`), `critic`
(`history_length=1`, corruption off), `actor_current` (`history_length=0`, corruption inherited),
`amp` (no history, `enable_corruption=False`, both terms `noise=None`).

**Verdict: DIVERGES on history-at-reset, MATCHES on noise treatment of the AMP group.**
- History-at-reset: G1 carries stale pre-reset frames; mjlab backfills identical post-reset frames.
  Real difference in the actor's input at every episode start; the port does not replicate G1 and
  arguably should not. Does **not** reach the discriminator (the `amp` group has no history).
- AMP group: un-noised, un-scaled, un-clipped-per-term on both sides. ✔ matches G1's raw
  `dof_pos.clone()`.
- **Observations for a just-reset env reflect POST-reset state on BOTH sides** — G1 because
  `compute_observations()` is called after `reset_idx()`, mjlab because `observation_manager.compute`
  is called after `_reset_idx` + `sim.forward()`. ✔ This part matches.

---

<a name="phase-8"></a>
## Phase 8 — terminal-observation capture and the 7-tuple

### Isaac Gym

`compute_termination_observations(env_ids)` (`:434-463`) is called at `:233`, **between**
`check_termination()` and `reset_idx()`. It recomputes the *privileged* observation vector from the
still-pre-reset state and returns `current_obs[env_ids]`. Differences vs. the privileged branch of
`compute_observations`: none in content — same terms, same order, no noise, no ball visibility mask.
It is purely a "capture this now, before I destroy it" hook.

`him_on_policy_runner.py:182-183` then does:

```python
next_critic_obs = critic_obs.clone().detach()
next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()
```

so the *critic's* next-state is correct across episode boundaries. Note what it is **not** used for:
the AMP pair (see Phase 11).

### mjlab

There is **no hook**. With `auto_reset=True` (the only mode this project runs), `step()` resets in
place and every returned observation is post-reset — stated in `ManagerBasedRlEnvCfg.auto_reset`'s
own docstring (`manager_based_rl_env.py:144-155`) and in mjlab's changelog. The only alternative is
`auto_reset=False`, which requires the caller to drive `reset(env_ids=...)` itself.

`AMPEnvWrapper.step(..., not_amp=False)` (`amp_wrapper.py:62-94`) *copies the 7-tuple shape*:

```python
terminal_amp_states = obs_dict.get(self._amp_group, obs).clamp(-500, 500)   # :79  POST-RESET
reset_env_ids = torch.where(dones)[0]                                       # :82
return (obs, privileged_obs, rew, dones, extras,
        reset_env_ids, terminal_amp_states[reset_env_ids])                  # :86-94
```

`terminal_amp_states` is sliced out of the **same post-reset `obs_dict`**, so
`him_amp_on_policy_runner.py:217-218`'s

```python
next_amp_obs_with_term = torch.clone(next_amp_obs)
next_amp_obs_with_term[reset_env_ids] = terminal_amp_states
```

assigns a tensor to itself. Confirmed a no-op (documented at `multi_disc_amp_ppo.py:188-207`,
live-verified byte-identical for 64/64 forced terminations).

Also note the port has **no critic-side equivalent** of G1's `next_critic_obs[termination_ids] = ...`;
`_mini_batch_generator_with_next` instead takes `storage.privileged_observations[1:]` (post-reset)
and relies on `cont_batch` to zero the mix weight for done transitions
(`multi_disc_amp_ppo.py:259-274, 445-447`). Functionally equivalent for the smoothness regularizer,
since G1's mix weight is likewise `cont_batch * ...` (`him_ppo.py:235`).

**Verdict: DIVERGES — port does NOT handle it.** Mitigated (not fixed) on 2026-09-15 by refusing to
insert done transitions into the AMP replay buffer (`multi_disc_amp_ppo.py:208-213`).
**Important nuance for prioritisation:** G1 does **not** substitute the terminal state into its AMP
pair either — `him_on_policy_runner.py:161-163` builds `cat([old_amp_state, amp_state])` where
`amp_state` is the **post-reset** `dof_pos`. So on the AMP path specifically, G1 has the *same*
boundary contamination, unfiltered. Our port now *over*-corrects relative to G1 by dropping those
samples. That makes this a real framework difference but a **weak** candidate for the saturation.

---

<a name="phase-9"></a>
## Phase 9 — extras, `time_outs`, episode logging

### Isaac Gym

`extras["episode"]` and `extras["time_outs"]` are written **inside `reset_idx`**, after the
`len(env_ids) == 0` early return (`:277-278`, `:313-322`). Consequences:
- On a step where **no** env resets, `extras` retains the **previous** step's contents, including a
  stale `time_outs` mask. `him_ppo.process_env_step` (`:157-158`) unconditionally applies
  `rewards += gamma * values * infos['time_outs']` whenever the key is present — so a stale mask
  can bootstrap a step that did not time out. Upstream quirk.
- `extras["episode"]` is a fresh dict each reset; the runner appends it to `ep_infos`.
- `episode_sums[key][env_ids] / clip(episode_length_buf[env_ids], min=1) / dt` → a per-second rate,
  and `episode_length_buf` is zeroed **after** this division.

### mjlab

`extras["log"] = dict()` is reset at the **top of every step** (`:418`) and at the top of `reset()`
(`:368`). Every manager's `reset()` merges its stats into `extras["log"]`.
`extras["time_outs"] = truncated` is written **fresh every step** by the wrapper
(`vecenv_wrapper.py:70-71`, `amp_wrapper.py:83-84`), gated on `not cfg.is_finite_horizon`.
`RewardManager.reset` divides by `max_episode_length_s`, **not** by the actual episode length —
a per-second rate normalised by the *maximum* episode, not the realised one.

**Verdict: DIVERGES; the port is more correct than G1.**
- `time_outs` freshness: port ✔ correct, G1 buggy. No action needed.
- Episode-reward normalisation denominator differs (max length vs. realised length). Affects log
  interpretation only — already documented in CLAUDE.md's "Reading TensorBoard" section.
- `infos["episode"]` vs `infos["log"]`: the port appends **both** (`him_amp_on_policy_runner.py:228-231`).
  mjlab only ever produces `"log"`, so `"episode"` is dead. Harmless.

---

<a name="phase-10"></a>
## Phase 10 — per-env indexing / ordering consistency

### Isaac Gym

All buffers are `[num_envs, ...]` with row *i* = env *i*, for the whole run. The only place ordering
could break is the Gym actor-index arithmetic: with 2 actors per env (robot, ball), the robot is
global actor `2*i` and the ball `2*i+1` (`:684-689`, `:731-735`) — and
`_reset_dofs` builds `env_ids_int32` twice, discarding the first (`:684` then `:686`), keeping the
robot-only version. `_reset_root_states` uses the interleaved `[2*ids, 2*ids+1]`. Consistent.
`termination_ids` is `reset_buf.nonzero().flatten()` and
`termination_privileged_obs = current_obs[env_ids]`, so the two are index-aligned by construction,
and the runner's `next_critic_obs[termination_ids] = termination_privileged_obs` is a correct
advanced-index scatter.

### mjlab / port

Same invariant: every manager buffer is `[num_envs, ...]`, `reset_env_ids = reset_buf.nonzero()`,
and the wrapper's `terminal_amp_states[reset_env_ids]` is aligned with `reset_env_ids`
(`amp_wrapper.py:82,93`). `region_id` is read as `critic_obs[:, -1].long()`
(`him_amp_on_policy_runner.py:220`, index configured at `:108-109`) and captured into
`self._pending_region` in `act()` **before** the step (`multi_disc_amp_ppo.py:169`), so the region
used to route the reward and the region used to route the replay-buffer insert are the **same
pre-step region** for the same env. ✔

One asymmetry worth recording: G1 recovers its motion id from the *scaled, clipped* critic obs —
`motion_ids = 3 * critic_obs[:, num_one_step_obs + 3]` (`him_on_policy_runner.py:168`,
`him_ppo.py:248`), reading back the `end_regions/3` column after it has passed through
`clip(±clip_observations)`. That round-trip is float-exact for values 0…5/3, so it works, but it is
fragile. The port reads a dedicated, unscaled `region_gt` term instead.

**Verdict: MATCHES.** Both sides maintain a stable env-major layout across obs/reward/done/extras.
No indexing hazard found on either side.

---

<a name="phase-11"></a>
## Phase 11 — AMP policy transition assembly (`s_t`, `s_{t+1}`)

### Isaac Gym

`get_amp_observations()` is **one line** (`legged_robot.py:159-163`):

```python
def get_amp_observations(self):
    return self.dof_pos.clone()
```

Unconditional, ungated, un-noised, un-scaled, raw absolute joint positions, all 29 DOF.
`cfg.amp.num_obs = 29*2 = 58` (`g1_29_config.py:363`), `num_steps = 2`.

Rollout loop (`him_on_policy_runner.py:151-163`):

```python
actions = self.alg.act(obs, critic_obs)
old_amp_state = amp_state                      # dof_pos BEFORE this step
obs, ..., = self.env.step(actions)
amp_state = self.env.get_amp_observations()    # dof_pos AFTER this step (POST-RESET if it reset)
amp_state_ = cat([old_amp_state, amp_state], dim=1)
self.alg.process_amp_state(amp_state_)         # stored on the SAME transition as the action
```

So the pair is exactly (dof_pos at control step *t*, dof_pos at control step *t+1*) — a genuine
20 ms transition. **It can cross an episode boundary**, and G1 does nothing about it: at a reset,
`amp_state` is the freshly-written reset pose. G1 pushes that pair into the discriminator batch
like any other.

### mjlab / port

`AMPEnvWrapper.get_amp_observations()` (`amp_wrapper.py:57-58`) returns
`observation_manager.compute()["amp"]` — the cached post-step buffer.

Rollout loop (`him_amp_on_policy_runner.py:197-225`):

```python
actions = self.alg.act(obs, obs_history, critic_obs, amp_obs)   # amp_obs = s_t, stashed at :168
... = self.env.step(actions, not_amp=False)
next_amp_obs = self.env.get_amp_observations()                  # s_{t+1}
next_amp_obs_with_term = clone(next_amp_obs); next_amp_obs_with_term[reset_env_ids] = terminal  # no-op
rewards, ... = self.alg.predict_region_routed_amp_reward(amp_obs, next_amp_obs_with_term, region_id, raw)
amp_obs = clone(next_amp_obs)
self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
```

`process_env_step` (`multi_disc_amp_ppo.py:208-213`) inserts `(self._pending_amp_obs[mask], amp_obs[mask])`
into the per-region `ReplayBuffer`, `mask = (region == r) & (dones == 0)`.

Structurally identical to G1 **except** for the done filter and for the *content* of the amp group.

#### The content of the `amp` group — the critical part

`goalkeeper_multidisc_amp_cfg.py:147-162` registers two terms:

```python
"joint_pos": ObservationTermCfg(func=gk_obs.joint_pos_abs_arms_masked_by_region,
                                noise=None, params={"far_region_ids": (0,1,2,3)}),
"joint_vel": ObservationTermCfg(func=gk_obs.joint_vel_abs_arms_masked_by_region,
                                noise=None, params={"far_region_ids": (0,1,2,3)}),
```

Both functions (`observations.py:385-496`) do **two** things beyond returning raw state:

1. **Arm-column masking** — arm joints forced to `default_joint_pos` / `0.0` for envs whose
   `_region_id ∈ far_region_ids`, which is now **all four regions**. The expert side is masked to
   match (`MotionDatasetCfg.freeze_joint_names=_ARM_JOINT_NAMES`, applied at
   `motion_dataset.py:126-129`). **Symmetric → fine.**

2. **Whole-vector freeze on `_softstop_flag`** — `observations.py:430-432` and `:478-480`:

   ```python
   softstop_fired = getattr(env, "_softstop_flag", None)
   if softstop_fired is not None and softstop_fired.any():
       joint_pos[softstop_fired] = default_joint_pos[softstop_fired]   # ALL 21 joints
       joint_vel[softstop_fired] = 0.0                                 # ALL 21 joints
   ```

   `_softstop_flag` is a **sticky per-episode latch**: set inside `_reward_softstop`
   (`rewards.py:3377-3379` allocation) and cleared only at `episode_length_buf <= 1`
   (`rewards.py:3408-3411`). So from the save instant to the end of the episode, **every** AMP
   observation for that env is the *identical constant vector*
   `[default_joint_pos, 0]` — and therefore every AMP *transition* for that env is the identical
   constant pair `[default_joint_pos, 0, default_joint_pos, 0]`.

   **G1 has no analogue of this whatsoever.** Its `get_amp_observations` is ungated.

   Order-of-magnitude: `episode_length_s = 3.0` → 150 steps; ball flight ≤ ~1.45 s (≈72 steps);
   save rate reported ~78%. If a typical save lands around step 50–75, 50–65% of that episode's
   transitions are the frozen constant, and ~0.78 × that ≈ **40–50% of all policy-side
   discriminator samples are literally one point in state space that no expert sample can ever
   equal**.

3. **Stale-flag edge case at the reset boundary.** The flag is cleared inside the *reward* function,
   which runs **before** `_reset_idx` and before observation computation. On the terminating step,
   `episode_length_buf` is not yet ≤ 1 when rewards run, so the flag is still set when the
   post-reset observation is computed → the just-reset env's AMP obs is **still the frozen
   default**. On the *next* step, `episode_length_buf == 1`, the flag clears, and the obs becomes the
   real post-reset pose. That next transition has `dones == 0`, so the 2026-09-15 filter does **not**
   exclude it, and the pair `(frozen default, real randomized reset pose)` **is inserted** into the
   replay buffer. Every saved episode contributes one such artificial teleport at its start.

**Verdict: DIVERGES — port does NOT handle it. Highest-priority finding.** See #1 and #2 in the
ranked list.

---

<a name="phase-12"></a>
## Phase 12 — AMP expert transition sampling

### Isaac Gym — `MotionLib.get_expert_obs` (`g1_utils.py:158-190`)

```python
motion_ids  = randint(0, num_motion, (B,))                 # uniform over MOTION FILES
start/end   = motion_start_ids/motion_end_ids[motion_ids]
time_in_prop= rand(B).clamp(0, 1 - num_steps/motion_len)   # uniform within the clip
motion_ids  = start_ids + floor(time_in_prop * (end - start))
motion_dof  = motion_dof_pos[motion_ids]                   # exact integer frame, NOT interpolated
ratio = (fps / env_fps) * (rand(B) * 1.0 + 0.25)           # U(0.25, 1.25) × 30/50
for i in 1..num_steps-1:
    next_pos = motion_ids + i * ratio
    floor/ceil clamped to the GLOBAL concatenated dataset bound
    motion_dof_next = lerp(motion_dof_pos[floor], motion_dof_pos[ceil], next_pos - floor)
    motion_dof = cat([motion_dof, motion_dof_next])
```

Three properties matter: (a) the base frame `t` is an exact stored frame, only `t+1` is interpolated;
(b) the playback ratio is randomized per sample, so expert transitions carry a **wide spread of
displacement magnitudes** (0.15–0.75 source frames ≈ 5–25 ms of motion) rather than a single fixed
speed; (c) the floor/ceil clamp is to the **global** bound, so near a clip's tail G1 can blend two
unrelated motion files.

### mjlab / port — `MotionDataset.build_transition` (`motion_dataset.py:388-442`)

```python
fps   = self._frame_fps[t]
ratio = (fps / self.env_fps) * (rand_like(fps) * 1.0 + 0.25)     # U(0.25, 1.25) — same
next_pos  = t + ratio
max_idx   = self._frame_traj_max_idx[t]        # OWN-trajectory bound, not global
floor/ceil = min(floor(next_pos), max_idx), min(floor+1, max_idx)
linear_ratio = (next_pos - floor).clamp(0, 1)
_t   = values[t]                                # exact stored frame
_tp1 = lerp(values[floor], values[ceil], linear_ratio)
```

`sample_batch` draws uniformly over *transitions* (i.e. by frame count), except in
`WeightedMotionDataset` where per-file weights apply (used for the 3 far-region clips,
`goalkeeper_multidisc_amp_cfg.py:312, 396`). `env_fps` is derived from `env.step_dt` = 50 Hz
(`motion_dataset.py:298-316`), matching G1's hardcoded 50.

**Verdict: MATCHES (with one deliberate, better deviation).** The randomized-playback-ratio
mechanism is a faithful port; clamping to the sampled frame's own trajectory instead of the global
dataset bound avoids a real cross-clip-blend artefact that G1 has. Sampling weights differ by design
(G1 samples uniformly over *files*; the port samples over *frames* with optional per-file weights) —
documented.

**BUT — the `joint_vel` term interacts badly with this (see ranked #3):** the expert's `joint_vel`
columns are the NPZ's stored per-frame finite-difference velocities. Interpolating the *position*
frame at a random ratio `r` and simultaneously reading the *velocity* at the interpolated index
produces a pair whose positional displacement is `r × Δt`-sized while its velocity column still
describes 1.0× playback. The `s_t` half's velocity is likewise the unscaled stored value.
G1 sidesteps this entirely by using `joint_pos` only.

---

<a name="phase-13"></a>
## Phase 13 — normalizer

Both sides use the **same** `Normalizer(RunningMeanStd)` class (`clip_obs=10.0`, `epsilon=1e-4`),
and both share **one** normalizer instance across all discriminators.

### Isaac Gym (`him_ppo.py:287-305`)

```python
amp_expert_obs_batch = self.amp_normalizer.normalize_torch(amp_expert_obs_batch_mask, device)
amp_obs_batch        = self.amp_normalizer.normalize_torch(amp_obs_batch, device)
...                                            # discriminator loss on the normalized tensors
self.amp_normalizer.update(amp_obs_batch.cpu().detach().numpy())          # <-- ALREADY NORMALIZED
self.amp_normalizer.update(amp_expert_obs_batch.cpu().detach().numpy())   # <-- ALREADY NORMALIZED
```

Note `amp_obs_batch` has been **rebound** to the normalized tensor two lines earlier. So G1 feeds
already-whitened data back into the running statistics. The fixed point of that recursion is
`mean → 0`, `var → 1`, i.e. **G1's normalizer degenerates toward the identity map**, and the
discriminator ends up seeing essentially raw `dof_pos` (values well inside the ±10 clip).

### Port (`multi_disc_amp_ppo.py:608-613`)

```python
for policy_state, policy_next_state, expert_state, expert_next_state in normalizer_states:
    self.amp_normalizer.update(policy_state.cpu().numpy())        # RAW
    self.amp_normalizer.update(policy_next_state.cpu().numpy())
    self.amp_normalizer.update(expert_state.cpu().numpy())
    self.amp_normalizer.update(expert_next_state.cpu().numpy())
```

`normalizer_states` stores the **pre-normalization** tensors (`:571`), so the port maintains a
genuine, converging whitener over the combined policy+expert distribution, updated once per region
per minibatch.

**Verdict: DIVERGES — the port is "more correct", but the behavioural consequence is real and
under-appreciated.** A true whitener divides each dimension by its combined-population σ. For a
dimension with tiny combined variance but a consistent expert-vs-policy offset, that offset is
**amplified** to many σ. This directly magnifies exactly the class of tell described in ranked #1/#3
(a frozen constant, or a velocity-statistics mismatch), whereas G1's degenerate near-identity
normalizer would leave the same offset small in absolute terms. It also means the port's
discriminator input scale drifts over training while G1's converges to a constant.

---

<a name="phase-14"></a>
## Phase 14 — discriminator batching, replay, loss, reward mixing

| Aspect | Isaac Gym | Port | Verdict |
|---|---|---|---|
| Policy-side batch source | `amp_obs_batch` straight out of the **on-policy** `HIMRolloutStorage` minibatch, cleared every `update()` | per-region `ReplayBuffer(250_000)` (`goalkeeper_multidisc_amp_cfg.py:460`), **never cleared** since 2026-09-14 (`multi_disc_amp_ppo.py:628-647`) | **DIVERGES**, deliberate |
| Policy/expert region routing | mask on `motion_ids` derived from critic obs (`him_ppo.py:248-284`) | mask on `gt_region` (`multi_disc_amp_ppo.py:466-471`) | MATCHES |
| Sample reuse cadence | one permutation per `update()`, replayed across `num_learning_epochs` | same, after the 2026-07-21 fix (`replay_buffer.py:66-113`) | MATCHES |
| Draw size | region's own masked share | region's own share (`num_samples // num_mini_batches`) | MATCHES |
| LSGAN loss | `MSE(expert_d, +1) + MSE(policy_d, -1)`, unweighted sum | identical (`:484-485, 563`) | MATCHES |
| R1 grad penalty | `compute_grad_pen(..., lambda_=5) * 0.1` → effective 0.5 | `lambda_=5` × 0.1 → effective 0.5 (reverted to G1-exact 2026-09-14, `:541`) | MATCHES |
| Logit L2 reg | none | `0.05 * amp_linear.module.weight.pow(2).sum()` (`:561`) | DIVERGES, deliberate |
| Spectral norm | class exists in `amp.py` but is **never applied** | applied to every trunk layer **and** `amp_linear` (`amp_discriminator.py:127-137`) | DIVERGES, deliberate |
| Discriminator width | `[512, 256]` | `[512, 256]` | MATCHES |
| Init | first trunk layer default, all later layers + head `U(-1, 1)`, zero bias | identical | MATCHES |
| Reward formula | `amp_reward_coef * clamp(1 - 0.25*min_over_20_perturbed(d-1)², 0)` | identical (`amp_discriminator.py:169-205`) | MATCHES |
| Reward mixing | `rewards = amp_reward*0.4 + raw*0.6`, with an extra `*0.5` on the predicted reward (`him_on_policy_runner.py:177,185`; `amp_coef = 0.4`) | `task_reward_lerp=0.6` → `0.4*amp + 0.6*task`, `amp_reward_coef=0.5` | MATCHES in structure; peak per-step AMP contribution is 0.20 (port) vs 0.08 (G1) — a 2.5× magnitude difference |
| Optimizer groups | one `actor_critic` group + per-disc `trunk` (wd 1e-3) / `head` (wd 1e-1) | identical (`:112-118`) | MATCHES |
| Grad clip | `clip_grad_norm_(actor_critic.parameters())` only | identical (`:579`) | MATCHES |
| LR schedule | `adaptive`, all param groups rescaled incl. the estimators | identical (`:384-398`) | MATCHES |
| PPO batch | `num_envs 6144 × num_steps_per_env 100`, 6 discriminators | `6144 × 24`, 4 discriminators | DIVERGES, tested and reverted |

---

<a name="ranked"></a>
## Ranked divergences — most to least plausible as the cause of a permanently saturated discriminator

The ranking criterion is: *does this make policy transitions **systematically, uniformly** separable
from expert transitions, independently of hyperparameters?*

### 1. `_softstop_flag` freezes the entire AMP observation to a constant — policy side only. **CRITICAL**

`observations.py:430-432, 478-480`. Once a save is registered, every AMP observation for that env,
for the rest of the episode, is exactly `[default_joint_pos, 0]`. The transition fed to the
discriminator is therefore exactly `[default_joint_pos, 0, default_joint_pos, 0]` — a single point
that **cannot** occur in expert data (no reference clip sits at the robot's default pose with
identically zero velocity on all 21 joints). Plausibly 40–50% of all policy-side samples.

G1's `get_amp_observations()` (`legged_robot.py:159-163`) is a bare `return self.dof_pos.clone()`.
There is no gate of any kind.

A discriminator can drive its output to the LSGAN floor on that entire mass with zero risk, which
is precisely the "pinned at worst possible output, immune to grad-penalty/architecture/batch-size
changes" symptom. It also explains why every previous fix (R1 coefficient, spectral norm, logit reg,
batch size, replay scheme, sampling cadence) failed — none of them touch a linearly separable point
mass.

**Port status: NOT handled.**
**Cheap confirmation:** during a rollout, log
`(amp_obs == default_joint_pos_broadcast).all(dim=-1).float().mean()` — i.e. the fraction of
policy AMP samples that are the frozen vector — and the same statistic restricted to the replay
buffer. If it is materially above a few percent, this is the bug.
**Cheapest fix consistent with G1:** delete the softstop gate from the two AMP observation terms
(the gate exists to stop post-save motion from being style-scored; the G1-faithful way to do that is
to *not* score it, i.e. exclude those transitions from the replay-buffer insert, never to substitute
a fake observation).

### 2. Stale `_softstop_flag` at the reset boundary injects a fake "teleport" transition that **is** stored. **HIGH**

The flag clears at `episode_length_buf <= 1` inside the reward function (`rewards.py:3408-3411`),
which runs *before* `_reset_idx` and *before* observation computation. So the terminating step's
post-reset AMP obs is still frozen, and the **following** step's transition is
`(frozen default-pose constant → real randomized reset pose)` with `dones == 0` — not caught by the
2026-09-15 done filter (`multi_disc_amp_ppo.py:208-213`). One such artificial discontinuity per
saved episode, concentrated at a systematic point in the episode.

**Port status: NOT handled.** Resolved automatically if #1 is fixed.

### 3. `joint_vel` in the AMP observation: two independent policy-vs-expert statistical tells. **HIGH**

Deliberate 2026-07-22 divergence (`goalkeeper_multidisc_amp_cfg.py:103-127`); G1 uses `joint_pos`
only, 58 dims, so it never faces either problem.

- (a) **Source mismatch.** Expert `joint_vel` is a finite difference of retargeted mocap at clip fps
  — smooth, band-limited, no contact transients. Policy `joint_vel` is MuJoCo `qvel` — contains
  contact-impulse spikes, PD chatter and reset transients. Even with identical *positions*, the
  velocity **spectra** differ enough to classify on.
- (b) **Internal inconsistency in the expert pair.** `build_transition` interpolates the *position*
  of the `t+1` frame at a random playback ratio `r ∈ [0.15, 0.75]` source-frames
  (`motion_dataset.py:422-439`), but the `joint_vel` columns of both `s_t` and `s_{t+1}` are the
  **stored 1.0×-playback** velocities. So an expert transition whose positional delta corresponds to
  0.25× speed carries velocities describing 1.0× speed. Policy transitions are always internally
  consistent (`Δq ≈ q̇ · dt`). *That consistency relation itself is a one-line separator.*
- (c) Amplified by Phase 13: the port's genuine whitener rescales each velocity dimension by its
  combined σ, so a systematic scale offset becomes a many-σ offset.

**Port status: NOT handled.**
**Cheap confirmation:** for expert and policy batches separately, compute
`corr( (s_{t+1} - s_t)[joint_pos_slice],  s_t[joint_vel_slice] * dt )` per dimension. If the expert
correlation is much lower than the policy's, (b) is live.
**Cheap test:** temporarily set the `amp` group to `joint_pos` only (G1-exact, 42 dims) and see
whether `mean_discri_logits` unpins.

### 4. 100% randomized-default reset pose vs. G1's ~80% live-donor copy. **MEDIUM-HIGH**

G1's `_reset_dofs` takes the `continue_keep` donor branch on ~80% of resets
(`legged_robot.py:669-670`) — copying a **currently-running env's actual joint configuration**, i.e.
an in-distribution, mid-motion pose. SGK's `rsi_fraction = 0.0`, so 100% of resets take the other
branch: `default_joint_pos * U(0.5, 1.5) + U(-0.1, 0.1)` per joint
(`events.py:338-345`). A 0.5×–1.5× per-joint scaling of the standing pose is far off both the
expert manifold and the policy's own behaviour manifold, and the robot then spends ~5–15 steps
settling out of it — every single episode, in every env.

Compounded by the absence of G1's warm-up PD hold (Phase 1): G1 holds the robot at `init_dof_pos`
through the first few steps, so its post-reset AMP frames are a *settling* robot near a *held*
target; SGK's are a policy-driven robot flailing out of a random configuration.

**Port status: NOT handled** (the branch probability is a documented, deliberate choice; its
interaction with AMP is not).

### 5. Persistent, never-cleared AMP replay buffer vs. G1's strictly on-policy batch. **MEDIUM**

`multi_disc_amp_ppo.py:628-647` (deliberate, 2026-09-14). 250k capacity per region; at
`4096 × 24 / 4 ≈ 24.6k` inserts per region per iteration that is ~10 iterations of history.
G1's discriminator only ever sees the current policy. On its own this is a *stabiliser*, and NVIDIA's
own AMP implementations do it — but it **compounds** #1 and #2: once frozen-constant or teleport
samples enter the buffer they persist for ~10 iterations, and the discriminator is retrained against
them repeatedly. If #1 is fixed, this is likely benign.

### 6. Normalizer semantics: genuine whitener (port) vs. degenerate near-identity (G1). **MEDIUM**

Phase 13. Not a bug in the port — G1's version is the buggy one — but it is a real change in what
the discriminator's input space looks like, and it **amplifies** any low-variance systematic offset
between the two populations by up to `1/σ`. Directly relevant to #1 and #3. Worth being aware of
before concluding that "we match G1's discriminator recipe".

### 7. Reward/termination ordering swap. **LOW-MEDIUM**

G1: reward → termination. mjlab: termination → reward (`manager_based_rl_env.py:436-440`). Any SGK
reward that consults a termination result sees this step's value; the G1 equivalent would see the
previous step's. Also G1's `_reward_termination` reads a one-step-stale `reset_buf` by construction.
No AMP impact; a real semantic difference for any termination-coupled reward.
**Port status: NOT handled** (nor easily handleable without subclassing `step()`).

### 8. Derived-quantity staleness (one physics substep) at reward/termination time. **LOW-MEDIUM**

Phase 3. Uniform across envs and steps, so no policy-vs-expert tell, but it shifts every
threshold-crossing latch (`sharpforce`, `stopball`/`softstop` `delta_vx`, `_ball_is_behind`) by 5 ms.
Since `_softstop_flag` gates the AMP freeze in #1, this staleness *also* shifts where the frozen
block begins. **Port status: NOT handled** (mjlab-architectural).

### 9. Action-delay distribution and rate. **LOW-MEDIUM**

Phase 1. G1: `U{0 … decimation-1}` substeps, resampled once per control step, applied as a
last→current step change, **zero lag possible**. Port: mjlab `DelayBuffer(min_lag=1, max_lag=3)`
appended **once per physics substep**, so the lag is 1–3 substeps and never zero, and targets carry
across control-step boundaries. Different distribution, similar magnitude. Also missing entirely:
G1's `joint_injection` and `actuation_offset` torque perturbations, and the per-reset
`Kp_factors`/`Kd_factors` randomization. **Port status: partially handled / documented.**

### 10. Observation history at reset: stale window (G1) vs. backfilled copies (mjlab). **LOW** (for AMP)

Phase 7. G1's rolling `obs_buf` is never cleared, so a just-reset env's actor input contains 9
pre-reset frames — genuinely misleading data the G1 policy learned to tolerate. mjlab's
`CircularBuffer.reset` + backfill gives 10 identical copies of the post-reset frame. Real difference
in the actor's learning problem; **zero** effect on the discriminator (the `amp` group has no
history). **Port status: NOT replicated; the port's behaviour is the saner one.**

### 11. `mean_discri_logits` is an episode **sum**, not a mean. **LOW (interpretation)**

`him_amp_on_policy_runner.py:234, 239` accumulates `d_logits` per step into `cur_discri_sum` and
flushes it on `done`. A reported value of −50…−85 over a ~150-step episode is ≈ −0.35…−0.55 per
step, i.e. near but not exactly at the LSGAN policy target of −1. Worth normalising by
`mean_episode_length` before drawing conclusions about how saturated the discriminator actually is;
the same caveat applies to `Train/mean_amp_reward`.

### 12. `terminal_amp_states` / the 7-tuple no-op. **LOW** (as an AMP explanation)

Phase 8. Real, confirmed, and correctly described — but G1 does **not** substitute a terminal state
into its AMP pair either (`him_on_policy_runner.py:161-163` uses the post-reset `amp_state`
unconditionally). The port's 2026-09-15 mitigation makes it *stricter* than G1. It bounds at
~1/mean_episode_length ≈ 1–2% of transitions. Keep the fix; do not expect it to explain saturation.

### 13. Cosmetic / inert differences. **LOWEST**

- `time_out` comparison `>=` (mjlab) vs `>` (G1) — one step per episode.
- `extras["time_outs"]` fresh every step (port) vs. stale between resets (G1). Port is correct.
- `Episode_Reward/*` divided by `max_episode_length_s` (mjlab) vs. realised episode length (G1).
  Logging only.
- `tick_catchstep` is an `"interval"` event and mjlab runs the interval block **after** the reset
  block (`manager_based_rl_env.py:458-461`), so a just-reset env's `_catchstep` is decremented on the
  same step it was set. G1 does `catchstep -= 1` at the top of `post_physics_step`, before reset.
  One-step offset in the ball-visibility warm-up.
- mjlab's published architecture doc lists the `"step"`/`"interval"` events *before* the reset block;
  the installed source runs them after. Use the source.
- `infos["episode"]` handling in the port is dead code (mjlab only emits `infos["log"]`).
- G1 recovers `motion_ids` by reading back a scaled, clipped critic-obs column
  (`3 * critic_obs[:, num_one_step_obs + 3]`); the port reads a dedicated `region_gt` term. Port is
  cleaner.
- Peak per-step AMP reward contribution: 0.20 (port) vs 0.08 (G1), a 2.5× magnitude difference in
  the task/style balance despite the identical 0.4/0.6 mixing structure.

---

## Suggested order of investigation

1. Measure the frozen-sample fraction (#1). One line in the rollout loop; decides everything else.
2. If confirmed, remove the softstop gate from `joint_pos_abs_arms_masked_by_region` /
   `joint_vel_abs_arms_masked_by_region` and instead **exclude** post-save transitions from the
   replay-buffer insert in `process_env_step` (same place the `not_done` mask already lives). That
   preserves the original design intent — "don't style-score the post-save recovery" — without ever
   handing the discriminator a synthetic observation. #2 disappears with it.
3. Independently, run one short A/B with the `amp` group reduced to `joint_pos` only (G1-exact) to
   isolate #3.
4. Only then revisit #4 (`rsi_fraction`) and #5 (replay persistence).

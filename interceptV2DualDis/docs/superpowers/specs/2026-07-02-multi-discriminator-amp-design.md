# Multi-Discriminator AMP with HIM-Style Estimators — Design Spec

## Context

`interceptV2DualDis` is a standalone copy of `SimpleGoalKeeper` (feet-only Booster T1 goalkeeper,
mjlab + beyondAMP), created 2026-07-02 specifically to prototype a region-conditioned,
multi-discriminator AMP setup modeled on the original `Humanoid-Goalkeeper` paper implementation's
`HIMPPO`/`ActorCritic` (G1, Isaac Gym, hands). SimpleGoalKeeper currently uses `beyondAMP`'s
single-discriminator `AMPPPO` (one discriminator, one motion buffer, no region concept, no
observation history, no privileged-info estimator heads) — none of what's described below exists
there today.

**Scope is `interceptV2DualDis` only.** `SimpleGoalKeeper/` and the shared `beyondAMP` upstream
clone are not modified by this work. New code is written as forked/new classes inside
`interceptV2DualDis/src/...`, not as edits to `interceptV2DualDis/beyondAMP/` in place (keeps that
clone diffable against upstream beyondAMP).

## Reference material

Ported from `Humanoid-Goalkeeper/rsl_rl/rsl_rl/`:
- `modules/actor_critic.py` — `ActorCritic` with `history_encoder`/`ball_estimator`/`region_estimator`
- `modules/amp.py` — single-discriminator `AMP` module (trunk + `amp_linear` head, GAIL-style loss, `predict_reward`)
- `algorithms/him_ppo.py` — `HIMPPO`: dict of 6 discriminators, dict of 6 motion buffers, region-masked loss
- `runners/him_on_policy_runner.py` — rollout loop: region-masked `predict_reward`, reward blending, logging

And `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py`:
- `self.end_regions` static partition (`legged_robot.py:916-924`)
- per-region `command_ranges` from `cfg.commands.ranges_N` (`legged_robot.py:933-958`)
- `compute_observations`/`compute_termination_observations` — history stacking, ball-obs vanish gate

G1 uses **6** regions (`lefthand, righthand, leftjump, rightjump, leftstep, rightstep`).
`interceptV2DualDis` uses **4** regions (`left_near, left_far, right_near, right_far`), split by
side and by the existing `|cross_y|` 0.5 m threshold already used elsewhere in SGK's RSI/pool code.

## A. Env-side region partitioning

- At env construction, split `num_envs` into 4 equal, contiguous, **permanent** blocks — same
  pattern as G1's `six = num_envs // 6` (`legged_robot.py:916`), just 4-way:
  `quarter = num_envs // 4`; env indices `[0, quarter)` → `left_near`, `[quarter, 2*quarter)` →
  `left_far`, `[2*quarter, 3*quarter)` → `right_near`, `[3*quarter, num_envs)` → `right_far`.
  Stored as `self.region_id: torch.Tensor[int64]` of shape `(num_envs,)`, values 0-3, fixed for the
  process lifetime (never reassigned on reset — matches G1).
- Each block's ball-spawn `reset_ball_rolling` call is parameterized so the spawned ball's
  predicted crossing distance actually lands in that block's category:
  - `left_near`: side biased left, `|cross_y|` sampled in `[0, 0.5)`.
  - `left_far`: side biased left, `|cross_y|` sampled in `[0.5, existing max)`.
  - `right_near` / `right_far`: mirrored for the right side.
  - Implementation: reuse the existing `y_start_range`/`y_end_range` machinery in
    `goalkeeper_env_cfg.py`/`events.py`, but resolve the range **per env-slot from `region_id`**
    instead of one shared range for all envs — same shape as G1's per-region `command_ranges`
    lookup, adapted to SGK's ball-spawn parameterization instead of G1's width/height command.
- Critic (privileged) observation gains two new appended fields, written every step:
  - Ground-truth ball state, 4 dims: `(ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y)` in robot
    body frame (matches the 4-dim ball_estimator target decided below) — analogous to G1's
    `critic_obs_batch[:, -13:-7]` (6-dim there; 4-dim here since SGK stays 2D).
  - `region_id` (as a float, unnormalized 0-3) — analogous to G1's
    `critic_obs_batch[:, -14]` (`3 *` scaling there decodes the 0-5 range; here it's `region_id`
    directly, no rescale needed since the raw int is already small and exact in float32).
  - These are supervision targets only — never fed to the actor.

## B. Actor-critic: observation history + HIM estimator heads

New class `HimActorCritic` (or similar), forked from `rsl_rl_amp/modules/actor_critic.py`'s plain
`ActorCritic`, adding G1's three auxiliary MLPs operating on a stacked observation history:

- **History window**: 10 steps (`actor_history_length = 10`, matching G1's `num_actor_history`
  default in `g1_29_config.py:7`; SGK runs the same `dt=0.02`, so this is a like-for-like port).
  Requires a new rolling history buffer maintained per env in the env wrapper (SGK currently has no
  observation history at all — this is new plumbing, not present anywhere in `rsl_rl_amp` or
  `beyondAMP` today).
- **`history_encoder`**: `Linear(one_step_dim * 10, 128) → ReLU → Linear(128, 64) → ReLU →
  Linear(64, 16)`. Same shape as G1 (`actor_critic.py:132-138`).
- **`ball_estimator`**: `Linear(one_step_dim * 10, 128) → ReLU → Linear(128, 32) → ReLU →
  Linear(32, 4)` — **4 dims** (`ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y`), not G1's 6
  (no Z, matching SGK's existing XY-only deployment convention). Trained via MSE against the
  ground-truth ball state in critic obs (mirrors G1's `est_loss`).
- **`region_estimator`**: `Linear(one_step_dim * 10, 128) → ReLU → Linear(128, 32) → ReLU →
  Linear(32, 4)` — **4-way** logits (not G1's 6). Trained via cross-entropy against `region_id`
  in critic obs (mirrors G1's `region_loss`).
- **Actor input** = `concat(last_raw_one_step_obs, history_latent[16], estimated_ball[4],
  argmax(region_logits)[1])` — same composition as G1 (`actor_critic.py:232`), dimension count
  differs only in the ball/region pieces (4+1 here vs 6+1 there).
- Critic keeps its existing full-privileged-obs input unchanged (still gets true ball state +
  region id directly, no estimation needed on the critic side — matches G1, where only the actor
  is deployment-constrained).

## C. AMP: 4 discriminators, region-routed

New algorithm class (forked from `rsl_rl_amp/algorithms/amp_ppo/amp_ppo.py`'s single-discriminator
`AMPPPO`, restructured to match `HIMPPO`'s dict-based pattern):

- `self.discriminators = {"left_near": AMPDiscriminator(...), "left_far": ..., "right_near": ...,
  "right_far": ...}` — 4 independent instances (own `trunk` + head, no shared weights), each added
  to the shared optimizer as separate param groups (mirrors `HIMPPO.__init__`, `him_ppo.py:99-115`;
  keep G1's differentiated weight decay: `10e-4` on trunk params, `10e-2` on head params).
- `self.motion_buffers = {"left_near": MotionDataset(motion_files=["LeftStep"]), "left_far":
  MotionDataset(motion_files=["LeftDoubleStep"]), "right_near": MotionDataset(motion_files=
  ["Rightstep"]), "right_far": MotionDataset(motion_files=["RightDoubleStep"])}` — each discriminator
  only ever trains against its own region's motion clip. `LeftTripleStep`/`RightTripleStep` are
  loaded by **no** buffer — fully excluded from AMP in `interceptV2DualDis`. This exclusion is
  expressed in `interceptV2DualDis`'s own `goalkeeper_amp_cfg.py`-equivalent config, not in
  `beyondAMP` or `SimpleGoalKeeper`.
- **Update-time loss** (`update()`, mirrors `him_ppo.py:244-305`): for each region, mask the
  minibatch by `region_id == r`, compute that region's `compute_loss(policy_amp_obs[mask],
  expert_amp_obs[mask])` against that region's discriminator + expert buffer, sum across the 4
  regions into `amp_loss`. No cross-region leakage — a `left_far` sample's AMP loss only ever comes
  from the `left_far` discriminator/buffer.
- **Rollout-time reward** (mirrors `him_on_policy_runner.py:161-178`): per step, mask by
  `region_id`, call `self.discriminators[region].predict_reward(amp_state[mask], normalizer)` per
  region, assemble into one `amp_reward` tensor, then blend with task reward using SGK's existing
  `amp_reward_coef`/`amp_task_reward_lerp` knobs (same linear-blend concept as G1's
  `amp_reward * amp_coef + raw_rewards * (1 - amp_coef)`).

## D. Code organization

All new/forked code lives under `interceptV2DualDis/src/simple_goalkeeper/` (exact module names TBD
at plan time), pulling in `beyondAMP`'s existing motion-loading/replay-buffer utilities where they
still fit (no need to refork what isn't changing — e.g. NPZ loading, `WeightedMotionDataset`'s
transition-sampling machinery can likely be reused per-region-buffer as-is, just instantiated 4x
with a single-motion-file list each instead of one shared instance).

New pieces, roughly:
- `mdp/events.py` (or new `mdp/region.py`): static region assignment, per-region ball-spawn range
  resolution, ground-truth critic-obs fields.
- New actor-critic module (`HimActorCritic`) + history buffer plumbing in the env wrapper.
- New multi-discriminator algorithm class + runner glue (may be able to subclass
  `AMPOnPolicyRunner`/`AMPPPO` rather than fully rewrite, depending on how cleanly the region-masked
  loop drops in — to be assessed during planning/implementation, not decided here).
- `tasks/goalkeeper_amp_cfg.py`: 4-discriminator config replacing the current single `amp_data`
  block; explicit motion-file lists per region (no `TripleStep` anywhere).

## E. Testing / verification plan

- Unit tests (mirrors the recent AMP-motion-weighting test pattern): region assignment is exactly
  a 4-way contiguous partition of `num_envs`; each region's motion buffer contains only its
  assigned single motion file; `TripleStep` files never appear in any buffer; ball/region estimator
  output shapes match (4-dim, 4-way) at construction time; region-masked loss/reward routing never
  mixes regions (e.g. a synthetic batch with known region_id never produces gradient in the wrong
  discriminator).
- Smoke test: short training run (few iterations, small `num_envs`, mirrors the standalone-copy
  smoke test already done for `interceptV2DualDis`) confirming the env builds, all 4 discriminators
  receive nonzero-gradient updates, and estimator losses (`est_loss`, `region_loss`) are finite and
  decrease from their initial values over a short run.

## Non-goals / open risks

- Not porting G1's `num_regions=6` hand/jump/step split — deliberately simplified to the 4-way
  near/far × side split requested for this prototype.
- Not touching `SimpleGoalKeeper/` or the shared `beyondAMP/` clone under either project.
- History-window plumbing is new infrastructure for this codebase (SGK has never stacked
  observations before) — this is the largest net-new engineering surface in the plan, bigger than
  the discriminator-splitting itself.
- Static per-env-slot region assignment means each of the 4 groups only ever trains on 1/4 of
  `num_envs` — with a fixed total `num_envs`, effective per-region batch size drops accordingly;
  worth watching in the smoke test / early training curves.

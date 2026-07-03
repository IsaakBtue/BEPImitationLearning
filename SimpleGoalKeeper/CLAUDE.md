# SimpleGoalKeeper — CLAUDE.md

## Phase 1 Scope

**Phase 1 focuses exclusively on foot-based goalkeeping.** The robot must intercept incoming balls using its feet only. There are no hand rewards, no arm-specific observations, and no hand-related AMP body names.

Hand rewards and arm observations are explicitly out of scope and should not be added until Phase 2 is started with a new design review.

## Project Purpose

Standalone, simplified goalkeeper training environment for the Booster T1 humanoid using:
- **mjlab** (MuJoCo-Warp RL framework)
- **beyondAMP** (simplified AMP integration — no custom 6-discriminator runner)
- **21-DOF headless T1** (head joints removed from action/observation space)

## Project Origin

`SimpleGoalKeeper` is a **foot-only** experimental track, distinct from `Humanoid-Goalkeeper` (the original paper) and `Imitationlearningbooster` (the T1 hand-catching port). Both of those use hands/arms; here the robot may only use its feet. AMP motion priors encourage natural bipedal motion while the task rewards focus entirely on foot-ball contact.

## Design Rule: Always Verify Against Humanoid-Goalkeeper First

**Before adding, changing, or removing ANY reward term, spawn parameter, observation, termination condition, curriculum stage, or training hyperparameter**, you MUST:

1. **Read the upstream G1 code** in `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/` — specifically `base/legged_robot.py` (all reward functions, reset logic, behind/deflection conditions) and `g1/g1_29_config.py` (all weights, ranges, thresholds).
2. **Confirm the G1 equivalent exists** and quote the exact line/function. If it does not exist in G1, that is a red flag — explain why it's needed.
3. **Explicitly justify every divergence** — state WHY the upstream G1 value is wrong for this setup (MuJoCo vs PhysX, feet vs hands, mjlab API difference, etc.).
4. **Document it** in the "Divergences from G1 Upstream" table below.

This rule exists because G1 is the only proven working reference. Every undocumented divergence is a potential source of a local optimum or training failure. "It seems reasonable" is not a justification — G1 must be the baseline.

## Training / Play Parity Rule

**Training and play ball spawn parameters must always match.** Whenever `dist_range`, `speed_range`, `y_start_range`, `y_end_range`, or `spawn_z` are changed in the training `reset_ball` block (`goalkeeper_env_cfg.py` around line 515), the play block (around line 589) must be updated to the same values in the same commit. A policy evaluated on a different distribution than it was trained on gives misleading results.

## Git Commit & Push Rule

**Every push must include ALL modified files** (`git add -A`) unless the user explicitly says otherwise. Never leave uncommitted working-tree changes behind when pushing.

## Change Approval Workflow

**Before making ANY code changes**, you MUST:

1. **List all changes** you plan to make in a clear, bullet-point format
2. **Show this list to the user** and wait for explicit approval
3. **DO NOT apply changes until approved**
4. **Only after approval**: apply changes and document them

This prevents accidental modifications, keeps the user informed of scope, and ensures changes match the actual request.

## Divergences from G1 Upstream

| Parameter | G1 value | SimpleGoalKeeper value | Justification |
|-----------|----------|------------------------|---------------|
| **Robot reset (RSI)** | Fixed standing pose | **Full RSI: root Z + quat + lin/ang vel + joints from random NPZ frame; XY kept from env_origins** | **CRITICAL 2026-06-15 (fixed v1):** `reset_base` added so root position/yaw reset. **CRITICAL 2026-06-15 (fixed v2):** RSI was joint-state-only — root velocity was zero at reset. A robot in a mid-step pose with zero root velocity is physically impossible: the simulator immediately jerks it back to standing. Fix: `MotionResetManager` now also loads `body_pos_w[:,0,:]`, `body_quat_w[:,0,:]`, `body_lin_vel_w[:,0,:]`, `body_ang_vel_w[:,0,:]` from NPZ and writes root Z + quat + velocity from the sampled motion frame. Root XY stays at env_origins (goal line fixed). Mirrors BoosterT1mjlab `_write_reset_state` exactly. |
| Ball spawn height | 0.15–0.4 m (chest-compatible) | **0.05–0.35 m (foot-to-shin)** | **CRITICAL 2026-06-14:** Ball spawn was too high (0.2–0.7 m), forcing upper-body contact. Lowered to foot/shin level for feet-only goalkeeping. Also lowered arrive height from 0.1–0.4 m to 0.05–0.25 m. |
| `ang_vel_xy` weight | -0.1 (roll+pitch only, `[:, :2]`) | -0.1 same | No change |
| `ang_vel_z` (yaw) | **Not penalized** (free) | **-0.5** | G1 yaws to extend arm reach. Feet have ~10 cm reach radius — yawing does not help. Observed training run 9000: robot learned to spin around (0,0). |
| `_ball_is_behind` threshold | `delta_vx > 2.0` (`legged_robot.py:1377`) | `delta_vx > 1.0` | MuJoCo soft contact produces smaller impulses than Isaac Gym PhysX. At 2.0 m/s threshold, slow-ball saves (ball at -1 → +0.5 m/s) never fire `stopball` or activate post-save rewards. Lowered to 1.0 m/s, same as ILB. Structure is identical: `(x<0) \| (delta_vx > threshold)`. |
| `ball_positive_vx` | Not in G1 | **Removed** | Was a SGK-only addition. Caused robot to chase ball after saving it (continuous reward never deactivates). G1/ILB rely on `stopball` + post-save recovery; no continuous vx reward needed. |
| Ball spawn frame | Robot-local | **World (global) frame** (`reset_ball_rolling`) | **FIX 2026-06-20:** Changed from robot-local to world frame. Ball always approaches along world -X regardless of robot yaw. Keeps `stopball` (world-X velocity), `stayonline` (world-X position), `ball_exit_termination` (world-X) exactly aligned with the ball trajectory. Robot faces world +X (enforced by `ang_vel_z=-2.0`), so the difference is minimal in practice but the frames are now consistent. `events.py:reset_ball_rolling`. |
| Ball spawn angular velocity | N/A | **Pure-rolling ω at spawn** (`ωy=vx/r, ωx=-vy/r`) | **FIX 2026-06-20:** Ball was spawned with zero angular velocity, causing a sliding phase that loses `2/7 × v0` to friction before reaching rolling. At max spawn speed 3.5 m/s, `delta_vx_friction = 1.0 m/s` — exactly equal to the `stopball` threshold, firing a false-positive save reward with no robot contact. Fix: set angular velocity to pure-rolling no-slip condition at spawn. With `condim=3` (no rolling friction), ball then rolls at constant speed and contributes zero delta_vx. `events.py:reset_ball_rolling`. |
| RSI foot floor penetration | N/A | **+0.030 m root Z offset in MotionResetManager** | **FIX 2026-06-20:** `pkl_to_npz` aligns the foot_link **body** center to Z=0, but the capsule collision geoms sit at Z=-0.01 in the foot frame with radius=0.02 → lowest contact point is 0.03 m **below** the floor. At reset, MuJoCo violently pushes the robot up via contact forces ("pump-up"), causing large sharpforce readings and early terminations. Fix: `MotionResetManager._write_rsi_state()` adds `_FOOT_CONTACT_BELOW_BODY = 0.030` to the motion root Z so the contact surface lands at floor level. `events.py`. |
| RSI motion selection | Random frame from all 6 motions | **Distance-conditioned: ball crossing Y → (side × step-count) pool** | **FIX 2026-06-20:** Random RSI picked triple-step frames for close balls and single-step frames for far balls, giving the policy no correlated hint about which motion to use. Fix: `reset_ball` now fires **before** `reset_from_motion_data` (see `goalkeeper_env_cfg.py` event order). `MotionResetManager` reads the new ball velocity, computes predicted goal-line crossing Y, and routes each resetting env to one of 6 pools (left/right × single/double/triple). Thresholds: \|cross_y\| < 0.35 m → single, 0.35–0.65 m → double, ≥ 0.65 m → triple. AMP discriminator unchanged (still sees all motions). `events.py:MotionResetManager`. |
| Ball spawn `y_start_range` | ±0.8 m (ILB) | **±0.3 m** | **FIX 2026-06-20:** Reduced from ±1.0 m. ±0.5 m lateral spawn combined with ±1.5 m target created extreme diagonal shots (>60°) the robot could not physically reach. ±0.3 m keeps approach angles reasonable while still training off-centre trajectories. `goalkeeper_env_cfg.py:reset_ball`. |
| Ball spawn `y_end_range` (goal target) | Region-conditioned `command_ranges`, curriculum-expanded ±0.3/update up to region bounds; the low/feet region (`ranges_4`) reaches **±1.8 m** at max curriculum (`g1_29_config` maxw, `legged_robot.py:333-336`) | **±1.1 m with dead zone at d=0** | **FIX 2026-06-20:** Reduced from ±1.5 m. **FIX 2026-06-20b:** Added two-sided dead zone: at d=0 ball arrives 0.15–0.35 m to left or right of center (never dead center through the legs). Dead zone shrinks linearly to 0 at d=1. Mirrors G1's ±0.2 m hard minimum center offset. **UPDATE 2026-07-03:** widened ±0.9 → ±1.1 (train + play, parity rule) — user request to force double-stepping: wide balls are beyond single-lunge reach, and the policy was not converging to double-steps at ±0.9. No stepping-specific reward added: G1 has none — its footwork emerges from wide targets (feet region to ±1.8) + step motions in the prior; a hand-crafted step reward invites foot-shuffle hacking. The 4× double/triple AMP boost was also NOT restored: at 6 files the double/triple motions already hold 77% of frames uniformly, and 4× would crush the near-standing share to 7%, re-creating the post-save AMP conflict fixed on 2026-07-03. `events.py:reset_ball_rolling`, `goalkeeper_env_cfg.py` both reset_ball blocks. |
| `stopball`/`softstop` `in_front` threshold | N/A | **`ball_x_local > -0.3`** (was `> 0.0`) | **FIX 2026-06-19:** Ball deflection accumulates gradually — by the time `delta_vx` crosses 1 m/s, the ball may have rolled a few cm past x=0. The strict `> 0.0` check silently missed saves that completed just after the goal line. Relaxed to `-0.3 m` grace margin. `rewards.py:184, 214`. |
| RSI in play mode | RSI active | **RSI now active in play mode too** | **FIX 2026-06-20:** Play mode was popping `reset_from_motion_data` (stale comment from commit 6832e64 when RSI was briefly removed). After RSI was re-added in subsequent commits, the pop was never removed. Fix: removed the `cfg.events.pop("reset_from_motion_data", None)` line from play mode so play episodes also start from random motion frames, matching training distribution. `goalkeeper_env_cfg.py` play block. |
| `feet_slippage` ball-contact gate | N/A | **Suppressed when ball within 0.5 m of either foot** | **FIX 2026-06-19:** `feet_contact` sensor cannot distinguish ground from ball contact. Without gating, the reward penalized foot movement during ball contact, training the policy to plant feet passively rather than sweep into the ball (fighting `stopball`). Now returns 1.0 (neutral) when `min_dist_foot_to_ball < 0.5 m`. `rewards.py:feet_slippage`. |
| `footreach`/`foot_proximity` phase2 target | Live ball when close, frozen when far (G1) | **FIX 2026-06-24:** Restored G1 live-ball switch. G1 `post_physics_step` lines 203-206: `end_target = ball_states[:, :3]` when `ball_x_local < 0.5 m`. Previously SGK froze the crossing point for the full episode (FIX 2026-06-20), which broke phase2 coupling to the actual ball. Now: far (≥ 0.5 m) → frozen crossing_y; close (< 0.5 m) → live ball Y/Z. X stays at `goal_x_w` (stayonline keeps foot at goal line). `rewards.py:footreach,foot_proximity`. | 
| `footreach` `vel_sigma` multiplier | G1: 3.0 (max 10×) | **3.0 (max 10×) — matches G1** | Restored to G1 value 2026-06-24 after Fix A added live-ball tracking for ball < 0.5m. Previously reduced to 1.0 because vel_sigma applied to a fully frozen target rewarded sprinting without contact. Now that the target snaps to the live ball in the critical close range (same structure as G1's end_target), the 3.0 multiplier is safe. Note: intentional remaining divergence — G1 sets `vel_sigma[behind]=2.0` (reward continues post-save at 2×); SGK zeros via `(~behind).float()` to prevent post-save footreach farming. `rewards.py:footreach`. |
| `stopball` `delta_vel_threshold` | G1: 2.0 m/s | **0.6 m/s** | **FIX 2026-06-24:** Previous SGK threshold was 1.0 m/s (documented in `_ball_is_behind` row). At minimum ball speed 0.8 m/s, stationary foot COR≈0.7 gives max delta_vx = 1.7 × 0.8 = 1.36 m/s. The 1.0 threshold was achievable but marginal. More critically, at 0.5 m/s balls (previous min speed), delta_vx_max = 0.85 — below 1.0 → stopball was physically impossible. New threshold 0.6 m/s ensures stopball fires reliably at all speeds ≥ 0.8 m/s. G1's 2.0 m/s is for high-speed hand catches (3–6 m/s balls) — SGK balls max 2.0 m/s so 2.0 threshold would require a near-perfect stop. `goalkeeper_env_cfg.py:stopball`. Note: this also lowers the `_ball_is_behind` sb_flag threshold from 1.0 to 0.6. |
| Ball spawn timing (speed) | G1: `t_flight = 0.4 + 0.6×rand()` s, derive velocity from distance/t_flight | **`t_flight_range`: easy (0.9–1.3 s) → hard (0.7–1.1 s). Speed = horiz_dist / t_flight, giving ~1.8–5.7 m/s at full difficulty.** | **FIX 2026-06-27:** Previous approach sampled speed directly (`speed_range=(2.0,4.0)`), making reaction time depend on distance — a far diagonal shot at max speed gave only 0.57 s reaction time. Switched to G1's t_flight-first approach: sample reaction time directly, derive speed from trajectory length. Guarantees ≥0.7 s reaction time at all difficulties. `_EASY_T=(0.9,1.3)`, `t_flight_range=(0.7,1.1)`. `events.py:reset_ball_rolling`, `goalkeeper_env_cfg.py`. |
| `stopball` curriculum | G1 lines 363-364: `weight = stop_init × (1 + 0.5 × cu)` | **Added 2026-06-24: base 20 → max 50 at cu=3** | **FIX 2026-06-24:** Previously stopball had no curriculum (fixed weight 20). G1 includes stopball in the episode-length curriculum (same formula as eereach). Added `stopball_curriculum` with base 20 → max 50. softstop stays 100 → 250 (5× higher at all levels). `goalkeeper_env_cfg.py`. |
| Episode length (training) | ILB: 3 s | **SGK: 3 s** (was 6 s) | **FIX 2026-06-20:** 6 s was wasted compute. The `~behind` gate zeros `footreach` once the ball is deflected (~75 steps in), so the remaining 225 steps of a 6 s episode contributed zero footreach and diluted `stopball`'s per-step signal. At 6 s (300 steps), max stopball/step = weight(100)/300 = 0.33. At 3 s (150 steps), max = 100/150 = 0.67 — matching the observed 0.7 from near-100% save rate. Matches ILB episode_length_s=3.0. `goalkeeper_env_cfg.py:episode_length_s`. |
| **RSI split** | 100% RSI (previous SGK) | **80% RSI + 20% default-pose branch** | **FIX 2026-06-20:** Mirrors ILB `reset_robot_rsi` 80/20 split. Pure 100% RSI caused AMP artefact: after a diving save, the next episode's RSI mid-motion start looks like a discontinuous jump from a dive pose — the discriminator penalises this as an impossible transition. The 20% branch gives AMP a natural standing→step transition that IS in the reference data. **UPDATE 2026-07-01:** this row's 80/20 claim was silently wrong in practice from 2026-06-20 until today — `reset_from_motion_data` hardcoded `rsi_fraction=0.5`, a 50/50 split, caught by an independent fidelity audit and fixed to restore the intended 80/20. See the RSI mechanism row below for full current behavior. `events.py:MotionResetManager.reset`. |
| `penalize_sharpcontact` threshold | 1000 N | **1200 N** | **FIX 2026-06-20:** 1000 N was too sensitive — normal rapid lateral stepping during ball interception triggered the penalty, preventing the robot from moving aggressively. At 1000 N, the training run (iter 7143 and 18944) showed catastrophic crashes where the penalty fired continuously (every step of a 150-step episode). 1200 N still penalises ground-slamming but tolerates normal stepping dynamics. `rewards.py:penalize_sharpcontact`, `goalkeeper_env_cfg.py`. |
| `ball_difficulty` curriculum `ep_len_divisor` | 48 | **47** | **FIX 2026-06-29:** `ball_difficulty_curriculum` used divisor 48 while all reward curricula (softstop, footreach, stopball) used 47. This caused ball difficulty to advance at a slightly different episode-length threshold (ep_len=48 vs 47 for cu=1), making the curves appear desynchronised in WandB. Changed to 47 to match reward curricula exactly. `goalkeeper_env_cfg.py`. |
| WandB logging | N/A | **Batch per iteration (single `wandb.log` call)** | **FIX 2026-06-29:** `WandbSummaryWriter` called `wandb.log({tag: val}, step=it)` separately for EACH metric. WandB's internal step counter auto-increments on every `wandb.log()` call, so metrics for the same training iteration appeared at different WandB x-axis positions. Fixed by buffering all `add_scalar` calls for the same step and flushing in a single `wandb.log(all_metrics, step=it)` call. `beyondAMP/source/rsl_rl_amp/rsl_rl_amp/utils/wandb_utils.py`. |
| `airborne_at_save` reward | N/A | **Removed 2026-06-29** | **FIX 2026-06-29:** User request. The reward optimised for the correct foot being airborne at the save moment, but this is a secondary quality metric that adds training noise without clear benefit at early stages. Removed the `RewardTermCfg` and `airborne_at_save_curriculum` from config. `goalkeeper_env_cfg.py`. |
| RSI mechanism (`reset_from_motion_data`) | `continue_keep` branch (80%): single batch-level coin flip; copy `dof_pos` from `torch.randint(0, num_envs, ...)` (any live env, no tier matching, no exclusion of current reset batch, no maturity check), no clamp. Else branch (20%, `randomize_initial_joint_pos=True` in G1's active g1_29_config): `standpos * U(0.5, 1.5) + U(-0.1, 0.1)`, clipped to HARD `dof_pos_limits`. `dof_vel` always zeroed both branches; root untouched (G1 also randomizes root velocity ±0.3 unconditionally in `_reset_root_states`, separate from this). | **Literal port**, matching every point above: one `torch.rand(1)` draw per `reset()` call, donor from any of `env.num_envs` with no restriction/guard and no clamp; else branch scales/offsets `default_joint_pos` and clips to `robot.data.joint_pos_limits` (hard, not soft); `joint_vel` always zeroed; root left to `reset_base` (root velocity still zero there — known, separate, out-of-scope divergence). | **2026-07-01:** Initially implemented as a side/tier-scoped variant, then explicitly reverted to a literal, unscoped port per user request. A follow-up independent fidelity audit (dispatched per user request to check "is this actually similar to G1") caught three further divergences in that first literal-port attempt, all now fixed: (1) the clamp was applied to the donor-copy branch, where G1 has none, and was missing from the else branch, where G1's active config actually clips; (2) the else branch was a flat `default_joint_pos` copy with zero randomization, omitting G1's active `randomize_initial_joint_pos=True` scale/offset entirely; (3) `reset_from_motion_data` hardcoded `rsi_fraction=0.5`, silently running a 50/50 split instead of G1's 80/20 the whole time since 2026-06-20. The old side/tier NPZ-pool system (`self.pools`, `_write_rsi_state`, `_STEM_TO_POOL`) still exists and loads at startup but is no longer used by `reset()` — kept only for the `sgk_play_rsi` diagnostic script. See `docs/superpowers/plans/2026-07-01-live-env-rsi.md` for the full audit trail. |
| AMP discriminator motion sampling | N/A | **Per-motion weighting added 2026-07-02: double/triple-step files boosted 4× vs. uniform-by-frame-count** | **FEAT 2026-07-02:** User request (weight raised from an initial 3× to 4× same day). `MotionDatasetCfg`/`MotionDataset.sample_batch` (`beyondAMP/.../motion/motion_dataset.py`) previously sampled AMP transitions uniformly across all concatenated motion frames with no per-motion weighting field at all. Separately, `WeightedMotionDataset` (`weighted_motion_dataset.py`) — which `mjlab/rsl_rl/amp_wrapper.py` already wraps every `MotionDatasetCfg` in by default — accepted a `traj_weights` constructor arg but silently discarded it (`self.weights` was always `torch.ones(...)`, and `_build_transition_weights_from_traj` referenced `self._traj_lengths`, which the base class never set — dead/broken code, never exercised). Fixed both: base class now stores `self._traj_lengths`; `WeightedMotionDataset.__init__` now actually converts `traj_weights` into per-transition weights when provided. Added `motion_weights: list[float] | None` field to `MotionDatasetCfg`, forwarded from cfg to `WeightedMotionDataset` via `amp_wrapper._motion_weights_from_cfg()`. `goalkeeper_amp_cfg.py:_motion_weights()` (`_DOUBLE_TRIPLE_STEP_WEIGHT = 4.0`) sets weight 4.0 for any motion file with `DoubleStep`/`TripleStep` in its name, 1.0 otherwise — applies to `LeftDoubleStep`, `RightDoubleStep`, `LeftTripleStep`, `RightTripleStep` (4 of 14 files, previously ~36% of AMP transitions by frame count alone). Covered by `tests/simple_goalkeeper/test_weighted_motion_dataset.py`, `test_amp_wrapper_motion_weights.py`, `test_amp_motion_weights.py`. Not yet validated against a training run — next checkpoint should be compared against pre-change runs for AMP naturalness on double/triple-step saves. **UPDATE 2026-07-02:** dataset restricted to the 4 double/triple-step files only, weights dropped (881e063). **UPDATE 2026-07-03:** `LeftStep_own`/`Rightstep_own` re-added (6 files, `Safe*` still excluded, uniform sampling) — the standing-free dataset made AMP fight the post* recovery rewards (post-save walking in play, run 2026-07-02_22-56-40); G1's dataset contains `leftstep.pt`/`rightstep.pt`. See `docs/BugFixes.md` 2026-07-03. |
| Actor ball-obs post-save release | `flying` mask: visible only while ball in front cone (`x_body` 0.05–3.4, \|y\|<2, z<1.8), still approaching, AND within the 50-step catch window (`catchstep > 0`, counts DOWN from 50 at launch; `legged_robot.py:397-428`) — the back ~2/3 of every 3 s episode is blind; plus warmup blackout (3–10 steps at episode start, `initial_vanish`, actions overridden to the reset pose via line 643) and noise-mode `random_vanish` (ball cut 0–30 steps early). Noise added BEFORE the mask (lines 425-426), so hidden = exact zeros. | **v2 gate (2026-07-03): actor's `ball_pos_xy_b` zeroed when ball is behind the torso (`x_body < 0.05`, G1's front edge) OR episode step > 75 (G1's catch window, resized from 50 because SGK flights are 0.7–1.3 s vs G1's 0.4–1.0 s — the window closes only after the latest possible arrival). Noise moved inside the term before the mask (`noise_scale=0.05`, manager noise off; play sets 0.0) so hidden = exact zeros, matching G1's ordering. NOT ported per user constraint (no pre-save blindness): warmup blackout, random vanish, approach/cone checks, and G1's 3.4 m far edge (SGK spawns at 2–4 m — G1's own cone hides far balls at spawn). Critic ungated.** | **FIX 2026-07-03 (v2), replaces FEAT 2026-07-02 (v1):** v1 gated on the `_ball_is_behind` save flags — a mechanism G1 does not have. Audit of run 2026-07-02_22-56-40 (first v1 run) found 17% of iterations diverged (mean reward to -9.5e5, value loss to 5e10, action_rate flailing) vs zero divergent iters in the gate-free control 20-18-38, and play showed post-save walking. Root causes vs G1: (1) v1 kept the ball visible through soft grazes below the 0.6 m/s flag threshold, so `footreach` kept paying post-contact chase; (2) v1's `x<0` trigger was un-latched and could re-toggle; (3) mjlab manager noise was applied AFTER the zeroing, so the policy trained against a phantom ±5 cm ball at its feet instead of zeros; (4) no catch window — G1 policies spend ~2/3 of every episode blind from iter 1, which is where post-save standing is actually learned. The earlier full-port failure (8803b6e, run 17-53-27) misread `catchstep` as counting up, making the warmup ~43 steps instead of 3–10 — the G1 mechanism itself was never correctly tested. `observations.py:ball_pos_xy_b`, `goalkeeper_env_cfg.py` actor term + play block. |
| Effector type | Hands | Feet only | Phase 1 scope. |

## Frame Convention

Ball spawning uses **world (global) frame** (`reset_ball_rolling` in `mdp/events.py`).
Ball spawns at `(env_origin_x + x_start, env_origin_y + y_start, floor_z + spawn_z)` in
world frame, aimed at `(env_origin_x - 0.3, env_origin_y + y_end)` — 0.3 m behind the
goal line so the ball retains forward momentum at interception.

Ball always approaches in world **-X** direction. All reward terms that gate on ball direction
also use world-X (stopball: `root_link_lin_vel_w[:, 0]`; stayonline: world-X position;
ball_exit_termination: world-X). Observations (`ball_pos_b`, `ball_vel_b`) are in robot
body frame — consistent because `ang_vel_z=-2.0` keeps robot facing world +X.

Key frame notes:
- `noretreat`: body-frame X velocity (correct even when robot yaws during a dive)
- `stopball`/`softstop`: world-X velocity — consistent with world-frame ball spawn
- `footreach`/`crossing_y`: world Y alignment (ball_y_w − robot_y_w)

## Key Files

| File | Purpose |
|------|---------|
| `src/simple_goalkeeper/robots/t1_constants.py` | T1 actuator configs, action scale, home keyframe |
| `src/simple_goalkeeper/robots/xmls/` | T1 headless XML + ball XML + STL assets |
| `src/simple_goalkeeper/mdp/observations.py` | ball_pos_b, ball_vel_b (visibility system), foot positions |
| `src/simple_goalkeeper/mdp/events.py` | reset_ball_local_frame, tick_catchstep |
| `src/simple_goalkeeper/mdp/rewards.py` | 5 goalkeeper reward terms (feet-only) |
| `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` | Full env config |
| `src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py` | AMPRunnerCfg |
| `src/simple_goalkeeper/tasks/__init__.py` | Task registration |
| `src/simple_goalkeeper/motions/data/` | NPZ motion files (converted from PKL) |
| `src/simple_goalkeeper/scripts/train.py` | Training entry point |
| `src/simple_goalkeeper/scripts/play.py` | Play/evaluation entry point |
| `src/simple_goalkeeper/scripts/pkl_to_npz.py` | PKL→NPZ motion converter |

## beyondAMP Location

Cloned at `./beyondAMP/`. The four packages are installed as editable:
- `beyondAMP/source/beyondAMP` → `beyondAMP` package
- `beyondAMP/source/rsl_rl_amp` → `rsl-rl-amp` package
- `beyondAMP/source/amp_tasks` → `amp-tasks` package
- `beyondAMP/source/amp_tasks_mjlab` → `amp-tasks-mjlab` package

## Motion Files

NPZ format, 21-DOF headless T1 joint order. Expected arrays:
- `fps`: sampling rate
- `joint_pos` (T, 21): joint positions (absolute, matching T1 default pose reference)
- `joint_vel` (T, 21): joint velocities via finite differences
- `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`: body kinematics from FK

Convert PKL → NPZ:
```bash
uv run sgk_convert --input-dir /path/to/Motions --output-dir src/simple_goalkeeper/motions/data
```

## Reward Design

Phase 1 reward structure (ported from proven Imitationlearningbooster pattern):

| Term | Weight | Purpose |
|------|--------|---------|
| `stopball` | +100→250 (curriculum) | One-time bonus when ball is deflected (delta_vx > 1 m/s). Primary signal. |
| `footreach` | +10→20 (curriculum) | Phase1: lateral alignment. Phase2: sigmoid reach × vel_sigma (1–10×). Deactivates on deflection. |
| `stayonline` | -2.0 | Penalty for drifting away from goal line (X displacement) |
| `noretreat` | -2.0 | Penalty for retreating backward (negative body-frame X velocity) |
| `feetorientation` | +3.0 | Flat feet (gravity aligned with foot Z) |
| `postorientation` | +3.0 | Upright posture recovery, active only when ball is behind |
| `postangvel` | +3.0 | Low XY angular velocity reward, active only when ball is behind |
| `postlinvel` | +1.0 | Low forward velocity reward, active only when ball is behind |
| `postupperdofpos` | +1.0 | exp(-err) arm recovery to default, active only when ball is behind |
| `postwaistdofpos` | +1.0 | exp(-err) waist recovery to default, active only when ball is behind |
| `penalize_kneeheight` | -100.0 | Penalty when shank drops below 15 cm above floor (prevents kneeling) |
| `penalize_sharpcontact` | -100.0 | Binary penalty when mean foot contact force > 1000 N (requires `feet_contact` sensor) |
| `penalize_self_collision` | -50.0 | Binary penalty on any Trunk-subtree self-collision (requires `self_collision` sensor) |
| `feet_slippage` | +3.0 | exp(-10*contactvel) — rewards stable foot contact, penalises sliding (requires `feet_contact` sensor) |
| `dof_pos_limits` | -3.0 | Joint limit violation penalty |
| `dof_vel_limits` | -2.0 | Penalty for joint velocity > 10 rad/s (sum of squared excess) |
| `torque_limits` | -3.0→-9.0 (curriculum) | Per-joint torque limit violation; penalises hip-yaw spin torques |
| `ang_vel_xy` | -0.1 | Penalise rolling/pitching |
| `ang_vel_z` | -0.5 | Penalise yaw rotation — a goalkeeper should face the field |
| `deviation_waist_joint` | -0.001 | Waist joint regularisation (always active) |
| `torques` | -1e-5 | Normalized torque L2: sum((torque/kp)^2) — dimensionless across joints |
| `action_rate_l2` | -0.3 | Action smoothness |
| `action_acc_l2` | -0.1 | Action jerk penalty (second-order smoothness) |
| `dof_vel` | -5e-4 | Joint velocity regularisation |
| `dof_acc` | -2.5e-7 | Joint acceleration penalty (jerk reduction; matches ILB) |

**Terminations:** `time_out`, `bad_orientation` (>57°), `base_height` (<0.4 m), `ball_exit` (behind goal -0.5 m), `sharpforce` (>1500 N mean foot force).

**`_ball_is_behind` semantics:** `(ball_x < 0) | (delta_vx > 1.0)` — matches ILB exactly. Fires the moment stopball fires (deflection), deactivating `footreach` (no post-save chasing) and activating all post-save recovery rewards immediately.

**Removed (created stand-still or wrong local optimum):**
- `ball_positive_vx`: caused robot to chase ball after already saving it (continuous vx reward never turns off); `_ball_is_behind` + `stopball` are sufficient
- `successland`: with feet-only goalkeeping this became a ball-chasing reward; removed in favour of `stopball`
- `ball_vx_reduction`: peaked when ball stopped naturally — rewarded doing nothing
- `foot_to_ball` (std=0.15): zero gradient at 2–4 m spawn distance
- `posture`: AMP handles motion naturalness; posture+regularisation incentivised standing still

**Ball visibility:** actor's `ball_pos_b` uses `always_visible=True` (full visibility
during the entire approach and save) plus the v2 post-save release gate:
`hide_behind_torso=True` (zero once ball `x_body < 0.05`, G1's flying-mask front edge)
and `hide_after_steps=75` (zero once the episode step exceeds the window — G1's
`catchstep` analog, sized past SGK's max 1.3 s flight so it never blinds a ball still
en route). Ball-obs noise is applied inside the term BEFORE the mask (`noise_scale=0.05`
in training, 0.0 in play; manager noise off) so a hidden ball is exact zeros — G1's
noise-then-mask ordering (`legged_robot.py:425-426`). The full G1 visibility system
(warmup + flying cone + random vanish, `_compute_ball_visibility`) exists in
`observations.py` but is not active in training or play (see the post-save release
divergence row). Critic's `ball_pos_b`/`ball_vel_b` remain `always_visible=True`.

## Training Commands

```bash
# Convert motions (once):
uv run sgk_convert --input-dir /home/isaak/BEPImitationlearning/Motions --output-dir src/simple_goalkeeper/motions/data

# Train:
uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1 --num-envs 4096

# Play (zero policy sanity check):
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1

# Play (trained checkpoint):
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --checkpoint-file logs/rsl_rl/simple_goalkeeper/<run>/model_500.pt
```

## Reading TensorBoard / WandB Episode Reward Metrics

mjlab's `reward_manager` logs `Episode_Reward/X` with **two scaling factors** baked in:

```
Episode_Reward/X = (Σ over episode of [reward_fn(obs) × weight × dt]) / max_episode_length_s
```

Where `dt = 0.02 s` and `max_episode_length_s = 3.0 s` (150 steps).

**The logged value is NOT a raw per-episode sum.** It is divided by `max_episode_length_s`, so it represents "reward per second" rather than "reward per episode". This is done in `reward_manager.py`:
```python
value = value * term_cfg.weight * scale          # scale = dt = 0.02
self._episode_sums[name] += value
# on episode end:
extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
```

### Converting logged values to meaningful metrics

**One-shot rewards** (fire once per episode: `softstop`, `stopball`, `single_foot_save`):
```
event_rate = logged_value × max_episode_length_s / (weight × dt)
           = logged_value × 3.0 / (weight × 0.02)
```
Example: softstop logged = 1.09, weight = 210 → rate = 1.09 × 3.0 / (210 × 0.02) = **77.9%** of episodes

**Per-step rewards** (e.g., `ang_vel_xy`, `feetorientation`): logged value ≈ average per-step value × weight (already divided by max_episode_length_s cancels the dt accumulation for continuous rewards).

**Binary termination-linked rewards** (e.g., `penalize_sharpcontact` at 1700 N): same formula as one-shot, since they also fire on at most one step per episode.

This scaling is the source of the "softstop ≈ 1.09 looks tiny" confusion — 1.09 is actually a 78% save rate. Always apply the formula before interpreting one-shot metrics.

## Standalone Constraint

**No runtime imports from `Imitationlearningbooster`, `BoosterT1mjlab`, or `HandWavingMotion`.**
All needed assets and constants are copied into this folder.

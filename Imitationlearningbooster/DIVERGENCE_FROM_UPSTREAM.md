# Divergence from Upstream (Humanoid-Goalkeeper)

This document tracks substantive changes where the Booster T1 adaptation deviates from the G1 tracking task pipeline.

## 2026-05-14 — Fix reward axis mismatch after 90° rotation (ball from +Y not +X)

**File:** `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/mdp/rewards.py`

**What:** Changed `_ball_is_behind`, `stopball`, `stayonline`, and `noretreat` from using world X (index 0) to world Y (index 1).

**Why:** The original Humanoid-Goalkeeper had the ball coming from +X and the robot facing +X, so all rewards used world X. When the T1 port rotated the setup 90° (ball from +Y, robot faces +Y), the reward functions were never updated. As a result:
- `stopball` (weight=100) **never fired** — ball ends at negative X so `ball_x_local > 0` was always false. The entire main task reward was dead.
- `_ball_is_behind` was true from the start of every episode (ball starts at negative X), so `postorientation`/`postangvel`/`postlinvel` activated immediately and at the wrong time.
- `noretreat` penalised moving in world -X (sideways to the ball) rather than -Y (actually retreating from the ball), giving the policy no signal to prevent retreating.
- These misaligned rewards likely caused the observed clockwise rotation: no penalty for drifting/spinning, and the dominant 100-weight reward not firing at all.

**Impact:** Full retrain required. `stopball` will now actually fire, which is the primary learning signal.

## 2026-05-14 — Fix bang-bang ankle/leg control and PPO rollout length

**Files:**
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/robots/t1_constants.py`
- `my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/tasks/goalkeeper_ppo_cfg.py`

**What:**
1. Applied the `0.25` action-scale factor to all non-arm joints (waist, hips, knees, ankles). Previously only arms had this factor.
2. Set `num_steps_per_env=24` (original upstream used 100) and `gamma=0.99` (was 0.998, now matches upstream).

**Why — action scale:**
Without the 0.25 factor, every non-arm joint saturated at max torque when the policy output ±1.0. Because the initial policy is Gaussian(0,1), outputs above ±1 are common, causing bang-bang (on/off) torque switching — visible as fast ankle jitter and micro-bouncing. The G1 reference applies `0.25 × effort/stiffness` to ALL joints so saturation only occurs at action=4.0.

**Why — gamma:**
0.998 was an unexplained deviation from both the original Humanoid-Goalkeeper (0.99) and the G1 mjlab config (0.99). Reverted to 0.99.

**Why — num_steps_per_env=24 (tradeoff, may need revisiting):**
The original upstream used 100, which equals 2 seconds of experience per rollout at 50 Hz. 24 steps = 0.48 seconds. Using 24 makes each training iteration ~4× faster in wall-clock time (fewer simulation steps per gradient update cycle), which is valuable for experimentation.

The risk: a 5-second episode spans ~10 rollouts at 24 steps. The dominant reward (`stopball`, weight=100) only fires near the end. The value function must propagate that credit backward through 10 rollout boundaries using bootstrapped estimates. With `gamma=0.99`, early-episode value estimates are already discounted to ~0.37 of the final reward — so the signal is diluted. In practice this may cause the policy to react late to the ball rather than pre-positioning early.

**Switch back to `num_steps_per_env=100` if:** after a solid training run (5000+ iterations) the robot only dives at the last second and never anticipates the ball's trajectory early.

**Impact:**
- Ankle pitch: action=1 now produces 5 Nm (was 20 Nm = full saturation). Control stays in PD linear regime.
- Smooth foot contact expected; micro-bouncing should be eliminated.
- **All prior checkpoints are incompatible** (action space meaning changed). Full retrain required.

## 2026-05-03 — Fully autonomous goalkeeper (no motion input at play time)

**Files:** 
- `my_mjlab_project/src/my_mjlab_project/mdp/resets.py` (new)
- `my_mjlab_project/src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

**What:** 
1. Created `reset_ball_autonomous()` - standalone ball reset function with no motion tracking dependency
2. Removed motion command entirely from play config
3. Added autonomous ball reset as startup event
4. Removed 6 motion-dependent reward + 3 motion-dependent termination terms

**Why:** The end goal is a **100% autonomous goalkeeper**:
- **Training:** Learn from all 6 motion types (left/right hand, jump, step) via RSI (RL + imitation)
- **Observations:** Ball position, ball velocity, joint state **only** (no motion in obs)
- **Play:** Policy autonomously chooses best response to any incoming ball

By removing the motion command, the policy receives no motion input at inference time. The autonomous reset function ensures the ball still gets randomized trajectories.

**Impact:** 
- Policy trained on diverse motion examples but is **completely autonomous at play time**
- No `--motion-file` argument needed
- Ball resets with random trajectory **every 10 seconds** (3-5m away, random y/z, timed arc)
- Episode auto-resets every 10 seconds to generate new ball trajectory
- Play command: `uv run python -m mjlab.scripts.play goalkeeper --checkpoint-file logs/.../model_N.pt`
- Play mode runs stably, policy continuously faces new ball trajectories
- Robot returns to random pose on each reset (domain randomization)

## 2026-05-02 — Remove motion-reference observations from actor/critic

**File:** `my_mjlab_project/src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

**What:** Removed `command`, `motion_anchor_pos_b`, `motion_anchor_ori_b` from
the actor and critic observation groups in `goalkeeper_env_cfg()`.

**Why:** The upstream G1 tracking task treats reference motion as an explicit
input to the policy. This requires a motion file at inference time. For an
autonomous goalkeeper agent that decides its own response based on ball
position, the motion reference must not appear in the observation space.
The 6 motion files are retained for RSI (reference state initialisation) and
style-shaping rewards during training only.

**Impact:** All checkpoints trained before 2026-05-02 are incompatible with the
new observation space (actor dim changed). Full retrain required.

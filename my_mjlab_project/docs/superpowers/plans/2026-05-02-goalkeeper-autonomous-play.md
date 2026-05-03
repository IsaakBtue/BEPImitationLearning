# Goalkeeper Autonomous Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain a goalkeeper policy that plays autonomously from ball + joint state alone — no motion file needed at inference.

**Architecture:** Remove motion-reference observations (`command`, `motion_anchor_pos_b`, `motion_anchor_ori_b`) from the training actor/critic obs. The 6 motion files remain active during training for RSI (reference state initialisation) and style rewards only. The play config simply removes the motion command; no further obs cleanup is needed because training obs are already clean.

**Tech Stack:** Python 3.12, mjlab, MuJoCo Warp, PyTorch, uv

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Modify | `src/my_mjlab_project/tasks/goalkeeper_env_cfg.py` | Remove motion-ref obs from training; simplify play cfg |
| Modify | `commands.txt` | Update play command (no --motion-file) |
| Modify | `../../Imitationlearningbooster/DIVERGENCE_FROM_UPSTREAM.md` | Document obs-space change |

---

### Task 1: Remove motion-reference observations from training config

**Files:**
- Modify: `src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

The base config (`unitree_g1_flat_tracking_env_cfg`) adds `command`, `motion_anchor_pos_b`, and `motion_anchor_ori_b` to both actor and critic observation groups. We remove them inside `goalkeeper_env_cfg()` immediately after building the base config, before any other customisation.

- [ ] **Step 1: Open the file and locate `goalkeeper_env_cfg`**

Read `src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`. The function starts at line ~39.

- [ ] **Step 2: Add obs removal block after `cfg = unitree_g1_flat_tracking_env_cfg(play=play)`**

In `goalkeeper_env_cfg()`, directly after the line:
```python
cfg = unitree_g1_flat_tracking_env_cfg(play=play)
```
add:
```python
# Remove motion-reference observations so the policy trains and infers
# from ball + joint state only — no motion file needed at play time.
_motion_obs = ["command", "motion_anchor_pos_b", "motion_anchor_ori_b"]
for _obs in _motion_obs:
    cfg.observations["actor"].terms.pop(_obs, None)
    cfg.observations["critic"].terms.pop(_obs, None)
```

- [ ] **Step 3: Verify the obs list is correct**

Run:
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project && python3 - << 'EOF'
from my_mjlab_project.tasks.goalkeeper_env_cfg import goalkeeper_env_cfg
cfg = goalkeeper_env_cfg(play=False)
print("Actor obs:", list(cfg.observations["actor"].terms.keys()))
print("Critic obs:", list(cfg.observations["critic"].terms.keys()))
EOF
```

Expected actor obs (9 terms, no motion refs):
```
['base_lin_vel', 'base_ang_vel', 'joint_pos', 'joint_vel', 'actions',
 'ball_pos_b', 'ball_vel_b', 'left_hand_pos_b', 'right_hand_pos_b']
```

Expected critic obs (11 terms, adds body_pos + body_ori):
```
['body_pos', 'body_ori', 'base_lin_vel', 'base_ang_vel', 'joint_pos',
 'joint_vel', 'actions', 'ball_pos_b', 'ball_vel_b',
 'left_hand_pos_b', 'right_hand_pos_b']
```

- [ ] **Step 4: Commit**

```bash
git add src/my_mjlab_project/tasks/goalkeeper_env_cfg.py
git commit -m "feat: remove motion-ref obs from training — autonomous play obs space"
```

---

### Task 2: Simplify the play config

**Files:**
- Modify: `src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

`goalkeeper_play_env_cfg()` currently tries to pop the motion-ref obs, which are now already gone from training. It also removes the motion command so the play script doesn't require `--motion-file`.

- [ ] **Step 1: Replace the body of `goalkeeper_play_env_cfg()`**

Find the current `goalkeeper_play_env_cfg()` function and replace its body with:
```python
def goalkeeper_play_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = goalkeeper_env_cfg(play=True)
    cfg.scene.num_envs = 1
    # Remove motion command so play script doesn't require --motion-file.
    # Obs are already clean (motion refs removed in goalkeeper_env_cfg).
    cfg.commands.pop("motion", None)
    return cfg
```

- [ ] **Step 2: Verify play config builds and has no motion command**

```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project && python3 - << 'EOF'
from my_mjlab_project.tasks.goalkeeper_env_cfg import goalkeeper_play_env_cfg
cfg = goalkeeper_play_env_cfg()
print("Commands:", list(cfg.commands.keys()))
print("Num envs:", cfg.scene.num_envs)
print("Actor obs:", list(cfg.observations["actor"].terms.keys()))
EOF
```

Expected:
```
Commands: []
Num envs: 1
Actor obs: ['base_lin_vel', 'base_ang_vel', 'joint_pos', 'joint_vel', 'actions',
            'ball_pos_b', 'ball_vel_b', 'left_hand_pos_b', 'right_hand_pos_b']
```

- [ ] **Step 3: Commit**

```bash
git add src/my_mjlab_project/tasks/goalkeeper_env_cfg.py
git commit -m "feat: simplify play cfg — no motion command, no obs cleanup needed"
```

---

### Task 3: Smoke-test training starts without crash

**Files:** none (validation only)

Verify a 2-environment, 3-iteration training run completes without error.

- [ ] **Step 1: Run smoke test**

```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project && \
  uv run python -m mjlab.scripts.train goalkeeper \
    --env.scene.num-envs 2 \
    --runner.max-iterations 3 \
    --gpu-ids '[0]'
```

Expected: prints iteration 1, 2, 3 with `mean_episode_length` > 1.0, no traceback.

- [ ] **Step 2: Note the run ID from output (format: `YYYY-MM-DD_HH-MM-SS`)**

It will appear in `logs/rsl_rl/g1_goalkeeper/`. Keep this for Task 4.

- [ ] **Step 3: Commit smoke-test result as comment in commands.txt**

```bash
git add commands.txt
git commit -m "docs: update commands.txt with verified smoke-test run"
```

---

### Task 4: Smoke-test play runs without --motion-file

**Files:** none (validation only)

Use the checkpoint from Task 3 smoke test.

- [ ] **Step 1: Find the model_0.pt from the smoke-test run**

```bash
ls /home/isaak/BEPImitationlearning/my_mjlab_project/logs/rsl_rl/g1_goalkeeper/
```

Pick the most recent run folder.

- [ ] **Step 2: Launch play**

```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project && \
  uv run python -m mjlab.scripts.play goalkeeper \
    --checkpoint-file logs/rsl_rl/g1_goalkeeper/<run_id>/model_0.pt
```

Expected: MuJoCo viewer opens, robot stands, ball spawns and approaches, robot attempts to intercept. No crash, no `--motion-file` required.

- [ ] **Step 3: If crash occurs, read traceback and identify which obs term still references the motion command**

If `AssertionError` in `generated_commands` or similar, the obs term is still present. Go back to Task 1 Step 2 and add it to `_motion_obs`.

---

### Task 5: Update commands.txt

**Files:**
- Modify: `commands.txt` at repo root `/home/isaak/BEPImitationlearning/commands.txt`

- [ ] **Step 1: Overwrite commands.txt with the verified command set**

```
# ========== TRAINING COMMANDS ==========
# Obs space: joint_pos/vel, base_lin/ang_vel, ball_pos/vel, hand_pos L+R, actions
# Motion files used for RSI + style reward only — NOT in observations
# Full retrain required after obs-space change (old checkpoints incompatible)

# Full GPU training (1020 envs on RTX 3070 Laptop)
cd /home/isaak/BEPImitationlearning/my_mjlab_project && uv run python -m mjlab.scripts.train goalkeeper --gpu-ids '[0]'

# Quick smoke test (2 envs, 3 iterations)
cd /home/isaak/BEPImitationlearning/my_mjlab_project && uv run python -m mjlab.scripts.train goalkeeper --env.scene.num-envs 2 --runner.max-iterations 3 --gpu-ids '[0]'

# Reduced-env training for faster iteration
cd /home/isaak/BEPImitationlearning/my_mjlab_project && uv run python -m mjlab.scripts.train goalkeeper --env.scene.num-envs 64 --gpu-ids '[0]'

# ========== PLAY COMMANDS ==========
# No --motion-file needed — policy infers autonomously from ball + joint state

cd /home/isaak/BEPImitationlearning/my_mjlab_project && uv run python -m mjlab.scripts.play goalkeeper --checkpoint-file logs/rsl_rl/g1_goalkeeper/<run_id>/model_<iter>.pt

# ========== WHAT WAS FIXED ==========
# BUG 1 (CRITICAL): convert.py
#   - body_pos_w saved worldbody (index 0) instead of robot bodies (index 1+)
#   - RSI placed robot at z=0 instead of z≈1.15m; all envs terminated on step 1
#   FIX: Skip worldbody, save xpos[1:] and xquat[1:], num_bodies=30
#
# BUG 2 (MINOR): commands.py _reset_ball
#   - z_start did not include env_origins[:,2] offset
#   FIX: ball_pos_w z uses origins[:,2] + z_start
#
# BUG 3 (DESIGN): goalkeeper_env_cfg.py
#   - Motion-reference obs in actor forced --motion-file at play time
#   FIX: Removed command/motion_anchor_pos_b/motion_anchor_ori_b from training obs
#        Play cfg removes motion command; no --motion-file needed
#
# TRANSFORM: commands.py
#   - Motion data lowered 0.39m (feet to z=0) + rotated -90° yaw (goalkeeper alignment)
```

- [ ] **Step 2: Commit**

```bash
git add /home/isaak/BEPImitationlearning/commands.txt
git commit -m "docs: update commands.txt — clean play command, bug fix history"
```

---

### Task 6: Document divergence from upstream

**Files:**
- Modify: `../../Imitationlearningbooster/DIVERGENCE_FROM_UPSTREAM.md`

- [ ] **Step 1: Append entry**

Append to `Imitationlearningbooster/DIVERGENCE_FROM_UPSTREAM.md`:

```markdown
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

**Impact:** All checkpoints trained before this date are incompatible with the
new observation space. Full retrain required.
```

- [ ] **Step 2: Commit**

```bash
git add ../../Imitationlearningbooster/DIVERGENCE_FROM_UPSTREAM.md
git commit -m "docs: log obs-space divergence from upstream tracking task"
```

---

### Task 7: Full training run

**Files:** none (execution only)

- [ ] **Step 1: Launch full training**

```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project && \
  uv run python -m mjlab.scripts.train goalkeeper --gpu-ids '[0]'
```

Training runs for 200k iterations by default. Monitor W&B for:
- `mean_episode_length` climbing above 50 steps (robot survives)
- `rew_eereach` increasing (robot reaches toward ball)
- `rew_catch_success` > 0 (robot makes contact)

- [ ] **Step 2: Once a checkpoint looks promising (e.g. model_2000.pt), run play to inspect**

```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project && \
  uv run python -m mjlab.scripts.play goalkeeper \
    --checkpoint-file logs/rsl_rl/g1_goalkeeper/<run_id>/model_2000.pt
```

Expected: robot stands, ball approaches, robot intercepts. No motion file needed.

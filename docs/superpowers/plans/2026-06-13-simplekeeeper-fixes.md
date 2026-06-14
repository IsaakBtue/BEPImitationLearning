# SimpleGoalKeeper Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 12 correctness and completeness issues in SimpleGoalKeeper's reward design, termination logic, observation space, and AMP config.

**Architecture:** All fixes are localised to four files under `SimpleGoalKeeper/src/simple_goalkeeper/` — `mdp/rewards.py`, `mdp/observations.py`, `mdp/events.py`, `tasks/goalkeeper_env_cfg.py`, and `tasks/goalkeeper_amp_cfg.py`. The `mdp/__init__.py` export list is updated once at the end.

**Tech Stack:** Python 3.12, mjlab (MuJoCo-Warp RL), beyondAMP, rsl_rl_amp. All commands run from `/home/isaak/BEPImitationlearning/SimpleGoalKeeper/` with `uv run`.

---

## Sanity-check command (use after every task)

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1
```
Expected: env loads and runs 250 steps without error, then exits.

---

## File map

| File | What changes |
|---|---|
| `src/simple_goalkeeper/mdp/rewards.py` | Add `_ball_is_behind`, gate `footreach`, add `successland`, `penalize_kneeheight`, `dof_vel_limits`, `postupperdofpos`, `postwaistdofpos`; tune `feetorientation`, `dof_vel` weight hints removed (done in cfg) |
| `src/simple_goalkeeper/mdp/observations.py` | Add `base_lin_vel` |
| `src/simple_goalkeeper/mdp/events.py` | Add `ball_exit_termination` |
| `src/simple_goalkeeper/mdp/__init__.py` | Export new symbols |
| `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` | Wire new obs, rewards, termination, curriculum; tune weights |
| `src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py` | Fix `amp_task_reward_lerp` 0.7→0.9 |

---

## Task 1: Add `_ball_is_behind` helper and gate `footreach`

**Files:**
- Modify: `src/simple_goalkeeper/mdp/rewards.py`

**Problem:** `footreach` currently runs forever — even after the ball passes the robot (failed save) the policy still gets rewarded for lateral alignment. The fix: add a `_ball_is_behind` helper (mirrors Imitationlearningbooster exactly) and zero-out `footreach` when it's True.

- [ ] **Step 1: Add `_ball_is_behind` before `footreach` in rewards.py**

Insert after the `_DEFAULT_ROBOT_CFG` line (line 16) and before `def footreach`:

```python
def _ball_is_behind(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Bool mask (N,): ball has passed goal line OR been deflected.

    Mirrors Imitationlearningbooster exactly:
      behind = (ball_x_local < 0) | (delta_vx > 1.0)
    """
    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    ball_x_vel = ball.data.root_link_lin_vel_w[:, 0]
    init_vx = getattr(env, "_sb_init_vx", None)
    if init_vx is not None:
        delta_vx = ball_x_vel - init_vx
        return (ball_x_local < 0.0) | (delta_vx > 1.0)
    return (ball_x_local < 0.0) | (ball_x_vel > 1.0)
```

- [ ] **Step 2: Gate `footreach` return with `~behind`**

In `footreach`, find the final return (currently `return taskrew * upright`) and replace it:

```python
    behind = _ball_is_behind(env, ball_name)
    return taskrew * upright * (~behind).float()
```

- [ ] **Step 3: Run sanity check**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1
```
Expected: runs 250 steps, no error.

- [ ] **Step 4: Commit**

```bash
cd /home/isaak/BEPImitationlearning
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py
git commit -m "fix(goalkeeper): gate footreach reward with _ball_is_behind helper"
```

---

## Task 2: Add ball exit termination

**Files:**
- Modify: `src/simple_goalkeeper/mdp/events.py`
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`

**Problem:** Episodes run to the full 4s timeout even after a failed save (ball past goal) or successful deflection. The fix: terminate when `ball_x_local < -0.5` (ball clearly past goal) or `stopball` has fired AND ball is moving away (deflected).

- [ ] **Step 1: Add `ball_exit_termination` to events.py**

Append at the end of `events.py`:

```python
def ball_exit_termination(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    behind_threshold: float = -0.5,
) -> torch.Tensor:
    """Terminate when ball has clearly passed the goal line or been deflected.

    Fires when:
      - ball_x_local < behind_threshold (ball behind robot by > 0.5 m), OR
      - stopball has fired (deflection registered) AND ball moving in +X (moving away)

    This ends episodes quickly after the outcome is decided, freeing sim time.
    """
    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    ball_x_vel = ball.data.root_link_lin_vel_w[:, 0]

    passed = ball_x_local < behind_threshold
    sb_flag = getattr(env, "_sb_flag", None)
    if sb_flag is not None:
        deflected_away = sb_flag & (ball_x_vel > 0.5)
    else:
        deflected_away = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return (passed | deflected_away).float()
```

- [ ] **Step 2: Wire termination into goalkeeper_env_cfg.py**

In `goalkeeper_env_cfg.py`, in the `cfg.terminations = {...}` block (around line 286), add a new entry:

```python
        "ball_exit": TerminationTermCfg(
            func=gk_mdp.ball_exit_termination,
            params={"ball_name": BALL_NAME, "behind_threshold": -0.5},
            time_out=False,
        ),
```

So the full terminations block becomes:

```python
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mjlab_mdp.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=mjlab_mdp.bad_orientation,
            params={"limit_angle": 1.0, "asset_cfg": _ROBOT_CFG},
        ),
        "base_height": TerminationTermCfg(
            func=mjlab_mdp.root_height_below_minimum,
            params={"minimum_height": 0.4, "asset_cfg": _ROBOT_CFG},
        ),
        "ball_exit": TerminationTermCfg(
            func=gk_mdp.ball_exit_termination,
            params={"ball_name": BALL_NAME, "behind_threshold": -0.5},
            time_out=False,
        ),
    }
```

- [ ] **Step 3: Run sanity check**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1
```
Expected: env loads, episodes reset more frequently (ball exits in ~0.5s at 3-7 m/s), no errors.

- [ ] **Step 4: Commit**

```bash
cd /home/isaak/BEPImitationlearning
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/events.py
git add SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py
git commit -m "fix(goalkeeper): terminate episode when ball exits (passed goal or deflected)"
```

---

## Task 3: Fix `amp_task_reward_lerp` and add `stopball` reward curriculum

**Files:**
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py`
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`

**Problems:**
1. `amp_task_reward_lerp=0.7` means AMP contributes 30% of combined signal — 3× beyondAMP's default of 0.9. The task reward (intercept ball) needs to dominate.
2. `stopball` weight is static at 100.0. Imitationlearningbooster ramps it 100→175→250 as the task gets harder.

- [ ] **Step 1: Fix `amp_task_reward_lerp` in goalkeeper_amp_cfg.py**

In `goalkeeper_amp_cfg.py`, in the `AMPRunnerCfg(...)` call, change:

```python
        amp_task_reward_lerp=0.7,
```
to:
```python
        amp_task_reward_lerp=0.9,
```

- [ ] **Step 2: Add `stopball` reward curriculum in goalkeeper_env_cfg.py**

In the curriculum section (around line 94), add `stopball_curriculum` alongside `ball_difficulty`:

```python
    _num_steps = 24
    cfg.curriculum.clear()
    if not play:
        cfg.curriculum["ball_difficulty"] = CurriculumTermCfg(
            func=gk_mdp.ball_difficulty_curriculum,
            params={
                "stages": [
                    {"step": 0,                    "difficulty": 0.0},
                    {"step": 600  * _num_steps,    "difficulty": 0.5},
                    {"step": 1200 * _num_steps,    "difficulty": 1.0},
                ],
            },
        )
        cfg.curriculum["stopball_curriculum"] = CurriculumTermCfg(
            func=mjlab_mdp.reward_curriculum,
            params={
                "reward_name": "stopball",
                "stages": [
                    {"step": 0,                    "weight": 100.0},
                    {"step": 600  * _num_steps,    "weight": 175.0},
                    {"step": 1200 * _num_steps,    "weight": 250.0},
                ],
            },
        )
```

Note: `mjlab_mdp.reward_curriculum` is already imported via `import mjlab.envs.mdp as mjlab_mdp`. Verify it exists:

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run python -c "import mjlab.envs.mdp as m; print(m.reward_curriculum)"
```
Expected: prints the function object. If missing, remove the `stopball_curriculum` block and note for later.

- [ ] **Step 3: Run sanity check**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
cd /home/isaak/BEPImitationlearning
git add SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py
git add SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py
git commit -m "fix(goalkeeper): amp_task_reward_lerp 0.7→0.9; add stopball reward curriculum 100→175→250"
```

---

## Task 4: Add `base_lin_vel` observation

**Files:**
- Modify: `src/simple_goalkeeper/mdp/observations.py`
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`

**Problem:** The policy can't directly observe its own linear velocity. Without it, `noretreat` penalty is hard to learn because the policy doesn't see what it's being penalised for. Imitationlearningbooster includes `base_lin_vel` in actor observations.

- [ ] **Step 1: Add `base_lin_vel` function to observations.py**

Append at the end of `observations.py`:

```python
def base_lin_vel(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    """Robot base linear velocity in robot body frame. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    return robot.data.root_link_lin_vel_b
```

- [ ] **Step 2: Add to actor and critic obs in goalkeeper_env_cfg.py**

In the `actor_terms` dict (around line 114), add `base_lin_vel` alongside the other terms:

```python
        "base_lin_vel": ObservationTermCfg(
            func=gk_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
```

The critic_terms loop copies from actor_terms automatically, so no separate change needed there.

- [ ] **Step 3: Export from mdp/__init__.py**

In `mdp/__init__.py`, add `base_lin_vel` to the observations import:

```python
from .observations import ball_pos_b, ball_vel_b, left_foot_pos_b, right_foot_pos_b, base_lin_vel
```

- [ ] **Step 4: Run sanity check and verify obs shape changed**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run python -c "
import mjlab.tasks
import simple_goalkeeper.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
cfg = load_env_cfg('Mjlab-BeyondAMP-Goalkeeper-T1')
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
obs, _ = env.reset()
print('actor obs shape:', obs['actor'].shape)
env.close()
"
```
Expected: actor shape `(1, N)` where N is 3 larger than before (was 10×(3+3+21+21+21+3+3+3+3)=10×81=810, now 10×(81+3)=840). The exact number depends on what was there before — the key is it should be 30 (3 dims × 10 history frames) larger.

- [ ] **Step 5: Commit**

```bash
cd /home/isaak/BEPImitationlearning
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/observations.py
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py
git add SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py
git commit -m "feat(goalkeeper): add base_lin_vel to actor/critic observations"
```

---

## Task 5: Add `successland`, `penalize_kneeheight`, `dof_vel_limits` rewards

**Files:**
- Modify: `src/simple_goalkeeper/mdp/rewards.py`
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`

**Problems:**
- `successland` (w=+4.0): Imitationlearningbooster has a dense contact signal for when a foot is close to the ball. `stopball` fires on Δvx only, which needs a clean deflection. `successland` rewards proximity before the deflection, providing a denser gradient.
- `penalize_kneeheight` (w=-100.0): Prevents the robot from kneeling (both shanks drop below floor+15 cm). Without it, the policy can get stuck in kneeling states that would damage real hardware.
- `dof_vel_limits` (w=-2.0): Penalises joint velocities above 10 rad/s (conservative limit below all T1 actuator velocity limits). Prevents degenerate high-speed joint motions.

- [ ] **Step 1: Add `_DEFAULT_KNEE_CFG` and three reward functions to rewards.py**

Near the top of `rewards.py`, after `_DEFAULT_ROBOT_CFG`:

```python
_DEFAULT_KNEE_CFG = SceneEntityCfg("robot", body_names=("Shank_Left", "Shank_Right"))
```

Then append after `deviation_waist_joint`:

```python
def successland(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    contact_th: float = 0.12,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Dense reward for foot proximity to ball before ball is behind.

    Fires when either foot is within contact_th of the ball AND the ball
    is still in front (not yet behind). Provides a denser gradient than
    stopball alone — mirrors Imitationlearningbooster successland (w=4.0).
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    ball_pos_w = ball.data.root_link_pos_w                              # (N, 3)
    dist = torch.norm(foot_pos_w - ball_pos_w[:, None, :], dim=-1)     # (N, 2)
    min_dist = dist.min(dim=-1).values                                   # (N,)
    behind = _ball_is_behind(env, ball_name)
    return (min_dist < contact_th).float() * (~behind).float()


def penalize_kneeheight(
    env: "ManagerBasedRlEnv",
    min_height: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_KNEE_CFG,
) -> torch.Tensor:
    """Penalise shank bodies dropping below min_height above floor.

    Detects kneeling/falling states that would damage real hardware.
    Returns sum of excess below threshold across both shanks.
    """
    robot: Entity = env.scene[asset_cfg.name]
    shank_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    floor_z = env.scene.env_origins[:, 2]                                # (N,)
    shank_z_local = shank_pos_w[:, :, 2] - floor_z[:, None]             # (N, 2)
    violation = torch.clamp(min_height - shank_z_local, min=0.0)        # (N, 2)
    return violation.sum(dim=-1)


def dof_vel_limits(
    env: "ManagerBasedRlEnv",
    vel_threshold: float = 10.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Penalise joint velocities above vel_threshold (rad/s).

    10 rad/s is below all T1 actuator velocity limits (min is arm at ~9.3 rad/s).
    Returns sum of squared excess across all joints.
    """
    robot: Entity = env.scene[asset_cfg.name]
    vel = robot.data.joint_vel[:, asset_cfg.joint_ids]                  # (N, J)
    excess = torch.clamp(vel.abs() - vel_threshold, min=0.0)            # (N, J)
    return excess.pow(2).sum(dim=-1)
```

- [ ] **Step 2: Wire new rewards into goalkeeper_env_cfg.py**

In `goalkeeper_env_cfg.py`, add a new `_KNEE_BODY_CFG` near the existing SceneEntityCfg lines at the top of the function (around line 35):

```python
_KNEE_BODY_CFG = SceneEntityCfg("robot", body_names=("Shank_Left", "Shank_Right"))
```

Then in `cfg.rewards = {...}`, add three new entries after the `footreach` entry:

```python
        "successland": RewardTermCfg(
            func=gk_mdp.successland,
            weight=4.0,
            params={"ball_name": BALL_NAME, "contact_th": 0.12, "asset_cfg": _FEET_CFG},
        ),
        "penalize_kneeheight": RewardTermCfg(
            func=gk_mdp.penalize_kneeheight,
            weight=-100.0,
            params={"min_height": 0.15, "asset_cfg": _KNEE_BODY_CFG},
        ),
        "dof_vel_limits": RewardTermCfg(
            func=gk_mdp.dof_vel_limits,
            weight=-2.0,
            params={"vel_threshold": 10.0, "asset_cfg": _ALL_JOINTS_CFG},
        ),
```

- [ ] **Step 3: Run sanity check**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
cd /home/isaak/BEPImitationlearning
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py
git add SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py
git commit -m "feat(goalkeeper): add successland, penalize_kneeheight, dof_vel_limits rewards"
```

---

## Task 6: Add post-save recovery rewards (`postupperdofpos`, `postwaistdofpos`)

**Files:**
- Modify: `src/simple_goalkeeper/mdp/rewards.py`
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`

**Problem:** After a successful save, the robot has no incentive to recover upright. Imitationlearningbooster uses `postupperdofpos` and `postwaistdofpos` (both w=1.0) to encourage the arms and waist to return to the default pose once the ball is behind.

- [ ] **Step 1: Add `_ARM_JOINT_CFG` constant and two recovery functions to rewards.py**

After `_DEFAULT_KNEE_CFG` (top of rewards.py):

```python
_ARM_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=(
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    ),
)
_WAIST_JOINT_CFG_RECOVERY = SceneEntityCfg("robot", joint_names=("Waist",))
```

Note: the existing `_WAIST_JOINT_CFG` in `goalkeeper_env_cfg.py` is a module-level constant there, not in rewards.py — this new one lives in rewards.py for recovery use.

Then append two functions after `dof_vel_limits`:

```python
def postupperdofpos(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _ARM_JOINT_CFG,
) -> torch.Tensor:
    """Penalise arm joint deviation from default AFTER ball is behind.

    Encourages the robot to recover its arm pose after a save or failed save.
    Active only when _ball_is_behind is True.
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    delta = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    err = torch.sum(torch.square(delta), dim=-1)
    return err * behind.float()


def postwaistdofpos(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _WAIST_JOINT_CFG_RECOVERY,
) -> torch.Tensor:
    """Penalise waist deviation from default AFTER ball is behind.

    Companion to postupperdofpos — encourages trunk recovery after save.
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    delta = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    err = torch.sum(torch.square(delta), dim=-1)
    return err * behind.float()
```

- [ ] **Step 2: Add new `_RECOVERY_ARM_CFG` and `_RECOVERY_WAIST_CFG` to goalkeeper_env_cfg.py**

Near the top `SceneEntityCfg` definitions in `goalkeeper_env_cfg`:

```python
_RECOVERY_ARM_CFG = SceneEntityCfg(
    "robot",
    joint_names=(
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    ),
)
_RECOVERY_WAIST_CFG = SceneEntityCfg("robot", joint_names=("Waist",))
```

In `cfg.rewards`, add after `deviation_waist_joint`:

```python
        "postupperdofpos": RewardTermCfg(
            func=gk_mdp.postupperdofpos,
            weight=-1.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _RECOVERY_ARM_CFG},
        ),
        "postwaistdofpos": RewardTermCfg(
            func=gk_mdp.postwaistdofpos,
            weight=-1.0,
            params={"ball_name": BALL_NAME, "asset_cfg": _RECOVERY_WAIST_CFG},
        ),
```

Note: weight is **negative** because the function returns error magnitude (deviation from default); negative weight makes it a penalty that is minimised when the robot returns to default.

- [ ] **Step 3: Run sanity check**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
cd /home/isaak/BEPImitationlearning
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/rewards.py
git add SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py
git commit -m "feat(goalkeeper): add post-save recovery rewards postupperdofpos + postwaistdofpos"
```

---

## Task 7: Tune weights and add `action_acc_l2`

**Files:**
- Modify: `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py`

**Changes:**
- `feetorientation` weight: 1.5 → 3.0 (match Imitationlearningbooster; flat feet critical for deflection quality)
- `dof_vel` weight: -0.001 → -5e-4 (halve it; over-penalising joint velocity discourages fast saves)
- Add `action_acc_l2` (w=-0.1) alongside existing `action_rate_l2` (second-order jerk penalty; `mjlab_mdp.action_acc_l2` exists in ILB)

- [ ] **Step 1: Verify `action_acc_l2` is available in mjlab_mdp**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run python -c "import mjlab.envs.mdp as m; print(hasattr(m, 'action_acc_l2'))"
```
- If `True`: proceed with adding it.
- If `False`: skip adding `action_acc_l2` (only do the weight changes).

- [ ] **Step 2: Update weights in `cfg.rewards` in goalkeeper_env_cfg.py**

Change the `feetorientation` entry:
```python
        "feetorientation": RewardTermCfg(
            func=gk_mdp.feetorientation,
            weight=3.0,   # was 1.5
            params={"asset_cfg": _FEET_CFG},
        ),
```

Change the `dof_vel` entry:
```python
        "dof_vel": RewardTermCfg(
            func=mjlab_mdp.joint_vel_l2,
            weight=-5e-4,   # was -0.001
            params={"asset_cfg": _ALL_JOINTS_CFG},
        ),
```

If `action_acc_l2` is available, add after `action_rate_l2`:
```python
        "action_acc_l2": RewardTermCfg(
            func=mjlab_mdp.action_acc_l2,
            weight=-0.1,
        ),
```

- [ ] **Step 3: Run sanity check**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
cd /home/isaak/BEPImitationlearning
git add SimpleGoalKeeper/src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py
git commit -m "tune(goalkeeper): feetorientation 1.5→3.0, dof_vel -0.001→-5e-4, add action_acc_l2"
```

---

## Task 8: Update `mdp/__init__.py` exports

**Files:**
- Modify: `src/simple_goalkeeper/mdp/__init__.py`

All new symbols need to be exported so `goalkeeper_env_cfg.py` can reach them via `gk_mdp.<name>`.

- [ ] **Step 1: Replace `mdp/__init__.py` with the full updated exports**

```python
from . import observations, events, rewards, commands
from .observations import ball_pos_b, ball_vel_b, left_foot_pos_b, right_foot_pos_b, base_lin_vel
from .events import (
    reset_ball_local_frame,
    tick_catchstep,
    ball_difficulty_curriculum,
    ball_exit_termination,
)
from .rewards import (
    foot_to_ball, ball_vx_reduction, ball_positive_vx, posture, ang_vel_xy_l2,
    stayonline, noretreat, feetorientation, deviation_waist_joint,
    footreach, stopball,
    successland, penalize_kneeheight, dof_vel_limits,
    postupperdofpos, postwaistdofpos,
)
from .commands import GhostMotionCommand, GhostMotionCommandCfg
```

- [ ] **Step 2: Run sanity check**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1
```
Expected: no import errors, env runs 250 steps cleanly.

- [ ] **Step 3: Verify all reward terms are registered**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run python -c "
import mjlab.tasks
import simple_goalkeeper.tasks
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg('Mjlab-BeyondAMP-Goalkeeper-T1')
print(sorted(cfg.rewards.keys()))
"
```
Expected output contains all of: `footreach, stopball, successland, penalize_kneeheight, dof_vel_limits, stayonline, noretreat, feetorientation, ang_vel_xy, deviation_waist_joint, postupperdofpos, postwaistdofpos, dof_pos_limits, action_rate_l2, dof_vel, ball_positive_vx`.

- [ ] **Step 4: Commit**

```bash
cd /home/isaak/BEPImitationlearning
git add SimpleGoalKeeper/src/simple_goalkeeper/mdp/__init__.py
git commit -m "chore(goalkeeper): update mdp/__init__.py exports for all new reward/obs/event symbols"
```

---

## Task 9: Update DIVERGENCE_FROM_UPSTREAM.md

**Files:**
- Modify: `SimpleGoalKeeper/CLAUDE.md` (training commands section)
- Append: `Imitationlearningbooster/DIVERGENCE_FROM_UPSTREAM.md` (if it exists for SimpleGoalKeeper)

- [ ] **Step 1: Update CLAUDE.md training commands**

The commands section in `SimpleGoalKeeper/CLAUDE.md` already shows the correct commands. Update the "Motion Files" section to reflect the new dataset source:

Find the line:
```
uv run sgk_convert --input-dir /home/isaak/BEPImitationlearning/Motions --output-dir src/simple_goalkeeper/motions/data
```
Replace with:
```
# Convert new motions (run from SimpleGoalKeeper/):
uv run python -c "
from simple_goalkeeper.scripts.pkl_to_npz import main
main(
    input_dir='/home/isaak/BEPImitationlearning/SimpleGoalKeeper/Motions',
    output_dir='src/simple_goalkeeper/motions/data',
    output_fps=50,
    speed_factor=2.0,
)
"
```

- [ ] **Step 2: Final full sanity check with num_envs=64**

```bash
cd /home/isaak/BEPImitationlearning/SimpleGoalKeeper
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 64
```
Expected: 64-env run completes 250 steps, no CUDA errors, no shape mismatches.

- [ ] **Step 3: Commit**

```bash
cd /home/isaak/BEPImitationlearning
git add SimpleGoalKeeper/CLAUDE.md
git commit -m "docs(goalkeeper): update training commands and dataset notes"
```

---

## Summary of all changes

| Fix | File | Type |
|---|---|---|
| `_ball_is_behind` helper | `rewards.py` | Bug fix |
| `footreach` ball-behind gate | `rewards.py` | Bug fix |
| `ball_exit_termination` | `events.py` + env cfg | Feature |
| `amp_task_reward_lerp` 0.7→0.9 | `goalkeeper_amp_cfg.py` | Bug fix |
| `stopball` reward curriculum 100→175→250 | env cfg | Feature |
| `base_lin_vel` observation | `observations.py` + env cfg | Feature |
| `successland` reward (w=+4.0) | `rewards.py` + env cfg | Feature |
| `penalize_kneeheight` (w=-100) | `rewards.py` + env cfg | Safety |
| `dof_vel_limits` (w=-2.0) | `rewards.py` + env cfg | Safety |
| `postupperdofpos` (w=-1.0) | `rewards.py` + env cfg | Feature |
| `postwaistdofpos` (w=-1.0) | `rewards.py` + env cfg | Feature |
| `feetorientation` 1.5→3.0 | env cfg | Tuning |
| `dof_vel` -0.001→-5e-4 | env cfg | Tuning |
| `action_acc_l2` (w=-0.1) | env cfg | Tuning |
| Export new symbols | `mdp/__init__.py` | Plumbing |

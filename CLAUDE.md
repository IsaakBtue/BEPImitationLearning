# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This workspace contains two goalkeeper research tracks:

### `Humanoid-Goalkeeper/` — Original paper implementation (G1, Isaac Gym)
The original InternRobotics Humanoid-Goalkeeper from the research paper. Targets the **Unitree G1** robot using **Isaac Gym** and **HIM-PPO**. Uses **both hands** to catch balls. **Do NOT modify this directory** — it is the frozen upstream reference.

### `Imitationlearningbooster/` — Booster T1 port of the paper (Isaac Gym + AMP)
Adaptation of `Humanoid-Goalkeeper/` to target **Booster Robotics T1** using **Isaac Gym** and **AMP motion priors**. Still uses **hands** for goalkeeping. All changes must be justified against `Humanoid-Goalkeeper/` and documented in `DIVERGENCE_FROM_UPSTREAM.md`.

### `SimpleGoalKeeper/` — Foot-only goalkeeper (MuJoCo-Warp + beyondAMP)
A new, simplified goalkeeper experiment targeting **Booster T1** with **feet only** (no hand catching). Uses **mjlab** (MuJoCo-Warp) and **beyondAMP** instead of Isaac Gym. The goal is to achieve robust foot-based ball interception before adding arm motions. This track diverges intentionally from the paper approach: it uses a simpler single-discriminator AMP, MuJoCo physics, and foot-only rewards.

**Key distinction**: `Humanoid-Goalkeeper/` and `Imitationlearningbooster/` use hands; `SimpleGoalKeeper/` uses feet only. When reading patterns from the upstream paper code, check whether a decision (reward weight, spawn range, observation term) was designed for hands and needs adaptation for feet.

---

## Project Goal

Adapt the **InternRobotics Humanoid-Goalkeeper** pipeline (originally targeting Unitree G1) to target the **Booster Robotics T1** humanoid instead.

## Command Output Rule

**Every time a command is given to the user:**
1. Clear `commands.txt` (overwrite it from scratch).
2. Write the command(s) into `commands.txt` at the repo root (`/home/isaak/BEPImitationlearning/commands.txt`).
3. Also show the command in the chat response.

## SSH / Git Push

SSH key passphrase for this machine: `Isaak`

## Critical Constraints

1. **Do NOT modify `Humanoid-Goalkeeper/`** — treat it as a frozen upstream reference for G1 behavior. Read it for patterns; never edit it.
1b. **Do NOT modify `Imitationlearningbooster/`** — this is the user's active project. Do not change it without explicit permission.
0. **Always check `Humanoid-Goalkeeper/` first** — before adding or changing any reward, spawn range, observation, termination, or training hyperparameter in `Imitationlearningbooster/` or `SimpleGoalKeeper/`, read the corresponding G1 code and verify the decision. G1 is the proven baseline. Divergences must be explicitly justified and documented.
2. **All Booster-specific code goes under `Imitationlearningbooster/`** — wrappers, forks, or vendored subtrees only.
3. **Every change in `Imitationlearningbooster/` must be explicitly justified against `Humanoid-Goalkeeper/`** — before making any design decision (reward weights, observation schema, training hyperparams, reset logic, command ranges), read the corresponding code in `Humanoid-Goalkeeper/` and document whether the change mirrors G1, adapts G1 for T1's different kinematics, or is a known deliberate divergence. If you cannot point to where in `Humanoid-Goalkeeper/` the decision comes from, treat it as a red flag requiring review.
4. **Document every divergence** — append a dated entry to `Imitationlearningbooster/DIVERGENCE_FROM_UPSTREAM.md` for every substantive change.
5. **Log changes inside `Humanoid-Goalkeeper/`** — if upstream context must be noted, append "what" and "why" to `Humanoid-Goalkeeper/changes.md`.
6. **License is CC BY-NC-SA 4.0** — non-commercial research only.
7. **Document every fix immediately** — after every bug fix, reward change, or config change, update `Imitationlearningbooster/DIVERGENCE_FROM_UPSTREAM.md` in the same commit. Include: what changed, why it was wrong, what the correct value is, and what evidence (training data, error messages) confirmed the fix was needed.
8. **Do NOT modify anything inside `booster_deploy/`** — treat it as a frozen upstream deployment framework. Read it for patterns; never edit it. All goalkeeper-specific deployment code lives in `goalkeeper_deploy/`.

## Installation

Requires Python 3.8, conda, and an NVIDIA GPU with Isaac Gym support.

```bash
conda create -n gk python=3.8 && conda activate gk
cd Humanoid-Goalkeeper/isaacgym/python && pip install -e .
cd ../../rsl_rl && pip install -e .
cd ../legged_gym && pip install -e .
pip install -r ../requirements.txt
```

## Running

```bash
# Train (from Humanoid-Goalkeeper/)
python legged_gym/legged_gym/scripts/train.py --exptid=<name>

# Evaluate
python legged_gym/legged_gym/scripts/play.py --exptid=<name>
```

Pretrained weights are in `legged_gym/resources/weight/`.

## Architecture

### Component Overview

| Component | Role |
|---|---|
| `isaacgym/` | NVIDIA GPU-accelerated physics simulator (parallel envs) |
| `legged_gym/` | Environment wrapper; defines observations, rewards, reset logic |
| `rsl_rl/` | HIM-PPO training framework (Hybrid Internal Model PPO) |
| `Boosterversion/` | Booster T1 URDF/MJCF assets + migration plan |
| `Imitationlearningbooster/` | All new Booster-specific code lives here |

### Training Pipeline

`train.py` → `task_registry.py` (registers env + config) → `HimOnPolicyRunner` → `HimPPO` (actor-critic with AMP motion priors) → Isaac Gym parallel envs.

The policy uses **proprioceptive observations** (joint angles/velocities/forces) plus **privileged observations** (ball position, target state) during training only. The AMP module provides adversarial motion priors from reference datasets in `legged_gym/resources/datasets/`.

### Key Files for Booster Migration

- **Config:** create `Imitationlearningbooster/booster_t1_config.py` mirroring `legged_gym/envs/g1/g1_29_config.py`
- **Utils:** create `Imitationlearningbooster/booster_t1_utils.py` (joint indices, AMP dataset loading, observation schema)
- **Assets:** place Booster URDFs in `legged_gym/resources/robots/booster_t1/` (source from `Boosterversion/booster_t1/`)
- **Registry:** register new task without touching existing G1 registration

### Goalkeeper Deployment (`goalkeeper_deploy/`)

Deploys `my_mjlab_project_booster_t1/logs/rsl_rl/g1_goalkeeper/2026-05-23_18-35-15/model_2000.pt`
using the `booster_deploy` framework **without modifying any file in `booster_deploy/`**.

| File | Purpose |
|---|---|
| `goalkeeper_deploy/deploy.py` | Wrapper entry-point; sets sys.path correctly so `tasks/` resolves to `goalkeeper_deploy/tasks/` before `booster_deploy/tasks/` |
| `goalkeeper_deploy/export_model.py` | Converts rsl_rl checkpoint → TorchScript (run once) |
| `goalkeeper_deploy/tasks/goalkeeper/task.py` | `GoalkeeperPolicy` + `GoalkeeperT1ControllerCfg` |
| `goalkeeper_deploy/tasks/goalkeeper/controller.py` | `GoalkeeperMujocoController` (builds scene with ball, overrides `update_state`/`ctrl_step`) |
| `goalkeeper_deploy/tasks/goalkeeper/models/goalkeeper_t1_2000.pt` | Exported TorchScript actor |

**Sim2sim (MuJoCo):**
```bash
cd goalkeeper_deploy
python deploy.py --task goalkeeper_t1 --mujoco
```

**Re-export model (if checkpoint changes):**
```bash
python goalkeeper_deploy/export_model.py
```

**Key design facts:**
- Observation: 87 dims/step × 10 history = 870 input to network
- Joints: 23 DOF in MuJoCo XML order = T1_23DOF_CFG.joint_names order (no remapping)
- Default pose: T1_STANDING_KEYFRAME (bent legs, right arm counterbalance)
- Action scale: per-joint `0.25 * effort / stiffness` (matching training T1_ACTION_SCALE)
- PD gains: match training `BuiltinPositionActuatorCfg` (kp=stiffness, kd≈0.0637*kp)

### Known Upstream Modifications (in `changes.md`)

- `num_envs` reduced 6144 → 1020 for 8 GB GPU memory
- W&B entity is configurable via `cfg.wandb_entity` instead of only env var

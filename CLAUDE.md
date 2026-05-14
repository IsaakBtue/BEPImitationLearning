# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Adapt the **InternRobotics Humanoid-Goalkeeper** pipeline (originally targeting Unitree G1) to target the **Booster Robotics T1** humanoid instead.

## Command Output Rule

**Every time a command is given to the user:**
1. Clear `commands.txt` (overwrite it from scratch).
2. Write the command(s) into `commands.txt` at the repo root (`/home/isaak/BEPImitationlearning/commands.txt`).
3. Also show the command in the chat response.

## Critical Constraints

1. **Do NOT modify `Humanoid-Goalkeeper/`** — treat it as a frozen upstream reference for G1 behavior. Read it for patterns; never edit it.
2. **All Booster-specific code goes under `Imitationlearningbooster/`** — wrappers, forks, or vendored subtrees only.
3. **Document every divergence** — append a dated entry to `Imitationlearningbooster/DIVERGENCE_FROM_UPSTREAM.md` for every substantive change.
4. **Log changes inside `Humanoid-Goalkeeper/`** — if upstream context must be noted, append "what" and "why" to `Humanoid-Goalkeeper/changes.md`.
5. **License is CC BY-NC-SA 4.0** — non-commercial research only.
6. **Document every fix immediately** — after every bug fix, reward change, or config change, update `Imitationlearningbooster/DIVERGENCE_FROM_UPSTREAM.md` in the same commit. Include: what changed, why it was wrong, what the correct value is, and what evidence (training data, error messages) confirmed the fix was needed.

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

### Known Upstream Modifications (in `changes.md`)

- `num_envs` reduced 6144 → 1020 for 8 GB GPU memory
- W&B entity is configurable via `cfg.wandb_entity` instead of only env var

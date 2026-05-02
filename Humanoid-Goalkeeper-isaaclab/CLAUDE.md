# Humanoid-Goalkeeper-isaaclab — Claude Instructions

## RULE #1: Always check the original before changing any code

Before modifying ANY logic, reward, observation, config value, or behaviour in this repo,
you MUST first read the corresponding file in the original upstream project:

```
/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper/legged_gym/legged_gym/envs/
  base/legged_robot.py        ← all env logic (rewards, resets, observations, PD control)
  base/legged_robot_config.py ← base config defaults (sim.dt, physx, PPO params)
  g1/g1_29_config.py          ← G1-specific overrides (gains, reward scales, obs dims)
  g1/g1_utils.py              ← MotionLib, joint mapping, AMP dataset loading
```

The goal of this port is to stay **as close as possible to the original behaviour**.
Any divergence must be justified by an Isaac Lab API constraint — not preference.
If the original does X, this port must also do X (adapted for the new API, not reimagined).

## RULE #2: Read migration guide before any API changes

Always read `context/isaaclab_migration_guide.md` before editing Isaac Lab API calls.
It contains critical mappings, quaternion convention changes, and architectural decisions.

## RULE #3: Document every divergence

Any substantive difference from the original must be noted in `docs/FULL_DOCUMENTATION.md`
under "Things That Changed vs the Original".

---

## Project Goal
Isaac Lab 0.54.3 / Isaac Sim 5.1.0 / rsl_rl 5.0.1 port of the Humanoid-Goalkeeper project (Unitree G1).
- **Source (frozen, read-only):** `/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper/`
- **This folder:** all Isaac Lab-specific code — do NOT touch the source

## Running
```bash
conda activate /home/isaak/miniconda3/envs/env_isaaclab

# Training with GUI (small env count)
python -u scripts/train.py --num_envs=2 --max_iterations=200000

# Training headless (full scale)
python -u scripts/train.py --headless --num_envs=512 --max_iterations=200000

# Evaluate trained policy
python -u scripts/play.py --num_envs=4 --checkpoint=logs/rsl_rl/goalkeeper/YYYY-MM-DD_HH-MM-SS_run_name/model_XXXX.pt

# Smoke test
python -u scripts/test_env.py --headless --num_envs=16 --steps=50
```

Always use `python -u` (unbuffered) so print output isn't hidden behind Isaac Sim logs.

## Deep Documentation

Read these **before** modifying any reward scale, observation, or training parameter:

- `docs/PORT_COMPLETION_SUMMARY.md` — **START HERE** after running smoke test. Documents the 4 critical fixes (gamma, entropy_coef, AMP dims, mass DR), design decisions, structural equivalence checklist, and known pitfalls. Essential reference for understanding what changed and why.
- `docs/ORIGINAL_VS_PORT_COMPARISON.md` — Full line-by-line audit of HIM-PPO original vs standard PPO port.
  Covers: algorithm gaps (AMP, auxiliary tasks, history encoding), reward scale tables with per-step/per-second breakdowns, domain randomization status, root-cause analysis of high rew_dof_vel_limits, and prioritised fix list.
- `docs/FULL_DOCUMENTATION.md` — Port architecture overview and validated smoke-test results
- `context/isaaclab_migration_guide.md` — Isaac Lab API reference (quaternions, joint ordering, contact sensors)

## Key Architecture

| File | Corresponds to original |
|---|---|
| `goalkeeper/goalkeeper_env_cfg.py` | `g1/g1_29_config.py` + `base/legged_robot_config.py` |
| `goalkeeper/goalkeeper_env.py` | `base/legged_robot.py` |
| `goalkeeper/goalkeeper_utils.py` | `g1/g1_utils.py` |
| `goalkeeper/agents/rsl_rl_ppo_cfg.py` | `g1/g1_29_config.py` → `G129CfgPPO` |
| `scripts/train.py` | `legged_gym/scripts/train.py` |

## Critical API Differences (Isaac Lab vs Isaac Gym)

### Quaternions & Math
1. Quaternions are **wxyz** in IsaacLab (was xyzw in IsaacGym)
2. Use `quat_apply_inverse` not `quat_rotate_inverse` (same function, new name)

### Scene & Simulation
3. Joint ordering is **breadth-first** (was depth-first in IsaacGym)
4. `joint_pos_limits` / `joint_vel_limits` (not `joint_limits` / `joint_velocity_limits`)
5. Ball is a `RigidObject` (SphereCfg), not a URDF actor

### Physics & Forces
6. Manual PD control: stiffness=0, damping=0, apply torques via `set_joint_effort_target()`
7. External forces: use `RigidObject.set_external_force_and_torque(forces, torques, is_global=True)` (NOT `permanent_wrench_composer`)

### Observations & Control
8. `_get_observations()` must return `{"policy": ..., "critic": ...}`
9. Contact forces come from `ContactSensor.data.net_forces_w` (not gym tensors)
10. `self.actor_history_buf` = rolling obs history (renamed from `obs_buf` to avoid parent collision)

### RSL-RL Config (rsl_rl 5.0.1)
11. Use `RslRlMLPModelCfg` with separate `actor` and `critic` fields (NOT the deprecated `RslRlPpoActorCriticCfg` / `policy` field)
12. Actor config requires `distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar")` for stochastic output; critic has `distribution_cfg=None` (deterministic)
13. **Required** runner fields: `obs_groups = {"actor": ["policy"], "critic": ["critic"]}` and `empirical_normalization = False`
14. Always call `handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl_rl_lib"))` in train.py AND play.py — it cleans up MISSING deprecated stochastic fields from the model configs before serialization via `to_dict()`

## License
CC BY-NC-SA 4.0 — non-commercial research only

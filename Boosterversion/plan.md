# BoosterT1 Migration Plan

## Goal
Convert this repository from its current G1-specific setup into a working BoosterT1 training and evaluation pipeline, using assets in `Boosterversion/booster_t1/`.

## Current Situation (What We Have)
- Booster assets exist:
  - `Boosterversion/booster_t1/T1_locomotion.urdf`
  - `Boosterversion/booster_t1/T1_serial.urdf`
  - `Boosterversion/booster_t1/T1_locomotion.xml`
  - `Boosterversion/booster_t1/T1_serial.xml`
- The active task path is currently G1-specific in multiple places:
  - Config task class and registration are tied to `g1_29_config.py` and task `"29"`.
  - Core env code imports from `legged_gym/envs/g1/g1_utils.py`.
  - Reward/observation assumptions are tuned for current G1 joints, body names, and motion dataset format.

## Migration Strategy
Use a staged approach:
1. Get BoosterT1 loading in simulation.
2. Make reset/step stable.
3. Align observations/actions/rewards.
4. Train a baseline.
5. Validate and harden.

This avoids a large one-shot rewrite and makes failures easier to isolate.

## Phase 1 - Asset and Kinematic Readiness
### Objective
Ensure BoosterT1 URDF can be loaded cleanly by Isaac Gym and mapped to expected actuator/joint structures.

### Tasks
- Choose canonical robot asset for training (`T1_locomotion.urdf` likely primary).
- Place Booster assets under `legged_gym/resources/robots/booster_t1/` with consistent relative mesh paths.
- Verify URDF references (mesh paths, inertials, collision geometry, joint limits, axis directions).
- Build a joint inventory table:
  - joint name
  - type
  - limit range
  - intended control mode
- Compare Booster joint set against current policy action dimension assumptions.

### Deliverables
- Booster asset folder in expected runtime location.
- Booster joint mapping table (source of truth for downstream config).

## Phase 2 - Create Booster Task Config
### Objective
Introduce a separate Booster configuration without breaking existing G1 training.

### Tasks
- Add new Booster config module (parallel to G1 config), including:
  - asset path
  - `num_actions`, `num_dofs`, observation dimensions
  - PD gains and default joint angles
  - reward scales and domain randomization defaults
  - safe `num_envs` for your GPU
- Register a new task name in env registry (do not overwrite `"29"`).
- Keep G1 path intact to allow side-by-side testing.

### Deliverables
- New Booster task config and registry entry.
- New train/play entry usage documented for Booster task.

## Phase 3 - Decouple G1-Specific Logic
### Objective
Remove hard dependencies on `g1_utils` and G1-only assumptions in base environment path.

### Tasks
- Identify all G1-coupled code paths in `legged_robot.py` and related helpers:
  - joint/body naming assumptions
  - motion dataset loading format
  - ball/goalkeeper-specific state shaping that depends on G1 layout
- Introduce robot/task-specific abstraction points:
  - utility hooks per robot/task
  - configurable body indices and keypoints via config instead of hardcoded names
- Implement Booster-specific utility module mirroring required interfaces.

### Deliverables
- Base env can run with either G1 or Booster-specific utility logic.
- No G1-only import blocking Booster task startup.

## Phase 4 - Observation, Action, and Reward Alignment
### Objective
Make policy I/O consistent with Booster morphology and task goals.

### Tasks
- Recompute observation schema from Booster DOF/joint count.
- Recompute privileged observation schema.
- Align action scaling and clipping to Booster actuator limits.
- Update reward terms using Booster kinematics:
  - posture
  - smoothness
  - ball interception behavior
  - stability and anti-fall
- Recalibrate termination conditions and reset behavior.

### Deliverables
- Booster-consistent obs/action/reward definitions.
- Environment resets and runs rollouts without index or shape errors.

## Phase 5 - Dataset and Motion Prior Compatibility
### Objective
Ensure AMP/motion components are valid for Booster skeleton (or disable safely).

### Tasks
- Audit current dataset assumptions in `g1_utils.py`.
- Decide one of two paths:
  1. **Quick-start path:** temporarily disable AMP dependencies and train task-only PPO baseline.
  2. **Full path:** retarget motion data to Booster joint topology and regenerate mapping files.
- If using full path, create Booster joint mapping file and validate sequence loading.

### Deliverables
- Working training mode (baseline PPO or full AMP).
- No runtime warnings/errors from incompatible motion dimensions.

## Phase 6 - Bring-Up and Debug Loop
### Objective
Reach a stable first training run for BoosterT1.

### Tasks
- Start headless with small env count (e.g., divisible-by-6 if region partition logic still applies).
- Run with debug flags for deterministic error localization:
  - `CUDA_LAUNCH_BLOCKING=1` when needed
- Fix startup blockers in order:
  1. asset load
  2. reset indexing
  3. tensor shape mismatches
  4. reward NaNs / exploding actions
- Validate W&B logging path and checkpoint save/load path.

### Deliverables
- Booster run reaches multiple learning iterations.
- Checkpoints are saved and loadable.

## Phase 7 - Evaluation and Play
### Objective
Enable `play.py` for Booster checkpoints with expected behavior.

### Tasks
- Add Booster task selection support in play flow.
- Validate checkpoint naming and log directory conventions.
- Run visual playback and verify:
  - physically plausible motion
  - no immediate collapse
  - task interaction (ball behavior) matches intent

### Deliverables
- `play.py` works for Booster checkpoint(s).
- Basic qualitative validation completed.

## Risks and Mitigations
- **Risk: hardcoded G1 assumptions in base env cause repeated index errors.**
  - Mitigation: centralize robot-specific indices in config and utility layer.
- **Risk: motion prior is incompatible with Booster skeleton.**
  - Mitigation: first establish PPO-only baseline before retargeting AMP.
- **Risk: GPU memory/runtime instability.**
  - Mitigation: conservative `num_envs`, incremental scaling after stable startup.
- **Risk: URDF mesh/collision issues from exported CAD model.**
  - Mitigation: simplify collisions and verify inertials if simulation is unstable.

## Acceptance Criteria (Definition of Done)
- Booster task is selectable and registered independently.
- Training starts, runs, and saves checkpoints without runtime exceptions.
- W&B logging works for Booster runs.
- `play.py` loads a Booster checkpoint and runs visibly stable episodes.
- G1 workflow still functions (no regression).

## Suggested Execution Order (Practical)
1. Phase 1 + Phase 2 (new Booster config + registration).
2. Phase 3 minimal decoupling needed just to start/reset.
3. Phase 4 tensor and reward alignment.
4. Phase 6 debug loop until stable iterations.
5. Phase 7 play/evaluation.
6. Phase 5 full AMP retargeting as a second milestone.

# IsaacGym → Isaac Lab Migration Decisions Log

Date: 2026-04-25  
Source: `Humanoid-Goalkeeper/legged_gym/` (frozen, Isaac Gym / rsl_rl HIM-PPO)  
Target: `Humanoid-Goalkeeper-isaaclab/` (Isaac Lab 2.2.1 / Isaac Sim 5.0.0)

---

## Architecture Changes

### 1. Base class: `LeggedRobot` → `DirectRLEnv`

**Original:** `LeggedRobot(BaseTask)` in `legged_gym/envs/base/legged_robot.py`  
**New:** `GoalkeeperEnv(DirectRLEnv)` in `goalkeeper/goalkeeper_env.py`

The IsaacGym `BaseTask` used `gym.acquire_*_tensor()` + `gym.refresh_*()` to read state. Isaac Lab's `DirectRLEnv` auto-updates all state at every physics step via `scene.update()`. The method mapping is:

| IsaacGym | Isaac Lab |
|---|---|
| `create_sim()` | `_setup_scene()` |
| `pre_physics_step()` | `_pre_physics_step()` |
| `post_physics_step()` | split into `_get_dones()`, `_get_rewards()`, `_get_observations()` |
| `reset_idx()` | `_reset_idx()` |
| `_apply_action()` | `_apply_action()` (new explicit hook) |

---

### 2. Quaternion Convention: xyzw → wxyz

**Original:** all quaternion operations assumed `[x, y, z, w]`  
**New:** Isaac Lab uses `[w, x, y, z]`

All calls to `quat_rotate_inverse`, `quaternion_to_matrix`, etc. now use the wxyz convention. The `euler_from_quaternion_wxyz()` utility in `goalkeeper_utils.py` was written for this convention.

---

### 3. Joint Ordering: depth-first → breadth-first

**Original:** Isaac Gym ordered joints depth-first in the URDF tree  
**New:** Isaac Lab orders joints breadth-first

This means the index mapping of 29 G1 joints is different. The code uses `_robot.find_joints(name)` to build explicit name→index maps in `_build_joint_maps()`, so it is robust to ordering differences. No hard-coded index arrays carried over.

---

### 4. Robot Asset: URDF actor → `ArticulationCfg` (UrdfFileCfg)

**Original:** `gym.load_asset()` with URDF path  
**New:** `ArticulationCfg(spawn=UrdfFileCfg(...))` — Isaac Lab auto-converts URDF to USD on first run

Key config differences:
- `joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="none", gains=PDGainsCfg(stiffness=0.0, damping=0.0))` — disables internal PD drives in the USD so we can apply manual torques
- `activate_contact_sensors=True` — required for `ContactSensor` to work (IsaacGym had this implicitly)
- `ImplicitActuatorCfg(stiffness=0.0, damping=0.0, effort_limit_sim=400.0)` — runtime actuator with no PD; torques are applied manually via `set_joint_effort_target()`

---

### 5. Ball: URDF sphere actor → `RigidObjectCfg` (SphereCfg)

**Original:** ball was a URDF actor loaded via `gym.load_asset()`  
**New:** `RigidObjectCfg(spawn=sim_utils.SphereCfg(radius=0.1, mass=0.1kg, restitution=0.8))`

No URDF for the ball — Isaac Lab's primitive spawner handles it directly. Ball drag is applied via `self._ball.set_external_force_and_torque(forces, zeros)`.

---

### 6. Contact Forces: gym tensors → `ContactSensor`

**Original:** `gym.refresh_net_contact_force_tensor()` → `net_contact_forces`  
**New:** `ContactSensor(prim_path=".../Robot/.*")` registered in `_setup_scene()`; read as `self._contact_sensor.data.net_forces_w`

Requires `activate_contact_sensors=True` in the robot's spawn config (see §4 above).

---

### 7. PD Control: built-in drives → manual torques

**Original:** `dof_props["stiffness"]` / `dof_props["damping"]` set on the actor  
**New:** stiffness/damping are set to 0 in the USD; manual PD is computed each step:

```python
torques = p_gains * Kp_factors * (target - joint_pos) - d_gains * Kd_factors * joint_vel
torques = clip(torques + actuation_offset + joint_injection, -torque_limits, torque_limits)
self._robot.set_joint_effort_target(torques)
```

---

### 8. Training Framework: HIM-PPO → standard PPO (rsl_rl 3.x)

**Original:** custom `HimOnPolicyRunner` + `HimPPO` with AMP motion priors (adversarial motion prior discriminator loss)  
**New:** standard `OnPolicyRunner` + `PPO` from `rsl_rl 3.x`

**Why:** `rsl_rl 3.x` does not ship `HimPPO`. The AMP discriminator and motion priors are not available out-of-the-box. The motion dataset (`MotionLib`) is still loaded and available in the env (for future re-integration), but the adversarial loss term is not currently connected to the optimizer.

**Impact:** Training may converge more slowly without AMP motion priors. The policy still observes ball position, target region, and rich proprioception — just without the reference motion regularization.

---

### 9. Observation Buffer: `obs_buf` rename

**Original:** single `obs_buf` tensor for actor history  
**New:** renamed to `actor_history_buf` to avoid collision with `DirectRLEnv.obs_buf`

Isaac Lab's `DirectRLEnv` sets `self.obs_buf = self._get_observations()` after every step, which overwrites any instance attribute named `obs_buf`. The internal 10-frame history buffer is now stored as `self.actor_history_buf`.

---

### 10. `PhysxCfg` — removed invalid field

**Original (bug):** `sim_utils.PhysxCfg(solver_type="TGS", ..., max_depenetration_velocity=1.0)`  
**Fixed:** `solver_type` is `Literal[0, 1]` (not a string), and `max_depenetration_velocity` is not a valid field in Isaac Lab 2.2.1's `PhysxCfg`. Removed the invalid field; moved depenetration settings to `RigidBodyPropertiesCfg` on the individual asset.

---

### 11. `_setup_dof_limits()` call order fix

`torque_limits` (from `joint_effort_limits`) was referenced in `_init_buffers()` before `_setup_dof_limits()` was called. Fixed by moving `_setup_dof_limits()` call to before the actuation noise setup block.

---

### 12. `filter_collisions`: `_terrain.prim_path` → `cfg.terrain.prim_path`

Isaac Lab's `TerrainImporter` does not expose a `.prim_path` property. The correct way to get the terrain prim path is from the config: `self.cfg.terrain.prim_path`.

---

### 13. Indexing fix: 2D fancy indexing with env_ids

`self.actuation_offset[env_ids, curriculum_dof_indices]` fails when the two index tensors have different lengths. Fixed with:

```python
self.actuation_offset[env_ids[:, None], self.curriculum_dof_indices] = 0.0
```

This broadcasts `env_ids` as column and `curriculum_dof_indices` as row to produce a `(N_envs, N_joints)` selection.

---

### 14. Deprecated API fixes

| Deprecated | Replacement |
|---|---|
| `joint_limits` | `joint_pos_limits` |
| `joint_velocity_limits` | `joint_vel_limits` |
| `effort_limit=...` | `effort_limit_sim=...` |
| `quat_rotate_inverse` | `quat_apply_inverse` (not yet replaced — functional but deprecated) |

---

### 15. Terrain: `prim_path="/World/ground"` via `TerrainImporterCfg(terrain_type="plane")`

**Original:** Isaac Gym had a built-in plane — `gym.add_ground_plane()`  
**New:** `TerrainImporterCfg(terrain_type="plane")` creates a flat ground plane at the USD path `/World/ground`.

---

## Files Not in the Original

| File | Purpose |
|---|---|
| `goalkeeper/__init__.py` | Registers `Isaac-Goalkeeper-Direct-v0` with `gymnasium.register()` |
| `goalkeeper/agents/rsl_rl_ppo_cfg.py` | `RslRlOnPolicyRunnerCfg` subclass for rsl_rl 3.x |
| `goalkeeper/agents/__init__.py` | Package init |
| `scripts/test_env.py` | Headless smoke test |
| `scripts/train.py` | Training entry point (replaces `legged_gym/scripts/train.py`) |
| `scripts/play.py` | Evaluation entry point |
| `setup.py` | Pip-installable package |
| `context/isaaclab_migration_guide.md` | Migration reference (fetched from official docs) |
| `docs/MIGRATION_DECISIONS.md` | This file |

---

## Known Limitations / Future Work

1. **AMP motion priors not connected**: `MotionLib` loads reference data but the adversarial discriminator loss is not wired into the PPO optimizer. Re-implementing HIM-PPO on top of rsl_rl 3.x would recover this.

2. **Domain randomization — friction/restitution**: The `set_material_properties` API is not yet implemented. Currently friction and restitution randomization is a no-op (the values are sampled but not applied). Isaac Lab provides `write_physics_material_properties_to_sim()` for this.

3. **Observation noise from motion dataset warning**: `load_imitation_dataset` returns a list of tensors when only one file type is present; the MotionLib loading loop expects a dict. Minor refactor needed.

4. **Velocity/torque penalty scales increased (2026-05-02)**: Without AMP motion priors (item 1), the original penalty scales (`rew_dof_vel=-5e-4`, `rew_dof_vel_limits=-2.0`, `rew_torque_limits=-3.0`) are insufficient to prevent unnatural high-velocity motion. After observing `rew_dof_vel_limits≈−1145/s` (≈20 rad/s excess per joint on average) during 6h training runs, scales were increased: `rew_dof_vel→−5e-3` (×10), `rew_dof_vel_limits→−5.0` (×2.5), `rew_torque_limits→−5.0` (×1.67). Restore original values if/when AMP is connected.

5. **AMP motion priors connected (2026-05-02)**: Implemented 6 independent AMP discriminators (one per region/motion type: lefthand, righthand, leftjump, rightjump, leftstep, rightstep) inside `GoalkeeperEnv`. Discriminators are trained using a separate Adam optimizer (`disc_lr=1e-4`) every `disc_update_interval=16` policy steps, using GAIL loss + gradient penalty (λ=5). AMP reward is blended into task reward at `amp_coef=0.4`. This approximates the original HIM-PPO AMP integration but keeps rsl_rl as the PPO backbone (discriminator is trained independently, not end-to-end with policy gradient). Fixed `_load_motion` bug: original .pt files use `joint_position` key, not `dof_pos`/`joint_pos`. Added DFS→BFS joint permutation when building expert observations.

6. **Friction/restitution DR applied (2026-05-02)**: Added `_randomize_materials()` called in `_reset_idx()`. Uses `robot.root_physx_view.set_material_properties()` to apply per-reset friction/restitution randomization. Wrapped in try/except for API compatibility across Isaac Lab versions.

# Observation and Reward Consistency Verification

## Fixed Issue: Distribution Shift Between Training and Play

**Problem:** Training was using tracking observations (robot_body_pos_b, robot_body_ori_b, etc.) that were removed at play time, causing distribution shift.

**Solution:** Remove tracking observations in the base `goalkeeper_env_cfg()` so they're unavailable in all modes consistently.

## Consistency Matrix

### Observations (Actor & Critic)
All three modes now use identical observations:
```
✓ actions
✓ ball_pos_b          (ball position in robot frame)
✓ ball_vel_b          (ball velocity in robot frame)
✓ base_ang_vel        (base angular velocity)
✓ base_lin_vel        (base linear velocity)
✓ joint_pos           (joint positions)
✓ joint_vel           (joint velocities)
✓ left_hand_pos_b     (left hand position in robot frame)
✓ right_hand_pos_b    (right hand position in robot frame)
```

### Rewards by Mode

| Reward | Training | Play | Play+Overlay | Reason |
|--------|----------|------|--------------|--------|
| motion_global_root_pos | ✓ | ✗ | ✓ | Only used during training |
| motion_global_root_ori | ✓ | ✗ | ✓ | Only used during training |
| motion_body_pos | ✓ | ✗ | ✓ | Only used during training |
| motion_body_ori | ✓ | ✗ | ✓ | Only used during training |
| motion_body_lin_vel | ✓ | ✗ | ✓ | Only used during training |
| motion_body_ang_vel | ✓ | ✗ | ✓ | Only used during training |
| eereach | ✓ | ✓ | ✓ | Used for reaching evaluation |
| catch_success | ✓ | ✓ | ✓ | Used for catching evaluation |
| stopball | ✓ | ✓ | ✓ | Primary task objective |
| stayonline | ✓ | ✓ | ✓ | Stability reward |
| noretreat | ✓ | ✓ | ✓ | Stability reward |
| feetorientation | ✓ | ✓ | ✓ | Stability reward |
| postorientation | ✓ | ✓ | ✓ | Stability reward |
| postlinvel | ✓ | ✓ | ✓ | Stability reward |
| postangvel | ✓ | ✓ | ✓ | Stability reward |
| action_rate_l2 | ✓ | ✓ | ✓ | Regularization |
| joint_limit | ✓ | ✓ | ✓ | Joint safety |
| self_collisions | ✓ | ✓ | ✓ | Self-collision penalty |

### Commands by Mode

| Command | Training | Play | Play+Overlay | Reason |
|---------|----------|------|--------------|--------|
| motion | ✓ | ✗ | ✓ | Reference trajectory for tracking (removed in eval-only mode) |

## Impact of Fix

1. **No distribution shift:** Policy learns and evaluates with identical observation set
2. **Correct motion tracking:** Training can focus on motion alignment via explicit rewards
3. **Clean play mode:** Play mode (no overlay) has no unnecessary motion infrastructure
4. **Visualization mode:** Play+overlay preserves motion visualization capability

## Verified Configs

- ✓ `goalkeeper_env_cfg(play=False)` — Training configuration
- ✓ `goalkeeper_play_env_cfg()` — Play without visualization  
- ✓ `goalkeeper_play_withoverlay_env_cfg()` — Play with motion overlay

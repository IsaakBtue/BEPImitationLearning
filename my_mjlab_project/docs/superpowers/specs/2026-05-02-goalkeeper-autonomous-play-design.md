# Goalkeeper RL Pipeline — Autonomous Play Design

**Date:** 2026-05-02  
**Status:** Approved  
**Author:** Isaak Bouwmeester

---

## Goal

Train a single humanoid goalkeeper policy (G1 robot) that stops a soccer ball using natural-looking defensive movements. The 6 reference motion files guide movement style during training only. At inference the policy acts autonomously from ball + joint state alone — no motion file, no W&B run required.

---

## Problem With Current Architecture

The policy was trained with explicit motion-reference observations in the actor input:
- `command` — motion tracking command vector
- `motion_anchor_pos_b` — anchor body position in robot frame
- `motion_anchor_ori_b` — anchor body orientation in robot frame

This makes the motion file a required input at play time. Removing these observations at play time causes a shape mismatch with the trained model weights. A retrain is required.

---

## Architecture

### Observation Space (Training = Play — identical)

| Term | Dim | Description |
|---|---|---|
| `joint_pos` | 29 | Actuated joint positions |
| `joint_vel` | 29 | Actuated joint velocities |
| `base_lin_vel` | 3 | Root linear velocity (body frame) |
| `base_ang_vel` | 3 | Root angular velocity (body frame) |
| `ball_pos_b` | 3 | Ball position in robot base frame |
| `ball_vel_b` | 3 | Ball velocity in robot base frame |
| `left_hand_pos_b` | 3 | Left wrist position in base frame |
| `right_hand_pos_b` | 3 | Right wrist position in base frame |
| `actions` | 29 | Previous frame joint targets |
| **Total actor** | **105** | |

Critic adds privileged terms (`body_pos`, `body_ori`) during training only.

**Removed from actor:** `command`, `motion_anchor_pos_b`, `motion_anchor_ori_b`

### What the 6 Motion Files Are Used For

| Use | Training | Play |
|---|---|---|
| RSI — robot starts in a pose from the motion | Yes | No (starts standing) |
| Ball trajectory coupling — ball aimed at motion zone | Yes | No (ball spawned freely) |
| Motion tracking rewards — style shaping | Yes | No (rewards not needed) |
| Motion reference in observations | **No** | **No** |

The motions shape *how* the robot moves (via reward), not *what it sees*.

### Reward Terms

| Term | Weight | Purpose |
|---|---|---|
| `eereach` | 10.0 | Hand approaches ball |
| `catch_success` | 5.0 | Hand within 0.3m of ball |
| `stopball` | 2.0 | Ball deflected/stopped |
| `stayonline` | −2.0 | Stay on goal line |
| `noretreat` | −2.0 | Don't back away |
| `feetorientation` | 3.0 | Feet stay flat |
| `postorientation` | 3.0 | Upright posture after ball passes |
| `postangvel` | 3.0 | Low angular velocity after pass |
| `postlinvel` | 1.0 | Low forward velocity after pass |
| `motion_body_pos/ori/vel` | inherited | Style shaping from 6 motions |
| `action_rate_l2` | −0.1 | Smooth actions |
| `joint_limit` | −10.0 | No joint limit violations |
| `self_collisions` | −10.0 | No self-collisions |

---

## Files Changed

### `tasks/goalkeeper_env_cfg.py`

1. Remove `command`, `motion_anchor_pos_b`, `motion_anchor_ori_b` from **training** `actor` and `critic` observations.
2. Simplify `goalkeeper_play_env_cfg()` — remove motion command only; observation cleanup is no longer needed since training obs are already clean.
3. No standing-pose reset needed: `sampling_mode="start"` in play already uses frame 0 (standing).

### `mdp/commands.py`

No logic changes required. `MultiMotionCommand` keeps:
- RSI from random motion frame
- Ball trajectory coupling via `_BALL_END_RANGES`
- −90° yaw rotation + 0.39m z-correction (already applied)

### `docs/`

- This spec file
- Updated `DIVERGENCE_FROM_UPSTREAM.md` entry

---

## Play Behaviour

```
Robot initialises standing (frame 0 of random motion clip)
↓
Ball spawns 3–5 m ahead, aimed at a random zone
↓
Policy observes: ball pos/vel + joint state + hand pos
↓
Policy outputs: 29 joint position targets
↓
Robot intercepts ball autonomously
```

### Play Command

```bash
cd my_mjlab_project && uv run python -m mjlab.scripts.play goalkeeper \
    --checkpoint-file logs/rsl_rl/g1_goalkeeper/<run_id>/model_<iter>.pt
```

No `--motion-file` argument required.

---

## Training Command

```bash
cd my_mjlab_project && uv run python -m mjlab.scripts.train goalkeeper --gpu-ids '[0]'
```

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Robot moves unnaturally without reference obs | Motion tracking rewards stay as style signal |
| Train/play gap (RSI in training, standing in play) | `sampling_mode="start"` uses frame 0 ≈ standing |
| Current checkpoints incompatible | Full retrain required — old obs space |
| Motion selection unknown to user at play time | By design: policy infers internally from ball position |

---

## Success Criteria

1. `uv run python -m mjlab.scripts.play goalkeeper --checkpoint-file <path>` runs without `--motion-file`
2. Robot starts in standing position
3. Ball spawns and approaches
4. Robot moves to intercept using natural-looking motion
5. No crashes, no shape mismatch errors

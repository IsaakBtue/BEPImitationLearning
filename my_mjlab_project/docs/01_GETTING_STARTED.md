# Getting Started - Goalkeeper Task in MuJoCo Lab

## Quick Start

### Training Commands

**Smoke test (2 envs, quick validation):**
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project
uv run python -m mjlab.scripts.train goalkeeper --env.scene.num-envs 2 --runner.max-iterations 3 --gpu-ids '[0]'
```

**Full GPU training (1020 envs, recommended):**
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project
uv run python -m mjlab.scripts.train goalkeeper --gpu-ids '[0]'
```

**Reduced-env training (64 envs, faster iteration):**
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project
uv run python -m mjlab.scripts.train goalkeeper --env.scene.num-envs 64 --gpu-ids '[0]'
```

**Training with video recording:**
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project
uv run python -m mjlab.scripts.train goalkeeper --video True --video-interval 500 --video-length 200 --gpu-ids '[0]'
```

### Play Commands

**Autonomous goalkeeper (fully independent, no motion input at play time):**
```bash
cd /home/isaak/BEPImitationlearning/my_mjlab_project
uv run python -m mjlab.scripts.play goalkeeper --checkpoint-file logs/rsl_rl/g1_goalkeeper/2026-05-03_10-47-38/model_200.pt
```

**Controls in viewer:**
- **ENTER:** Reset environment and re-randomize ball trajectory
- **r:** Toggle debug visualization (NOT reset)
- **Mouse drag:** Pan camera
- **Scroll:** Zoom

## Architecture Overview

### Components
| Component | Role |
|-----------|------|
| **Training data** | 6 motion clips (left/right hand, jump, step) in NPZ format |
| **Observations** | Joint angles/velocities + ball position/velocity + hand positions (172-dim) |
| **Rewards** | 18 terms: motion tracking (6), goalkeeper task (9), regularization (3) |
| **Physics** | MuJoCo with bouncy ball (solref=[0.05, 0.0001]) |
| **RL Algorithm** | PPO with 1020 parallel environments |

### Key Design: Autonomous Goalkeeper
- **Training:** Policy learns from all 6 motion types simultaneously (multi-motion imitation learning)
- **Observations:** Ball position/velocity, joint state **only** (no motion type indicator)
- **Play time:** Policy is 100% autonomous—chooses best response based on incoming ball trajectory
- **No motion files needed at inference**

## File Structure

```
src/my_mjlab_project/
├── tasks/
│   ├── goalkeeper_env_cfg.py    ← Environment config (observations, rewards, physics)
│   └── goalkeeper_ppo_cfg.py    ← RL hyperparameters
├── mdp/
│   ├── commands.py              ← MultiMotionCommand (loads all 6 motions)
│   ├── observations.py          ← Ball/hand position observations
│   ├── rewards.py               ← 18 reward functions
│   └── resets.py                ← Autonomous ball reset
└── motions/
    ├── data/                    ← 6 NPZ motion files
    └── convert.py               ← Motion format converter (one-time use)

docs/
├── 01_GETTING_STARTED.md        ← This file
├── 02_ARCHITECTURE.md           ← Full technical architecture
├── 03_BUG_FIXES.md              ← History of bugs and fixes
├── 04_REFERENCE_COMPARISON.md   ← Comparison with Unitree implementation
├── 05_SESSION_2026_05_03.md     ← Session notes (ball physics, autonomous play)
└── FUTURE_ROADMAP.md            ← Future improvements and pitfalls
```

## Observation Space (172-dim)

- **Command** (58-dim): Reference motion trajectory
- **Motion anchor** (6-dim): Reference pose
- **IMU** (6-dim): Base linear + angular velocity
- **Joint state** (58-dim): 29 positions + 29 velocities
- **Ball** (6-dim): Position + velocity in robot frame
- **Hand positions** (6-dim): Left + right wrist positions in robot frame
- **Actions** (29-dim): Previous action

## Reward Structure (18 terms)

### Motion Tracking (6 terms, weights 0.5–1.0)
- motion_global_root_pos, motion_global_root_ori
- motion_body_pos, motion_body_ori
- motion_body_lin_vel, motion_body_ang_vel

### Goalkeeper Task (9 terms, weights -2.0 to 10.0)
- **eereach:** Nearest hand reaches toward ball (+10.0)
- **catch_success:** Hand within 0.3m of ball (+5.0)
- **stopball:** Ball velocity decreases (-2.0 penalty if increasing)
- **stayonline:** Stay between goal lines (-2.0 penalty if wandering)
- **noretreat:** Don't back away from ball (-2.0 penalty if retreating)
- **feetorientation:** Keep feet flat (+3.0)
- **postorientation:** Face the ball (+3.0)
- **postangvel:** Minimize body angular velocity (+3.0)
- **postlinvel:** Minimize retreat velocity (+1.0)

### Regularization (3 terms, weights -0.1 to -10.0)
- action_rate_l2: Penalize jerky movements (-0.1)
- joint_pos_limits: Penalize joint limit violations (-10.0)
- self_collisions: Penalize robot self-collision (-10.0)

## Training Convergence

### Expected Performance
- **Smoke test (2 envs, CPU):** Episodes reach ~7 steps within 3 iterations
- **Full training (1020 envs, GPU):** 
  - Iteration time: ~0.1–0.5 seconds (MuJoCo)
  - Full 200k iterations: ~8–40 hours depending on GPU
  - Mean reward: Improves from -100+ to -10 to 0+ range
  - Episode length: Grows from 7 → 50 → 200+ steps

## Debugging

### Common Issues

**Ball not bouncing?**
- Check `get_ball_spec()` in `goalkeeper_env_cfg.py`
- Current settings: `solref=[0.05, 0.0001]` (stiff, minimal damping)
- To adjust: Increase dampratio (more damping, less bounce)

**Episodes terminating immediately?**
- Check `error_anchor_pos` in W&B metrics
- If >0.25m: body indexing issue (see 03_BUG_FIXES.md)
- Verify motion files are 30 bodies (not 31)

**Slow training?**
- MuJoCo is CPU-bound; GPU doesn't accelerate physics
- Reduce `num_envs` to test faster (e.g., `--env.scene.num-envs 64`)
- On RTX 3070 Laptop: ~11k steps/s with 1020 envs

**Motion files not found?**
- Verify path in `goalkeeper_env_cfg.py` line ~87
- Ensure convert.py has been run: `python src/my_mjlab_project/motions/convert.py`

## Monitoring

### W&B Dashboard
https://wandb.ai/i-p-b-bouwmeester-eindhoven-university-of-technology/mjlab

### Key Metrics
- **mean_episode_length:** Should grow from ~7 toward 200+
- **mean_reward:** Should improve (become less negative)
- **error_anchor_pos:** Should stay < 0.25m (no terminations)
- **episode_reward/eereach, catch_success:** Should improve from ~0

## Next Steps

1. **Run full training** (GPU):
   ```bash
   uv run python -m mjlab.scripts.train goalkeeper --gpu-ids '[0]'
   ```

2. **Monitor** W&B metrics in real-time

3. **Evaluate** after 1k–10k iterations with play script

4. **Tune** reward weights if needed (see 02_ARCHITECTURE.md)

## Resources

- **MuJoCo Docs:** https://mujoco.readthedocs.io/
- **MjLab Docs:** https://mjlab.readthedocs.io/
- **RSL-RL Docs:** https://rsl-rl.readthedocs.io/
- **Original Isaac Gym Port:** `/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper-isaaclab`


# Architecture - Goalkeeper Task in MuJoCo Lab

## System Overview

The goalkeeper task combines:
1. **Multi-Motion Imitation Learning** – Policy trains on 6 diverse motion clips simultaneously
2. **Ball Interaction** – Custom reward terms for catching/blocking soccer ball
3. **Autonomous Play** – At inference, policy receives NO motion input, only ball state

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training (1020 parallel envs)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────┐  ┌──────────────────────┐             │
│  │ Environment Manager  │  │  MultiMotionCommand  │             │
│  ├──────────────────────┤  ├──────────────────────┤             │
│  │ • 1020 G1 robots     │  │ • lefthand.npz       │             │
│  │ • 1020 soccer balls  │  │ • righthand.npz      │             │
│  │ • Flat terrain       │  │ • leftjump.npz       │             │
│  │ • Contact detection  │  │ • rightjump.npz      │             │
│  │                      │  │ • leftstep.npz       │             │
│  │ RL Freq: 50 Hz       │  │ • rightstep.npz      │             │
│  │ Motion Freq: 30 Hz   │  │                      │             │
│  │                      │  │ Samples 1 per reset  │             │
│  └──────────────────────┘  └──────────────────────┘             │
│           ▲                          ▲                           │
│           │                          │                           │
│  ┌────────┴──────────────────────────┴──────────┐               │
│  │  Observation Manager (172-dim per env)        │               │
│  ├──────────────────────────────────────────────┤               │
│  │ • Joint pos/vel (58)      │ • Command (58)    │               │
│  │ • Base vel (6)             │ • Anchor pose (6) │               │
│  │ • Ball pos/vel (6)         │ • Actions (29)    │               │
│  │ • Hand positions (6)       │                   │               │
│  └──────────────────────────────────────────────┘               │
│           ▼                                                       │
│  ┌──────────────────────────────────────────────┐               │
│  │  Reward Manager (18 terms)                    │               │
│  ├──────────────────────────────────────────────┤               │
│  │ Motion tracking (6):  Weight 0.5–1.0         │               │
│  │   • Root/body pos/ori, lin/ang vel           │               │
│  │                                              │               │
│  │ Goalkeeper task (9): Weight -2.0 to +10.0    │               │
│  │   • Reach (toward ball)                      │               │
│  │   • Catch (hand near ball)                   │               │
│  │   • Stop ball (reduce velocity)              │               │
│  │   • Stay on line (goal bounds)               │               │
│  │   • Don't retreat                            │               │
│  │   • Foot/post orientation                    │               │
│  │   • Body angular/linear velocity             │               │
│  │                                              │               │
│  │ Regularization (3): Weight -0.1 to -10.0     │               │
│  │   • Action rate, joint limits, collisions    │               │
│  └──────────────────────────────────────────────┘               │
│           ▼                                                       │
│  ┌──────────────────────────────────────────────┐               │
│  │  PPO Algorithm (RSL-RL)                       │               │
│  ├──────────────────────────────────────────────┤               │
│  │ • Learning rate: 1e-3      │ γ: 0.998         │               │
│  │ • Entropy coef: 0.005      │ Clip: 0.2        │               │
│  │ • Network: 512→256→128, ELU │ Batch: 32768    │               │
│  │ • Gradient steps per iter: 1 │                 │               │
│  └──────────────────────────────────────────────┘               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       Inference (Play Mode)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │  G1 Robot        │  │  Soccer Ball     │                    │
│  │                  │  │                  │                    │
│  │  Observations:   │  │  Reset every 10s │                    │
│  │  • Joint state   │  │  Random start pos│                    │
│  │  • Ball pos/vel  │  │  Random velocity │                    │
│  │  • No motion!    │  │  Compute arc to  │                    │
│  │                  │  │  random endpoint │                    │
│  └──────────────────┘  └──────────────────┘                    │
│           ▲                      ▲                               │
│           │                      │                               │
│           └──────────────────────┘                               │
│                    │                                              │
│                    ▼                                              │
│         ┌──────────────────────┐                                │
│         │  Policy (Trained)    │                                │
│         │  172-dim input       │                                │
│         │  29-dim output       │                                │
│         │  (joint torques)     │                                │
│         └──────────────────────┘                                │
│                                                                   │
│  Key: Policy is 100% AUTONOMOUS                                 │
│        No reference motion, no motion command                    │
│        Learns strategy for ANY incoming ball                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Implementation

### 1. Task Registration

**File:** `src/my_mjlab_project/tasks/__init__.py`

```python
from mjlab import register_mjlab_task
from .goalkeeper_env_cfg import goalkeeper_env_cfg, goalkeeper_play_env_cfg
from .goalkeeper_ppo_cfg import goalkeeper_ppo_runner_cfg
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

register_mjlab_task(
    task_id="goalkeeper",
    env_cfg=goalkeeper_env_cfg(play=False),
    play_env_cfg=goalkeeper_play_env_cfg(),
    rl_cfg=goalkeeper_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
```

### 2. Environment Configuration

**File:** `src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

**Key components:**

#### Ball Physics
```python
def get_ball_spec(radius: float = 0.11, mass: float = 0.42) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="ball_joint")
    geom = body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(radius, 0.0, 0.0),
        mass=mass,
        rgba=(1.0, 1.0, 0.0, 1.0),  # Yellow
    )
    geom.friction = (0.4, 0.005, 0.0001)
    # MuJoCo contact solver parameters for bouncing
    geom.solref = [0.05, 0.0001]  # [timeconst, dampratio]
    geom.solimp = [0.0001, 0.001, 0.0001, 0.5, 2.0]
    geom.margin = 0.001
    geom.gap = 0.0001
    return spec
```

**Why these parameters:**
- `solref[0]=0.05`: Stiff contact (0.05s time constant)
- `solref[1]=0.0001`: Minimal damping (near-zero energy loss)
- Result: Maximum bounciness while remaining stable

#### Multi-Motion Command
```python
cfg.commands["motion"] = MultiMotionCommandCfg(
    entity_name="robot",
    motion_files=(
        "lefthand.npz", "righthand.npz",
        "leftjump.npz", "rightjump.npz",
        "leftstep.npz", "rightstep.npz",
    ),
    ball_name="ball",
    anchor_body_name="torso_link",
    # ... other params
)
```

**Design:**
- All 6 files loaded at init (not CLI-selectable)
- Each reset: randomly sample 1 motion type
- Ball trajectory matched to motion type
- Policy learns all 6 simultaneously (no need to run 6 separate trainings)

#### Observations (Training vs Play)

**Training observations (172-dim):**
- Command (58): Motion trajectory reference
- Motion anchor (6): Reference pose
- Standard tracking obs (IMU, joints, actions)
- Ball obs (6): position + velocity
- Hand obs (6): left + right wrist positions

**Play observations (same 172-dim, but command zeroed):**
- Command (58): **All zeros** (no motion input)
- Ball obs (6): position + velocity
- Policy must learn to respond to ball without explicit motion guidance

### 3. Multi-Motion Loading

**File:** `src/my_mjlab_project/mdp/commands.py`

```python
class MultiMotionCommandCfg(ManagerBasedRLEnvCfg):
    """Load multiple motion files, sample one per reset."""
    
    def __init__(self, motion_files, ...):
        self.motion_files = motion_files  # Tuple of 6 NPZ paths
        self.motion_loaders = {}
        # At init, load all 6 motion clips
        for f in motion_files:
            loader = MotionLoader(f)
            self.motion_loaders[f] = loader
    
    def __call__(self, env):
        # Per reset: sample motion type
        motion_idx = torch.randint(0, len(self.motion_files), (env.num_envs,))
        selected_loaders = [self.motion_loaders[self.motion_files[i]] for i in motion_idx]
        # Spawn ball and set reference pose accordingly
        # ...
```

### 4. Reward Structure

**File:** `src/my_mjlab_project/mdp/rewards.py`

All 18 reward functions follow the pattern:

```python
def eereach(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _HAND_CFG,
    reach_th: float = 0.3,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Sigmoid reward: nearest hand reaches ball (smooth distance-based reward)."""
    ball = env.scene[ball_name]
    robot = env.scene[asset_cfg.name]
    
    # Distance from each hand to ball
    hand_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids]  # (N, 2, 3)
    ball_pos_w = ball.data.root_link_pos_w  # (N, 3)
    
    # Min distance to either hand
    dists = torch.norm(hand_pos_w - ball_pos_w.unsqueeze(1), dim=2)  # (N, 2)
    min_dist = torch.min(dists, dim=1).values  # (N,)
    
    # Sigmoid reward: max at reach_th, decays away
    reward = 1.0 / (1.0 + (min_dist / reach_th) ** sigma)
    return reward
```

### 5. Observation Space

**Actor observations (policy input, 172-dim):**
1. command (58): Reference motion trajectory
2. motion_anchor_pos_b (3): Reference anchor position
3. motion_anchor_ori_b (6): Reference anchor orientation (6D rotation repr)
4. base_lin_vel (3): Robot linear velocity (IMU)
5. base_ang_vel (3): Robot angular velocity (IMU)
6. joint_pos (29): Joint angles
7. joint_vel (29): Joint velocities
8. actions (29): Previous action
9. ball_pos_b (3): Ball position in robot frame
10. ball_vel_b (3): Ball velocity in robot frame
11. left_hand_pos_b (3): Left wrist position in robot frame
12. right_hand_pos_b (3): Right wrist position in robot frame

**Critic observations (298-dim, includes privileged state):**
- All actor terms (172)
- Plus: body_pos (42): All 14 body positions
- Plus: body_ori (84): All 14 body orientations (6D)

### 6. Motion Data Format

**File:** `src/my_mjlab_project/motions/data/*.npz`

Each NPZ contains:
```python
{
    'joint_pos': (N, 29),           # Joint angles
    'joint_vel': (N, 29),           # Joint velocities
    'body_pos_w': (N, 30, 3),       # 30 body positions (skip worldbody)
    'body_quat_w': (N, 30, 4),      # 30 body orientations (wxyz)
    'body_lin_vel_w': (N, 30, 3),   # 30 body linear velocities
    'body_ang_vel_w': (N, 30, 3),   # 30 body angular velocities
}
```

**Why 30 bodies (not 31)?**
- MuJoCo model has 31 bodies (0=worldbody, 1-30=robot)
- mjlab's entity.body_names is 0-indexed (skips worldbody)
- NPZ data must skip worldbody to align indices
- See 03_BUG_FIXES.md for details

### 7. Play-Mode Configuration

**File:** `src/my_mjlab_project/tasks/goalkeeper_env_cfg.py`

```python
def goalkeeper_play_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = goalkeeper_env_cfg(play=True)
    cfg.scene.num_envs = 1
    
    # Auto-reset every 10 seconds for new ball trajectory
    cfg.auto_reset = True
    cfg.episode_length_s = 10.0
    
    # Remove motion command (policy is autonomous)
    cfg.commands.pop("motion", None)
    
    # Add autonomous ball reset event
    cfg.events["reset_ball_autonomous"] = EventTermCfg(
        func=gk_resets.reset_ball_autonomous,
        mode="reset",  # Run every episode reset
        params={"ball_name": "ball"},
    )
    
    # Remove motion-dependent obs/rewards/terms
    # (see file for full list)
```

## Training Hyperparameters

**File:** `src/my_mjlab_project/tasks/goalkeeper_ppo_cfg.py`

```python
class GoalkeeperPPORunnerCfg(RslRlPpoActorCriticCfg):
    # Learning
    learning_rate = 1e-3
    num_learning_epochs = 4
    num_mini_batches = 4
    gamma = 0.998               # Discount factor (long horizon)
    gae_lambda = 0.95           # GAE discount
    entropy_coef = 0.005        # Exploration bonus
    clip_param = 0.2            # PPO clip range
    
    # Network
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation_fn = "elu"
    
    # Flags
    normalize_observations = True
    normalize_advantage = True
    use_running_mean_std = True
```

## Differences from Original (Isaac Gym)

| Aspect | Isaac Gym | MuJoCo Lab |
|--------|-----------|-----------|
| Physics engine | PhysX | MuJoCo |
| Environment count | 6144 (GPU-native) | 1020 (configurable) |
| Motion handling | Single file per run | All 6 files simultaneously |
| Motion reference format | `.pt` (PyTorch) | `.npz` (NumPy) |
| Joint ordering | Breadth-first | Depth-first (MuJoCo native) |
| Ball bounciness | restitution=0.8 | solref=[0.05, 0.0001] |
| Play mode | Requires motion file | Fully autonomous |
| Deployment | ONNX export | Python policy |

## Performance Characteristics

### Training
- **Iteration time (1020 envs, CPU):** 5–10 seconds
- **Iteration time (1020 envs, GPU):** 0.1–0.5 seconds
- **Full training (200k iterations, GPU):** 8–40 hours
- **Memory per env:** ~100 MB
- **Total memory (1020 envs):** ~100 GB (recommend 4× RAM = 24 GB+ GPU)

### Inference
- **Single env step:** <10 ms
- **Policy latency:** <1 ms
- **Ball reset latency:** <5 ms

## Known Limitations

1. **Motion files hardcoded** – Cannot change at CLI, must edit config
2. **No ONNX export** – Python policy only (vs Isaac Lab reference with ONNX)
3. **MuJoCo is CPU-bound** – GPU doesn't accelerate physics, only RL
4. **Contact handling** – May need tuning for different ball/robot sizes

## Future Improvements

See FUTURE_ROADMAP.md for planned enhancements and architectural decisions.


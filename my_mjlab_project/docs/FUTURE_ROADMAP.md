# Future Roadmap & Pitfalls

## 🎯 Next Major Features

### 1. Skill Chooser Module (Proposed)

**Goal:** Let the policy explicitly choose which motion type (skill) to execute, rather than learning implicitly across all 6.

**Current behavior:**
- Policy trains on all 6 motions simultaneously
- Learns a single policy that handles all 6 implicitly
- No explicit "which skill am I executing?" feedback

**Proposed behavior:**
```
Input: (ball_pos, ball_vel, joint_state)
    ↓
    [Policy Network]
    ↓ (splits into two streams)
    ├─ Skill Selector: outputs 6-dim distribution (softmax)
    │  → Predicts which motion type best handles this ball
    │  → Can be interpreted/logged for analysis
    │
    └─ Control Output: outputs 29-dim action
       → Uses selected skill's reference trajectory as prior
       → Conditionally gates/biases based on skill_id
```

**Benefits:**
- Interpretability: Can see which skill policy chooses for each situation
- Curriculum learning: Train skill selector separately from control
- Transfer learning: Reuse skill selector for new robots
- Better generalization: Explicitly learning skill associations

**Implementation approach:**
1. Add skill_selector head to PPO actor network
2. Add skill embedding to reward calculations
3. Log skill choice distribution to W&B for analysis
4. Gate control network: `action = base_action + skill_embedding[skill_id]`

**Pitfall:** If skill selection is not aligned with actual trajectory quality, policy may "hack" by always choosing one skill. Solution: Add auxiliary loss that encourages diverse skill usage.

---

### 2. AMP-Style Skill Structure (Alternative Architecture)

**Goal:** Replace multi-motion training with adversarial motion priors (AMP), letting the policy learn more naturally.

**Current behavior (Motion Imitation):**
- Reference trajectories from data are hard constraints
- Policy penalized for deviating from reference motion
- Limited to pre-recorded motion variety

**Proposed behavior (AMP-like):**
```
Training Loop:
┌──────────────────────────────────────────────────────┐
│                                                      │
│  RL Policy learns goalkeeper task                   │
│  ├─ Direct objective: Catch ball                    │
│  └─ No explicit motion tracking                     │
│                                                      │
│  Parallel: Motion Discriminator                     │
│  ├─ Learns to distinguish policy motion from data  │
│  └─ Provides reward bonus for data-like motion      │
│                                                      │
│  Result: Policy discovers realistic motion styles  │
│  naturally while optimizing for task               │
│                                                      │
└──────────────────────────────────────────────────────┘
```

**Benefits:**
- More natural motion emergence (no "unnatural" optimization)
- Better generalization (learns principles, not rote trajectories)
- Smaller motion file requirement (discriminator learns features)
- Easier to extend to new behaviors

**Challenges:**
- Requires training discriminator in parallel
- Balancing task reward vs. motion reward is non-trivial
- Original Humanoid-Goalkeeper used AMP+HIM-PPO; porting to MuJoCo Lab is complex
- Would need to rewrite RL loop substantially

**Research reference:** 
- Peng et al. 2021: "AMP: Adversarial Motion Priming for Legged Robot Learning"
- Zhang et al. 2022: "Learning Human-like Running on Natural Terrain"

**Estimated effort:** 2–4 weeks of focused development

---

## 🚨 Known Pitfalls & How to Avoid Them

### Pitfall 1: Contact Parameter Interactions

**Problem:** MuJoCo contact behavior depends on BOTH objects, not one.
- Ball solref good but ground solref bad → no bounce
- Friction values interact with contact stiffness
- Contact margin/gap must be calibrated together

**Solution:**
- When tuning ball physics, profile ground contact too
- Test in isolation first (drop ball on floor, verify bounce)
- If bounciness suddenly stops, check for ground property changes
- Document any changes to contact handling

**Prevention:**
```python
# Always test physics in a minimal scenario
def test_ball_bouncing():
    """Unit test for ball physics before training."""
    env = GoalkeeperEnv(num_envs=1)
    ball_init_h = 2.0
    for _ in range(100):
        _, _, terminated, _ = env.step(zero_actions)
        if terminated:
            break
    # If ball bounces > 1m high, contact params are reasonable
```

---

### Pitfall 2: Multi-Motion Synchronization Issues

**Problem:** Training on 6 motions simultaneously means:
- Each motion has different phase/speed
- Reward weights may favor certain motions
- Some motions might be "easier" to learn

**Example:** Jump motion might be naturally higher-reward (big movements catch ball) vs. subtle hand deflection from step motion.

**Solution:**
- Monitor reward contribution by motion type (log per-motion rewards)
- Balance reward scales if some motions dominate
- Consider curriculum: Start with single motion, gradually add others
- Check W&B for convergence per-motion

**Prevention:**
```python
# Track per-motion statistics
class MotionMetrics:
    def __init__(self):
        self.rewards_by_motion = defaultdict(list)
    
    def log(self, motion_id, reward):
        self.rewards_by_motion[motion_id].append(reward)
    
    def summary(self):
        for mid, rewards in self.rewards_by_motion.items():
            print(f"Motion {mid}: avg_reward={np.mean(rewards):.3f}")
```

---

### Pitfall 3: Autonomous Play Coupling to Training Config

**Problem:** Play config is manually created by removing motion-dependent code. If training config changes, play config might break.

**Example:** Add new reward term that uses `cmd.anchor_pos_w` → play config crashes because motion command doesn't exist.

**Solution:**
- Add automated checks: `assert "motion" in cfg.commands or play_mode==True`
- Keep track of ALL motion dependencies in a config
- Consider making motion command optional rather than removing it
- Add CI test: "Can play config be created without errors?"

**Prevention:**
```python
def validate_play_config(cfg):
    """Check that play config has no hidden motion dependencies."""
    forbidden_terms = [
        "motion_global_root_pos", "motion_body_pos",  # Rewards
        "anchor_pos", "anchor_ori",  # Terminations
        "command", "motion_anchor_pos_b",  # Observations
    ]
    
    for term in forbidden_terms:
        if term in cfg.rewards:
            raise ValueError(f"Play config has motion-dependent reward: {term}")
        if term in cfg.observations["actor"].terms:
            raise ValueError(f"Play config has motion-dependent obs: {term}")
```

---

### Pitfall 4: Ball Spawn Timing Race Conditions

**Problem:** Ball reset happens at episode reset, but if physics update hasn't finished, ball might spawn inside robot.

**Example:** 
- Step 1: episode resets
- Ball spawn position written
- Physics step hasn't run yet
- Ball overlaps with robot → undefined behavior

**Solution:**
- Call `env.physics.reset()` BEFORE setting ball pose
- Or spawn ball at high position (z=5m) to guarantee clearance
- Check for penetration errors in first few steps

**Prevention:**
```python
def reset_ball_autonomous(env, env_ids, ball_name="ball"):
    """Reset with safety checks."""
    ball = env.scene[ball_name]
    
    # Ensure physics is stable before spawning
    for _ in range(10):
        env.physics_step()  # Settle any collisions
    
    # Spawn high to guarantee clearance
    z_safe = 2.0 + sample_uniform(...)
    
    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    
    # Verify no penetration
    for _ in range(5):
        env.physics_step()
        # Log if ball fell through floor (negative z)
```

---

### Pitfall 5: Reward Weight Saturation

**Problem:** With 18 reward terms, some naturally dwarf others.

**Example:**
- motion_body_pos (weight=1.0, always active) → avg 0.5 per step
- catch_success (weight=5.0, rare event) → avg 0.01 per step
- Total reward dominated by tracking, goalkeeper task invisible

**Solution:**
- Normalize rewards before summing (unit variance)
- Use curriculum: Start with high motion weight, gradually increase task weight
- Log individual reward terms separately in W&B
- Consider reward clipping: `reward = tanh(reward / scale)`

**Prevention:**
```python
# Track reward contribution histogram
def analyze_reward_balance(runner):
    """Print which rewards dominate."""
    total_reward = 0
    for term_name, term_weight in runner.cfg.rewards.items():
        avg_term = wandb.api.run(run_id).history(step=-1)[f"reward/{term_name}"]
        contribution = avg_term * term_weight / total_reward
        print(f"{term_name:20s} {contribution*100:5.1f}%")
```

---

### Pitfall 6: Observation Normalization Drift

**Problem:** Training observations come from running policy, but in play mode observations come from random ball trajectories.

**Example:**
- Training: Policy generates "natural" joint velocities (normalized to mean=0, std=1)
- Play: Ball throws robot around → joint velocities 10× larger → completely outside trained distribution
- Policy outputs garbage

**Solution:**
- Use wider observation bounds: `obs_min/max = percentile(training_obs, [0.1, 99.9])`
- Test play mode on low-reward trajectories first (gentle ball throws)
- Monitor observation norms: If they spike, something is wrong
- Consider observation clipping: `obs = torch.clamp(obs, obs_min, obs_max)`

**Prevention:**
```python
# Log observation statistics during training
class ObsMonitor:
    def __init__(self, expected_obs_size):
        self.mean = torch.zeros(expected_obs_size)
        self.std = torch.ones(expected_obs_size)
    
    def update(self, obs_batch):
        self.mean = 0.99 * self.mean + 0.01 * obs_batch.mean(0)
        self.std = 0.99 * self.std + 0.01 * obs_batch.std(0)
    
    def log_to_wandb(self):
        wandb.log({
            "obs_mean": wandb.Histogram(self.mean),
            "obs_std": wandb.Histogram(self.std),
        })
```

---

### Pitfall 7: Motion File Conversion Gotchas

**Problem:** Converting from Isaac Gym `.pt` to MjLab `.npz` format has subtle bugs.

**Known issues:**
- Joint ordering differs (BFS vs DFS)
- Body count: 31 (with worldbody) vs 30 (without)
- Quaternion convention: xyzw vs wxyz
- Frame rate interpolation needed

**Solution:**
- Verify converted motion file by visual inspection:
  ```bash
  python -c "
  import numpy as np
  data = np.load('lefthand.npz')
  print('Keys:', data.keys())
  print('Shapes:', {k: data[k].shape for k in data})
  print('Joint range:', data['joint_pos'].min(), data['joint_pos'].max())
  "
  ```
- Cross-check first frame with original Isaac Gym data
- Visual sim: Replay converted motion in MuJoCo viewer

**Prevention:**
```python
def validate_motion_file(npz_path):
    """Check motion file integrity."""
    data = np.load(npz_path)
    
    # Verify structure
    assert 'joint_pos' in data
    assert 'body_pos_w' in data
    
    # Verify shapes
    n_frames = data['joint_pos'].shape[0]
    assert data['joint_pos'].shape == (n_frames, 29)
    assert data['body_pos_w'].shape == (n_frames, 30, 3)  # NOT 31!
    
    # Verify ranges
    assert data['joint_pos'].min() > -2 and data['joint_pos'].max() < 2
    assert data['body_pos_w'][:, 0, 2].min() > 0.5  # Pelvis above ground
    
    print(f"✓ {npz_path} is valid")
```

---

## 📋 Improvement Checklist

- [ ] Implement skill chooser module (optional auxiliary output)
- [ ] Add per-motion reward logging to W&B
- [ ] Automate play config validation
- [ ] Add physics unit tests (ball bounce height)
- [ ] Monitor observation statistics during training
- [ ] Document all contact parameter choices
- [ ] Test on heterogeneous terrain (non-flat)
- [ ] Measure generalization to unseen ball speeds
- [ ] Implement curriculum learning (single motion → 6 motions)
- [ ] Add ONNX export wrapper for deployment

---

## 🔮 Long-Term Vision

### Phase 1: Current (Done)
✅ Multi-motion imitation learning with 6 motion types
✅ Autonomous goalkeeper (no motion input at play time)
✅ Bouncy ball physics tuned for visibility

### Phase 2: Next (3–6 months)
- [ ] Skill chooser with interpretability
- [ ] Curriculum learning: 1 motion → 6 motions → new motions
- [ ] Real-time ball trajectory prediction
- [ ] Evaluate on hardware (Unitree G1)

### Phase 3: Research (6–12 months)
- [ ] AMP-style motion prior (no hard motion tracking)
- [ ] Heterogeneous terrain (slopes, sand)
- [ ] Multi-robot coordination (2 goalkeepers)
- [ ] Sim-to-real transfer learning

### Phase 4: Production (12+ months)
- [ ] Deployed model (ONNX/TensorRT)
- [ ] Real-time inference on robot
- [ ] Continuous learning from failures
- [ ] Integration with motion planning stack

---

## 📚 Research References

1. **Multi-Motion Learning:**
   - Peng et al. 2022: "Learning Agile and Dynamic Motor Skills for Legged Robots"
   - Chiu et al. 2023: "Universality in a Single Module: Learning to Teach with One Network"

2. **Skill Learning & Discovery:**
   - Eysenbach et al. 2019: "Diversity is All You Need" (unsupervised skill discovery)
   - Zhang et al. 2020: "Learning Transferable Representations for Unsupervised Adaptation"

3. **AMP (Adversarial Motion Priors):**
   - Peng et al. 2021: "AMP: Adversarial Motion Priming for Legged Robot Learning"
   - Chiu et al. 2023: "Emergence of Locomotion Behaviours in Rich Environments"

4. **MuJoCo & Contact Dynamics:**
   - Todorov et al. 2012: "MuJoCo: A Physics Engine for Model-Based Control"
   - Freeman et al. 2021: "Brax – A Differentiable Physics Engine for JAX"

---

## 🎓 Key Metrics to Track

As you progress through future work, monitor these:

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Mean episode length | 7–50 steps | 100–200 steps | More robust policy |
| Mean reward | -10 to -5 | 0 to +10 | Better task performance |
| Catch success rate | ~5% | 20–30% | Primary task metric |
| Motion tracking error | <0.2m | <0.1m | Better imitation |
| Inference latency | <10ms | <5ms | Real-time capability |
| Sim-to-real gap | TBD | <10% | Hardware transfer |

---

## 🔗 Related Documents

- **01_GETTING_STARTED.md** – Commands and quick reference
- **02_ARCHITECTURE.md** – Implementation details
- **03_BUG_FIXES.md** – Known issues and solutions
- **04_REFERENCE_COMPARISON.md** – Comparison with Unitree reference
- **05_SESSION_2026_05_03.md** – Latest session notes


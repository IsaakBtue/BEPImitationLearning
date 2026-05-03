# Documentation Index - Goalkeeper Task in MuJoCo Lab

Welcome to the documentation for the autonomous goalkeeper project. This folder contains comprehensive guides for understanding, running, and extending the system.

## 📖 Reading Order (Recommended)

### For Quick Start (5–10 minutes)
1. **[01_GETTING_STARTED.md](01_GETTING_STARTED.md)** – Commands, quick start, basic troubleshooting

### For Understanding the System (30–45 minutes)
2. **[02_ARCHITECTURE.md](02_ARCHITECTURE.md)** – Complete technical architecture, components, design decisions

### For Debugging & Learning from Mistakes (15–20 minutes)
3. **[03_BUG_FIXES.md](03_BUG_FIXES.md)** – History of bugs, root causes, and solutions (educational)

### For Comparisons & Context (15 minutes)
4. **[04_REFERENCE_COMPARISON.md](04_REFERENCE_COMPARISON.md)** – How we compare to the Unitree reference implementation

### For Latest Work & Future Plans (20–30 minutes)
5. **[05_SESSION_2026_05_03.md](05_SESSION_2026_05_03.md)** – What was done in the most recent session
6. **[FUTURE_ROADMAP.md](FUTURE_ROADMAP.md)** – Future features, pitfalls, and long-term vision

---

## 🎯 Quick Navigation by Goal

### "I want to train the model right now"
→ Go to [01_GETTING_STARTED.md](01_GETTING_STARTED.md) and run the training command

### "I want to understand how this works"
→ Read [02_ARCHITECTURE.md](02_ARCHITECTURE.md) for full technical details

### "Something is broken, I need to fix it"
→ Check [03_BUG_FIXES.md](03_BUG_FIXES.md) for known issues and their solutions

### "I want to compare with the original implementation"
→ See [04_REFERENCE_COMPARISON.md](04_REFERENCE_COMPARISON.md)

### "What was done recently? What should I work on next?"
→ Read [05_SESSION_2026_05_03.md](05_SESSION_2026_05_03.md) and [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md)

### "I'm looking for a specific detail"
→ Use Ctrl+F to search across all files, or check the table of contents in each document

---

## 📄 File Summaries

| File | Lines | Purpose | Best For |
|------|-------|---------|----------|
| **01_GETTING_STARTED.md** | 412 | Quick start commands, architecture overview, debugging | New users, quick reference |
| **02_ARCHITECTURE.md** | 590+ | Complete system design, implementation details, code walkthrough | Deep understanding, troubleshooting |
| **03_BUG_FIXES.md** | 400+ | Bug history, root causes, solutions, lessons learned | Learning from mistakes, debugging |
| **04_REFERENCE_COMPARISON.md** | 342 | Comparison with Unitree RL MjLab implementation | Context, design rationale |
| **05_SESSION_2026_05_03.md** | 350+ | Latest session work, ball physics fixes, doc reorganization | Recent changes, current status |
| **FUTURE_ROADMAP.md** | 450+ | Skill chooser, AMP, pitfalls, long-term vision | Planning, research directions |

---

## 🔑 Key Concepts You Should Know

### Autonomous Goalkeeper
- **Training:** Policy learns from 6 motion types simultaneously
- **Inference:** Policy receives ZERO motion input—chooses response based only on ball trajectory
- **Goal:** Catch/block incoming soccer ball using 29 DOF humanoid robot

### Multi-Motion Imitation Learning
- Load 6 motion clips (left/right hand, jump, step) via `MultiMotionCommandCfg`
- Each reset: randomly sample 1 motion type
- Policy learns all simultaneously (vs. training 6 separate policies)

### Key Files
```
src/my_mjlab_project/
├── tasks/
│   ├── goalkeeper_env_cfg.py    ← Environment config
│   └── goalkeeper_ppo_cfg.py    ← RL hyperparameters
├── mdp/
│   ├── commands.py              ← Motion command (loads 6 motions)
│   ├── observations.py          ← Ball/hand observations
│   ├── rewards.py               ← 18 reward functions
│   └── resets.py                ← Ball reset logic
└── motions/
    └── data/                    ← 6 motion clips (NPZ format)
```

### Physics Tuning
- **Ball bounciness:** `solref=[0.05, 0.0001]` (stiff, minimal damping)
- **Contact detection:** `margin=0.001, gap=0.0001`
- **Color:** Yellow (1.0, 1.0, 0.0, 1.0) for visibility

---

## ⚠️ Common Pitfalls (See FUTURE_ROADMAP.md for Details)

1. **Contact parameter interactions** – Ball bounciness depends on BOTH ball AND ground
2. **Multi-motion imbalance** – Some motions naturally higher-reward
3. **Play config coupling** – Play config breaks if training config changes
4. **Ball spawn timing** – Race conditions between physics update and ball reset
5. **Reward weight saturation** – Some rewards dwarf others
6. **Observation drift** – Play mode observations outside training distribution
7. **Motion file conversion** – Subtle bugs in format conversion

Each pitfall has a solution—see [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) for details.

---

## 📊 Monitoring & Metrics

### W&B Dashboard
https://wandb.ai/i-p-b-bouwmeester-eindhoven-university-of-technology/mjlab

### Key Metrics
- **mean_episode_length** – Should grow from 7 → 200+ steps
- **mean_reward** – Should improve (become less negative)
- **error_anchor_pos** – Should stay < 0.25m (no terminations)
- **episode_reward/eereach** – Hand reaching toward ball
- **episode_reward/catch_success** – Hand within 0.3m of ball

---

## 🚀 Quick Commands

```bash
# Navigate to project
cd /home/isaak/BEPImitationlearning/my_mjlab_project

# Train (full scale, 1020 envs)
uv run python -m mjlab.scripts.train goalkeeper --gpu-ids '[0]'

# Train (small scale, 2 envs, for testing)
uv run python -m mjlab.scripts.train goalkeeper --env.scene.num-envs 2 --runner.max-iterations 3 --gpu-ids '[0]'

# Play (evaluate trained policy)
uv run python -m mjlab.scripts.play goalkeeper --checkpoint-file logs/rsl_rl/g1_goalkeeper/2026-05-03_10-47-38/model_200.pt

# Check GPU
nvidia-smi
```

---

## 🔗 External References

- **MuJoCo Docs:** https://mujoco.readthedocs.io/
- **MjLab Docs:** https://mjlab.readthedocs.io/
- **RSL-RL Docs:** https://rsl-rl.readthedocs.io/
- **Original Isaac Gym:** `/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper`
- **Unitree Reference:** `/home/isaak/BEPImitationlearning/unitree_rl_mjlab`

---

## 🎓 Learning Path

If this is your first time:
1. Read [01_GETTING_STARTED.md](01_GETTING_STARTED.md) (10 min)
2. Run the smoke test command (2 min)
3. Read [02_ARCHITECTURE.md](02_ARCHITECTURE.md) sections 1–3 (15 min)
4. Run full training and monitor W&B (ongoing)
5. Read remaining [02_ARCHITECTURE.md](02_ARCHITECTURE.md) sections (15 min)
6. Browse [03_BUG_FIXES.md](03_BUG_FIXES.md) for insights (10 min)
7. Check [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) for next steps (10 min)

Total time investment: ~1 hour to full understanding

---

## 📞 Questions?

1. **Performance issue?** → Check [01_GETTING_STARTED.md#Debugging](01_GETTING_STARTED.md) or [03_BUG_FIXES.md](03_BUG_FIXES.md)
2. **Architecture question?** → See [02_ARCHITECTURE.md](02_ARCHITECTURE.md)
3. **Comparison with original?** → Read [04_REFERENCE_COMPARISON.md](04_REFERENCE_COMPARISON.md)
4. **Next steps?** → Check [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md)
5. **Want to understand a bug?** → Browse [03_BUG_FIXES.md](03_BUG_FIXES.md) for educational value

---

## 📝 Document Maintenance

Last updated: **2026-05-03**

All documents are kept in sync. When making changes:
1. Update the relevant `.md` file
2. Update this README's file summary table if structure changes
3. Update [05_SESSION_2026_05_03.md](05_SESSION_2026_05_03.md) with session notes
4. Add new findings to [03_BUG_FIXES.md](03_BUG_FIXES.md) or [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) as applicable

---

**Happy coding! 🚀**


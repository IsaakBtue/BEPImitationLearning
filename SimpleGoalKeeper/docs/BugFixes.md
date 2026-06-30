
---

## 2026-06-30 — Ghost overlay spawns in ground for DoubleStep/TripleStep motions

**What changed:** Added `body_pos_w` property override in `GhostMotionCommand` (`mdp/commands.py`) that adds +0.030 m to the Z coordinate of all ghost body positions.

**Why it was wrong:** `pkl_to_npz` stores NPZ `body_pos_w` with the foot capsule contact surface at Z = -0.030 m (`_CONTACT_Z_TARGET = -0.030`). The RSI system compensates at runtime by adding `_FOOT_CONTACT_BELOW_BODY = +0.030` to the root Z in `MotionResetManager._write_rsi_state()`. The ghost overlay (`GhostMotionCommand`) replays positions directly from the NPZ via the inherited `MotionCommand.body_pos_w` property without applying this offset, so the ghost robot spawned 0.030 m into the floor.

**Correct value:** Ghost `body_pos_w` = NPZ positions + env_origins + (0, 0, +0.030). Matches what RSI writes to sim.

**Evidence:** All 4 overlay clips (LeftDoubleStep, RightDoubleStep, LeftTripleStep, RightTripleStep) visually embedded in the floor.

---

## 2026-06-30 — Ghost overlay in ground (v2): baked Z offset into NPZ files

**What changed:**
- Added +0.030 m to `body_pos_w[:, :, 2]` in all 14 NPZ motion files in `src/simple_goalkeeper/motions/data/`
- Removed runtime `_FOOT_CONTACT_BELOW_BODY = 0.030` addition from `_write_rsi_state` in `events.py`
- Changed `_CONTACT_Z_TARGET` in `pkl_to_npz.py` from `-0.030` to `0.0` so future conversions output correct Z directly
- Reverted ghost command override in `commands.py` (previous fix was wrong approach)

**Why it was wrong:** The original fix tried to offset positions at runtime inside `GhostMotionCommand.body_pos_w`. The correct fix is to store the correct floor-relative Z in the NPZ files themselves, so all consumers (ghost overlay, RSI, AMP discriminator) see consistent data without any runtime patches.

**Evidence:** All 4 step motions (LeftDoubleStep, RightDoubleStep, LeftTripleStep, RightTripleStep) spawning in the floor during overlay visualization.

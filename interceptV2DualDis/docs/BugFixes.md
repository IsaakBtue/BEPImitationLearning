
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

---

## 2026-07-05 — Ported SimpleGoalKeeper's post-save ball-visibility gate v2

**What changed:** `ball_pos_xy_b` (`mdp/observations.py`) gains `hide_behind_torso`, `hide_after_steps`, and `noise_scale` params. The actor's `ball_pos_b` term (`goalkeeper_env_cfg.py`) now sets `hide_behind_torso=True`, `hide_after_steps=75`, `noise_scale=0.05`, and the manager-level `Unoise` on that term is removed (noise now applied inside the term, before the mask). Play mode explicitly zeroes `noise_scale` since `enable_corruption=False` only disables manager-level noise.

**Why it was wrong (in SimpleGoalKeeper, the source of this port):** The actor's ball observation was fully visible for the entire episode with no post-save release, so the policy never learned to stop reacting once the ball left play — it kept tracking/chasing the ball after a save. An earlier attempt (`hide_when_behind`, a latched-flag gate) destabilized a training run (17% of iterations diverged). This project (interceptV2DualDis) never had a post-save release at all — its own `always_visible=True` came from reverting a *different, stricter* mistake (`8803b6e`: hiding the ball for the whole episode via G1's train-time vanish gate, not just post-save).

**Correct value:** Full visibility during the approach and save (ball visible until `x_body < 0.05` behind the torso, or until episode step 75 — sized for this project's 0.7–1.3 s flight-time range and 3 s episodes, identical to SimpleGoalKeeper's). Neither condition can fire while the ball is still en route.

**Evidence:** Not yet validated against a training run in this project (ported from SimpleGoalKeeper's `ce69f36`, which required and got a fresh run there). Next checkpoint here should be checked in play for post-save tracking/chasing behavior.

---

## 2026-07-05 — Ported SimpleGoalKeeper's `cleanstop` threshold tightening

**What changed:** `cleanstop`'s `speed_threshold` param (`goalkeeper_env_cfg.py`) lowered from `0.25` to `0.10` m/s.

**Why it was wrong:** After a deflection the ball is typically sliding, and translational friction during the slide-to-roll transition bleeds speed on its own — at 0.25 m/s this decay alone could cross the threshold with no genuine foot-trap, i.e. `cleanstop` could fire from friction rather than a real save.

**Correct value:** `speed_threshold=0.10`. Harder for pure friction decay to clear within the post-save window, still reachable by an actual foot-trap. Mechanism otherwise unchanged (one-shot bonus, weight 25.0, requires `softstop` already fired + correct-foot contact at that moment).

**Evidence:** Ported from SimpleGoalKeeper (`f974942`), not yet independently validated against a training run in this project.

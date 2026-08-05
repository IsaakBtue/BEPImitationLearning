---
name: editing-amp-motion-data
description: Use when creating, viewing, or hand-correcting AMP reference motion NPZ clips (motions/data/*.npz) for this project -- e.g. a swing leg looks over-extended, a foot looks angled/not flat, joint angles need damping toward neutral, or any other pose adjustment to an existing clip. Also covers which viewer actually shows the real scene vs. a bare skeleton.
---

# Editing AMP Motion Data (NPZ Reference Clips)

## Overview

This project's AMP reference clips (`src/simple_goalkeeper/motions/data/*.npz`)
are hand-corrected on top of raw mocap, not just raw conversion output. Editing
one joint group (e.g. Hip/Knee_Pitch) silently breaks an invariant a DIFFERENT
joint group was tuned to satisfy (e.g. ankle flatness), because they're all
links in the same kinematic chain. The core lesson from this skill: **passing
NaN/floor-penetration/velocity-spike checks does NOT mean the edit is correct**
-- those checks say nothing about body ORIENTATION, and orientation is exactly
what a downstream joint like the ankle is responsible for.

## When to Use

- User reports a limb/foot "looks angled," "not flat," "pointing the wrong
  way," or "weird" in the real mjlab viewer, for a specific reference clip.
- You're about to damp/scale/offset ANY joint angle in an existing NPZ
  (e.g. "make the swing leg straighter," "reduce that arm swing") -- read
  the Kinematic Chain Rule below BEFORE editing, not after the user reports
  a second problem.
- Creating a brand-new corrected clip from raw PKL, or hand-patching an
  existing NPZ in place.
- A foot/leg needs to be moved CLOSER TO a target position (not just
  flatness) -- e.g. "the trailing foot sticks out too far forward" -- see
  "Correcting Foot Position" below, a different and harder class of edit
  than pure orientation fixes.

**Don't use for:** reward-function bugs (`penalize_wrong_foot_ball_contact`
etc. -- see `debugging-mujoco-contact-sensors`), or training/curriculum
issues unrelated to the reference clip data itself.

## NPZ Format Recap

21-DOF headless T1 joint order (matches `t1_headless.xml` joint declaration
order, head joints excluded):
```
0 Left_Shoulder_Pitch   4 Right_Shoulder_Pitch   8  Waist
1 Left_Shoulder_Roll    5 Right_Shoulder_Roll     9  Left_Hip_Pitch
2 Left_Elbow_Pitch      6 Right_Elbow_Pitch       10 Left_Hip_Roll
3 Left_Elbow_Yaw        7 Right_Elbow_Yaw         11 Left_Hip_Yaw
                                                    12 Left_Knee_Pitch
                                                    13 Left_Ankle_Pitch
                                                    14 Left_Ankle_Roll
                                                    15-20 mirror 9-14 for Right_*
```
Body order in `body_pos_w`/`body_quat_w` (index 0 = world skipped, so index 0
= Trunk/root): fetch live via
`mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)` for `i in range(model.nbody)`,
offset by -1 -- **don't hardcode an assumed order**, verify against
`t1_headless.xml` every time (same discipline as
`debugging-mujoco-contact-sensors`'s "never assume order" rule -- it applies
here too, not just to `SceneEntityCfg.body_ids`).

Arrays: `fps` (scalar), `joint_pos`/`joint_vel` `(T,21)`, `body_pos_w`/
`body_quat_w` `(T,B,3/4 wxyz)`, `body_lin_vel_w`/`body_ang_vel_w` `(T,B,3)`.
Foot-contact convention in this project's files: the **foot body's Z sits at
exactly 0.0300 m** when grounded (not 0.0) -- this is the capsule-radius
offset baked in by `pkl_to_npz.py`'s pass-1 z-correction, not a bug.

## The Kinematic Chain Rule (the actual lesson this skill exists for)

**Any edit to a joint changes the WORLD orientation/position of every body
downstream of it in the kinematic chain, even if you don't touch the
downstream joints' own values.** Concretely in this project:

`Trunk -> Waist -> Hip_Pitch/Roll/Yaw -> Knee_Pitch -> Ankle_Pitch/Roll -> foot`

Damping `Hip_Pitch`/`Knee_Pitch` (to make a swing leg "straighter") leaves
`Ankle_Pitch`/`Ankle_Roll`'s own VALUES untouched, but the foot's WORLD
orientation still changes, because it's the composition of the whole chain.
If the original mocap had the ankle specifically tuned to keep the foot flat
against a *different* hip/knee trajectory (it did -- see below), editing
hip/knee without re-solving the ankle silently reintroduces exactly the tilt
the original leveling pass removed.

**This is not hypothetical -- it happened in this project on 2026-07-31.**
Swing-leg hip/knee damping (see `docs/BugFixes.md`) passed every existing
sanity check (no NaN, no floor penetration, no velocity spikes) and still
broke foot flatness, because none of those checks look at orientation. The
user had to notice it visually a second time ("the feet are angled again")
before it was caught.

### Detecting a flatness violation

Same metric `rewards.py:feetorientation` uses -- gravity vector expressed in
the foot's own local frame; `(0, 0)` XY components means perfectly flat:

```python
def tilt_xy(quat_wxyz):
    w, x, y, z = quat_wxyz
    q_inv = np.array([w, -x, -y, -z])
    g = _quat_rot(q_inv, np.array([0.0, 0.0, -1.0]))  # full quat-rotate, see template's _quat_rot
    return g[:2]
```
(`tilt_xy`/`_quat_rot` are defined exactly this way in `flatten_and_damp_template.py`.)

**Always measure this on the TRUE ORIGINAL clip first.** In this project the
original mocap's leveling pass made every frame, both feet, the ENTIRE clip
read `|tilt| ~ 0.000` (machine precision) -- not just at footstrike. That
told us the correction target was "exactly flat everywhere a foot exists in
the chain," not some looser approximation.

### Fixing it: per-frame Newton solve against real FK, not an assumed formula

At the T1 standing neutral pose, `Hip_Pitch(-0.3) + Knee_Pitch(+0.6) +
Ankle_Pitch(-0.3) = 0` -- tempting to treat this as "the pitch angles just
sum to zero for a flat foot" and hand-correct `Ankle_Pitch` by the negative
of whatever delta you added to Hip/Knee. **Don't rely on this as the actual
fix** -- `Hip_Roll`/`Hip_Yaw` are usually nonzero too, and rotations about
different axes don't commute, so the true relationship has cross-coupling
the simple sum misses. Solve numerically instead, per frame:

1. FK the foot's current world quat with the joint angles as they stand
   after your edit (`mj_kinematics`, not `mj_forward` -- no physics needed).
2. Measure `tilt_xy`. If already ~0, skip (this makes the fix a no-op on
   stance frames and any frame you didn't otherwise touch).
3. Build a 2x2 Jacobian via two finite-difference FK evals (perturb
   `Ankle_Pitch` by `+1e-3`, then `Ankle_Roll` by `+1e-3`, re-measure tilt
   each time).
4. Solve `J @ delta = -tilt` and apply, clipped to the joint's real range
   (`model.jnt_range` -- fetch live, don't hardcode; T1's ankle pitch/roll
   ranges are asymmetric, e.g. `[-0.87, 0.35]` for pitch).
5. Iterate ~3-4 times (converges fast -- it's nearly linear in practice).

This is cheap: a few dozen frames x ~2 feet x 4 Newton iters x 3 FK calls
each is a fraction of a second. See `flatten_and_damp_template.py` (same
directory) for a full worked implementation, which also folds in the
stance/swing damping pattern below.

## Editing an Existing NPZ In Place (vs. re-running the raw-PKL pipeline)

This project's established pattern (confirmed by prior sessions, see
`docs/BugFixes.md` 2026-07-18/07-20 entries) is to **patch the already-built
NPZ directly** with a one-off script when only specific joints/frames need
correction, rather than re-deriving from `motions/raw/*.pkl` through
`pkl_to_npz.py` -- the raw pipeline doesn't reproduce prior hand-leveling
passes (there is no committed script that redoes the original "leveled,
ankle-flatness, hip-straightened" pass from `pkl_to_npz.py` alone; it was
done once by an earlier session and never captured as a reusable tool before
this skill).

Steps, in order:

1. **Recover the TRUE original** if `motions/data/` itself has already been
   overwritten by a previous correction pass (it usually has, if you're
   iterating on a damping strength): `git log --oneline -- <path>` to find
   the commit before your edits, then `git show <commit>~1:<path> > original.npz`.
   **Never re-read from `motions/data/` as your source when re-tuning a
   parameter (e.g. going from 0.5 to 0.65 to 0.8 damping)** -- that stacks a
   nonlinear correction on top of itself instead of applying one clean pass.
2. Detect **stance vs. swing** per leg from the clip's OWN foot-height
   trajectory in `body_pos_w` (`z > ~0.032` = airborne in this project's
   0.0300-grounded convention). **Never edit a stance-phase frame's leg
   joints** -- the root trajectory is being kept fixed, so changing a
   planted leg's hip/knee moves the foot out of its recorded ground contact
   via FK, with nothing to compensate.
3. Blend the edit in/out over a couple of frames at swing/stance
   transitions (box-filter the boolean mask) so the edit doesn't create a
   step discontinuity that shows up as a finite-difference velocity spike.
4. Apply the actual pose edit (e.g. damp toward `HOME_KEYFRAME` neutral:
   `new = neutral + (1-damping)*(old-neutral)`).
5. **Run the flat-foot Newton solve (above) on every frame** -- not just the
   ones you edited, since it's a correct no-op elsewhere and cheap enough
   not to bother special-casing.
6. Recompute `joint_vel` via central differences (`pkl_to_npz.py`'s
   `_finite_diff_vel`, same formula).
7. Refit `body_pos_w`/`body_quat_w` via `mj_kinematics` from the ORIGINAL,
   untouched root trajectory (position + quaternion, body index 0) plus the
   new `joint_pos` -- exactly mirrors `pkl_to_npz.py`'s own Pass 2. Then
   recompute `body_lin_vel_w`/`body_ang_vel_w` (finite-diff position,
   quaternion-log-map angular velocity).
8. **Re-level**: a joint edit near a stance/swing boundary can shift the
   stance foot's contact height by a couple mm even though you didn't
   directly edit that frame (the blend ramp still touches it slightly).
   Compute the new minimum foot-contact Z across both feet, shift the
   WHOLE clip's `body_pos_w` Z column by a single constant so the minimum
   returns to exactly `0.0300` -- this is a pure rigid translation, safe to
   apply directly to already-computed positions without re-running FK, and
   is the same idea as `pkl_to_npz.py`'s own pass-1 z-correction.
9. Sanity-check ALL of: no NaN, foot-contact-Z minimum restored, joint
   velocity did not spike, **AND `tilt_xy` is ~0 on every frame for both
   feet** -- the last check is the one this skill exists to remind you not
   to skip.

## Correcting Foot Position (not just orientation) -- IK is fragile, prefer damping

The flatness fix above only ever solves 2 DOF (`Ankle_Pitch`/`Ankle_Roll`)
for an ORIENTATION target -- that Newton solve is well-behaved because
ankle rotation barely moves the foot's position, so there's no real
multi-solution ambiguity. **Solving `Hip_Pitch`+`Knee_Pitch` for a
POSITION target (e.g. "pull this foot's X closer to the trunk") is a
different, much more fragile problem**: it's a classic 2-link IK setup
with two valid solution branches ("knee bent one way vs. the other"), and
a per-frame-independent Newton solve can jump between branches frame to
frame even though every individual frame reports full convergence. This
happened repeatedly correcting the far-region double-step clips' trailing
foot (2026-08-05, `docs/BugFixes.md`) -- five different mitigation
attempts each surfaced a new failure mode:

- **Exact-target IK** (drive rel-X to 0, or to a damped fraction of
  original): converges per-frame, but joint angles can snap ~15-20deg for
  a single frame relative to a smooth trend either side -- a real velocity
  discontinuity, not a false alarm.
- **Warm-starting each frame from the previous frame's solved values**
  (the standard fix for branch-jumping): made it WORSE here, oscillating
  chaotically instead of jumping once -- the required correction routes
  the knee near a singular (straight-leg) configuration for several
  consecutive frames, where the linearization is ill-conditioned
  regardless of starting point.
- **Revert a spiking frame to its original value**: creates a NEW seam,
  since original and corrected trajectories don't generally coincide at
  that boundary.
- **Interpolate a spiking frame from its corrected neighbors**: leaves a
  whack-a-mole residual -- fixing one frame in a tight cluster just makes
  an adjacent frame the new outlier.

**What actually worked:** damp `Hip_Pitch`/`Knee_Pitch` toward T1's
standing-neutral pose (`-0.3`/`0.6`) by a constant fraction -- NOT solving
IK at all, just linear interpolation of an already-smooth signal, which
cannot itself produce a discontinuity. If a position TARGET is still
needed (not just "move it somewhen closer"), use the damped pose's own
resulting position as the IK target (proven smooth) while forcing Z back
to the exact original height every frame (see next paragraph for why), then
clean up any remaining spike with a **uniform moving-average smooth over
the WHOLE clip's hip/knee columns**, iterated until velocity drops at or
below the original clip's own max -- a moving average of a bounded signal
has a hard bound on its own derivative, so it can't miss a case the way a
targeted single-frame patch can.

**Different legs need different damping strengths, and more damping is
NOT always better.** The trailing foot's large offset needs strong
damping; the leading foot's small offset gets pushed the WRONG way
(overshoots past the target) by the same strong damping. Grid-search each
leg's own damping factor independently against its own foot's resulting
`max|trunk-relative-X|` -- don't assume a single shared factor is optimal
for both.

**Always verify ground-contact TIMING survived, not just flatness/NaN/
velocity.** A damping-only pass (no explicit Z target) passed every
existing check -- including zero velocity spike -- and still silently
swapped which foot touches the ground when: the trailing foot's genuine
footfall disappeared (min height rose ~8mm, well past the 0.033m grounded
threshold) while the leading foot gained a footfall that never existed in
the original capture. Caught only by explicitly diffing the boolean
grounded mask (`z<0.033`, frame-by-frame) old vs. new -- a metric this
skill's own flatness-focused checklist did not previously mention. Add it:
**compare `body_pos_w[:,foot_idx,2] < 0.033` old vs. new and confirm no
systematic shift**, not just that the minimum is still near 0.030
somewhere in the clip.

## Viewing Clips: Use the Real Viewer, Not the Bare One

`sgk_view <path.npz>` uses a bare `mujoco.viewer.launch_passive` call --
skeleton only, no ground plane, no ball, no lighting context. **This is the
wrong tool for judging whether a clip looks natural** -- a user explicitly
flagged it as "wrong visualiser" in this project's history.

Use the real mjlab viewer via the `-WithOverlay` task variant instead, with
`--agent zero --no-terminations True` so the policy doesn't drive the robot
and the ghost overlay just plays the reference clip back in the actual
scene:

```bash
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc-WithOverlay \
  --agent zero --no-terminations True \
  --motion-file src/simple_goalkeeper/motions/data/<clip>.npz
```

When comparing an edit against the original, point two separate invocations
at the two files (there's no side-by-side mode) -- keep the true original
around at a scratch path per the recovery step above so both commands stay
runnable throughout an iteration cycle.

**Where to keep the "old" copy for a live comparison session:** a sibling
directory of `motions/data/` (e.g. `motions/data_pre_<fixname>_backup/`),
NOT a path outside the repo -- every motion loader in this project globs
non-recursively (`_MOTIONS_DIR.glob("*.npz")`), so a sibling directory is
never picked up by training/AMP/the diagnostic RSI script, but both `old`
and `new` commands stay one relative-path edit apart, which matters when
you're iterating through several rounds of "try a fix, compare, try
again" in the same session (2026-08-05's far-region double-step fix went
through 5+ rounds). Delete the backup dir once the user has confirmed the
fix (it's scratch, never commit it) -- or recover the true original from
git history (`git show <commit>:<path>`) if the backup was already
cleaned up and you need to re-compare.

**Programmatic comparison, not just eyeballing the viewer:** for a whole-
body position/trajectory question (root drift, foot excursion, "does this
loop look right"), a top-down (bird's-eye) matplotlib plot of `world X`
vs. `world Y` for the trunk and both feet -- old and new side by side --
is far more diagnostic than watching the 3D viewer, especially for
catching a SHAPE regression (e.g. a new loop/zigzag a fix introduced)
that's easy to miss in real-time playback but obvious as a static trace:

```python
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRUNK, LFOOT, RFOOT = 0, 17, 23  # body indices -- verify live, don't hardcode (see NPZ Format Recap)
for label, path in [("old", old_path), ("new", new_path)]:
    d = np.load(path)
    bp = d["body_pos_w"]
    plt.plot(bp[:,TRUNK,0], bp[:,TRUNK,1], 'k.-', label="trunk")
    plt.plot(bp[:,LFOOT,0], bp[:,LFOOT,1], 'b.-', label="left foot")
    plt.plot(bp[:,RFOOT,0], bp[:,RFOOT,1], 'r.-', label="right foot")
    # ... start/end markers, axvline(0), aspect('equal') -- see docs/BugFixes.md
    # 2026-08-05 entries for the full worked version that caught two
    # separate regressions this way before they were ever applied for real.
```
Frame-label the points (`ax.annotate(str(t), ...)`) when zooming into a
specific window -- this is what actually identified which frame was the
true outlier in a joint-angle discontinuity, not just that one existed.

## Quick Reference

| Question | How to answer it |
|---|---|
| Is this frame stance or swing, this leg? | `body_pos_w[frame, foot_idx, 2] > 0.032` (this project's grounded convention) |
| Is this foot flat after my edit? | `tilt_xy(body_quat_w[frame, foot_idx])` ~ `(0, 0)`, NOT eyeballing joint angle sums |
| What's the real joint order / body order? | Fetch live from the MJCF via `mujoco.mj_id2name` -- never hardcode from memory, it silently drifts from the XML |
| Am I re-tuning a parameter (e.g. damping strength)? | Re-source from the TRUE original (git-recovered if needed), never from the already-corrected file -- stacking compounds nonlinearly |
| Did my edit only touch position, or orientation too? | Assume orientation is affected for every body downstream of the edited joint in the kinematic chain, and verify with FK -- don't assume "I didn't touch that joint" means "that joint's world orientation is unaffected" |
| Need to move a foot's POSITION, not just fix its orientation? | Prefer damping Hip/Knee toward neutral over solving IK for an exact target -- IK for position (unlike the 2-DOF ankle-orientation solve) is a real 2-link-arm problem with multiple solution branches and can jump between them frame to frame even when each frame converges |
| Did a position/damping edit change WHEN a foot touches the ground? | Diff the boolean `z<0.033` grounded mask, old vs. new, frame-by-frame -- a pass can look perfect on flatness/NaN/velocity and still silently swap which foot is grounded when |

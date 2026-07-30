---
name: debugging-mujoco-contact-sensors
description: Use when a MuJoCo/mjlab reward or sensor tied to body-part contact (feet, knees, chin, any geom-vs-geom touch) seems to fire on the wrong body part, fires too rarely or never, or its firing doesn't match what's visible in the viewer -- before trusting console flags, reward printouts, or your own reading of the geom XML.
---

# Debugging MuJoCo Contact Sensors

## Overview

Contact-sensor bugs in this codebase are almost never "the sensor is broken." They're
"the sensor is reporting something true that you mislabeled, misindexed, or
under-thresholded." Three real sessions on `interceptV2DualDis`'s
`penalize_wrong_foot_ball_contact` went through 3+ rounds of "still not
working" before landing on the actual cause each time -- this skill exists to
skip straight to the verification step that ends the loop.

## When to Use

- A `ContactSensorCfg` combining multiple geoms (e.g. one sensor matching
  both knee AND shin) fires, and you assumed it means one specific part.
- A reward/flag isn't firing even though the user can see contact happening
  in the viewer, or IS firing when the user swears nothing is touching.
- You're about to reason about which geom index means what from source code
  or regex patterns alone, without checking the live sensor object.
- You're about to widen or narrow a contact condition and want to know the
  real firing distribution before changing weights.

**Don't use for:** designing a new contact sensor from scratch with no bug to
diagnose (just write it and verify normally), or non-contact reward bugs.

## Core Pattern

**Never reason about contact sensors from the MJCF or regex pattern alone --
always interrogate the live sensor object.** Three things the source code
cannot tell you reliably:

1. **Geom index order.** `ContactSensorCfg.primary`'s regex match order is
   NOT alphabetical and NOT source-file order. Always fetch it directly:
   ```python
   sensor = env.scene["leg_ball_contact"]
   print(sensor.primary_names)
   # e.g. ['left_shin_collision', 'left_knee_collision',
   #       'right_knee_collision', 'right_shin_collision']
   ```
   A combined sensor's `found`/`force` tensor columns follow THIS order, not
   your assumption. If you find yourself writing `found[:, :2]` for "left"
   without having printed `primary_names` first, stop and print it.

2. **Which sub-geom actually fired.** A sensor named `leg_ball_contact`
   covering both knee and shin will read as "contact" when EITHER fires.
   Don't say "the knee touched" -- log the per-index `found`/`force` and
   check which column was nonzero. In this codebase's T1 model, "knee/shin"
   sensors fire almost exclusively on the SHIN (it sits lower, closer to a
   rolling ball's height) -- calling every firing "the knee" caused three
   rounds of user-reported "still not working" before this was caught.

3. **Whether "found" means what you think.** `found=True` requires actual
   geometric contact resolution (penetration within MuJoCo's margin) for
   that physics step. It under-fires relative to what looks "close" in the
   viewer or to a human's sense of "near." If you need a more permissive
   trigger (fires when close, not just touching), compute a direct
   ball-to-geom distance instead of reading `found` -- see Quick Reference.

## Quick Reference

| Question | How to answer it (not guess) |
|---|---|
| What geoms exist on this body part? | `grep -n "collision" *.xml \| grep -iE "part_name"` -- read `type`, `size`, `pos` (local offset from parent body), `fromto` for capsules given as segments |
| What does viewer group N show? | Check `<default class="...">` blocks in the XML for `group=` -- group numbers are arbitrary per-model, not a MuJoCo-wide convention. In `t1_headless.xml`: visual mesh=2, collision default=3, foot capsules explicitly override to 1. |
| What order are a sensor's geoms in? | `env.scene["<sensor_name>"].primary_names` -- always print, never assume |
| Is `found=True` a real touch or a near-miss? | Compute the actual surface gap (see Implementation) -- don't trust the boolean alone when debugging |
| Is my new/changed condition doing what I think? | Write a probe (see Implementation) that recomputes the expected value independently and asserts it matches the reward function's real output, over many steps of a REAL trained checkpoint -- not zero-action, not a handful of manual steps |

## Implementation

Use `probe_template.py` (same directory) as a starting point. It:
- Loads a real env + a real trained checkpoint (not zero-action -- a
  do-nothing policy rarely reproduces the contact patterns a trained policy
  produces).
- Prints `primary_names` for every sensor you're about to reason about.
- Computes an exact surface-gap distance (ball center to geom surface,
  accounting for local offset + orientation for oriented geoms like
  capsules) instead of trusting `found` alone -- this is what conclusively
  distinguished "shin touching" from "knee touching" when both looked
  identical from the console.
- Cross-checks a reward function's live output against an independently
  recomputed expected value every step, over thousands of steps, and prints
  any mismatch immediately. Zero mismatches over a real run is the actual
  proof a fix works -- not "it compiles" or "one manual step looked right."

Adapt the constants at the top (task id, checkpoint path, body/geom names,
radii) to the sensor you're debugging. Run with:
```bash
env -u PYTHONPATH uv run python probe_template.py
```

For a capsule-shaped geom with a local offset (e.g. a shin capsule sitting
0.12m below its parent body's origin), compute the nearest point on the
capsule's axis segment in the BODY's local frame, then rotate into world
frame with `quat_apply_inverse` before measuring distance -- template shows
this exactly (`_capsule_gap`).

## Common Mistakes

| Mistake | Fix |
|---|---|
| Calling a combined sensor by one sub-part's name ("the knee") in explanations to the user | Say what the sensor actually covers, or better, log which index fired and say that |
| Assuming `found[:, :2]` = left because the regex pattern lists left first | Print `primary_names`, verify the actual order |
| Debugging with a zero-action or random policy | Load the real trained checkpoint -- it's the only thing that reproduces the actual contact patterns being asked about |
| Trusting "it didn't crash" as verification | Assert the reward output matches an independently computed expected value across many real steps; report the exact count of matches/mismatches |
| Widening a contact condition to "closer" without checking if `found` under-fires | Compute a real distance-based proximity check instead of a bigger contact margin hack |

---
name: recording-training-videos
description: Use whenever the user asks how training is going, wants to see the current checkpoint's behavior, or asks for a video/recording of a checkpoint. Records a fixed number of episodes (each capped at a fixed number of seconds, however long the episode actually runs) to .mp4, WITH the same blue/orange/red/green waypoint markers play.py's interactive viewer draws, and sends it directly to the user. Covers real bugs hit building this: MUJOCO_GL set too late, why mjlab's own VideoRecorder wrapper doesn't fit this project's per-episode-capped requirement, the play-mode episode-length gotcha (10s, not the training-time 3s -- most episodes run much closer to 10s than 3s due to post-save recovery rewards), and how the offscreen debug-vis callback (`env.update_visualizers`) differs from play.py's interactive-viewer monkeypatch.
---

# Recording Training Videos

## What it is

`src/simple_goalkeeper/scripts/record_checkpoint_video.py` -- a diagnostic-only
script (no training impact) that loads a checkpoint, rolls it out live in a
single env, and hand-writes an `.mp4` via `mediapy.write_video` from manually
captured frames (`raw_env.render()`, `render_mode="rgb_array"`).

Built 2026-09-19 after the user asked to see the latest checkpoint's actual
behavior rather than just reward numbers. Originally used mjlab's own
`mjlab.utils.wrappers.VideoRecorder` -- rewritten same day, see "Why not
mjlab's VideoRecorder" below for why.

## Usage

```bash
uv run python src/simple_goalkeeper/scripts/record_checkpoint_video.py \
    --checkpoint logs/rsl_rl/.../model_9500.pt \
    --out-dir /tmp/goalkeeper_video \
    --num-episodes 6 --episode-seconds 3.0 \
    --difficulty 0.6878 --domain-rand 0.4161 \
    --azimuth 180 --distance 5.0 --elevation -30.0
```

- `--num-episodes` / `--episode-seconds`: records exactly `num_episodes`
  resets, each one's footage cut at `episode_seconds` real time (however
  long the episode actually runs past that point) -- NOT a flat total-frame
  budget. See "Why not mjlab's VideoRecorder" for why this distinction
  matters in practice.
- `--difficulty` should match (or be below) whatever `ball_difficulty` the
  checkpoint actually reached during training (check
  `Episode/Curriculum/ball_difficulty/ball_difficulty` in its tensorboard
  log) -- testing at a higher difficulty than it ever trained on shows
  out-of-distribution behavior, not representative behavior. `--domain-rand`
  is a SEPARATE curriculum (`Episode/Curriculum/domain_rand/domain_rand_curriculum`
  in the same tensorboard log) -- read and pass it too if representative
  behavior matters (it defaults to `None`, i.e. left at whatever the env's
  own default is, NOT forced to match the checkpoint automatically).
- `--azimuth`/`--distance`/`--elevation` control the camera (`ViewerConfig`
  defaults: azimuth=90, distance=5.0, elevation=-45.0). Neither rotation
  sign is what it looks like it should be -- both confirmed empirically
  2026-09-19, don't re-derive from first principles, just use these:
  - **Azimuth:** decreasing from the 90 default (90 -> 0) was the WRONG
    direction for "rotate 90 clockwise"; 90 -> 180 was correct. To rotate
    clockwise again from a non-default azimuth, ADD to the current value.
  - **Elevation:** decreasing from the -45.0 default (-45 -> -60, intending
    "tilt down a little") actually tilted the camera UPWARD instead. To
    tilt further DOWN, INCREASE elevation toward 0 (e.g. -45 -> -30).

## Playback speed / fps

Frames are captured 1:1 with `env.step()` calls -- i.e. at the sim's own
native rate, `render_fps = 1/step_dt` (**50fps** for this env, confirmed via
`env.step_dt == 0.02`, `env.metadata["render_fps"] == 50.0`). Writing the
video out at that native 50fps IS objectively real-time (verified
2026-09-19 via `ffprobe`: `nb_frames * step_dt == duration`, exactly) -- but
the user reported it "seems just really fast" at 50fps anyway. **Not an
encoding bug** -- 50fps is just an unusual container rate some
players/embeds handle poorly (frame-dropping that reads as sped-up even
though the file's own timestamps are correct).

**Fix:** `--output-fps` (default 25.0) downsamples by keeping every Nth
captured frame (`stride = round(native_fps / output_fps)`) and writing at
`native_fps / stride`, NOT by just relabeling the fps tag on the same frame
set -- that would actually change playback speed, not just the container
rate. This preserves the real-world duration (confirmed: 439 frames @
25fps = 17.56s, matching the un-downsampled capture's own elapsed sim
time) while using a more universally-supported frame rate. If a video still
looks sped up after this, verify with `ffprobe -show_entries
stream=r_frame_rate,nb_frames,duration` before assuming another bug --
`nb_frames / r_frame_rate` should equal `duration` (that's the actual
real-time check, independent of what any given player does with it).

## Why not mjlab's `VideoRecorder`

The first version used `mjlab.utils.wrappers.VideoRecorder` with a flat
`--video-length` (total frame budget), on the assumption that episodes run
close to the training-time 3s length (`goalkeeper_env_cfg.py`:
`cfg.episode_length_s = 10.0 if play else 3.0` -- 3s is training-only).
**Wrong in practice:** post-save recovery rewards (`postorientation`,
`postupperdofpos`, etc.) keep most play-mode episodes running much closer
to the full 10s cap before `time_out` fires. A budget sized for "6 episodes
x 3s = 900 steps" only fit ~3 real episodes (user report: "you only had 3
episodes in that video?"), because episodes were actually averaging ~6s
each, not 3s.

`VideoRecorder` has no "cap each episode at N seconds, keep going across M
episodes" mode -- only "stop at a total frame budget" (spans episode
boundaries unpredictably) or "stop at the first episode's end"
(`video_length=None`, one episode only). Neither matches "M fixed-length
episode clips." **Fix:** stopped using `VideoRecorder` entirely -- step the
env for real every tick via `AMPEnvWrapper` (so episodes progress/terminate
naturally), but only call `raw_env.render()` and append the frame for the
first `episode_seconds` worth of ticks after each reset (tracked via a
`frames_this_episode` counter, reset to 0 whenever `dones[0]` is true), and
stop once `episode_count == num_episodes`. Write the collected frames with
`mediapy.write_video` directly at the end (same call `VideoRecorder` itself
used internally, just invoked by hand).

One consequence: `raw_env.render()` must be called directly, not
`env.render()` -- neither `AMPEnvWrapper` nor its base `RslRlVecEnvWrapper`
define `render()` or a `__getattr__` passthrough, so calling it on the
wrapped env raises `AttributeError`.

## Waypoint markers (blue/orange/red/green squares + spheres)

The script draws the same live waypoint markers `play.py`'s interactive
viewer does (2026-09-19, user request, "make the video also with the
rectangulars for the green ball and blue ball visualisations etc"), via a
standalone `_draw_intercept_markers(scn, raw_env)` function in the script
itself -- NOT by reusing `play.py`'s `_patch_viewer_intercept_vis` directly,
because that function is wired into `NativeMujocoViewer._update_debug_
visualizers`, a method specific to the INTERACTIVE viewer. The offscreen
`mujoco.Renderer` path uses a different hook entirely: `ManagerBasedRlEnv.
render()` calls `self.update_visualizers` (an env attribute, only invoked
`if hasattr(self, "update_visualizers")` -- nothing sets this by default)
with a `DebugVisualizer` wrapping the renderer's own `mujoco.MjvScene`
(`visualizer.scn`, same raw-geom API as the interactive viewer's
`viewer_handle.user_scn`). Set it once before the rollout loop:

```python
raw_env.update_visualizers = lambda visualizer: _draw_intercept_markers(visualizer.scn, raw_env)
```

`_draw_intercept_markers` is a deliberately simplified SUBSET of play.py's
full marker set (skips the btg/stb live-transition spheres and the
diagnostic landing-radius rings) -- covers what was actually asked for: the
blue/orange/red/green waypoint squares/spheres. It reads the exact same
live-cached env attributes `rewards.py` itself sets every tick
(`_blue_dbg_half_off`, `_blue_landing_half_side_current`,
`_success_rect_x_neg/_x_pos/_y_half`, `_orange_dbg_offset`,
`_orange_landing_half_side_current`, `_red_landing_half_side_current`,
`_blue_wide`, `_blue_landed`, `_red_active`) rather than a second
recomputed copy -- same "can't silently desync from what's actually
rewarded" discipline `play.py`'s own markers follow (see `docs/BugFixes.md`,
2026-09-17 entries, for the bug class this avoids). If `play.py` ever adds
a NEW marker (a 5th waypoint, a different color scheme, etc.), this
function will silently NOT show it -- update both places, or extract a
shared function in `play.py` itself if this drifts far enough to matter.

## Delivering the result

Send the `.mp4` straight to the user with `SendUserFile` (`display: "render"`)
instead of pushing it to GitHub or any external host -- simpler, immediate,
and avoids bloating the training repo with binary video files (unlike
checkpoints, which already have an established push convention). Only reach
for GitHub/external hosting if the user explicitly asks for a persistent
shareable link instead of an inline view.

## Bug: `MUJOCO_GL=egl` set too late

Setting `os.environ.setdefault("MUJOCO_GL", "egl")` inside `main()`, AFTER
the module-level `from mjlab.envs import ManagerBasedRlEnv` (and therefore
after mujoco's own rendering module) has already imported, is too late --
`mujoco.Renderer.__init__` fails with `mujoco.FatalError: an OpenGL platform
library has not been loaded into this process`. Every other probe script in
this project sets `MUJOCO_GL` inside `main()` and it never mattered, because
none of them actually construct a renderer (`render_mode=None` everywhere
else) -- this was the first script in the project to hit it. **Fix:** set
`MUJOCO_GL` as the very first lines of the file, before any `mjlab`/`mujoco`
import.

## Quick Reference

| Question | Answer |
|---|---|
| User asks "how's training going" / wants to see behavior | Run this script against the latest pushed (or local) checkpoint, `SendUserFile` the result alongside any numeric progress report |
| How long does a play-mode episode actually run? | Usually much closer to the 10s cap than the training-time 3s, due to post-save recovery rewards -- don't assume 3s when sizing `--episode-seconds`/reasoning about episode count |
| What `--difficulty`/`--domain-rand` should I use? | Whatever the checkpoint actually reached (read both from its own tensorboard log), not blind defaults |
| Camera not pointed where asked? | Adjust `--azimuth`/`--distance`/`--elevation`; increasing azimuth from 90 rotates clockwise, increasing elevation toward 0 tilts further down -- both empirically confirmed, both counter-intuitive |
| Video looks broken / OpenGL error | Check `MUJOCO_GL` is set before any mjlab/mujoco import, not inside `main()` |
| Got fewer episodes than asked for | Don't use a flat total-frame budget assuming a fixed episode length -- use the per-episode-capped manual capture loop (see "Why not mjlab's VideoRecorder") |
| `AttributeError` calling `env.render()` | Call `raw_env.render()` (i.e. `env.unwrapped.render()`) instead -- the AMP/RslRl wrappers don't pass it through |
| Video "looks sped up" | Not necessarily an encoding bug -- verify with `ffprobe` (`nb_frames / r_frame_rate == duration`) before assuming one. `--output-fps` (default 25.0) downsamples from the sim's native 50fps to a more universally-supported rate, same real-time duration |

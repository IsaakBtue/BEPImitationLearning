---
name: recording-training-videos
description: Use whenever the user asks how training is going, wants to see the current checkpoint's behavior, or asks for a video/recording of a checkpoint. Records a few episodes of a live rollout to .mp4 via mjlab's own VideoRecorder wrapper and sends it directly to the user. Covers two real import/wiring bugs hit building this (MUJOCO_GL set too late, VideoRecorder wrapping order vs AMPEnvWrapper) and the play-mode episode-length gotcha (10s, not the training-time 3s).
---

# Recording Training Videos

## What it is

`src/simple_goalkeeper/scripts/record_checkpoint_video.py` -- a diagnostic-only
script (no training impact) that loads a checkpoint, rolls it out live in a
single env, and records the rollout to an `.mp4` using mjlab's built-in
`mjlab.utils.wrappers.VideoRecorder`, the same mechanism mjlab's own
`scripts/play.py --video` uses.

Built 2026-09-19 after the user asked to see the latest checkpoint's actual
behavior rather than just reward numbers.

## Usage

```bash
uv run python src/simple_goalkeeper/scripts/record_checkpoint_video.py \
    --checkpoint logs/rsl_rl/.../model_9500.pt \
    --out-dir /tmp/goalkeeper_video \
    --video-length 600 \
    --difficulty 0.688 \
    --azimuth 180 --distance 5.0 --elevation -45.0
```

- `--video-length` is in env steps, not seconds (`dt=0.02` -> 50 steps/s).
  Play-mode episodes run up to **10s (500 steps)**, not the training-time 3s
  (`goalkeeper_env_cfg.py`: `cfg.episode_length_s = 10.0 if play else 3.0`) --
  don't assume the 3s training cap when sizing a recording. Most real
  episodes end sooner via `ball_exit`/save terminations anyway.
- `--difficulty` should match (or be below) whatever `ball_difficulty` the
  checkpoint actually reached during training (check
  `Episode/Curriculum/ball_difficulty/ball_difficulty` in its tensorboard
  log) -- testing at a higher difficulty than it ever trained on shows
  out-of-distribution behavior, not representative behavior.
- `--azimuth`/`--distance`/`--elevation` control the camera (`ViewerConfig`
  defaults: azimuth=90, distance=5.0, elevation=-45.0). Sign convention for
  "clockwise" wasn't obvious up front -- confirmed empirically 2026-09-19:
  decreasing azimuth from the 90 default (90 -> 0) was the WRONG direction
  for "rotate 90 clockwise"; 90 -> 180 was correct. If asked to rotate
  clockwise again from a non-default azimuth, ADD to the current value, not
  subtract.

## Delivering the result

Send the `.mp4` straight to the user with `SendUserFile` (`display: "render"`)
instead of pushing it to GitHub or any external host -- simpler, immediate,
and avoids bloating the training repo with binary video files (unlike
checkpoints, which already have an established push convention). Only reach
for GitHub/external hosting if the user explicitly asks for a persistent
shareable link instead of an inline view.

## Two real bugs hit building this (both fixed in the script, don't reintroduce)

### Bug 1: `MUJOCO_GL=egl` set too late

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

### Bug 2: `VideoRecorder` wrapping order vs `AMPEnvWrapper`

`VideoRecorder.step()` unpacks `obs, reward, terminated, truncated, info =
self._wrapped_env.step(action)` -- a 5-tuple, gymnasium-style. This
project's `AMPEnvWrapper` (used by every other probe/play script here)
returns a 4-tuple (`obs, rew, dones, extras`) instead. Wrapping order must
therefore be: raw `ManagerBasedRlEnv(render_mode="rgb_array")` -> \
`VideoRecorder(...)` -> `AMPEnvWrapper(...)` (matches mjlab's own
`scripts/play.py`: `ManagerBasedRlEnv` -> `VideoRecorder` ->
`RslRlVecEnvWrapper`). Wrapping in the other order, or wrapping
`AMPEnvWrapper` first, breaks `VideoRecorder`'s tuple unpacking.

## Quick Reference

| Question | Answer |
|---|---|
| User asks "how's training going" / wants to see behavior | Run this script against the latest pushed (or local) checkpoint, `SendUserFile` the result alongside any numeric progress report |
| What episode length should I assume for `--video-length` sizing? | 10s/500 steps max (play mode), not training's 3s -- see `goalkeeper_env_cfg.py`'s `episode_length_s` ternary |
| What `--difficulty` should I use? | Whatever `ball_difficulty` the checkpoint actually reached (read from its own tensorboard log), not a blind 1.0 |
| Camera not pointed where asked? | Adjust `--azimuth`/`--distance`/`--elevation`; increasing azimuth from 90 rotates clockwise (empirically confirmed), decreasing does not |
| Video looks broken / OpenGL error | Check `MUJOCO_GL` is set before any mjlab/mujoco import, not inside `main()` |
| `env.step()` unpacking error with `VideoRecorder` | Check wrapping order: raw env -> `VideoRecorder` -> `AMPEnvWrapper`, never the reverse |

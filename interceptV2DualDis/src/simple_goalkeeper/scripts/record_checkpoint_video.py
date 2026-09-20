"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Records a fixed number of episodes of a trained checkpoint to an .mp4, each
capped at a fixed number of SECONDS regardless of how long the episode
actually runs.

FIX 2026-09-19 (user report, "you only had 3 episodes in that video?"):
originally used mjlab's own `VideoRecorder` wrapper with a flat total-frame
budget (`--video-length`, e.g. 6 episodes x 3s assumed = 900 steps), betting
that episodes naturally run close to the training-time 3s length. They
don't in play mode -- post-save recovery rewards (postorientation,
postupperdofpos, etc.) keep most episodes running much closer to the 10s
play-mode cap before `time_out` fires, so a flat 900-step budget only fit
~3 real episodes, not 6. `VideoRecorder` has no "cap each episode at N
frames, keep going across M episodes" mode -- only "stop at frame budget"
(spans episode boundaries) or "stop at first episode end" (one episode
only). Replaced with manual frame capture: step the env for real every
tick (so episodes progress/terminate naturally), but only APPEND a frame to
the video for the first `episode_seconds` worth of ticks after each reset,
skip the rest of a longer episode, and stop once `num_episodes` resets have
happened. `VideoRecorder`/`AMPEnvWrapper` tuple-shape note from the old
approach no longer applies since `VideoRecorder` isn't used anymore --
`render_mode="rgb_array"` + manual `env.render()` calls only.

Usage:
    uv run python src/simple_goalkeeper/scripts/record_checkpoint_video.py \\
        --checkpoint logs/rsl_rl/.../model_9500.pt --num-episodes 6 \\
        --episode-seconds 3.0 --out-dir /tmp/goalkeeper_video
"""
from __future__ import annotations

import os

# Must be set before mujoco/mjlab are imported -- mujoco.Renderer reads this
# at import/first-use time to pick its GL backend. Setting it later (e.g.
# inside main()) is too late once mujoco's rendering module has already
# loaded, and mjr_makeContext fails with "an OpenGL platform library has not
# been loaded into this process".
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from dataclasses import dataclass
from pathlib import Path

import mediapy
import mujoco
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper

TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"


def _draw_intercept_markers(scn: "mujoco.MjvScene", raw_env) -> None:
    """Draws the same blue/orange/red/green waypoint spheres+squares
    play.py's `_patch_viewer_intercept_vis` draws for the interactive
    viewer, adapted for the offscreen VideoRecorder path (which calls
    `env.update_visualizers` with a `DebugVisualizer` wrapping the
    renderer's own scene, not a `NativeMujocoViewer` handle -- see
    `.claude/skills/recording-training-videos/` for why these are two
    separate rendering paths and why the interactive-viewer version can't
    just be reused directly).

    Deliberately a simplified subset of play.py's full marker set (skips
    the btg/stb live-transition spheres and the diagnostic landing-radius
    rings) -- covers what was actually asked for: the blue/orange/red/green
    waypoint squares. All positions/sizes read the SAME live-cached env
    attributes rewards.py itself sets every tick (`_blue_dbg_half_off`,
    `_blue_landing_half_side_current`, `_success_rect_*`,
    `_orange_dbg_offset`, `_orange_landing_half_side_current`,
    `_red_landing_half_side_current`), never a recomputed/hardcoded copy --
    same "can't silently desync from what's actually rewarded" discipline
    play.py's own markers follow (see docs/BugFixes.md, 2026-09-17 entries).
    """
    cross_y_t = getattr(raw_env, "_ball_crossing_y", None)
    if cross_y_t is None:
        return

    origins = raw_env.scene.env_origins[0].cpu().numpy()
    goal_x = float(origins[0])
    start_y = float(origins[1])
    cross_y = float(cross_y_t[0].item())
    floor_z = float(origins[2])
    sphere_z = floor_z + 0.12

    def _add_sphere(x: float, y: float, z: float, r: float, rgba) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        scn.ngeom += 1
        g = scn.geoms[scn.ngeom - 1]
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        mujoco.mjv_initGeom(
            geom=g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([r, r, r], dtype=np.float64),
            pos=np.array([x, y, z], dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).flatten(),
            rgba=np.array(rgba, dtype=np.float32),
        )

    def _add_line(from_: np.ndarray, to: np.ndarray, width: float, rgba) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        scn.ngeom += 1
        g = scn.geoms[scn.ngeom - 1]
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        mujoco.mjv_initGeom(
            geom=g, type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3, dtype=np.float64), pos=np.zeros(3, dtype=np.float64),
            mat=np.zeros(9, dtype=np.float64), rgba=np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(geom=g, type=mujoco.mjtGeom.mjGEOM_LINE, width=width, from_=from_, to=to)

    def _add_ground_rect(cx, cy, z, x_neg, x_pos, y_half, width, rgba) -> None:
        corners = [
            (cx - x_neg, cy - y_half), (cx + x_pos, cy - y_half),
            (cx + x_pos, cy + y_half), (cx - x_neg, cy + y_half),
        ]
        for i in range(4):
            p0 = np.array([*corners[i], z], dtype=np.float64)
            p1 = np.array([*corners[(i + 1) % 4], z], dtype=np.float64)
            _add_line(p0, p1, width, rgba)

    wide_t = getattr(raw_env, "_blue_wide", None)
    landed_t = getattr(raw_env, "_blue_landed", None)
    wide = bool(wide_t[0].item()) if wide_t is not None else False
    landed = bool(landed_t[0].item()) if landed_t is not None else False

    if wide and not landed:
        blue_off_t = getattr(raw_env, "_blue_dbg_half_off", None)
        mid_y = start_y + (float(blue_off_t[0].item()) if blue_off_t is not None else 0.0)
        _add_sphere(goal_x, mid_y, sphere_z, 0.08, [0.15, 0.4, 1.0, 0.75])
        _add_line(np.array([goal_x, mid_y, floor_z]), np.array([goal_x, mid_y, sphere_z]), 0.008, [0.15, 0.4, 1.0, 0.6])
        half_side = float(getattr(raw_env, "_blue_landing_half_side_current", 0.16))
        _add_ground_rect(goal_x, mid_y, floor_z + 0.002, half_side, half_side, half_side, 0.006, [0.15, 0.4, 1.0, 0.9])
    else:
        _add_sphere(goal_x, cross_y, sphere_z, 0.08, [0.1, 1.0, 0.2, 0.75])
        _add_line(np.array([goal_x, cross_y, floor_z]), np.array([goal_x, cross_y, sphere_z]), 0.008, [0.1, 1.0, 0.2, 0.6])
        x_neg = float(getattr(raw_env, "_success_rect_x_neg", 0.30))
        x_pos = float(getattr(raw_env, "_success_rect_x_pos", 0.05))
        y_half_raw = getattr(raw_env, "_success_rect_y_half", None)
        y_half = float(y_half_raw) if y_half_raw is not None else 0.08
        _add_ground_rect(goal_x, cross_y, floor_z + 0.002, x_neg, x_pos, y_half, 0.006, [0.1, 1.0, 0.2, 0.9])

    red_active_t = getattr(raw_env, "_red_active", None)
    red_active = bool(red_active_t[0].item()) if red_active_t is not None else False
    if wide and not red_active:
        orange_off_t = getattr(raw_env, "_orange_dbg_offset", None)
        orange_y = start_y + (float(orange_off_t[0].item()) if orange_off_t is not None else 0.0)
        _add_sphere(goal_x, orange_y, sphere_z, 0.08, [1.0, 0.55, 0.0, 0.75])
        _add_line(np.array([goal_x, orange_y, floor_z]), np.array([goal_x, orange_y, sphere_z]), 0.008, [1.0, 0.55, 0.0, 0.6])
        orange_half_side = float(getattr(raw_env, "_orange_landing_half_side_current", 0.12))
        _add_ground_rect(goal_x, orange_y, floor_z + 0.002, orange_half_side, orange_half_side, orange_half_side, 0.006, [1.0, 0.55, 0.0, 0.9])

    if wide and red_active:
        delta = cross_y - start_y
        red_y = start_y + 0.6 * delta
        _add_sphere(goal_x, red_y, sphere_z, 0.08, [0.9, 0.1, 0.1, 0.75])
        _add_line(np.array([goal_x, red_y, floor_z]), np.array([goal_x, red_y, sphere_z]), 0.008, [0.9, 0.1, 0.1, 0.6])
        red_half_side = float(getattr(raw_env, "_red_landing_half_side_current", 0.16))
        _add_ground_rect(goal_x, red_y, floor_z + 0.002, red_half_side, red_half_side, red_half_side, 0.006, [0.9, 0.1, 0.1, 0.9])


@dataclass(frozen=True)
class RecordConfig:
    checkpoint: str = ""
    out_dir: str = "/tmp/goalkeeper_video"
    num_episodes: int = 6
    episode_seconds: float = 3.0  # each episode's footage is cut here, however long it actually runs
    output_fps: float = 25.0  # downsampled from the sim's native 50fps (step_dt=0.02); see main()'s stride comment
    difficulty: float = 0.688
    domain_rand: float | None = None  # if None, left at the env's own default (does not force it)
    video_height: int = 480
    video_width: int = 640
    azimuth: float = 90.0  # ViewerConfig default; camera angle around lookat point
    distance: float = 5.0
    elevation: float = -45.0
    device: str | None = None


def main() -> None:
    cfg = tyro.cli(RecordConfig)
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    import simple_goalkeeper.tasks  # noqa: F401

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    assert isinstance(agent_cfg, dict)

    env_cfg.scene.num_envs = 1
    env_cfg.viewer.height = cfg.video_height
    env_cfg.viewer.width = cfg.video_width
    env_cfg.viewer.azimuth = cfg.azimuth
    env_cfg.viewer.distance = cfg.distance
    env_cfg.viewer.elevation = cfg.elevation

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = AMPEnvWrapper(env, clip_actions=None, motion_dataset=None)

    resume_path = Path(cfg.checkpoint)
    if not resume_path.is_absolute() and not resume_path.exists():
        resume_path = Path.cwd() / resume_path
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    print(f"[INFO] Loading checkpoint: {resume_path}", file=sys.stderr)

    runner_cls = load_runner_cls(TASK_ID)
    runner = runner_cls(env, agent_cfg, log_dir=None, device=device)
    runner.load(str(resume_path), load_optimizer=False)
    act_inference = runner.get_inference_policy(device=device)

    from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import (
        _get_actor_current_obs,
    )

    raw_env = env.unwrapped
    obs, _ = env.reset()
    raw_env._ball_difficulty = float(cfg.difficulty)
    if cfg.domain_rand is not None:
        raw_env._domain_rand_curriculum = float(cfg.domain_rand)
    # See `manager_based_rl_env.py`'s `render()`: it calls
    # `self.update_visualizers` as the offscreen debug-vis callback if the
    # attribute exists. Not set by anything else in this project (the
    # blue/orange/red/green markers are normally only wired into the
    # INTERACTIVE viewer via play.py's own monkeypatch) -- setting it here
    # is what makes them show up in the recorded video too.
    raw_env.update_visualizers = lambda visualizer: _draw_intercept_markers(visualizer.scn, raw_env)

    max_frames_per_episode = int(round(cfg.episode_seconds / raw_env.step_dt))
    frames: list[np.ndarray] = []
    episode_count = 0
    frames_this_episode = 0
    step = 0

    with torch.inference_mode():
        while episode_count < cfg.num_episodes:
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)
            step += 1

            if frames_this_episode < max_frames_per_episode:
                # env (AMPEnvWrapper/RslRlVecEnvWrapper) has no render()
                # passthrough -- call it on the raw env directly.
                frame = raw_env.render()
                if frame is not None:
                    frames.append(frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame)
                frames_this_episode += 1

            if bool(dones[0].item()):
                episode_count += 1
                frames_this_episode = 0
                print(f"[INFO] episode {episode_count}/{cfg.num_episodes} done at step {step}", file=sys.stderr)

    env.close()

    # Frames were captured 1:1 with env.step() calls, i.e. at the sim's own
    # native rate (render_fps = 1/step_dt, 50fps for this env). That's
    # objectively real-time if written back out at that same 50fps (confirmed
    # 2026-09-19 via ffprobe: nb_frames * step_dt == duration, exactly) --
    # but 50fps is an unusual container rate some players/embeds handle
    # poorly (frame-dropping that LOOKS sped up even though the file itself
    # is correctly timed). Downsample to a standard rate here instead of
    # just relabeling the fps tag -- that would actually change playback
    # speed. Keep every Nth frame (N = round(native_fps/output_fps)) so
    # real-world duration is preserved: fewer frames, same total seconds.
    native_fps = raw_env.metadata.get("render_fps", 1.0 / raw_env.step_dt)
    stride = max(1, round(native_fps / cfg.output_fps))
    video_frames = []
    for f in frames[::stride]:
        f = np.asarray(f)
        if f.dtype != np.uint8:
            f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        video_frames.append(f)
    effective_fps = native_fps / stride
    out_path = out_dir / "goalkeeper.mp4"
    mediapy.write_video(str(out_path), video_frames, fps=effective_fps)

    print(
        f"\n[SUMMARY] {episode_count} episodes, {len(video_frames)} frames "
        f"@ {effective_fps:.1f}fps ({len(video_frames) / effective_fps:.1f}s), "
        f"saved to: {out_path}"
    )


if __name__ == "__main__":
    main()

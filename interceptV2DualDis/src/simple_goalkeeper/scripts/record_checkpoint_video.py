"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Records a few episodes of a trained checkpoint to an .mp4, using mjlab's own
VideoRecorder wrapper (mjlab.utils.wrappers.VideoRecorder) the same way
mjlab's own scripts/play.py does it (render_mode="rgb_array" on the raw
ManagerBasedRlEnv, VideoRecorder wraps that BEFORE this project's own
AMPEnvWrapper -- VideoRecorder expects the raw 5-tuple gymnasium-style
env.step() return, not the 4-tuple AMPEnvWrapper produces).

Usage:
    uv run python src/simple_goalkeeper/scripts/record_checkpoint_video.py \\
        --checkpoint logs/rsl_rl/.../model_9500.pt --video-length 600 \\
        --out-dir /tmp/goalkeeper_video
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

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper

TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"


@dataclass(frozen=True)
class RecordConfig:
    checkpoint: str = ""
    out_dir: str = "/tmp/goalkeeper_video"
    video_length: int = 600  # ~4 episodes at 150 steps/episode
    difficulty: float = 0.688
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
    env = VideoRecorder(
        env,
        video_folder=out_dir,
        step_trigger=lambda step: step == 0,
        video_length=cfg.video_length,
        name_prefix="goalkeeper",
        disable_logger=False,
    )
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

    with torch.inference_mode():
        for step in range(cfg.video_length + 5):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)
            if step % 100 == 0:
                print(f"[INFO] step {step}/{cfg.video_length}", file=sys.stderr)

    env.close()
    print(f"\n[SUMMARY] video(s) saved under: {out_dir}")
    for f in sorted(out_dir.glob("*.mp4")):
        print(f"  {f}")


if __name__ == "__main__":
    main()

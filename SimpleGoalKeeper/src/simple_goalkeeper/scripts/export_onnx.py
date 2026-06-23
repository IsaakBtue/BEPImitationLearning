"""Export a SimpleGoalKeeper AMP checkpoint to ONNX with metadata.

Usage:
    uv run sgk_export logs/rsl_rl/simple_goalkeeper/2026-06-22_12-45-03_phase1/model_4250.pt

    # Custom output path:
    uv run sgk_export logs/rsl_rl/simple_goalkeeper/.../model_4250.pt --output /tmp/sgk.onnx
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata


def export_checkpoint(
    checkpoint_path: str,
    output_path: str | None = None,
    device: str = "cpu",
) -> Path:
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if output_path is None:
        onnx_path = ckpt_path.parent / f"{ckpt_path.parent.name}.onnx"
    else:
        onnx_path = Path(output_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    weight_keys = sorted(
        [k for k in state_dict if k.startswith("actor.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    num_actor_obs = state_dict[weight_keys[0]].shape[1]
    num_actions = state_dict[weight_keys[-1]].shape[0]
    hidden_dims = [state_dict[k].shape[0] for k in weight_keys[:-1]]
    print(f"[INFO] Actor: obs={num_actor_obs}, actions={num_actions}, hidden={hidden_dims}")

    actor_layers: list[nn.Module] = []
    actor_layers.append(nn.Linear(num_actor_obs, hidden_dims[0]))
    actor_layers.append(nn.ELU())
    for i in range(len(hidden_dims)):
        out = num_actions if i == len(hidden_dims) - 1 else hidden_dims[i + 1]
        actor_layers.append(nn.Linear(hidden_dims[i], out))
        if i < len(hidden_dims) - 1:
            actor_layers.append(nn.ELU())
    actor = nn.Sequential(*actor_layers)

    actor_state = {k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_state)
    actor.eval().to(device)

    dummy_obs = torch.zeros(1, num_actor_obs, device=device)
    torch.onnx.export(
        actor,
        dummy_obs,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
        dynamo=False,
    )
    print(f"[INFO] ONNX exported to: {onnx_path}")

    print("[INFO] Building env to extract metadata...")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg

    task_id = "Mjlab-BeyondAMP-Goalkeeper-T1"
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, AMPRunnerCfg)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env_wrapped = AMPEnvWrapper(env, clip_actions=agent_cfg.clip_actions, motion_dataset=agent_cfg.amp_data)

    metadata = get_base_metadata(env.unwrapped, str(ckpt_path.parent))
    env_wrapped.close()

    attach_metadata_to_onnx(str(onnx_path), metadata)
    print(f"[INFO] Metadata attached: {list(metadata.keys())}")
    print(f"[INFO] Done → {onnx_path}")
    return onnx_path


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import simple_goalkeeper.tasks  # noqa: F401

    parser = argparse.ArgumentParser(description="Export SimpleGoalKeeper checkpoint to ONNX")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument("--output", default=None, help="Output .onnx path (default: same dir as checkpoint)")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()

    export_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()

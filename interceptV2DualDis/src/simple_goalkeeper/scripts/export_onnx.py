"""Export a SimpleGoalKeeper/intercept AMP checkpoint to ONNX with metadata.

Handles two checkpoint families:
  * plain flat-MLP actor (state_dict has only "actor.*") -- unchanged from the
    original script.
  * HIM architecture (state_dict also has "history_encoder.*", "ball_estimator.*",
    "region_estimator.*", e.g. interceptV2DualDis/goalkeeper_multidisc_amp_cfg.py
    checkpoints) -- exports the FULL forward pass (history_encoder +
    ball_estimator + region_estimator + actor) behind a single raw obs_history
    input, not just the actor trunk. The old actor-only export produced a
    92-dim input with the estimator sub-networks missing entirely, which the
    rl_test deploy node cannot consume (92 % 71 != 0, its history-length
    auto-detect fails outright) even before considering that the required
    latent/estimate/region inputs have nowhere to come from outside the graph.

    Per-term observation scale (e.g. base_ang_vel*0.25, joint_vel*0.05 --
    goalkeeper_env_cfg.py's "obs-scaling audit", FIX 2026-07-20 item 21) is
    read directly from the live env_cfg and baked into the graph as its first
    op, so the exported ONNX accepts RAW, unscaled sensor values -- the same
    contract every other deploy policy uses -- and the deploy side never needs
    to know about or hand-sync to whatever scale convention a given training
    run used. This was a real, deployed bug: rl_test's C++ observation code
    (observation.cpp) has no per-term scaling anywhere, so any HIM checkpoint
    trained after the scaling fix landed would receive base_ang_vel and
    joint_vel roughly 4x/20x their trained-on magnitude -- confirmed as the
    root cause of a "robot has trouble lifting its foot" report on
    model_18500 (run 6144_ampvisiblehipflex_2026-08-01).

    All HIM hyperparameters (history_latent_dim, estimate_ball_dim,
    num_regions, per-term obs sizes/scale/order, history_length) are derived
    at export time from the checkpoint's own weight shapes and the live
    env_cfg -- nothing is hardcoded -- and cross-checked with assertions
    against the actor's actual input dimension, so a future config change
    (new obs term, different history length, different latent dims) fails
    loudly here instead of silently producing a broken ONNX.

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

# Only registered HIM/multi-discriminator task as of this fix -- see
# simple_goalkeeper/tasks/__init__.py. The original script's hardcoded
# "Mjlab-BeyondAMP-Goalkeeper-T1" is the plain (non-HIM) task and would load
# the wrong env_cfg (no region/ball estimator obs groups) for HIM checkpoints.
_PLAIN_TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1"
_HIM_TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"


def _build_mlp_from_prefix(prefix: str, state_dict: dict, activation: nn.Module) -> nn.Sequential:
    weight_keys = sorted(
        [k for k in state_dict if k.startswith(f"{prefix}.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    layers: list[nn.Module] = []
    for i, k in enumerate(weight_keys):
        w = state_dict[k]
        bias_key = k[: -len("weight")] + "bias"
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data = w.clone()
        lin.bias.data = state_dict[bias_key].clone()
        layers.append(lin)
        if i < len(weight_keys) - 1:
            layers.append(activation)
    return nn.Sequential(*layers)


class HimInterceptExportWrapper(nn.Module):
    """Full HIM forward pass behind a single raw (unscaled) flat obs_history
    input, term-major layout (matches rl_test's C++ deploy stacking AND
    mjlab's actual per-term history flattening -- see
    him_amp_on_policy_runner.py's "mjlab flattens history per-term rather
    than per-frame" note). obs_current is reconstructed as the newest (last)
    frame slice of each term's history sub-block: valid at deployment because
    real sensor data has no synthetic per-observation-group noise divergence
    between obs_current and obs_history's current slot (unlike training,
    where they're independently-sampled groups)."""

    def __init__(
        self,
        history_encoder: nn.Module,
        ball_estimator: nn.Module,
        region_estimator: nn.Module,
        actor: nn.Module,
        term_sizes: list[int],
        term_scales: list[float],
        history_length: int,
    ):
        super().__init__()
        self.history_encoder = history_encoder
        self.ball_estimator = ball_estimator
        self.region_estimator = region_estimator
        self.actor = actor
        self.history_length = history_length

        offsets = []
        pos = 0
        for sz in term_sizes:
            offsets.append((pos, sz))
            pos += sz * history_length
        self.offsets = offsets
        self.total_dim = pos

        scale_vec = torch.ones(self.total_dim)
        for (start, sz), s in zip(offsets, term_scales):
            if s != 1.0:
                scale_vec[start : start + sz * history_length] = s
        self.register_buffer("scale_vec", scale_vec)

    def forward(self, obs_history_raw: torch.Tensor) -> torch.Tensor:
        obs_history = obs_history_raw * self.scale_vec

        current_slices = []
        for start, sz in self.offsets:
            block = obs_history[:, start : start + sz * self.history_length]
            newest = block[:, (self.history_length - 1) * sz : self.history_length * sz]
            current_slices.append(newest)
        obs_current = torch.cat(current_slices, dim=-1)

        history_latent = self.history_encoder(obs_history)
        estimate_ball = self.ball_estimator(obs_history)
        estimate_region = self.region_estimator(obs_history)
        region_arg = torch.argmax(estimate_region, dim=-1, keepdim=True).float()

        actor_input = torch.cat([obs_current, history_latent, estimate_ball, region_arg], dim=-1)
        return self.actor(actor_input)


def _load_env(task_id: str, device: str):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, (AMPRunnerCfg, dict))
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    if isinstance(agent_cfg, dict):
        # Multi-disc tasks (e.g. Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc) register a
        # plain dict rl_cfg consumed by HimAMPOnPolicyRunner instead of the stock
        # AMPOnPolicyRunner -- mirrors play.py's run_play is_multidisc branch.
        # clip_actions has no override in the dict (keeps mjlab's own default,
        # None, same value used at training time); motion_dataset is a
        # per-region dict the single-dataset wrapper path can't consume and is
        # unused here anyway (this export path never reads AMP motion data).
        env_wrapped = AMPEnvWrapper(env, clip_actions=None, motion_dataset=None)
    else:
        env_wrapped = AMPEnvWrapper(env, clip_actions=agent_cfg.clip_actions, motion_dataset=agent_cfg.amp_data)
    return env, env_wrapped, env_cfg


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

    is_him = any(k.startswith("history_encoder.") for k in state_dict)

    if not is_him:
        # --- unchanged plain flat-MLP export path ---
        weight_keys = sorted(
            [k for k in state_dict if k.startswith("actor.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1]),
        )
        num_actor_obs = state_dict[weight_keys[0]].shape[1]
        num_actions = state_dict[weight_keys[-1]].shape[0]
        hidden_dims = [state_dict[k].shape[0] for k in weight_keys[:-1]]
        print(f"[INFO] Actor (plain): obs={num_actor_obs}, actions={num_actions}, hidden={hidden_dims}")

        actor_layers: list[nn.Module] = []
        actor_layers.append(nn.Linear(num_actor_obs, hidden_dims[0]))
        actor_layers.append(nn.ELU())
        for i in range(len(hidden_dims)):
            out = num_actions if i == len(hidden_dims) - 1 else hidden_dims[i + 1]
            actor_layers.append(nn.Linear(hidden_dims[i], out))
            if i < len(hidden_dims) - 1:
                actor_layers.append(nn.ELU())
        model: nn.Module = nn.Sequential(*actor_layers)
        actor_state = {k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")}
        model.load_state_dict(actor_state)
        model.eval().to(device)
        dummy_obs = torch.zeros(1, num_actor_obs, device=device)
        task_id = _PLAIN_TASK_ID
    else:
        print("[INFO] HIM architecture detected (history_encoder/ball_estimator/region_estimator present)")
        activation = nn.ReLU()  # him_actor_critic.py: encoder/estimator heads use ReLU
        history_encoder = _build_mlp_from_prefix("history_encoder", state_dict, activation)
        ball_estimator = _build_mlp_from_prefix("ball_estimator", state_dict, activation)
        region_estimator = _build_mlp_from_prefix("region_estimator", state_dict, activation)

        actor_weight_keys = sorted(
            [k for k in state_dict if k.startswith("actor.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1]),
        )
        actor_in_dim = state_dict[actor_weight_keys[0]].shape[1]
        num_actions = state_dict[actor_weight_keys[-1]].shape[0]
        actor = _build_mlp_from_prefix("actor", state_dict, nn.ELU())

        history_latent_dim = state_dict[
            sorted([k for k in state_dict if k.startswith("history_encoder.") and k.endswith(".weight")],
                   key=lambda k: int(k.split(".")[1]))[-1]
        ].shape[0]
        estimate_ball_dim = state_dict[
            sorted([k for k in state_dict if k.startswith("ball_estimator.") and k.endswith(".weight")],
                   key=lambda k: int(k.split(".")[1]))[-1]
        ].shape[0]
        print(f"[INFO] history_latent_dim={history_latent_dim}, estimate_ball_dim={estimate_ball_dim}, "
              f"actor_in_dim={actor_in_dim}, actions={num_actions}")

        task_id = _HIM_TASK_ID
        print(f"[INFO] Building env ({task_id}) to extract obs term config + metadata...")
        env, env_wrapped, env_cfg = _load_env(task_id, device)

        actor_group = env_cfg.observations["actor"]
        history_length = actor_group.history_length

        # NOTE: group_obs_term_dim["actor"] is a list positionally aligned with
        # ObservationManager's own active_terms["actor"] (both built by the same
        # single pass over group_cfg.terms.items() in observation_manager.py,
        # skipping None-valued terms) -- it is NOT keyed by term name. Must use
        # active_terms for term_names (not actor_group.terms.keys(), which can
        # include a None-valued term the manager itself skipped) so the name
        # list and the size list line up index-for-index.
        term_names = list(env.unwrapped.observation_manager.active_terms["actor"])
        term_scales = [(actor_group.terms[n].scale if actor_group.terms[n].scale is not None else 1.0)
                       for n in term_names]

        term_dims_lookup = getattr(env.unwrapped.observation_manager, "group_obs_term_dim", None)
        if term_dims_lookup is None:
            raise RuntimeError(
                "observation_manager has no group_obs_term_dim -- cannot derive per-term obs sizes "
                "automatically. Check mjlab's ObservationManager API (may have been renamed)."
            )
        actor_term_dims = term_dims_lookup["actor"]
        assert len(actor_term_dims) == len(term_names), (
            f"group_obs_term_dim['actor'] has {len(actor_term_dims)} entries but active_terms['actor'] "
            f"has {len(term_names)} -- ObservationManager's internal bookkeeping changed since this "
            f"export path was written, positional alignment can no longer be trusted."
        )
        # group_obs_term_dim reports the HISTORY-FLATTENED per-term size (e.g.
        # base_ang_vel: 3 per frame * history_length(10) = 30) when the group
        # applies uniform history stacking with flatten_history_dim=True (both
        # true here -- observation_manager.py inserts history_length into
        # obs_dims before flattening whenever group_cfg.history_length is set).
        # HimInterceptExportWrapper's own offset math (`pos += sz *
        # history_length`) expects the single-FRAME size instead, so divide it
        # back out here rather than double-multiplying by history_length below.
        flat_term_dims = [int(torch.tensor(dims).prod().item()) for dims in actor_term_dims]
        term_sizes = []
        for name, flat_dim in zip(term_names, flat_term_dims):
            assert flat_dim % history_length == 0, (
                f"Term '{name}' has flattened dim {flat_dim}, not divisible by "
                f"history_length={history_length} -- history stacking assumption broken."
            )
            term_sizes.append(flat_dim // history_length)
        num_one_step_obs = sum(term_sizes)
        print(f"[INFO] obs terms (order matches training concat order): "
              f"{list(zip(term_names, term_sizes, term_scales))}")
        print(f"[INFO] num_one_step_obs={num_one_step_obs}, history_length={history_length}")

        # actor_input = obs_current(num_one_step_obs) + history_latent + estimate_ball + region_arg(1)
        expected_actor_in = num_one_step_obs + history_latent_dim + estimate_ball_dim + 1
        assert expected_actor_in == actor_in_dim, (
            f"Derived actor input dim {expected_actor_in} (obs={num_one_step_obs} + "
            f"latent={history_latent_dim} + ball={estimate_ball_dim} + region_arg=1) != actor's actual "
            f"input dim {actor_in_dim} from checkpoint weights. Obs term config or HIM wiring has "
            f"changed since this export path was written -- fix the mismatch before trusting the export."
        )

        model = HimInterceptExportWrapper(
            history_encoder, ball_estimator, region_estimator, actor,
            term_sizes=term_sizes, term_scales=term_scales, history_length=history_length,
        )
        model.eval().to(device)
        assert model.total_dim == num_one_step_obs * history_length
        dummy_obs = torch.zeros(1, model.total_dim, device=device)

        env_wrapped.close()  # re-opened below for metadata via the normal path

    print(f"[INFO] Exporting ONNX (single input, dim={dummy_obs.shape[1]}) -> {onnx_path}")
    torch.onnx.export(
        model,
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
    env, env_wrapped, _env_cfg = _load_env(task_id, device)
    metadata = get_base_metadata(env.unwrapped, str(ckpt_path.parent))
    env_wrapped.close()

    attach_metadata_to_onnx(str(onnx_path), metadata)
    print(f"[INFO] Metadata attached: {list(metadata.keys())}")
    print(f"[INFO] Done -> {onnx_path}")
    return onnx_path


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import simple_goalkeeper.tasks  # noqa: F401

    parser = argparse.ArgumentParser(description="Export SimpleGoalKeeper/intercept checkpoint to ONNX")
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

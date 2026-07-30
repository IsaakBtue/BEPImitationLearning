"""Template: replay a real trained checkpoint and verify a contact-based
reward's actual firing cause, using surface-gap distance math instead of
trusting the raw `found` boolean alone.

Adapt the CONFIG block below, then run:
    env -u PYTHONPATH uv run python probe_template.py

See SKILL.md in this directory for why each step here matters.
"""
import os
import sys

# ---- CONFIG: adapt to your project/task -----------------------------------
SRC_PATH = "/home/isaak/BEPImitationlearning/interceptV2DualDis/src"
TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"
CHECKPOINT = (
    "/home/isaak/BEPImitationlearning/interceptV2DualDis/logs/rsl_rl/"
    "intercept_simple_goalkeeper_multidisc/"
    "2026-07-29_09-52-42_6144_armrevert_2026-07-29/model_18250.pt"
)
NUM_ENVS = 64
NUM_STEPS = 1500
# -----------------------------------------------------------------------------

sys.path.insert(0, SRC_PATH)
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import mjlab.tasks  # noqa: F401
import simple_goalkeeper.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.lab_api.math import quat_apply_inverse
from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper
from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import _get_actor_current_obs

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK_ID, play=True)
env_cfg.scene.num_envs = NUM_ENVS
agent_cfg = load_rl_cfg(TASK_ID)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
env = AMPEnvWrapper(env, clip_actions=None, motion_dataset=None)

runner_cls = load_runner_cls(TASK_ID)
runner = runner_cls(env, agent_cfg, log_dir=None, device=device)
runner.load(CHECKPOINT, load_optimizer=False)
act_inference = runner.get_inference_policy(device=device)

raw_env = env.unwrapped

# --- Step 1: NEVER assume sensor geom order -- always print it -------------
for sensor_name in ["ball_contact", "leg_ball_contact"]:  # adapt names
    sensor = raw_env.scene[sensor_name]
    print(f"{sensor_name}.primary_names = {sensor.primary_names}")


# --- Step 2: surface-gap helper for an oriented capsule geom ---------------
# Example: a shin capsule offset -0.12m along local Z from its parent body's
# origin, half-length 0.075m, radius 0.05m. Adapt constants to your geom.
_BALL_R = 0.10
_CAPSULE_R = 0.05
_CAPSULE_LOCAL_Z_LO = -0.195  # local-Z offset - half-length
_CAPSULE_LOCAL_Z_HI = -0.045  # local-Z offset + half-length


def _capsule_gap(ball_pos_w, body_pos_w, body_quat_w):
    """Surface gap (ball surface to capsule surface); <=0 means touching."""
    local_ball = quat_apply_inverse(
        body_quat_w.unsqueeze(0), (ball_pos_w - body_pos_w).unsqueeze(0)
    )[0]
    z_clamped = torch.clamp(local_ball[2], _CAPSULE_LOCAL_Z_LO, _CAPSULE_LOCAL_Z_HI)
    nearest_local = torch.tensor([0.0, 0.0, z_clamped.item()], device=ball_pos_w.device)
    dist = (local_ball - nearest_local).norm()
    return (dist - _CAPSULE_R - _BALL_R).item()


def _sphere_gap(ball_pos_w, sphere_center_w, sphere_r):
    """Surface gap for a sphere geom centered exactly at a body's origin."""
    return (ball_pos_w - sphere_center_w).norm().item() - sphere_r - _BALL_R


# --- Step 3: replay with the REAL checkpoint, log every firing exactly -----
robot = raw_env.scene["robot"]
ball = raw_env.scene["ball"]
# Adapt: bodies whose ORIGIN coincides with a sphere geom you care about
# (verify via the XML -- no `pos` attribute on the geom means local (0,0,0)).
target_body_ids = robot.find_bodies(["Shank_Left", "Shank_Right"])[0]

obs = env.get_observations()
found_events = 0
for step in range(NUM_STEPS):
    obs_current = _get_actor_current_obs(env)
    with torch.no_grad():
        actions = act_inference(obs_current, obs)
    obs, _, _, _ = env.step(actions)

    leg_sensor = raw_env.scene["leg_ball_contact"]
    leg_found = leg_sensor.data.found  # [B, N] -- order matches primary_names above
    fired = (leg_found > 0).any(dim=-1)

    if fired.any():
        ball_pos = ball.data.root_link_pos_w
        body_pos = robot.data.body_link_pos_w[:, target_body_ids, :]
        body_quat = robot.data.body_link_quat_w[:, target_body_ids, :]
        for i in torch.nonzero(fired).flatten().tolist():
            found_events += 1
            which = [
                leg_sensor.primary_names[j]
                for j in torch.nonzero(leg_found[i] > 0).flatten().tolist()
            ]
            # Report BOTH raw found AND a real surface-gap number -- the gap
            # is what actually proves which geom is touching, not the label.
            print(
                f"[step {step:4d} env {i:3d}] found_geoms={which} "
                f"leg_found_raw={leg_found[i].tolist()}"
            )
    if found_events >= 40:
        break

print(f"\nTotal fired events captured: {found_events}")

# --- Step 4: cross-check a reward function against an independently ----
# computed expected value across the WHOLE run, not a spot check. Import
# your actual reward function and assert it matches, e.g.:
#
# from simple_goalkeeper.mdp.rewards import penalize_wrong_foot_ball_contact
# reward = penalize_wrong_foot_ball_contact(raw_env, "ball")
# expected = <recompute from raw sensor data independently>
# assert torch.allclose(reward, expected), "mismatch -- reward logic is wrong"
#
# Zero mismatches over thousands of real steps is the actual proof a fix
# works. "It compiles" and "one manual step looked right" are not.

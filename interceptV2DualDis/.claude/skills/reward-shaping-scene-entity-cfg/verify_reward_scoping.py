"""Walks every registered reward term on a real env and flags any
SceneEntityCfg whose joint_ids/body_ids came back unresolved (slice(None))
despite the underlying function's default (or the params override) being
scoped to a subset of names -- see SKILL.md's Pitfall 2 for why this can
happen silently.

Run after adding or changing ANY asset-scoped reward, not just the one you
think you touched -- a scope change to a shared default object (e.g.
_ARM_JOINT_CFG) silently affects every consumer of that same object.

Usage:
    env -u PYTHONPATH uv run python .claude/skills/reward-shaping-scene-entity-cfg/verify_reward_scoping.py
"""
import sys
import os

sys.path.insert(0, "src")
os.environ.setdefault("MUJOCO_GL", "egl")

import mjlab.tasks  # noqa: F401
import simple_goalkeeper.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"


def _check_term_cfg(term_cfg, label: str, n_joints: int, n_bodies: int) -> list[str]:
    """Ground truth: after a genuine .resolve() call, `joint_names`/`body_names`
    hold the FULLY EXPANDED concrete name list (even for a regex like ".*" or
    a construction with no names at all) -- so `joint_ids == slice(None)` is
    only a bug when the expanded name count is a PROPER SUBSET of the
    robot's total joint/body count. A "select everything" cfg legitimately
    resolves to slice(None) too (scene_entity_config.py's own documented
    optimization) -- checking `bool(joint_names)` alone (as an earlier draft
    of this script did) false-positives on every one of those, e.g. this
    project's own `_ALL_JOINTS_CFG = SceneEntityCfg("robot", joint_names=
    (".*",))`, which expands to all 21 joints and correctly encodes that as
    slice(None)."""
    problems = []
    for key, value in term_cfg.params.items():
        if not isinstance(value, SceneEntityCfg):
            continue
        joint_subset = bool(value.joint_names) and len(value.joint_names) < n_joints
        body_subset = bool(value.body_names) and len(value.body_names) < n_bodies
        joint_unresolved = joint_subset and value.joint_ids == slice(None)
        body_unresolved = body_subset and value.body_ids == slice(None)
        if joint_unresolved or body_unresolved:
            problems.append(
                f"{label}: params['{key}'] declares a PROPER SUBSET "
                f"(joint_names={value.joint_names} [{len(value.joint_names or ())}/{n_joints}], "
                f"body_names={value.body_names} [{len(value.body_names or ())}/{n_bodies}]) "
                f"but the corresponding ids are still slice(None) -- "
                f"selecting the WHOLE entity, not the declared subset."
            )
    return problems


def check_reward_or_event_manager(manager, manager_name: str, n_joints: int, n_bodies: int) -> list[str]:
    # RewardManager.active_terms -> list[str]; EventManager.active_terms ->
    # dict[EventMode, list[str]], but get_term_cfg(term_name) is single-arg
    # for both, so flatten whichever shape active_terms comes back as.
    active = manager.active_terms
    term_names = active if isinstance(active, list) else [n for names in active.values() for n in names]
    problems = []
    for term_name in term_names:
        problems += _check_term_cfg(manager.get_term_cfg(term_name), f"{manager_name}.{term_name}", n_joints, n_bodies)
    return problems


def check_observation_manager(manager, n_joints: int, n_bodies: int) -> list[str]:
    # ObservationManager.active_terms -> dict[group_name, list[str]];
    # get_term_cfg needs BOTH group_name and term_name.
    problems = []
    for group_name, term_names in manager.active_terms.items():
        for term_name in term_names:
            term_cfg = manager.get_term_cfg(group_name, term_name)
            problems += _check_term_cfg(term_cfg, f"observation_manager.{group_name}.{term_name}", n_joints, n_bodies)
    return problems


def main() -> None:
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env.reset()
    zero_action = env.action_manager.action.clone()
    for _ in range(3):
        env.step(zero_action)

    robot = env.scene["robot"]
    n_joints = len(robot.joint_names)
    n_bodies = len(robot.body_names)

    problems = []
    problems += check_reward_or_event_manager(env.reward_manager, "reward_manager", n_joints, n_bodies)
    if hasattr(env, "observation_manager"):
        problems += check_observation_manager(env.observation_manager, n_joints, n_bodies)
    if hasattr(env, "event_manager"):
        problems += check_reward_or_event_manager(env.event_manager, "event_manager", n_joints, n_bodies)

    if problems:
        print(f"FOUND {len(problems)} unresolved-scope problem(s):\n")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print("OK -- every scoped SceneEntityCfg in params resolved correctly.")

    env.close()


if __name__ == "__main__":
    main()

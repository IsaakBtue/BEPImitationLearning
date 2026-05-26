#!/usr/bin/env python3
"""Goalkeeper deploy script.

Usage (from goalkeeper_deploy/ directory):
    python deploy.py --task goalkeeper_t1 --mujoco
    python deploy.py --list

This is a self-contained wrapper around the booster_deploy framework.
It adds goalkeeper_deploy/ first on sys.path so that our tasks/ package
takes precedence, then replicates the deploy.py dispatch logic while
honouring a _mujoco_controller_cls override on the task config.

Constraint: nothing in booster_deploy/ is modified.
"""
from __future__ import annotations

import argparse
import os
import sys

# ------------------------------------------------------------------
# Path setup – must happen before any booster_deploy imports
# ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_BOOSTER = os.path.join(_HERE, "..", "booster_deploy")

# goalkeeper_deploy/ first (our tasks/ overrides nothing in booster_deploy,
# they live in separate namespaces) – but putting it first ensures Python
# finds goalkeeper_deploy/tasks/ as the 'tasks' package.
sys.path.insert(0, _HERE)
sys.path.insert(1, os.path.abspath(_BOOSTER))

# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Deploy goalkeeper policy for T1.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--task", type=str, help="Task name to run.")
group.add_argument("-l", "--list", action="store_true", dest="list_tasks",
                   default=False, help="List available tasks.")
parser.add_argument("--net", type=str, default="127.0.0.1",
                    help="Network interface for SDK communication.")
parser.add_argument("--mujoco", action="store_true", default=False,
                    help="Run in MuJoCo simulation.")
parser.add_argument("--device", type=str, default="cpu",
                    help="Inference device (cpu / cuda).")
args = parser.parse_args()


def main() -> None:
    import pkgutil
    import tasks as tasks_pkg   # goalkeeper_deploy/tasks/

    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        try:
            __import__(mod_info.name)
        except Exception as exc:
            raise RuntimeError(f"Failed to import task module {mod_info.name}") from exc

    from booster_deploy.utils.registry import get_task, list_tasks

    if args.list_tasks:
        print("Available tasks:")
        for name, cfg in list_tasks().items():
            cls = type(cfg)
            print(f"  {name}  :  {cls.__module__}.{cls.__qualname__}")
        sys.exit(0)

    try:
        task_cfg = get_task(args.task)
    except KeyError:
        available = list(list_tasks().keys())
        print(f"Unknown task '{args.task}'. Available: {available}")
        sys.exit(1)

    task_cfg.policy.device = args.device

    if args.mujoco:
        # Honour per-task controller override if present, else fall back to
        # the standard MujocoController.
        ctrl_cls = getattr(task_cfg, "_mujoco_controller_cls", None)
        if ctrl_cls is None:
            from booster_deploy.controllers.mujoco_controller import MujocoController
            ctrl_cls = MujocoController
        ctrl_cls(task_cfg).run()
    else:
        try:
            from booster_robotics_sdk_python import ChannelFactory  # type: ignore
            ChannelFactory.Instance().Init(0, args.net)
        except ImportError:
            print(
                "Error: booster_robotics_sdk_python is not installed.\n"
                "Use --mujoco for simulation instead."
            )
            sys.exit(1)

        from booster_deploy.controllers.booster_robot_controller import BoosterRobotPortal
        with BoosterRobotPortal(task_cfg) as portal:
            portal.run()


if __name__ == "__main__":
    main()

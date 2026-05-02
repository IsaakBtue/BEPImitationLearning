#!/usr/bin/env python3
"""Train the goalkeeper task using mjlab's built-in train script.

Usage:
    cd /home/isaak/BEPImitationlearning/my_mjlab_project
    uv run python scripts/train.py                       # default 1 env (smoke test)
    uv run mjlab train goalkeeper                        # full 1020 envs
"""
import subprocess
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = ["uv", "run", "mjlab", "train", "goalkeeper"] + args
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd="/home/isaak/BEPImitationlearning/my_mjlab_project", check=True)

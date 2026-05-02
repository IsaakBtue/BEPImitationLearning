#!/usr/bin/env python3
"""Evaluate a trained goalkeeper checkpoint.

Usage:
    uv run python scripts/play.py --checkpoint logs/g1_goalkeeper/<run>/model_1000.pt
"""
import subprocess
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = ["uv", "run", "mjlab", "play", "goalkeeper"] + args
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd="/home/isaak/BEPImitationlearning/my_mjlab_project", check=True)

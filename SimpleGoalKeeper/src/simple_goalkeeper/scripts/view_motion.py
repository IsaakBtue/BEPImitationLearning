"""Replay an SGK NPZ motion file in a passive MuJoCo viewer (no physics).

Usage:
    uv run sgk_view src/simple_goalkeeper/motions/data/LeftDoubleStep_own_booster_t1.npz
"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

_XML = Path(__file__).parents[1] / "robots" / "xmls" / "t1_headless.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an SGK NPZ motion in MuJoCo viewer.")
    parser.add_argument("motion_file", type=str, help="Path to .npz motion file")
    args = parser.parse_args()

    path = Path(args.motion_file)
    if not path.exists():
        raise FileNotFoundError(path)

    d = np.load(path)
    fps       = float(d["fps"])
    joint_pos = d["joint_pos"]    # (T, 21)
    body_pos  = d["body_pos_w"]   # (T, B, 3) — index 0 = Trunk (root)
    body_quat = d["body_quat_w"]  # (T, B, 4) wxyz — index 0 = Trunk
    n_frames  = joint_pos.shape[0]
    dt        = 1.0 / fps

    model = mujoco.MjModel.from_xml_path(str(_XML))
    mdata = mujoco.MjData(model)

    print(f"File  : {path.name}")
    print(f"Frames: {n_frames} @ {fps:.0f} fps = {n_frames / fps:.2f} s")
    print("Press Ctrl-C or close window to exit.\n")

    with mujoco.viewer.launch_passive(model, mdata) as viewer:
        while viewer.is_running():
            for i in range(n_frames):
                t0 = time.perf_counter()

                mdata.qpos[0:3] = body_pos[i, 0, :]
                mdata.qpos[3:7] = body_quat[i, 0, :]  # wxyz
                mdata.qpos[7:]  = joint_pos[i]          # 21 DOFs
                mdata.qvel[:]   = 0.0

                mujoco.mj_forward(model, mdata)
                viewer.sync()

                remaining = dt - (time.perf_counter() - t0)
                if remaining > 0:
                    time.sleep(remaining)

                if not viewer.is_running():
                    break


if __name__ == "__main__":
    main()

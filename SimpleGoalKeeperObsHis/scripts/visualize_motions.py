"""Replay T1 NPZ motion files in a MuJoCo passive viewer.

Usage:
  python visualize_motions.py                  # list available motions
  python visualize_motions.py lefthand_t1      # play one motion (loop)
  python visualize_motions.py --all            # play all motions in sequence
  python visualize_motions.py lefthand_t1 --fps 25   # half speed
"""
import argparse
import glob
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

MOTIONS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "src", "imitationlearningbooster", "motions", "data",
)
XML_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "src", "imitationlearningbooster", "assets", "booster_t1", "T1_serial_clean.xml",
)
DEFAULT_FPS = 50  # motions were resampled to 50 Hz by convert_all.py


def list_motions():
    files = sorted(glob.glob(os.path.join(MOTIONS_DIR, "*.npz")))
    if not files:
        print(f"No NPZ files found in {MOTIONS_DIR}")
        return []
    print("Available motions:")
    for f in files:
        d = np.load(f)
        T = d["joint_pos"].shape[0]
        print(f"  {os.path.basename(f)[:-4]:<25}  {T} frames  ({T / DEFAULT_FPS:.2f}s)")
    return files


def resolve_path(name: str) -> str:
    if os.path.isabs(name) and name.endswith(".npz"):
        return name
    if not name.endswith(".npz"):
        name = name + ".npz"
    candidate = os.path.join(MOTIONS_DIR, name)
    if os.path.isfile(candidate):
        return candidate
    if os.path.isfile(name):
        return name
    raise FileNotFoundError(f"Motion not found: {name!r}\n"
                            f"  Looked in {MOTIONS_DIR}")


def play_motion(viewer, model, data, npz_path: str, fps: float, loop: bool = False):
    d = np.load(npz_path)
    joint_pos = d["joint_pos"]   # (T, 23)
    body_pos_w = d["body_pos_w"]  # (T, 24, 3)  body 0 = Trunk
    body_quat_w = d["body_quat_w"]  # (T, 24, 4) WXYZ
    T = joint_pos.shape[0]
    name = os.path.basename(npz_path)
    dt = 1.0 / fps

    print(f"\nPlaying {name}  ({T} frames @ {fps} Hz = {T/fps:.2f}s)"
          + ("  [looping, Ctrl-C to stop]" if loop else ""))

    while True:
        for frame in range(T):
            if not viewer.is_running():
                return
            t0 = time.perf_counter()

            # Reconstruct root state from body_pos_w / body_quat_w (body 0 = Trunk)
            data.qpos[:3] = body_pos_w[frame, 0]
            data.qpos[3:7] = body_quat_w[frame, 0]  # WXYZ — MuJoCo convention
            data.qpos[7:] = joint_pos[frame]
            mujoco.mj_kinematics(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - t0
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

        if not loop:
            break


def main():
    parser = argparse.ArgumentParser(description="Visualize T1 NPZ motions in MuJoCo.")
    parser.add_argument("motion", nargs="?",
                        help="Motion name (e.g. lefthand_t1) or path to NPZ. "
                             "Omit to list available motions.")
    parser.add_argument("--all", action="store_true",
                        help="Play all motions in sequence.")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the selected motion (ignored with --all).")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help=f"Playback FPS (default: {DEFAULT_FPS}).")
    args = parser.parse_args()

    if not args.motion and not args.all:
        list_motions()
        return

    model = mujoco.MjModel.from_xml_path(os.path.abspath(XML_PATH))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        if args.all:
            files = sorted(glob.glob(os.path.join(MOTIONS_DIR, "*.npz")))
            if not files:
                print("No motions found.")
                return
            try:
                while viewer.is_running():
                    for f in files:
                        if not viewer.is_running():
                            break
                        play_motion(viewer, model, data, f, args.fps, loop=False)
            except KeyboardInterrupt:
                pass
        else:
            npz_path = resolve_path(args.motion)
            try:
                play_motion(viewer, model, data, npz_path, args.fps,
                            loop=args.loop)
            except KeyboardInterrupt:
                pass

        print("\nDone.")
        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()

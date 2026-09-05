"""Diagnostic-only: compare candidate ball solref/solimp settings side by side.

NOT wired into training. Drops 5 balls (r=0.0..1.0, log-interpolated between a
"dead" endpoint and the current production bouncy endpoint) onto a flat floor
and reports measured bounce restitution (v_up_after / v_down_before at first
bounce) for each. Run with --viewer to also watch it live in MuJoCo's
interactive viewer.

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_ball_restitution.py
    uv run python src/simple_goalkeeper/scripts/probe_ball_restitution.py --viewer
"""
from __future__ import annotations

import argparse
import math

import mujoco
import mujoco.viewer
import numpy as np

# Endpoints for the r in [0,1] sweep.
# r=0 ("dead"): MuJoCo's own recommended default solref (timeconst, dampratio) —
#   critically damped, no bounce.
# r=1 ("bouncy"): current production ball.xml value (elastic bounce).
_DEAD_SOLREF = (0.02, 1.0)
_BOUNCY_SOLREF = (0.002, 0.0001)
_R_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
_DROP_HEIGHT = 0.5
_BALL_RADIUS = 0.10
_X_SPACING = 0.4


def _lerp_log(a: float, b: float, t: float) -> float:
    return math.exp((1 - t) * math.log(a) + t * math.log(b))


def _solref_for_r(r: float) -> tuple[float, float]:
    # Single free parameter: timeconst fixed at production's own value
    # (0.002 -- confirmed stable there), only dampratio swept. Two-parameter
    # log-interpolation (varying both timeconst and dampratio together) gave
    # a non-monotonic bounce curve (0.225 -> 0.146 -> 0.343 across
    # r=0.25/0.50/0.75) -- isolating one parameter is more likely to behave.
    tc = _BOUNCY_SOLREF[0]
    dr = _lerp_log(_DEAD_SOLREF[1], _BOUNCY_SOLREF[1], r)
    return tc, dr


def _build_xml() -> str:
    balls = []
    for i, r in enumerate(_R_VALUES):
        tc, dr = _solref_for_r(r)
        x = i * _X_SPACING
        balls.append(f'''
    <body name="ball_r{i}" pos="{x} 0 {_DROP_HEIGHT + _BALL_RADIUS}">
      <joint type="free" name="ball_freejoint_r{i}" damping="0.0005"/>
      <inertial mass="0.42" pos="0 0 0" diaginertia="0.00168 0.00168 0.00168"/>
      <geom name="ball_geom_r{i}" type="sphere" size="{_BALL_RADIUS}" rgba="1 0.9 0 1"
            friction="0.4 0.005 0.0001" solref="{tc} {dr}"
            solimp="0.0001 0.001 0.0001 0.5 2" margin="0.001" gap="0.0001"
            contype="1" conaffinity="1"/>
    </body>''')
    # No priority/solref/friction override on the floor -> plain MuJoCo
    # compiled defaults (solref=[0.02,1.0], friction=[1,0.005,0.0001]),
    # matching the real foot/terrain "collision" class exactly (neither sets
    # solref, friction, or priority anywhere in this repo -- checked
    # t1_headless.xml's collision default). MuJoCo then solmix/max-mixes
    # ball vs. floor/foot every contact, same as real training.
    #
    # CONFIRMED LIVE this matters a lot: an earlier probe draft gave the
    # floor the BALL's own friction (0.4 slide) instead of leaving it at
    # MuJoCo's default (1.0 slide) -- with everything else identical, that
    # alone flipped the measured r=1.0 restitution from a sane 0.478 to an
    # unphysical 1.833 (energy-gaining). This contact is numerically close
    # to an instability edge; don't add attributes to the floor/foot side
    # that aren't actually present in the real robot/terrain XML.
    return f'''<mujoco model="restitution_probe">
  <worldbody>
    <light pos="1 1 3"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.6 0.6 0.6 1"/>
    {''.join(balls)}
  </worldbody>
</mujoco>'''


def _apply_production_sim_options(model: mujoco.MjModel) -> None:
    """Match mjlab/velocity_env_cfg.py's real SimulationCfg (no override in
    this project) -- timestep=0.005, integrator=implicitfast, iterations=10,
    ls_iterations=20. Confirmed live: without this, a plain-mujoco default
    (Euler, iterations=100) run gave IDENTICAL results to the implicitfast
    run for this scene, so it likely doesn't change the qualitative outcome
    here -- kept anyway for fidelity."""
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = 10
    model.opt.ls_iterations = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="Open the interactive MuJoCo viewer instead of running headless.")
    parser.add_argument("--sim-seconds", type=float, default=3.0)
    args = parser.parse_args()

    xml = _build_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    _apply_production_sim_options(model)
    data = mujoco.MjData(model)

    print("r -> (timeconst, dampratio):")
    for r in _R_VALUES:
        print(f"  r={r:.2f} -> {_solref_for_r(r)}")

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            import time
            t0 = time.time()
            while viewer.is_running() and time.time() - t0 < args.sim_seconds * 20:
                mujoco.mj_step(model, data)
                viewer.sync()
        return

    # Headless: use the REAL mujoco_warp engine (not plain CPU mj_step) --
    # confirmed live this scene is numerically stiff enough that CPU and GPU
    # stepping disagree sharply near r=1 (CPU alone gave an unphysical >1
    # restitution there; mujoco_warp, the actual engine training runs on,
    # gives a physical ~0.48). mj_step (used by --viewer below) is CPU-only
    # and may look visibly different from these numbers at high r -- for
    # visual sanity-checking shape/timing only, trust these numbers for the
    # actual calibration.
    import mujoco_warp as mjwarp

    mujoco.mj_forward(model, data)
    m = mjwarp.put_model(model)
    d = mjwarp.put_data(model, data)

    n_balls = len(_R_VALUES)
    dof_addrs = [model.jnt_dofadr[model.body_jntadr[model.body(f"ball_r{i}").id]] for i in range(n_balls)]
    prev_vz = np.zeros(n_balls)
    min_vz_before_bounce = np.zeros(n_balls)
    measured_restitution = [None] * n_balls
    bounced = [False] * n_balls
    n_steps = int(args.sim_seconds / model.opt.timestep)

    for step in range(n_steps):
        mjwarp.step(m, d)
        qvel_row = d.qvel.numpy()[0]  # single world, shape (nv,)
        for i in range(n_balls):
            vz = float(qvel_row[dof_addrs[i] + 2])
            if not bounced[i]:
                min_vz_before_bounce[i] = min(min_vz_before_bounce[i], vz)
                if prev_vz[i] < -0.05 and vz > 0.05:
                    measured_restitution[i] = vz / abs(min_vz_before_bounce[i])
                    bounced[i] = True
            prev_vz[i] = vz

    print("\nMeasured first-bounce restitution (v_up / v_down), real mujoco_warp engine:")
    for i, r in enumerate(_R_VALUES):
        mr = measured_restitution[i]
        print(f"  r={r:.2f}: {'no bounce detected' if mr is None else f'{mr:.3f}'}")


if __name__ == "__main__":
    main()

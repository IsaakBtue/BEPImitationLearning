"""AMP dataset (2026-07-03): the four double/triple-step motions plus the two
near-standing Step motions feed the discriminator; Safe* files stay excluded;
no per-motion weighting (uniform by frame count). Mirrors G1's dataset, which
contains leftstep.pt/rightstep.pt alongside the save motions."""
import os


def test_motion_files_are_step_and_double_triple_only():
    from simple_goalkeeper.tasks.goalkeeper_amp_cfg import _motion_files

    basenames = sorted(os.path.basename(f) for f in _motion_files())
    assert basenames == [
        "LeftDoubleStep_own_booster_t1.npz",
        "LeftStep_own_booster_t1.npz",
        "LeftTripleStep_own_booster_t1.npz",
        "RightDoubleStep_own_booster_t1.npz",
        "RightTripleStep_own_booster_t1.npz",
        "Rightstep_own_booster_t1.npz",
    ]


def test_no_safe_files_in_dataset():
    from simple_goalkeeper.tasks.goalkeeper_amp_cfg import _motion_files

    for f in _motion_files():
        assert "Safe" not in os.path.basename(f), (
            f"unexpected Safe motion file in AMP dataset: {f}"
        )


def test_goalkeeper_amp_runner_cfg_wires_dataset():
    from simple_goalkeeper.tasks.goalkeeper_amp_cfg import goalkeeper_amp_runner_cfg

    cfg = goalkeeper_amp_runner_cfg()
    assert len(cfg.amp_data.motion_files) == 6
    assert cfg.amp_data.motion_weights is None

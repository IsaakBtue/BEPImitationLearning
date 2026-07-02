"""AMP dataset restriction (2026-07-02 experiment): only the four
double/triple-step motions feed the discriminator; no per-motion weighting."""
import os


def test_motion_files_are_only_the_four_step_motions():
    from simple_goalkeeper.tasks.goalkeeper_amp_cfg import _motion_files

    basenames = sorted(os.path.basename(f) for f in _motion_files())
    assert basenames == [
        "LeftDoubleStep_own_booster_t1.npz",
        "LeftTripleStep_own_booster_t1.npz",
        "RightDoubleStep_own_booster_t1.npz",
        "RightTripleStep_own_booster_t1.npz",
    ]


def test_no_safe_or_single_step_files_in_dataset():
    from simple_goalkeeper.tasks.goalkeeper_amp_cfg import _motion_files

    for f in _motion_files():
        name = os.path.basename(f)
        assert "DoubleStep" in name or "TripleStep" in name, (
            f"unexpected motion file in AMP dataset: {name}"
        )


def test_goalkeeper_amp_runner_cfg_wires_restricted_dataset():
    from simple_goalkeeper.tasks.goalkeeper_amp_cfg import goalkeeper_amp_runner_cfg

    cfg = goalkeeper_amp_runner_cfg()
    assert len(cfg.amp_data.motion_files) == 4
    assert cfg.amp_data.motion_weights is None

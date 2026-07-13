def test_double_and_triple_step_files_get_boosted_weight():
    """2026-07-13: DoubleStep and TripleStep both now the 2.5x-retimed clips,
    but weighted differently -- DoubleStep 5.0, TripleStep 4.0 -- both above
    the 1.0 baseline."""
    from simple_goalkeeper.tasks.goalkeeper_amp_cfg import _motion_weights

    files = [
        "/data/LeftDoubleStep_own_booster_t1_2p5x.npz",
        "/data/LeftSafe1_booster_t1.npz",
        "/data/LeftTripleStep_own_booster_t1_2p5x.npz",
        "/data/RightDoubleStep_own_booster_t1_2p5x.npz",
        "/data/RightSafeFar1_booster_t1.npz",
        "/data/RightTripleStep_own_booster_t1_2p5x.npz",
    ]

    weights = _motion_weights(files)

    assert weights == [5.0, 1.0, 4.0, 5.0, 1.0, 4.0]


def test_motion_weights_length_matches_motion_files():
    from simple_goalkeeper.tasks.goalkeeper_amp_cfg import _motion_files, _motion_weights

    files = _motion_files()
    assert len(_motion_weights(files)) == len(files)


def test_goalkeeper_amp_runner_cfg_wires_motion_weights():
    from simple_goalkeeper.tasks.goalkeeper_amp_cfg import goalkeeper_amp_runner_cfg

    cfg = goalkeeper_amp_runner_cfg()
    assert cfg.amp_data.motion_weights is not None
    assert len(cfg.amp_data.motion_weights) == len(cfg.amp_data.motion_files)
    for f, w in zip(cfg.amp_data.motion_files, cfg.amp_data.motion_weights):
        if "DoubleStep" in f or "TripleStep" in f:
            assert w > 1.0
        else:
            assert w == 1.0

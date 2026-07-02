from types import SimpleNamespace


def test_motion_weights_from_cfg_returns_declared_weights():
    from beyondAMP.mjlab.rsl_rl.amp_wrapper import _motion_weights_from_cfg

    cfg = SimpleNamespace(motion_weights=[1.0, 3.0, 3.0])
    assert _motion_weights_from_cfg(cfg) == [1.0, 3.0, 3.0]


def test_motion_weights_from_cfg_defaults_to_none_when_undeclared():
    from beyondAMP.mjlab.rsl_rl.amp_wrapper import _motion_weights_from_cfg

    cfg = SimpleNamespace(motion_files=["a.npz"])
    assert _motion_weights_from_cfg(cfg) is None

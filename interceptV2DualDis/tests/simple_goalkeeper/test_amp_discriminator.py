"""Regression tests for AMPDiscriminator/SpectralNorm
(rsl_rl_amp/modules/amp_discriminator.py) -- no direct test existed for this
class before, despite it being the actual component whose saturation this
whole 2026-09-15 investigation is about. Written as part of that
investigation's independent verification pass.

Specifically covers the newly-added `disc_logit_reg` mechanism (2026-09-14,
`multi_disc_amp_ppo.py`): it reads `discr.amp_linear.module.weight`, which
only exists as a live, differentiable tensor AFTER at least one forward pass
through the SpectralNorm wrapper (see SpectralNorm._update_u_v -- it
`setattr`s a fresh, non-parameter `weight` tensor onto `.module` every
forward call; before the first forward call `.module` has no `weight`
attribute at all, since `_make_params` deletes the original Parameter).
The real code always calls the discriminator forward before computing
logit_reg, but that ordering dependency was previously untested -- these
tests lock it in AND confirm gradients genuinely reach the learnable
`weight_bar` parameter, not just that `.module.weight` doesn't crash.
"""
import torch

from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator, SpectralNorm


def _make_discr(input_dim=10, hidden=(16, 8)):
    return AMPDiscriminator(
        input_dim=input_dim, amp_reward_coef=1.0,
        hidden_layer_sizes=list(hidden), device="cpu", task_reward_lerp=0.0,
    )


def test_amp_linear_module_weight_unavailable_before_first_forward():
    """Documents the real, load-bearing ordering constraint: accessing
    .module.weight before any forward pass fails. If this ever stops
    raising, SpectralNorm's internals changed and disc_logit_reg's own
    ordering assumption (forward always runs first in multi_disc_amp_ppo.py)
    should be re-checked."""
    discr = _make_discr()
    with __import__("pytest").raises(AttributeError):
        _ = discr.amp_linear.module.weight


def test_disc_logit_reg_gradient_reaches_the_real_learnable_parameter():
    """The actual mechanism added in commit bfe9f4a: after a forward pass,
    `discr.amp_linear.module.weight` must be a differentiable tensor whose
    gradient flows back to `weight_bar` (the real nn.Parameter SpectralNorm
    keeps under the hood) -- not a detached copy that would make the whole
    disc_logit_reg term a silent no-op."""
    discr = _make_discr()
    x = torch.randn(4, discr.input_dim)
    _ = discr(x)  # forward pass -- populates .module.weight, matching real usage order

    weight = discr.amp_linear.module.weight
    assert weight.requires_grad, "amp_linear.module.weight lost gradient tracking"

    logit_reg = 0.05 * weight.pow(2).sum()
    logit_reg.backward()

    weight_bar = discr.amp_linear.amp_linear.weight_bar if hasattr(discr.amp_linear, "amp_linear") \
        else discr.amp_linear.module.weight_bar
    assert weight_bar.grad is not None, "gradient never reached weight_bar -- disc_logit_reg is a no-op"
    assert weight_bar.grad.abs().sum().item() > 0.0


def test_compute_grad_pen_is_finite_nonnegative_and_weight_sensitive():
    """The R1 gradient penalty must be a real, finite, non-negative scalar
    (it's a squared-norm by construction) that actually reacts to changes in
    the discriminator's weights -- confirms autograd.grad in compute_grad_pen
    is genuinely differentiating through the live graph, not returning a
    detached/zero placeholder."""
    torch.manual_seed(0)
    discr = _make_discr()
    expert_state = torch.randn(8, discr.input_dim // 2)
    expert_next_state = torch.randn(8, discr.input_dim // 2)

    pen1 = discr.compute_grad_pen(expert_state, expert_next_state, lambda_=5)
    assert torch.isfinite(pen1)
    assert pen1.item() >= 0.0

    with torch.no_grad():
        for p in discr.trunk.parameters():
            p.add_(torch.randn_like(p) * 0.5)
        for p in discr.amp_linear.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    pen2 = discr.compute_grad_pen(expert_state, expert_next_state, lambda_=5)
    assert torch.isfinite(pen2)
    assert pen1.item() != pen2.item(), "grad penalty didn't change after perturbing weights"


def test_predict_amp_reward_noise_is_independent_per_sample():
    """The Gaussian-sample reward-smoothing trick (predict_amp_reward) must
    draw INDEPENDENT noise per (batch element, of the num_samples draws) --
    a broadcasting bug here would silently give every sample in the batch
    the exact same perturbation, defeating the whole point of the min-over-
    samples smoothing (see amp_discriminator.py's own 2026-07-08 FIX
    comment: it's supposed to match G1's per-sample Gaussian perturbation)."""
    torch.manual_seed(0)
    discr = _make_discr()
    state = torch.randn(5, discr.input_dim // 2)
    next_state = torch.randn(5, discr.input_dim // 2)
    task_reward = torch.zeros(5)

    # Let the spectral-norm power-iteration estimate converge first (a
    # freshly-initialized discriminator's sigma estimate is wildly off,
    # producing huge/unstable raw logits that clamp amp_reward to 0
    # regardless of noise -- not what this test is about).
    for _ in range(50):
        discr(torch.randn(4, discr.input_dim))

    reward1, d1, amp1 = discr.predict_amp_reward(state, next_state, task_reward, num_samples=20, sigma=0.3)
    reward2, d2, amp2 = discr.predict_amp_reward(state, next_state, task_reward, num_samples=20, sigma=0.3)

    # amp_reward (driven by the random noise samples) should differ between
    # two independent calls (different RNG draws) unless every sample in the
    # batch happened to land at the exact same min-error point -- vanishingly
    # unlikely with real random noise, would indicate a frozen/shared-noise
    # bug if it ever matches exactly.
    assert not torch.allclose(amp1, amp2, atol=1e-8)
    # d (the raw, unperturbed logit) IS identical across back-to-back calls
    # with unchanged input/weights -- only true since the SpectralNorm
    # eval-mode fix (test_spectral_norm_freezes_state_during_eval below);
    # before that fix this assertion failed (state drifted between calls).
    assert torch.allclose(d1, d2, atol=1e-5)


def test_spectral_norm_freezes_state_during_eval():
    """FIX 2026-09-15 (found + fixed via this independent verification
    pass): SpectralNorm.forward() used to call self._update_u_v()
    unconditionally -- no `if self.training:` guard, unlike the canonical
    spectral-norm technique (Miyato et al. 2018) and PyTorch's own
    torch.nn.utils.parametrizations.spectral_norm, both of which freeze u/v
    during eval mode specifically so inference-only forward passes don't
    perturb an estimate that's supposed to track TRAINING statistics.

    Concretely: AMPDiscriminator.predict_amp_reward wraps its forward calls
    in self.eval()/self.train() (amp_discriminator.py) -- clearly signaling
    "this is a read-only, non-training use" -- but SpectralNorm previously
    ignored that mode entirely, so every reward-computation call (happens
    every rollout step, independent of and far more often than actual
    discriminator training updates) still nudged the power-iteration
    estimate, and that state change persisted into the next genuine
    training forward pass. Now fixed: u/v are frozen during eval, only
    sigma (and therefore the differentiable normalized weight) still tracks
    live changes to w_bar.
    """
    torch.manual_seed(0)
    discr = _make_discr()
    x = torch.randn(4, discr.input_dim)

    # amp_linear maps hidden_dim -> 1, so its u vector is 1-D and converges
    # in a single call trivially -- use a trunk layer instead, which has a
    # genuine multi-dimensional u vector where power iteration would
    # otherwise keep visibly changing across calls if not frozen.
    first_trunk_sn = discr.trunk[0]
    assert isinstance(first_trunk_sn, SpectralNorm)

    discr.eval()
    u_before = first_trunk_sn.module.weight_u.clone()
    with torch.no_grad():
        discr(x)
        discr(x)
        discr(x)
    u_after = first_trunk_sn.module.weight_u.clone()

    assert torch.equal(u_before, u_after), (
        "SpectralNorm's power-iteration state changed during eval() mode -- "
        "the eval-mode guard regressed"
    )

    # Training mode must still refine u/v as before (the actual mechanism
    # SpectralNorm exists for) -- only eval-mode calls are frozen.
    discr.train()
    u_before_train = first_trunk_sn.module.weight_u.clone()
    discr(x)
    u_after_train = first_trunk_sn.module.weight_u.clone()
    assert not torch.equal(u_before_train, u_after_train), (
        "power-iteration refinement stopped happening in training mode too "
        "-- the eval-mode guard is too broad"
    )


def test_task_reward_lerp_matches_g1_amp_coef_formula():
    """G1's him_on_policy_runner.py:185 does
    `rewards = amp_reward * amp_coef + raw_rewards * (1 - amp_coef)`
    with amp_coef=0.4 (g1_29_config.py:364). This project's
    goalkeeper_multidisc_amp_cfg.py sets task_reward_lerp=0.6, and
    _lerp_reward computes `(1 - lerp) * disc_r + lerp * task_r`
    = 0.4 * disc_r + 0.6 * task_r -- confirms this is a genuine G1-parity
    match (blend-and-REPLACE, not additive -- G1's own file even has a
    commented-out additive alternative it deliberately did NOT use), not a
    port-introduced deviation. Locks in the exact numbers so a future
    config change is caught if it silently breaks parity."""
    discr = _make_discr()
    discr.task_reward_lerp = 0.6  # this project's actual configured value
    disc_r = torch.tensor([[2.0], [10.0]])
    task_r = torch.tensor([[100.0], [-4.0]])

    blended = discr._lerp_reward(disc_r, task_r)
    expected = 0.4 * disc_r + 0.6 * task_r
    assert torch.allclose(blended, expected)


def test_spectral_norm_bounds_layer_operator_norm_near_one():
    """SpectralNorm exists specifically to keep each layer's Lipschitz
    constant near 1 (see its own docstring, added 2026-09-05 to counter
    discriminator overconfidence). After the power-iteration estimate has
    had a few forward passes to converge, the wrapped layer's actual largest
    singular value should be close to 1 -- verifies the mechanism is
    genuinely doing its job, not just present in the code with no real
    effect (e.g. from a wrong axis/reshape in the power iteration)."""
    torch.manual_seed(0)
    linear = torch.nn.Linear(16, 16)
    sn = SpectralNorm(linear)

    x = torch.randn(32, 16)
    for _ in range(50):  # let the power-iteration estimate converge
        sn(x)

    weight = sn.module.weight.detach()
    true_largest_singular_value = torch.linalg.svdvals(weight)[0].item()
    assert 0.5 < true_largest_singular_value < 1.5, (
        f"spectral norm's power-iteration estimate has NOT converged near 1 "
        f"after 50 forward passes (true largest singular value = "
        f"{true_largest_singular_value}) -- the Lipschitz-bounding mechanism "
        "may not be working as intended"
    )

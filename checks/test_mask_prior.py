"""Masked prior preservation (Training tab: Outside the mask, keep the model's own picture): the loss arithmetic
against OneTrainer's, and the flag only for a LoRA."""
import pytest

torch = pytest.importorskip("torch")
from fizgig.families.driver import prior_base, weighted_loss, with_prior  # noqa: E402


class _Prior:
    """Stands in for train._MaskPrior: records that the forward ran with the LoRA off."""
    weight = 1.0

    def __init__(self):
        self.calls = 0

    def base(self, forward):
        self.calls += 1
        with torch.no_grad():
            return forward().detach().float()


def test_prior_base_only_with_a_prior_and_a_mask():
    p = _Prior()
    assert prior_base(None, torch.ones(1, 4), lambda: torch.zeros(1, 4)) is None
    assert prior_base(p, None, lambda: torch.zeros(1, 4)) is None
    assert p.calls == 0
    assert torch.equal(prior_base(p, torch.ones(1, 4), lambda: torch.full((1, 4), 2.0)), torch.full((1, 4), 2.0))
    assert p.calls == 1


def test_with_prior_matches_onetrainer():
    """OneTrainer's masked_losses_with_prior: clamp(m)*se/mean(clamp) + w*(1-clamp)*se_prior/mean(1-clamp)."""
    g = torch.Generator().manual_seed(0)
    se, se_p = torch.rand(1, 16, generator=g), torch.rand(1, 16, generator=g)
    mw = torch.rand(1, 16, generator=g).clamp(min=0.1)          # the floored mask the driver holds
    p = _Prior()
    p.weight = 0.7
    got = with_prior(weighted_loss(se, mw), p, torch.zeros(1), se_p, mw)
    want = (se * mw / mw.mean()).mean() + 0.7 * (se_p * (1 - mw) / (1 - mw).mean()).mean()
    assert torch.allclose(got, want, atol=1e-6)


def test_with_prior_adds_nothing_inside_an_all_white_mask():
    se, mw = torch.rand(1, 8), torch.ones(1, 8)
    loss = weighted_loss(se, mw)
    assert torch.allclose(with_prior(loss, _Prior(), torch.zeros(1), torch.rand(1, 8), mw), loss)


def test_mask_prior_flag_only_for_a_lora():
    from fizgig.families import launch, registry
    desc = registry.get("klein")
    assert launch.mask_prior(desc, {"MASK_PRIOR": "1"}) == 1.0
    assert launch.mask_prior(desc, {"MASK_PRIOR": "Off"}) == 0.0
    assert launch.mask_prior(desc, {"MASK_PRIOR": "1", "FAMILY_SLIDER": True}) == 0.0

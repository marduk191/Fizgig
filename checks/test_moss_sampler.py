"""MOSS-Image's sampler on a problem with a known answer: Gaussian data, whose exact denoiser is known, stands in for
the model. res_multistep (second order) must land closer to the ODE's answer than Euler at the same step count."""
import pytest

torch = pytest.importorskip("torch")
from fizgig.moss_image import sampling as S  # noqa: E402

A = 2.0                                       # the data's standard deviation


class _ExactDenoiser(torch.nn.Module):
    """v = noise - x0, with x0 the exact posterior mean of N(0, A^2) data at this sigma."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t, text, skip_layers=()):
        s = t / 1000.0
        x0 = x * (1 - s) * A * A / ((1 - s) ** 2 * A * A + s ** 2)
        return (x - x0) / s


def _run(steps, sampler, noise):
    return S.sample(_ExactDenoiser(), torch.zeros(4, 4096), 128, 128, steps=steps, seed=0, cfg=1.0, noise=noise,
                    sampler=sampler)


@pytest.fixture(scope="module")
def setup():
    noise = torch.randn(1, 128, 8, 8, generator=torch.Generator().manual_seed(0))
    return noise, _run(1000, "euler", noise)


@pytest.mark.parametrize("steps", [10, 20, 30, 50])
def test_res_multistep_beats_euler(setup, steps):
    noise, ref = setup
    err = lambda x: ((x - ref).norm() / ref.norm()).item()
    assert err(_run(steps, "res_multistep", noise)) < err(_run(steps, "euler", noise))


def test_schedule_ends_at_zero_and_decreases():
    s = S.schedule(30)
    assert float(s[-1]) == 0.0 and float(s[0]) <= 1.0
    assert all(float(a) > float(b) for a, b in zip(s[:-1], s[1:]))

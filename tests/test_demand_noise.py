import numpy as np

from modules.demand_simulator import DemandNoiseGenerator


def test_k_effective_uses_truncated_exponential_range() -> None:
    k_max = 0.8
    generator = DemandNoiseGenerator(
        k=k_max,
        k_exponential_lambda=10.0,
        seed=123,
    )

    values = np.array(
        [generator._sample_k_effective() for _ in range(2000)],
        dtype=float,
    )

    assert np.all(values >= 0.0)
    assert np.all(values <= k_max)
    assert values.mean() < k_max / 2.0


def test_k_effective_is_zero_when_k_is_zero() -> None:
    generator = DemandNoiseGenerator(k=0.0, k_exponential_lambda=10.0)

    assert generator._sample_k_effective() == 0.0

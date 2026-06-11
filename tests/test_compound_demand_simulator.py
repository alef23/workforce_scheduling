import numpy as np
import pytest

from modules.demand_simulator import CompoundDemandSimulator
from modules.workforce_engine.compound_actions import decode_action
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine
from modules.workforce_engine.schemas import ProblemSetup


def test_generates_four_compound_actions_per_resource() -> None:
    simulator = CompoundDemandSimulator(_build_setup(), seed=42)

    coverage, trajectory = simulator.compute_coverage(n_resources=4)

    assert coverage.shape == (24, 28)
    assert np.issubdtype(coverage.dtype, np.integer)
    assert np.all(coverage >= 0)
    assert int(coverage.sum()) > 0
    assert len(trajectory) == 16
    assert int(trajectory[0]["state"].remaining_stock.sum()) == 4
    assert all(sample["policy"].shape == (54,) for sample in trajectory)
    assert all(np.isclose(sample["policy"].sum(), 1.0) for sample in trajectory)


def test_every_action_changes_residual_demand() -> None:
    simulator = CompoundDemandSimulator(_build_setup(), seed=7)
    coverage, trajectory = simulator.compute_coverage(n_resources=5)

    assert np.array_equal(trajectory[0]["state"].residual_demand, coverage)

    for current, following in zip(trajectory, trajectory[1:]):
        current_residual = current["state"].residual_demand
        following_residual = following["state"].residual_demand
        assert not np.array_equal(current_residual, following_residual)
        assert np.all(following_residual <= current_residual)
        assert int(current_residual.sum() - following_residual.sum()) > 0

    last = trajectory[-1]
    last_coverage = _weekly_coverage(
        action_id=last["action_id"],
        week=last["state"].assignment_week,
    )
    assert np.array_equal(
        last["state"].residual_demand - last_coverage,
        np.zeros((24, 28), dtype=int),
    )


def test_modality_remains_fixed_during_each_resource() -> None:
    simulator = CompoundDemandSimulator(_build_setup(), seed=11)
    _, trajectory = simulator.compute_coverage(n_resources=8)

    for start in range(0, len(trajectory), 4):
        resource_samples = trajectory[start:start + 4]
        actions = [
            decode_action(sample["action_id"])
            for sample in resource_samples
        ]
        selected_modality = actions[0].modality

        assert len({action.modality for action in actions}) == 1
        assert resource_samples[0]["state"].current_modality is None
        assert [
            sample["state"].assignment_week
            for sample in resource_samples
        ] == [0, 1, 2, 3]
        assert all(
            sample["state"].current_modality == selected_modality
            for sample in resource_samples[1:]
        )
        assert all(
            action.entry_hour + action.modality <= 22
            for action in actions
        )


def test_stock_is_derived_from_first_action_of_each_resource() -> None:
    simulator = CompoundDemandSimulator(_build_setup(), seed=19)
    _, trajectory = simulator.compute_coverage(n_resources=20)
    initial_stock = trajectory[0]["state"].remaining_stock
    modality_counts = np.zeros(3, dtype=int)

    for start in range(0, len(trajectory), 4):
        action = decode_action(trajectory[start]["action_id"])
        modality_counts[action.modality_index] += 1

    assert np.array_equal(initial_stock, modality_counts)
    assert int(initial_stock.sum()) == 20

    remaining = initial_stock.copy()
    for start in range(0, len(trajectory), 4):
        assert np.array_equal(
            trajectory[start]["state"].remaining_stock,
            remaining,
        )
        modality_index = decode_action(
            trajectory[start]["action_id"]
        ).modality_index
        remaining = remaining.copy()
        remaining[modality_index] -= 1
    assert np.all(remaining == 0)


def test_policies_match_compound_engine_legal_actions() -> None:
    setup = _build_setup()
    simulator = CompoundDemandSimulator(setup, seed=23)
    engine = CompoundWorkforceEngine(setup)
    _, trajectory = simulator.compute_coverage(n_resources=10)

    for sample in trajectory:
        legal = engine.get_legal_actions(sample["state"])
        policy = sample["policy"]
        assert legal[sample["action_id"]]
        assert np.all(policy[~legal] == 0)
        assert np.all(policy[legal] > 0)
        assert np.isclose(policy.sum(), 1.0)


def test_seed_reproduces_coverage_and_actions() -> None:
    first = CompoundDemandSimulator(_build_setup(), seed=123)
    second = CompoundDemandSimulator(_build_setup(), seed=123)

    first_coverage, first_trajectory = first.compute_coverage(6)
    second_coverage, second_trajectory = second.compute_coverage(6)

    assert np.array_equal(first_coverage, second_coverage)
    assert [
        sample["action_id"] for sample in first_trajectory
    ] == [
        sample["action_id"] for sample in second_trajectory
    ]


@pytest.mark.parametrize("n_resources", [0, -1, 21])
def test_rejects_resource_count_outside_supported_range(
    n_resources: int,
) -> None:
    simulator = CompoundDemandSimulator(_build_setup(), seed=1)

    with pytest.raises(ValueError, match="entre 1 y 20"):
        simulator.compute_coverage(n_resources)


def _weekly_coverage(action_id: int, week: int) -> np.ndarray:
    action = decode_action(action_id)
    coverage = np.zeros((24, 28), dtype=int)
    week_start = week * 7
    for relative_day in range(7):
        if relative_day in action.days_off:
            continue
        coverage[
            action.entry_hour:action.entry_hour + action.modality,
            week_start + relative_day,
        ] += 1
    return coverage


def _build_setup() -> ProblemSetup:
    return ProblemSetup(
        mobile_days_off_count=1,
        fixed_day_off=6,
        allowed_entry_hours=[6, 12, 18],
        max_overcoverage_tolerance=0.1,
        closing_hour=22,
    )

import numpy as np
import pytest

from modules.demand_simulator import CompoundDemandSimulator
from modules.trajectory_generation import CompoundStockAdjuster
from modules.workforce_engine.compound_actions import decode_action
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine
from modules.workforce_engine.schemas import ProblemSetup


def test_no_reduction_returns_source_trajectory() -> None:
    setup = _build_setup()
    _, trajectory = CompoundDemandSimulator(
        setup,
        seed=42,
    ).compute_coverage(n_resources=5)

    result = CompoundStockAdjuster(
        CompoundWorkforceEngine(setup),
        p_stock=0.0,
    ).adjust(trajectory)

    assert result.trajectory is trajectory
    assert not result.stock_was_reduced
    assert np.array_equal(result.output_stock, result.original_stock)
    assert result.selected_chunk_indices == list(range(5))
    assert result.reordered_chunk_indices == list(range(5))
    assert result.consumed_action_count == 20


def test_reduction_samples_and_reorders_complete_chunks() -> None:
    setup = _build_setup(max_overcoverage_tolerance=1.0)
    _, trajectory = CompoundDemandSimulator(
        setup,
        seed=7,
    ).compute_coverage(n_resources=8)

    result = CompoundStockAdjuster(
        CompoundWorkforceEngine(setup),
        p_stock=1.0,
        seed=19,
    ).adjust(trajectory)

    assert result.stock_was_reduced
    assert len(result.selected_chunk_indices) < 8
    assert sorted(result.reordered_chunk_indices) == list(range(8))
    assert len(set(result.reordered_chunk_indices)) == 8
    assert result.source_action_count == 32
    assert result.consumed_action_count == 32
    assert not result.stopped_early

    expected_stock = np.zeros(3, dtype=int)
    for chunk_index in result.selected_chunk_indices:
        action_id = trajectory[chunk_index * 4]["action_id"]
        expected_stock[decode_action(action_id).modality_index] += 1

    assert np.array_equal(result.output_stock, expected_stock)
    assert np.array_equal(
        result.trajectory[0]["state"].remaining_stock,
        expected_stock,
    )

    adjusted_actions = [
        sample["action_id"]
        for sample in result.trajectory
    ]
    expected_actions = [
        trajectory[chunk_index * 4 + week]["action_id"]
        for chunk_index in result.reordered_chunk_indices
        for week in range(4)
    ]
    assert adjusted_actions == expected_actions


def test_zero_selected_chunks_starts_in_expansion_mode() -> None:
    setup = _build_setup()
    _, trajectory = CompoundDemandSimulator(
        setup,
        seed=3,
    ).compute_coverage(n_resources=1)

    result = CompoundStockAdjuster(
        CompoundWorkforceEngine(setup),
        p_stock=1.0,
        seed=1,
    ).adjust(trajectory)

    assert result.selected_chunk_indices == []
    assert np.all(result.output_stock == 0)
    assert result.first_expansion_step == 0
    assert result.trajectory[0]["state"].expansion_mode
    assert result.consumed_action_count == 4


def test_expansion_starts_after_selected_chunks() -> None:
    setup = _build_setup(max_overcoverage_tolerance=1.0)
    _, trajectory = CompoundDemandSimulator(
        setup,
        seed=15,
    ).compute_coverage(n_resources=6)

    result = CompoundStockAdjuster(
        CompoundWorkforceEngine(setup),
        p_stock=1.0,
        seed=8,
    ).adjust(trajectory)

    expected_step = len(result.selected_chunk_indices) * 4
    assert result.first_expansion_step == expected_step
    assert all(
        not sample["state"].expansion_mode
        for sample in result.trajectory[:expected_step]
    )
    assert result.trajectory[expected_step]["state"].expansion_mode


def test_rejects_incomplete_resource_chunk() -> None:
    setup = _build_setup()
    _, trajectory = CompoundDemandSimulator(
        setup,
        seed=5,
    ).compute_coverage(n_resources=2)

    with pytest.raises(ValueError, match="chunks completos"):
        CompoundStockAdjuster(
            CompoundWorkforceEngine(setup),
            p_stock=1.0,
        ).adjust(trajectory[:-1])


def test_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="entre 0 y 1"):
        CompoundStockAdjuster(
            CompoundWorkforceEngine(_build_setup()),
            p_stock=1.1,
        )


def _build_setup(
    max_overcoverage_tolerance: float = 0.1,
) -> ProblemSetup:
    return ProblemSetup(
        mobile_days_off_count=1,
        fixed_day_off=6,
        allowed_entry_hours=[6, 12, 18],
        max_overcoverage_tolerance=max_overcoverage_tolerance,
        closing_hour=22,
    )

import numpy as np
import pytest

from modules.demand_simulator import (
    CompoundDemandSimulator,
    DemandNoiseGenerator,
)
from modules.trajectory_generation import CompoundTrajectoryReplayer
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine
from modules.workforce_engine.schemas import ProblemSetup


def test_replays_compound_trajectory_after_noise() -> None:
    setup = _build_setup()
    coverage, base_trajectory = CompoundDemandSimulator(
        setup,
        seed=42,
    ).compute_coverage(n_resources=10)
    noise_result = DemandNoiseGenerator(
        k=0.8,
        seed=17,
    ).generate(coverage)
    engine = CompoundWorkforceEngine(setup)

    result = CompoundTrajectoryReplayer(engine).replay_trajectory(
        initial_demand=noise_result.initial_demand,
        source_trajectory=base_trajectory,
    )

    assert result["is_terminal"]
    assert result["consumed_action_count"] <= len(base_trajectory)
    assert result["source_action_count"] == len(base_trajectory)
    assert np.array_equal(
        result["trajectory"][0]["state"].residual_demand,
        noise_result.initial_demand,
    )
    assert np.isclose(
        result["final_reward"],
        engine.compute_reward(result["final_state"]),
    )
    assert all(
        sample["reward"] == result["final_reward"]
        for sample in result["trajectory"]
    )

    for sample in result["trajectory"]:
        legal = engine.get_legal_actions(sample["state"])
        assert legal[sample["action_id"]]
        assert sample["policy"].shape == (54,)
        assert np.all(sample["policy"][~legal] == 0)
        assert np.isclose(sample["policy"].sum(), 1.0)


def test_stops_when_engine_reaches_terminality_before_source_end() -> None:
    setup = _build_setup()
    _, base_trajectory = CompoundDemandSimulator(
        setup,
        seed=5,
    ).compute_coverage(n_resources=2)
    first_resource_demand = (
        base_trajectory[0]["state"].residual_demand
        - base_trajectory[4]["state"].residual_demand
    )

    result = CompoundTrajectoryReplayer(
        CompoundWorkforceEngine(setup)
    ).replay_trajectory(
        initial_demand=first_resource_demand,
        source_trajectory=base_trajectory,
    )

    assert result["is_terminal"]
    assert result["stopped_early"]
    assert result["source_action_count"] == 8
    assert result["consumed_action_count"] == 4
    assert len(result["trajectory"]) == 4


def test_supports_expansion_mode_from_zero_stock() -> None:
    setup = _build_setup()
    _, base_trajectory = CompoundDemandSimulator(
        setup,
        seed=9,
    ).compute_coverage(n_resources=2)
    first_resource_demand = (
        base_trajectory[0]["state"].residual_demand
        - base_trajectory[4]["state"].residual_demand
    )

    result = CompoundTrajectoryReplayer(
        CompoundWorkforceEngine(setup)
    ).replay_actions(
        initial_demand=first_resource_demand,
        initial_stock=np.zeros(3, dtype=int),
        actions=[
            sample["action_id"]
            for sample in base_trajectory[:4]
        ],
    )

    assert result["is_terminal"]
    assert result["trajectory"][0]["state"].expansion_mode
    assert np.all(result["final_state"].remaining_stock == 0)


def test_rejects_illegal_action_for_active_modality() -> None:
    setup = _build_setup()
    coverage, base_trajectory = CompoundDemandSimulator(
        setup,
        seed=13,
    ).compute_coverage(n_resources=1)
    actions = [sample["action_id"] for sample in base_trajectory]
    first_modality = actions[0] // 18
    different_modality = (first_modality + 1) % 3
    actions[1] = different_modality * 18

    with pytest.raises(ValueError, match="Acción ilegal durante replay"):
        CompoundTrajectoryReplayer(
            CompoundWorkforceEngine(setup)
        ).replay_actions(
            initial_demand=coverage,
            initial_stock=base_trajectory[0]["state"].remaining_stock,
            actions=actions,
        )


def test_can_return_non_terminal_partial_replay() -> None:
    setup = _build_setup()
    coverage, base_trajectory = CompoundDemandSimulator(
        setup,
        seed=21,
    ).compute_coverage(n_resources=2)

    result = CompoundTrajectoryReplayer(
        CompoundWorkforceEngine(setup)
    ).replay_actions(
        initial_demand=coverage,
        initial_stock=base_trajectory[0]["state"].remaining_stock,
        actions=[base_trajectory[0]["action_id"]],
        require_terminal=False,
    )

    assert not result["is_terminal"]
    assert result["final_reward"] == 0.0
    assert result["consumed_action_count"] == 1


def _build_setup() -> ProblemSetup:
    return ProblemSetup(
        mobile_days_off_count=1,
        fixed_day_off=6,
        allowed_entry_hours=[6, 12, 18],
        max_overcoverage_tolerance=0.1,
        closing_hour=22,
    )

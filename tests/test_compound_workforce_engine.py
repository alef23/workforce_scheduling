import numpy as np
import pytest

from modules.mcts.mcts import MCTS
from modules.mcts.mcts_schemas import MCTSConfig
from modules.workforce_engine.engine import WorkforceEngine
from modules.workforce_engine.compound_actions import (
    ACTION_SPACE_SIZE,
    decode_action,
    encode_action,
)
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine
from modules.workforce_engine.compound_schemas import CompoundWorkforceState
from modules.workforce_engine.schemas import ActionType, ProblemSetup, WorkforceState


def test_compound_action_round_trip_covers_all_ids() -> None:
    encoded_ids = []
    for modality_index in range(3):
        for entry_hour_index in range(3):
            for mobile_day_off in range(6):
                action_id = encode_action(
                    modality_index,
                    entry_hour_index,
                    mobile_day_off,
                )
                action = decode_action(action_id)
                encoded_ids.append(action_id)

                assert action.modality_index == modality_index
                assert action.entry_hour_index == entry_hour_index
                assert action.mobile_day_off == mobile_day_off

    assert sorted(encoded_ids) == list(range(ACTION_SPACE_SIZE))


def test_legal_actions_respect_stock_modality_and_closing_hour() -> None:
    engine = _build_engine()
    initial_state = _build_state(remaining_stock=np.array([1, 1, 1]))

    initial_legal = np.flatnonzero(engine.get_legal_actions(initial_state))
    assert len(initial_legal) == 42

    active_state = initial_state.copy_state(
        current_modality=4,
        assignment_week=1,
    )
    active_legal = np.flatnonzero(engine.get_legal_actions(active_state))
    assert len(active_legal) == 18
    assert {decode_action(int(action_id)).modality for action_id in active_legal} == {
        4
    }

    six_hour_state = active_state.copy_state(current_modality=6)
    assert int(engine.get_legal_actions(six_hour_state).sum()) == 12


def test_step_applies_one_week_of_coverage() -> None:
    engine = _build_engine()
    state = _build_state(residual_demand=np.full((24, 28), 10, dtype=int))
    action_id = encode_action(
        modality_index=0,
        entry_hour_index=0,
        mobile_day_off=0,
    )

    result = engine.step(state, action_id)

    assert not result.is_terminal
    assert result.reward == 0.0
    assert result.next_state.current_modality == 4
    assert result.next_state.assignment_week == 1
    assert np.all(result.next_state.residual_demand[6:10, 1:6] == 9)
    assert np.all(result.next_state.residual_demand[:, 0] == 10)
    assert np.all(result.next_state.residual_demand[:, 6] == 10)
    assert np.all(state.residual_demand == 10)


def test_fourth_week_closes_resource_and_discounts_stock() -> None:
    engine = _build_engine()
    state = _build_state(
        residual_demand=np.full((24, 28), 10, dtype=int),
        remaining_stock=np.array([1, 2, 3], dtype=int),
        current_modality=4,
        assignment_week=3,
    )
    action_id = encode_action(0, 0, 0)

    result = engine.step(state, action_id)

    assert result.next_state.current_modality is None
    assert result.next_state.assignment_week == 0
    assert np.array_equal(
        result.next_state.remaining_stock,
        np.array([0, 2, 3]),
    )
    assert not result.next_state.expansion_mode


def test_last_stock_enters_expansion_mode() -> None:
    engine = _build_engine()
    state = _build_state(
        residual_demand=np.full((24, 28), 10, dtype=int),
        remaining_stock=np.array([1, 0, 0], dtype=int),
        current_modality=4,
        assignment_week=3,
    )

    result = engine.step(state, encode_action(0, 0, 0))

    assert np.array_equal(result.next_state.remaining_stock, np.zeros(3, dtype=int))
    assert result.next_state.expansion_mode


def test_illegal_modality_change_is_rejected() -> None:
    engine = _build_engine()
    state = _build_state(current_modality=4, assignment_week=1)

    with pytest.raises(ValueError, match="modalidad activa"):
        engine.step(state, encode_action(1, 0, 0))


def test_engine_rejects_non_fixed_setup() -> None:
    setup = ProblemSetup(
        mobile_days_off_count=1,
        fixed_day_off=5,
        allowed_entry_hours=[6, 12, 18],
        max_overcoverage_tolerance=0.1,
        closing_hour=22,
    )

    with pytest.raises(ValueError, match="fixed_day_off=6"):
        CompoundWorkforceEngine(setup)


def test_compound_engine_implements_current_mcts_contract() -> None:
    class UniformEvaluator:
        action_space_size = ACTION_SPACE_SIZE

        def predict(self, _state):
            return (
                np.ones(ACTION_SPACE_SIZE, dtype=float) / ACTION_SPACE_SIZE,
                0.0,
            )

    engine = _build_engine()
    mcts = MCTS(
        engine=engine,
        evaluator=UniformEvaluator(),
        config=MCTSConfig(num_simulations=2, c_puct=1.0),
    )

    result = mcts.search(_build_state())

    assert result.policy.shape == (ACTION_SPACE_SIZE,)
    assert np.isclose(result.policy.sum(), 1.0)
    assert engine.get_legal_actions(_build_state())[result.selected_action_id]


@pytest.mark.parametrize(
    ("modality_index", "entry_hour_index", "mobile_day_off"),
    [
        (0, 0, 0),
        (0, 2, 5),
        (1, 0, 2),
        (1, 1, 4),
        (2, 0, 1),
        (2, 1, 5),
    ],
)
def test_weekly_transition_matches_legacy_engine(
    modality_index: int,
    entry_hour_index: int,
    mobile_day_off: int,
) -> None:
    setup = _build_setup()
    legacy_engine = WorkforceEngine(setup)
    compound_engine = CompoundWorkforceEngine(setup)
    residual = np.full((24, 28), 10, dtype=int)
    stock = np.array([2, 2, 2], dtype=int)
    legacy_state = WorkforceState(
        residual_demand=residual,
        remaining_stock=stock,
        expansion_mode=False,
        current_modality=None,
        current_entry_hour=None,
        assignment_week=0,
        initial_demand_total=int(residual.sum()),
    )
    compound_state = _build_state(
        residual_demand=residual,
        remaining_stock=stock,
    )

    legacy_result = _legacy_week_step(
        engine=legacy_engine,
        state=legacy_state,
        modality_index=modality_index,
        entry_hour_index=entry_hour_index,
        mobile_day_off=mobile_day_off,
    )
    compound_result = compound_engine.step(
        compound_state,
        encode_action(
            modality_index,
            entry_hour_index,
            mobile_day_off,
        ),
    )

    _assert_equivalent_results(legacy_result, compound_result)


def test_four_week_resource_matches_legacy_engine() -> None:
    setup = _build_setup()
    legacy_engine = WorkforceEngine(setup)
    compound_engine = CompoundWorkforceEngine(setup)
    residual = np.full((24, 28), 10, dtype=int)
    stock = np.array([2, 1, 1], dtype=int)
    legacy_state = WorkforceState(
        residual_demand=residual,
        remaining_stock=stock,
        expansion_mode=False,
        current_modality=None,
        current_entry_hour=None,
        assignment_week=0,
        initial_demand_total=int(residual.sum()),
    )
    compound_state = _build_state(
        residual_demand=residual,
        remaining_stock=stock,
    )
    weekly_actions = [
        (0, 0, 0),
        (0, 1, 2),
        (0, 2, 4),
        (0, 0, 5),
    ]

    for modality_index, entry_hour_index, mobile_day_off in weekly_actions:
        legacy_result = _legacy_week_step(
            engine=legacy_engine,
            state=legacy_state,
            modality_index=modality_index,
            entry_hour_index=entry_hour_index,
            mobile_day_off=mobile_day_off,
        )
        compound_result = compound_engine.step(
            compound_state,
            encode_action(
                modality_index,
                entry_hour_index,
                mobile_day_off,
            ),
        )
        _assert_equivalent_results(legacy_result, compound_result)
        legacy_state = legacy_result.next_state
        compound_state = compound_result.next_state

    assert np.array_equal(compound_state.remaining_stock, np.array([1, 1, 1]))
    assert compound_state.current_modality is None
    assert compound_state.assignment_week == 0


def _build_engine() -> CompoundWorkforceEngine:
    return CompoundWorkforceEngine(_build_setup())


def _build_setup() -> ProblemSetup:
    return ProblemSetup(
        mobile_days_off_count=1,
        fixed_day_off=6,
        allowed_entry_hours=[6, 12, 18],
        max_overcoverage_tolerance=0.1,
        closing_hour=22,
    )


def _build_state(
    residual_demand: np.ndarray | None = None,
    remaining_stock: np.ndarray | None = None,
    current_modality: int | None = None,
    assignment_week: int = 0,
) -> CompoundWorkforceState:
    residual = (
        np.ones((24, 28), dtype=int)
        if residual_demand is None
        else residual_demand
    )
    return CompoundWorkforceState(
        residual_demand=residual,
        remaining_stock=(
            np.array([1, 1, 1], dtype=int)
            if remaining_stock is None
            else remaining_stock
        ),
        expansion_mode=False,
        current_modality=current_modality,
        assignment_week=assignment_week,
        initial_demand_total=max(
            int(np.maximum(residual, 0).sum()),
            1,
        ),
    )


def _legacy_week_step(
    engine: WorkforceEngine,
    state: WorkforceState,
    modality_index: int,
    entry_hour_index: int,
    mobile_day_off: int,
):
    modality = (4, 6, 8)[modality_index]
    entry_hour = (6, 12, 18)[entry_hour_index]
    current_state = state

    if current_state.current_modality is None:
        modality_action = engine.encode_action(ActionType.MODALITY, modality)
        current_state = engine.step(current_state, modality_action).next_state

    entry_action = engine.encode_action(ActionType.ENTRY_HOUR, entry_hour)
    current_state = engine.step(current_state, entry_action).next_state
    day_off_action = engine.encode_action(
        ActionType.DAY_OFFS,
        (mobile_day_off, 6),
    )
    return engine.step(current_state, day_off_action)


def _assert_equivalent_results(legacy_result, compound_result) -> None:
    legacy = legacy_result.next_state
    compound = compound_result.next_state

    assert np.array_equal(legacy.residual_demand, compound.residual_demand)
    assert np.array_equal(legacy.remaining_stock, compound.remaining_stock)
    assert legacy.expansion_mode == compound.expansion_mode
    assert legacy.current_modality == compound.current_modality
    assert legacy.assignment_week == compound.assignment_week
    assert legacy.initial_demand_total == compound.initial_demand_total
    assert legacy_result.is_terminal == compound_result.is_terminal
    assert legacy_result.reward == compound_result.reward

import numpy as np

from modules.mcts.mcts import MCTS
from modules.mcts.mcts_schemas import MCTSConfig, MCTSMode
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine
from modules.workforce_engine.compound_schemas import CompoundWorkforceState
from modules.workforce_engine.schemas import ProblemSetup


class PreferredActionEvaluator:
    action_space_size = 54

    def __init__(self, preferred_action_id: int, value: float = 0.25) -> None:
        self.preferred_action_id = preferred_action_id
        self.value = value
        self.predict_calls = 0

    def predict(
        self,
        _state: CompoundWorkforceState,
    ) -> tuple[np.ndarray, float]:
        self.predict_calls += 1
        policy = np.zeros(self.action_space_size, dtype=float)
        policy[self.preferred_action_id] = 1.0
        return policy, self.value


def test_compound_mcts_generates_complete_four_week_trajectory() -> None:
    engine = CompoundWorkforceEngine(_build_setup())
    evaluator = PreferredActionEvaluator(preferred_action_id=0)
    mcts = MCTS(
        engine=engine,
        evaluator=evaluator,
        config=MCTSConfig(
            num_simulations=8,
            c_puct=1.5,
            mode=MCTSMode.INFERENCE,
        ),
    )
    state = _build_exact_four_week_state()
    samples: list[dict] = []
    final_reward = 0.0
    final_action_q = 0.0

    while True:
        result = mcts.search(state)
        action_id = int(result.selected_action_id)
        samples.append(
            {
                "state": state,
                "policy": result.policy.copy(),
                "action_id": action_id,
            }
        )

        assert result.policy.shape == (54,)
        assert np.isclose(result.policy.sum(), 1.0)
        assert action_id == 0

        step_result = engine.step(state, action_id)
        if step_result.is_terminal:
            final_action_q = float(result.root_stats[action_id]["q_value"])
        mcts.advance_root(action_id)
        state = step_result.next_state

        if step_result.is_terminal:
            final_reward = float(step_result.reward)
            break

    for sample in samples:
        sample["value"] = final_reward

    assert len(samples) == 4
    assert [sample["state"].assignment_week for sample in samples] == [0, 1, 2, 3]
    assert state.assignment_week == 0
    assert state.current_modality is None
    assert np.all(state.residual_demand <= 0)
    assert np.array_equal(state.remaining_stock, np.array([0, 0, 0]))
    assert np.isclose(final_reward, np.tanh(2.0))
    assert all(sample["value"] == final_reward for sample in samples)
    assert evaluator.predict_calls > 0
    assert final_action_q > evaluator.value


def _build_setup() -> ProblemSetup:
    return ProblemSetup(
        mobile_days_off_count=1,
        fixed_day_off=6,
        allowed_entry_hours=[6, 12, 18],
        max_overcoverage_tolerance=0.1,
        closing_hour=22,
    )


def _build_exact_four_week_state() -> CompoundWorkforceState:
    residual = np.zeros((24, 28), dtype=int)
    for week in range(4):
        week_start = week * 7
        residual[6:10, week_start + 1:week_start + 6] = 1

    return CompoundWorkforceState(
        residual_demand=residual,
        remaining_stock=np.array([1, 0, 0], dtype=int),
        expansion_mode=False,
        current_modality=None,
        assignment_week=0,
        initial_demand_total=int(residual.sum()),
    )

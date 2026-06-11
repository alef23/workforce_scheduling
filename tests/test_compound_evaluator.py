from pathlib import Path

import numpy as np
import pytest
import torch

from modules.evaluators.resnet.compound_evaluator import (
    CompoundResNetEvaluator,
)
from modules.evaluators.resnet.resnet_evaluator import WorkforceResNet
from modules.mcts.mcts import MCTS
from modules.mcts.mcts_schemas import MCTSConfig
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine
from modules.workforce_engine.compound_schemas import CompoundWorkforceState
from modules.workforce_engine.schemas import ProblemSetup


MODEL_CONFIG = {
    "input_channels": 11,
    "board_height": 28,
    "board_width": 28,
    "hidden_channels": 4,
    "num_res_blocks": 1,
    "action_space_size": 54,
    "policy_channels": 1,
    "value_channels": 1,
    "value_hidden_dim": 8,
}


def test_predict_returns_normalized_policy_and_value(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path, step=3)
    evaluator = CompoundResNetEvaluator(
        checkpoint_path=checkpoint_path,
        device="cpu",
    )

    policy, value = evaluator.predict(_build_state())

    assert evaluator.action_space_size == 54
    assert policy.shape == (54,)
    assert np.all(policy >= 0)
    assert np.isclose(policy.sum(), 1.0)
    assert -1.0 <= value <= 1.0


def test_predict_batch_handles_initial_and_active_states(tmp_path: Path) -> None:
    evaluator = CompoundResNetEvaluator(
        checkpoint_path=_save_checkpoint(tmp_path, step=1),
        device="cpu",
    )
    states = [
        _build_state(),
        _build_state(current_modality=6, assignment_week=2),
    ]

    policies, values = evaluator.predict_batch(states)

    assert policies.shape == (2, 54)
    assert values.shape == (2,)
    assert np.allclose(policies.sum(axis=1), 1.0)


def test_policy_can_be_filtered_by_compound_engine(tmp_path: Path) -> None:
    evaluator = CompoundResNetEvaluator(
        checkpoint_path=_save_checkpoint(tmp_path, step=1),
        device="cpu",
    )
    engine = CompoundWorkforceEngine(_build_setup())
    state = _build_state(remaining_stock=np.array([1, 0, 0]))

    policy, value = evaluator.predict(state)
    masked_policy = engine.legal_mask(state, policy)
    legal_actions = engine.get_legal_actions(state)

    assert np.isclose(masked_policy.sum(), 1.0)
    assert np.all(masked_policy[~legal_actions] == 0)
    assert np.all(masked_policy[legal_actions] > 0)
    assert -1.0 <= value <= 1.0


def test_evaluator_implements_mcts_contract(tmp_path: Path) -> None:
    evaluator = CompoundResNetEvaluator(
        checkpoint_path=_save_checkpoint(tmp_path, step=1),
        device="cpu",
    )
    engine = CompoundWorkforceEngine(_build_setup())
    mcts = MCTS(
        engine=engine,
        evaluator=evaluator,
        config=MCTSConfig(num_simulations=2, c_puct=1.0),
    )

    result = mcts.search(_build_state())

    assert result.policy.shape == (54,)
    assert np.isclose(result.policy.sum(), 1.0)
    assert engine.get_legal_actions(_build_state())[result.selected_action_id]


def test_reload_weights_returns_global_step(tmp_path: Path) -> None:
    first_path = _save_checkpoint(tmp_path, step=1)
    second_path = _save_checkpoint(tmp_path, step=7)
    evaluator = CompoundResNetEvaluator(
        checkpoint_path=first_path,
        device="cpu",
    )

    global_step = evaluator.reload_weights(second_path)

    assert global_step == 7
    assert evaluator.checkpoint_path == second_path


def test_checkpoint_directory_selects_highest_numeric_step(
    tmp_path: Path,
) -> None:
    _save_checkpoint(tmp_path, step=2)
    expected_path = _save_checkpoint(tmp_path, step=10)
    evaluator = CompoundResNetEvaluator(
        checkpoint_dir=tmp_path,
        device="cpu",
    )

    assert evaluator.checkpoint_path == expected_path


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_channels", 93, "input_channels=93"),
        ("action_space_size", 55, "action_space_size=55"),
    ],
)
def test_rejects_incompatible_checkpoint(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    model_config = {**MODEL_CONFIG, field: value}
    checkpoint_path = _save_checkpoint(
        tmp_path,
        step=1,
        model_config=model_config,
    )

    with pytest.raises(ValueError, match=message):
        CompoundResNetEvaluator(
            checkpoint_path=checkpoint_path,
            device="cpu",
        )


def _save_checkpoint(
    directory: Path,
    step: int,
    model_config: dict | None = None,
) -> Path:
    config = model_config or MODEL_CONFIG
    model = WorkforceResNet(**config)
    path = directory / f"workforce_resnet_compound_{step:06d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "training_state": {"global_step": step},
        },
        path,
    )
    return path


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
    demand = (
        residual_demand
        if residual_demand is not None
        else np.full((24, 28), 5, dtype=int)
    )
    stock = (
        remaining_stock
        if remaining_stock is not None
        else np.array([2, 2, 2], dtype=int)
    )
    return CompoundWorkforceState(
        residual_demand=demand,
        remaining_stock=stock,
        expansion_mode=bool(np.all(stock == 0)),
        current_modality=current_modality,
        assignment_week=assignment_week,
        initial_demand_total=int(np.maximum(demand, 0).sum()),
    )

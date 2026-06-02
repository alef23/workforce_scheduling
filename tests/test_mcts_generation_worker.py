from types import SimpleNamespace

import numpy as np

from modules.mcts.mcts_schemas import MCTSConfig
from modules.mcts_generation import (
    MCTSGenerationConfig,
    MCTSGenerationJob,
    MCTSGenerationWorker,
    MCTSStartMode,
    ReweightedPolicyConfig,
)


class _FakeTrajectoryBuffer:
    record = None

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def load(self, _trajectory_id):
        return self.record


class _DummyEvaluator:
    action_space_size = 55

    def predict(self, _state):
        policy = np.ones(55, dtype=np.float32) / 55
        return policy, 0.0


def test_worker_reweights_stock_trajectory_without_mcts(monkeypatch) -> None:
    from modules.mcts_generation import worker as worker_module

    state = {
        "residual_demand": np.ones((24, 28), dtype=int),
        "remaining_stock": np.array([1, 2, 3], dtype=int),
        "expansion_mode": False,
        "current_modality": 6,
        "current_entry_hour": None,
        "assignment_week": 0,
        "initial_demand_total": 672,
    }
    policy = np.zeros(55, dtype=np.float32)
    policy[[9, 15]] = 0.5
    _FakeTrajectoryBuffer.record = SimpleNamespace(
        trajectory_id="stock_raw_000000",
        problem_setup={
            "mobile_days_off_count": 0,
            "fixed_day_off": 5,
            "allowed_entry_hours": [6, 12],
            "max_overcoverage_tolerance": 0.1,
            "closing_hour": 22,
        },
        samples=[
            {
                "step_index": 0,
                "state": state,
                "policy": policy,
                "action_id": 15,
                "reward": 0.25,
            }
        ],
        final_reward=0.25,
    )
    monkeypatch.setattr(worker_module, "TrajectoryBuffer", _FakeTrajectoryBuffer)

    config = MCTSGenerationConfig(
        p_mcts=0.0,
        start_mode=MCTSStartMode.INITIAL_ONLY,
        max_seed_states=0,
        seed_state_probability=0.0,
        mcts_config=MCTSConfig(num_simulations=1, c_puct=1.0),
        reweighted_policy_config=ReweightedPolicyConfig(policy_weight=0.5),
    )
    worker = MCTSGenerationWorker(
        source_buffer_path="unused.zarr",
        config=config,
        evaluator=_DummyEvaluator(),
    )

    result = worker.run(
        MCTSGenerationJob(
            job_id="job_0",
            source_trajectory_id="stock_raw_000000",
            seed=123,
        )
    )

    assert not result.used_mcts
    assert len(result.trajectories) == 1

    trajectory = result.trajectories[0]
    sample = trajectory.trajectory[0]
    assert trajectory.trajectory_id == "reweighted_stock_raw_000000"
    assert sample["policy_weight"] == 0.5
    assert sample["metadata"]["sample_source"] == "stock_reweighted"
    assert np.isclose(sample["policy"].sum(), 1.0)
    assert np.isclose(sample["policy"][15], 1.0)
    assert np.isclose(sample["policy"][9], 0.0)


def test_worker_generates_mcts_trajectory(monkeypatch) -> None:
    from modules.mcts_generation import worker as worker_module

    state = {
        "residual_demand": np.ones((24, 28), dtype=int),
        "remaining_stock": np.array([1, 2, 3], dtype=int),
        "expansion_mode": False,
        "current_modality": None,
        "current_entry_hour": None,
        "assignment_week": 0,
        "initial_demand_total": 672,
    }
    policy = np.zeros(55, dtype=np.float32)
    policy[[0, 1, 2]] = 1 / 3
    _FakeTrajectoryBuffer.record = SimpleNamespace(
        trajectory_id="stock_raw_000000",
        problem_setup={
            "mobile_days_off_count": 0,
            "fixed_day_off": 5,
            "allowed_entry_hours": [6, 12],
            "max_overcoverage_tolerance": 0.1,
            "closing_hour": 22,
        },
        samples=[
            {
                "step_index": 0,
                "state": state,
                "policy": policy,
                "action_id": 1,
                "reward": 0.25,
            }
        ],
        final_reward=0.25,
    )

    def fake_generate_mcts_trajectory(initial_state, engine, mcts, debug=False):
        mcts_policy = np.zeros(55, dtype=np.float32)
        mcts_policy[1] = 1.0
        return (
            [
                {
                    "state": initial_state,
                    "policy": mcts_policy,
                    "action_id": 1,
                    "reward": 0.75,
                }
            ],
            0.75,
            initial_state,
        )

    monkeypatch.setattr(worker_module, "TrajectoryBuffer", _FakeTrajectoryBuffer)
    monkeypatch.setattr(
        worker_module,
        "generate_mcts_trajectory",
        fake_generate_mcts_trajectory,
    )

    config = MCTSGenerationConfig(
        p_mcts=1.0,
        start_mode=MCTSStartMode.INITIAL_ONLY,
        max_seed_states=0,
        seed_state_probability=0.0,
        mcts_config=MCTSConfig(num_simulations=1, c_puct=1.0),
        reweighted_policy_config=ReweightedPolicyConfig(policy_weight=0.5),
    )
    worker = MCTSGenerationWorker(
        source_buffer_path="unused.zarr",
        config=config,
        evaluator=_DummyEvaluator(),
    )

    result = worker.run(
        MCTSGenerationJob(
            job_id="job_0",
            source_trajectory_id="stock_raw_000000",
            seed=123,
        )
    )

    assert result.used_mcts
    assert len(result.trajectories) == 1
    sample = result.trajectories[0].trajectory[0]
    assert sample["policy_weight"] == 1.0
    assert sample["metadata"]["sample_source"] == "mcts"
    assert sample["metadata"]["source_step_index"] == 0
    assert sample["value"] == 0.75

from types import SimpleNamespace

import numpy as np

from modules.mcts.mcts_schemas import MCTSConfig
from modules.mcts_generation import (
    MCTSGenerationConfig,
    MCTSGenerationOrchestrator,
    MCTSOrchestratorConfig,
    MCTSStartMode,
    ReweightedPolicyConfig,
    build_mcts_generation_jobs,
)


class _DummyEvaluator:
    action_space_size = 55

    def predict(self, _state):
        return np.ones(55, dtype=np.float32) / 55, 0.0


def test_orchestrator_writes_reweighted_samples(monkeypatch, tmp_path) -> None:
    from modules.mcts_generation import worker as worker_module

    class FakeTrajectoryBuffer:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def load(self, _trajectory_id):
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
            return SimpleNamespace(
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

    monkeypatch.setattr(worker_module, "TrajectoryBuffer", FakeTrajectoryBuffer)

    generation_config = MCTSGenerationConfig(
        p_mcts=0.0,
        start_mode=MCTSStartMode.INITIAL_ONLY,
        max_seed_states=0,
        seed_state_probability=0.0,
        mcts_config=MCTSConfig(num_simulations=1, c_puct=1.0),
        reweighted_policy_config=ReweightedPolicyConfig(policy_weight=0.5),
    )
    orchestrator = MCTSGenerationOrchestrator(
        config=MCTSOrchestratorConfig(
            source_buffer_path="unused.zarr",
            sample_buffer_path=tmp_path / "samples.zarr",
            overwrite_samples=True,
            print_progress=False,
        ),
        generation_config=generation_config,
        evaluator=_DummyEvaluator(),
    )
    jobs = build_mcts_generation_jobs(["stock_raw_000000"], seed=123)

    report = orchestrator.run(jobs)

    assert report.completed_jobs == 1
    assert report.failed_jobs == 0
    assert report.saved_samples == 1
    assert report.reweighted_jobs == 1

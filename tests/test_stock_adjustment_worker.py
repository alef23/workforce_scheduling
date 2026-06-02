from types import SimpleNamespace

import numpy as np

from modules.dataset_generation import StockAdjustmentConfig, StockAdjustmentTrajectoryWorker
from modules.dataset_generation.schemas import GenerationJob


class _FakeTrajectoryBuffer:
    record = None

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def load(self, _trajectory_id):
        return self.record


def test_stock_worker_copies_raw_when_stock_not_reduced(monkeypatch) -> None:
    from modules.dataset_generation import stock_worker as worker_module

    _FakeTrajectoryBuffer.record = _record_with_samples(
        actions=[0, 9, 9, 9, 9, 1, 15, 15, 15, 15],
        final_reward=0.25,
    )
    monkeypatch.setattr(worker_module, "TrajectoryBuffer", _FakeTrajectoryBuffer)

    def fail_replay(*_args, **_kwargs):
        raise AssertionError("No deberia replayear si no reduce stock.")

    monkeypatch.setattr(worker_module, "replay_actions_as_trajectory", fail_replay)

    worker = StockAdjustmentTrajectoryWorker(
        source_buffer_path="unused.zarr",
        config=StockAdjustmentConfig(p_stock=0.0),
    )

    result = worker.run(_job(seed=123))

    assert result.metadata["stock_was_reduced"] is False
    assert result.metadata["output_stock"] == [1, 1, 0]
    assert result.metadata["final_reward"] == 0.25
    assert result.trajectories[0].trajectory_id == "stock_raw_000000"
    assert result.trajectories[0].trajectory[0]["action_id"] == 0


def test_stock_worker_reduces_stock_from_chunk_cut_without_shuffle(monkeypatch) -> None:
    from modules.dataset_generation import stock_worker as worker_module

    _FakeTrajectoryBuffer.record = _record_with_samples(
        actions=[0, 9, 9, 9, 9, 1, 15, 15, 15, 15, 2, 21, 21, 21, 21],
        final_reward=0.5,
    )
    monkeypatch.setattr(worker_module, "TrajectoryBuffer", _FakeTrajectoryBuffer)

    replay_calls = []

    def fake_replay(initial_demand, initial_stock, actions, engine, require_terminal):
        replay_calls.append(
            {
                "initial_stock": list(initial_stock),
                "actions": list(actions),
                "require_terminal": bool(require_terminal),
            }
        )
        return {
            "trajectory": [
                {
                    "state": SimpleNamespace(expansion_mode=False),
                    "policy": np.ones(55, dtype=np.float32) / 55,
                    "action_id": int(actions[0]),
                    "reward": 0.75,
                },
                {
                    "state": SimpleNamespace(expansion_mode=True),
                    "policy": np.ones(55, dtype=np.float32) / 55,
                    "action_id": int(actions[-1]),
                    "reward": 0.75,
                },
            ],
            "final_reward": 0.75,
        }

    monkeypatch.setattr(worker_module, "replay_actions_as_trajectory", fake_replay)

    worker = StockAdjustmentTrajectoryWorker(
        source_buffer_path="unused.zarr",
        config=StockAdjustmentConfig(p_stock=1.0),
    )

    result = worker.run(_job(seed=1))

    assert result.metadata["stock_was_reduced"] is True
    assert result.metadata["stock_cut_index"] in (1, 2)
    assert result.metadata["has_expansion_mode"] is True
    assert result.metadata["first_expansion_step"] == 1
    assert replay_calls
    assert replay_calls[0]["actions"] == [
        0, 9, 9, 9, 9,
        1, 15, 15, 15, 15,
        2, 21, 21, 21, 21,
    ]
    assert sum(replay_calls[0]["initial_stock"]) == result.metadata["stock_cut_index"]


def test_expansion_metadata_supports_dict_and_object_states() -> None:
    trajectory = [
        {"state": {"expansion_mode": False}},
        {"state": SimpleNamespace(expansion_mode=True)},
    ]

    assert StockAdjustmentTrajectoryWorker._expansion_metadata(trajectory) == (True, 1)


def _job(seed: int) -> GenerationJob:
    return GenerationJob(
        job_id="stock_000000",
        seed=seed,
        payload={"source_trajectory_id": "raw_000000"},
    )


def _record_with_samples(actions: list[int], final_reward: float):
    state = {
        "residual_demand": np.ones((24, 28), dtype=np.int32),
        "remaining_stock": np.array([1, 1, 0], dtype=np.int32),
        "expansion_mode": False,
        "current_modality": None,
        "current_entry_hour": None,
        "assignment_week": 0,
        "initial_demand_total": 672,
    }
    policy = np.ones(55, dtype=np.float32) / 55
    return SimpleNamespace(
        trajectory_id="raw_000000",
        problem_setup={
            "mobile_days_off_count": 0,
            "fixed_day_off": 5,
            "allowed_entry_hours": [6, 12, 18],
            "max_overcoverage_tolerance": 0.1,
            "closing_hour": 22,
        },
        samples=[
            {
                "step_index": index,
                "state": state,
                "policy": policy,
                "action_id": action_id,
                "reward": final_reward,
            }
            for index, action_id in enumerate(actions)
        ],
        final_reward=final_reward,
    )

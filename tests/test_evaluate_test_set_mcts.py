from pathlib import Path

import numpy as np

from modules.workforce_engine.schemas import WorkforceState
from scripts.evaluate_test_set_mcts import (
    build_run_summary,
    parse_checkpoint_step,
    resolve_checkpoint_path,
    write_run_summary,
    with_terminal_state,
)


def test_parse_checkpoint_step_from_filename() -> None:
    assert parse_checkpoint_step("workforce_resnet_000320.pt") == 320
    assert parse_checkpoint_step("model.pt") == 0


def test_resolve_checkpoint_path_uses_largest_numeric_step(tmp_path: Path) -> None:
    older = tmp_path / "workforce_resnet_000020.pt"
    newer = tmp_path / "workforce_resnet_000120.pt"
    non_numeric = tmp_path / "workforce_resnet_latest.pt"
    older.write_bytes(b"")
    newer.write_bytes(b"")
    non_numeric.write_bytes(b"")

    assert resolve_checkpoint_path(None, tmp_path) == newer


def test_with_terminal_state_appends_zero_policy_terminal_sample() -> None:
    state = WorkforceState(
        residual_demand=np.zeros((24, 28), dtype=np.int32),
        remaining_stock=np.array([0, 0, 0], dtype=np.int32),
        expansion_mode=False,
        current_modality=None,
        current_entry_hour=None,
        assignment_week=0,
        initial_demand_total=10,
    )
    trajectory = [
        {
            "state": state,
            "policy": np.ones((55,), dtype=np.float32) / 55,
            "action_id": 0,
            "reward": -0.5,
        }
    ]

    output = with_terminal_state(trajectory, state, -0.25)

    assert len(output) == 2
    assert output[0]["reward"] == -0.25
    assert output[0]["metadata"]["is_terminal"] is False
    assert output[1]["action_id"] == -1
    assert output[1]["reward"] == -0.25
    assert output[1]["metadata"]["is_terminal"] is True
    assert np.all(output[1]["policy"] == 0)


def test_build_run_summary_counts_positive_and_better() -> None:
    class Result:
        def __init__(self, metrics):
            self.metrics = metrics

    results = [
        Result(
            {
                "final_reward": 0.5,
                "original_value": 0.25,
                "value_error": 0.25,
                "is_better_than_original": True,
            }
        ),
        Result(
            {
                "final_reward": -0.1,
                "original_value": 0.0,
                "value_error": -0.1,
                "is_better_than_original": False,
            }
        ),
    ]

    summary = build_run_summary(
        run_id="run",
        sample_path=Path("samples.zarr"),
        trajectory_path=Path("trajectories.zarr"),
        checkpoint_path=Path("checkpoint.pt"),
        checkpoint_step=10,
        mcts_simulations=500,
        n_workers=2,
        total_jobs=2,
        results=results,
        errors=[],
        elapsed_seconds=1.5,
        saved_trajectories=2,
    )

    assert summary["completed_jobs"] == 2
    assert summary["saved_trajectories"] == 2
    assert summary["positive_count"] == 1
    assert summary["better_than_original_count"] == 1
    assert summary["worse_than_original_count"] == 1
    assert abs(summary["mean_value_error"] - 0.075) < 1e-12


def test_write_run_summary_writes_latest_and_history(tmp_path: Path) -> None:
    summary = {
        "run_id": "run_1",
        "completed_jobs": 10,
        "positive_count": 3,
    }

    write_run_summary(summary, tmp_path)
    write_run_summary({**summary, "run_id": "run_2"}, tmp_path)

    assert (tmp_path / "run_summary.json").exists()
    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"run_id": "run_1"' in lines[0]
    assert '"run_id": "run_2"' in lines[1]

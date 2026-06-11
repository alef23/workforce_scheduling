from pathlib import Path

import numpy as np
import pytest

from modules.dataset_generation import (
    CompoundDatasetOrchestrator,
    CompoundFullTrajectoryWorker,
    CompoundGenerationJob,
    CompoundOrchestratorConfig,
    NoiseGenerationConfig,
    build_compound_generation_jobs,
)
from modules.storage import CompoundTrajectoryBuffer
from modules.workforce_engine.schemas import ProblemSetup


def _build_worker(
    n_resources: int,
    p_stock: float,
) -> CompoundFullTrajectoryWorker:
    return CompoundFullTrajectoryWorker(
        problem_setup=ProblemSetup(
            mobile_days_off_count=1,
            fixed_day_off=6,
            allowed_entry_hours=[6, 12, 18],
            max_overcoverage_tolerance=0.1,
            closing_hour=22,
        ),
        n_resources=n_resources,
        p_stock=p_stock,
        noise_config=NoiseGenerationConfig(
            k_max=0.8,
            max_daily_peaks_min=1,
            max_daily_peaks_max=2,
            max_hourly_peaks_min=1,
            max_hourly_peaks_max=2,
        ),
    )


def test_compound_buffer_round_trip(tmp_path: Path) -> None:
    worker = _build_worker(n_resources=4, p_stock=0.0)
    result = worker.run(CompoundGenerationJob(job_id="buffer"))

    path = tmp_path / "compound_trajectories.zarr"
    writer = CompoundTrajectoryBuffer(path, mode="w")
    generated = result.generated
    writer.save(
        trajectory=generated.trajectory,
        problem_setup=generated.problem_setup,
        trajectory_id=generated.trajectory_id,
        metadata=generated.metadata,
    )

    reader = CompoundTrajectoryBuffer(path, mode="r")
    loaded = reader.load(generated.trajectory_id)
    original = generated.trajectory

    assert loaded.trajectory_id == generated.trajectory_id
    assert len(loaded.samples) == len(original)
    assert loaded.problem_setup["allowed_entry_hours"] == [6, 12, 18]
    assert (
        loaded.attrs["metadata.pipeline"]
        == "compound_raw_noise_stock"
    )
    assert loaded.final_reward == pytest.approx(original[-1]["reward"])

    for loaded_sample, original_sample in zip(loaded.samples, original):
        loaded_state = loaded_sample["state"]
        original_state = original_sample["state"]
        np.testing.assert_array_equal(
            loaded_state["residual_demand"],
            original_state.residual_demand,
        )
        np.testing.assert_array_equal(
            loaded_state["remaining_stock"],
            original_state.remaining_stock,
        )
        np.testing.assert_allclose(
            loaded_sample["policy"],
            original_sample["policy"],
        )
        assert loaded_sample["action_id"] == original_sample["action_id"]
        assert loaded_sample["reward"] == pytest.approx(
            original_sample["reward"]
        )


def test_full_worker_generates_final_stock_adjusted_trajectory() -> None:
    worker = _build_worker(n_resources=6, p_stock=1.0)
    result = worker.run(CompoundGenerationJob(job_id="worker"))

    generated = result.generated
    metadata = generated.metadata
    trajectory = generated.trajectory

    assert metadata["pipeline"] == "compound_raw_noise_stock"
    assert 1 <= metadata["n_resources"] <= 6
    assert metadata["max_n_resources"] == 6
    assert metadata["stock_was_reduced"] is True
    assert len(metadata["original_stock"]) == 3
    assert len(metadata["output_stock"]) == 3
    assert sum(metadata["original_stock"]) == metadata["n_resources"]
    assert sum(metadata["output_stock"]) < metadata["n_resources"]
    assert metadata["trajectory_length"] == len(trajectory)
    assert all(sample["policy"].shape == (54,) for sample in trajectory)
    assert all(sample["reward"] is not None for sample in trajectory)


def test_orchestrator_runs_workers_and_writes_one_buffer(tmp_path: Path) -> None:
    output_path = tmp_path / "compound_dataset.zarr"
    orchestrator = CompoundDatasetOrchestrator(
        worker=_build_worker(n_resources=3, p_stock=0.5),
        config=CompoundOrchestratorConfig(
            n_workers=2,
            output_path=output_path,
            overwrite=True,
            print_progress=False,
        ),
    )

    report = orchestrator.run(build_compound_generation_jobs(4))

    assert report.completed_jobs == 4
    assert report.failed_jobs == 0
    assert report.saved_trajectories == 4

    buffer = CompoundTrajectoryBuffer(output_path, mode="r")
    trajectory_ids = buffer.list_ids()

    assert len(trajectory_ids) == 4
    assert len(set(trajectory_ids)) == 4
    assert "mean_final_reward" in report.stats
    assert "mean_initial_demand_total" in report.stats
    assert "mean_input_resources" in report.stats

import argparse
import json
from pathlib import Path

from modules.learning import ResNetLearnerReport, ResNetTrainStepMetrics
from modules.mcts_generation import MCTSCycleReport, MCTSOrchestratorReport
from scripts.generate_mcts_samples import (
    MCTSGenerationRunLogger,
    parse_checkpoint_step,
    resolve_checkpoint_path,
)


def test_parse_checkpoint_step_from_filename() -> None:
    assert parse_checkpoint_step("workforce_resnet_000338.pt") == 338
    assert parse_checkpoint_step("model.pt") == -1


def test_resolve_checkpoint_path_uses_largest_numeric_step(tmp_path: Path) -> None:
    initial = tmp_path / "workforce_resnet_000.pt"
    lower = tmp_path / "workforce_resnet_000201.pt"
    latest = tmp_path / "workforce_resnet_000338.pt"
    initial.write_bytes(b"")
    lower.write_bytes(b"")
    latest.write_bytes(b"")

    assert resolve_checkpoint_path(None, tmp_path) == latest


def test_resolve_checkpoint_path_prefers_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "manual.pt"
    latest = tmp_path / "workforce_resnet_000338.pt"
    explicit.write_bytes(b"")
    latest.write_bytes(b"")

    assert resolve_checkpoint_path(explicit, tmp_path) == explicit


def test_mcts_generation_run_logger_writes_jsonl(tmp_path) -> None:
    logger = MCTSGenerationRunLogger(
        reports_dir=tmp_path,
        run_id="run_test",
    )
    cycle_report = MCTSCycleReport(
        cycle_index=0,
        completed_jobs=2,
        failed_jobs=0,
        saved_samples=350,
        generated_trajectories=2,
        used_mcts_jobs=1,
        reweighted_jobs=1,
        sample_start_index=0,
        sample_end_index=350,
    )
    learner_report = ResNetLearnerReport(
        checkpoint_path="checkpoints/workforce_resnet_000001.pt",
        global_step=1,
        trained_steps=1,
        sample_count=350,
        sample_start_index=0,
        sample_end_index=350,
        last_batch_size=350,
        metrics=[
            ResNetTrainStepMetrics(
                step=1,
                global_step=1,
                loss=1.0,
                policy_loss=0.8,
                value_loss=0.2,
                mean_policy_weight=0.75,
            )
        ],
    )
    orchestrator_report = MCTSOrchestratorReport(
        source_buffer_path="stock.zarr",
        sample_buffer_path="samples.zarr",
        total_jobs=2,
        completed_jobs=2,
        failed_jobs=0,
        generated_trajectories=2,
        saved_samples=350,
        used_mcts_jobs=1,
        reweighted_jobs=1,
        cycle_reports=[cycle_report],
        errors=[],
    )

    logger.log_cycle(cycle_report=cycle_report, learner_report=learner_report)
    logger.log_learner_steps(cycle_index=0, learner_report=learner_report)
    logger.log_run(
        status="completed",
        args=argparse.Namespace(workers=2, p_mcts=0.2),
        source_path="stock.zarr",
        sample_path="samples.zarr",
        source_trajectory_count=10,
        selected_trajectory_count=2,
        report=orchestrator_report,
    )

    cycle_lines = _read_jsonl(tmp_path / logger.cycles_filename)
    learner_lines = _read_jsonl(tmp_path / logger.learner_steps_filename)
    run_lines = _read_jsonl(tmp_path / logger.runs_filename)

    assert cycle_lines[0]["run_id"] == "run_test"
    assert cycle_lines[0]["cycle_id"] == "run_test_cycle_000"
    assert cycle_lines[0]["cycle"]["saved_samples"] == 350
    assert cycle_lines[0]["cycle"]["sample_start_index"] == 0
    assert cycle_lines[0]["cycle"]["sample_end_index"] == 350
    assert cycle_lines[0]["learner"]["checkpoint_path"].endswith("000001.pt")
    assert cycle_lines[0]["learner"]["last_metric"]["loss"] == 1.0

    assert learner_lines[0]["run_id"] == "run_test"
    assert learner_lines[0]["cycle_id"] == "run_test_cycle_000"
    assert learner_lines[0]["cycle_index"] == 0
    assert learner_lines[0]["sample_start_index"] == 0
    assert learner_lines[0]["sample_end_index"] == 350
    assert learner_lines[0]["metric"]["policy_loss"] == 0.8

    assert run_lines[0]["run_id"] == "run_test"
    assert run_lines[0]["status"] == "completed"
    assert run_lines[0]["args"]["workers"] == 2
    assert run_lines[0]["report"]["cycle_count"] == 1


def test_mcts_generation_run_logger_uses_correlative_ids(tmp_path) -> None:
    first = MCTSGenerationRunLogger(
        reports_dir=tmp_path,
        run_prefix="train_gpu_mid",
    )
    second = MCTSGenerationRunLogger(
        reports_dir=tmp_path,
        run_prefix="train_gpu_mid",
    )
    other_prefix = MCTSGenerationRunLogger(
        reports_dir=tmp_path,
        run_prefix="train_gpu_advanced",
    )

    assert first.run_id == "train_gpu_mid_001"
    assert second.run_id == "train_gpu_mid_002"
    assert other_prefix.run_id == "train_gpu_advanced_001"

    sequences = json.loads(
        (tmp_path / MCTSGenerationRunLogger.sequences_filename).read_text(
            encoding="utf-8"
        )
    )
    assert sequences == {
        "train_gpu_advanced": 1,
        "train_gpu_mid": 2,
    }


def test_manual_run_id_does_not_consume_sequence(tmp_path) -> None:
    manual = MCTSGenerationRunLogger(
        reports_dir=tmp_path,
        run_id="manual_run",
        run_prefix="train_gpu_mid",
    )
    automatic = MCTSGenerationRunLogger(
        reports_dir=tmp_path,
        run_prefix="train_gpu_mid",
    )

    assert manual.run_id == "manual_run"
    assert automatic.run_id == "train_gpu_mid_001"


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

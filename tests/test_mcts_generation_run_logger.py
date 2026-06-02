import argparse
import json

from modules.learning import ResNetLearnerReport, ResNetTrainStepMetrics
from modules.mcts_generation import MCTSCycleReport, MCTSOrchestratorReport
from scripts.generate_mcts_samples import MCTSGenerationRunLogger


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
    )
    learner_report = ResNetLearnerReport(
        checkpoint_path="checkpoints/workforce_resnet_000001.pt",
        global_step=1,
        trained_steps=1,
        sample_count=350,
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
    assert cycle_lines[0]["cycle"]["saved_samples"] == 350
    assert cycle_lines[0]["learner"]["checkpoint_path"].endswith("000001.pt")
    assert cycle_lines[0]["learner"]["last_metric"]["loss"] == 1.0

    assert learner_lines[0]["run_id"] == "run_test"
    assert learner_lines[0]["cycle_index"] == 0
    assert learner_lines[0]["metric"]["policy_loss"] == 0.8

    assert run_lines[0]["run_id"] == "run_test"
    assert run_lines[0]["status"] == "completed"
    assert run_lines[0]["args"]["workers"] == 2
    assert run_lines[0]["report"]["cycle_count"] == 1


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

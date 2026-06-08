from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.evaluators.centralized import (
    CentralizedEvaluatorClient,
    CentralizedEvaluatorConfig,
    CentralizedEvaluatorServer,
)
from modules.evaluators.resnet.resnet_state_evaluator import ResNetStateEvaluator
from modules.mcts.mcts import MCTS
from modules.mcts.mcts_schemas import MCTSConfig, MCTSMode
from modules.storage import SampleBuffer, TrajectoryBuffer
from modules.trajectory_generation import generate_mcts_trajectory
from modules.workforce_engine.engine import WorkforceEngine
from modules.workforce_engine.schemas import ProblemSetup, WorkforceState


ACTION_SPACE_SIZE = 55


@dataclass(frozen=True)
class EvaluationJob:
    job_id: str
    sample_index: int
    seed: int
    source_trajectory_id: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    job_id: str
    trajectory_id: str
    trajectory: list[dict[str, Any]]
    problem_setup: ProblemSetup
    metadata: dict[str, Any]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class WorkerError:
    job_id: str
    error: str


class EvaluationResultWriter:
    def __init__(
        self,
        trajectory_path: Path,
        reports_dir: Path,
        overwrite: bool,
    ) -> None:
        self.trajectory_path = Path(trajectory_path)
        self.reports_dir = Path(reports_dir)
        self.overwrite = bool(overwrite)
        self.trajectory_buffer: TrajectoryBuffer | None = None
        self.metrics_file = None
        self.saved_count = 0

    def __enter__(self) -> "EvaluationResultWriter":
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_buffer = TrajectoryBuffer(
            self.trajectory_path,
            mode="w" if self.overwrite else "a",
        )
        self.metrics_file = (self.reports_dir / "trajectories.jsonl").open(
            "w" if self.overwrite else "a",
            encoding="utf-8",
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.metrics_file is not None:
            self.metrics_file.close()

    def write(self, result: EvaluationResult) -> None:
        if self.trajectory_buffer is None or self.metrics_file is None:
            raise RuntimeError("EvaluationResultWriter no esta abierto.")

        self.trajectory_buffer.save(
            trajectory=result.trajectory,
            problem_setup=result.problem_setup,
            trajectory_id=result.trajectory_id,
            action_space_size=ACTION_SPACE_SIZE,
            metadata=result.metadata,
        )
        self.metrics_file.write(
            json.dumps(to_jsonable(result.metrics), sort_keys=True) + "\n"
        )
        self.metrics_file.flush()
        self.saved_count += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua el test set fijo generando trayectorias MCTS."
    )
    parser.add_argument(
        "--input-mode",
        choices=("initial", "partial"),
        default="initial",
        help=(
            "Origen de evaluacion: SampleBuffer inicial o TrajectoryBuffer parcial. "
            "Default: initial."
        ),
    )
    parser.add_argument(
        "--sample-path",
        default="datasets/test/initial_states.zarr",
        help="SampleBuffer de estados iniciales. Default: datasets/test/initial_states.zarr.",
    )
    parser.add_argument(
        "--partial-trajectory-path",
        default="datasets/test/partial_trajectories.zarr",
        help=(
            "TrajectoryBuffer fuente para --input-mode partial. "
            "Default: datasets/test/partial_trajectories.zarr."
        ),
    )
    parser.add_argument(
        "--tail-states",
        type=int,
        default=None,
        help=(
            "Cantidad de estados contados desde el final para elegir el inicio. "
            "Requerido con --input-mode partial."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Directorio raiz de salida. Defaults: datasets/evaluation/mcts_test "
            "o datasets/evaluation/mcts_partial."
        ),
    )
    parser.add_argument(
        "--trajectory-path",
        default=None,
        help="TrajectoryBuffer destino. Default: <output-root>/trajectories.zarr.",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directorio de reportes JSON. Default: <output-root>/reports.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Checkpoint ResNet a evaluar. Default: ultimo .pt por step numerico.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="modules/evaluators/resnet/checkpoints",
        help="Directorio donde buscar checkpoints si no se pasa --checkpoint-path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Cantidad de workers MCTS. Default: 1.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Cantidad opcional de samples del test set a evaluar. Default: todos.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Mezcla indices del test set antes de aplicar --n-samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opcional para ordenar jobs y MCTS. Default: None.",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=500,
        help="Simulaciones MCTS por decision. Default: 500.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.0,
        help="Constante de exploracion PUCT. Default: 1.0.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device del evaluator: auto, cpu o cuda. Default: auto.",
    )
    parser.add_argument(
        "--evaluator-batch-size",
        type=int,
        default=32,
        help="Batch maximo del evaluator centralizado. Default: 32.",
    )
    parser.add_argument(
        "--evaluator-batch-wait",
        type=float,
        default=0.01,
        help="Espera maxima para armar batch de inferencia. Default: 0.01.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Timeout opcional de requests al evaluator. Default: None.",
    )
    parser.add_argument(
        "--multiprocessing-start-method",
        default="fork",
        help="Metodo multiprocessing para workers>1. Default: fork.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recrea el buffer destino y reportes.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1,
        help="Frecuencia de impresion de progreso. Default: 1 job.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="ID opcional de corrida. Default: generado automaticamente.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.workers) <= 0:
        raise ValueError("--workers debe ser positivo.")
    if int(args.progress_interval) <= 0:
        raise ValueError("--progress-interval debe ser positivo.")
    if args.input_mode == "partial":
        if args.tail_states is None or int(args.tail_states) <= 0:
            raise ValueError(
                "--tail-states debe ser positivo con --input-mode partial."
            )
    elif args.tail_states is not None:
        raise ValueError("--tail-states solo se usa con --input-mode partial.")

    sample_path = (
        Path(args.partial_trajectory_path)
        if args.input_mode == "partial"
        else Path(args.sample_path)
    )
    output_root = Path(
        args.output_root
        or (
            "datasets/evaluation/mcts_partial"
            if args.input_mode == "partial"
            else "datasets/evaluation/mcts_test"
        )
    )
    trajectory_path = (
        Path(args.trajectory_path)
        if args.trajectory_path is not None
        else output_root / "trajectories.zarr"
    )
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir is not None
        else output_root / "reports"
    )
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=args.checkpoint_path,
        checkpoint_dir=args.checkpoint_dir,
    )
    checkpoint_step = parse_checkpoint_step(checkpoint_path)
    run_id = (
        args.run_id
        or (
            f"{'partial_mcts' if args.input_mode == 'partial' else 'test_mcts'}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
    )

    jobs = build_evaluation_jobs(
        sample_path=sample_path,
        input_mode=args.input_mode,
        n_samples=args.n_samples,
        shuffle=bool(args.shuffle),
        seed=args.seed,
    )

    config = _RunConfig(
        sample_path=sample_path,
        input_mode=args.input_mode,
        tail_states=args.tail_states,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
        run_id=run_id,
        mcts_config=MCTSConfig(
            num_simulations=int(args.mcts_simulations),
            c_puct=float(args.c_puct),
            mode=MCTSMode.INFERENCE,
        ),
    )

    started_at = time.monotonic()
    if int(args.workers) > 1:
        with EvaluationResultWriter(
            trajectory_path=trajectory_path,
            reports_dir=reports_dir,
            overwrite=bool(args.overwrite),
        ) as writer:
            results, errors = run_multiprocess(
                jobs=jobs,
                run_config=config,
                n_workers=int(args.workers),
                device=args.device,
                evaluator_batch_size=int(args.evaluator_batch_size),
                evaluator_batch_wait=float(args.evaluator_batch_wait),
                request_timeout=args.request_timeout,
                multiprocessing_start_method=args.multiprocessing_start_method,
                progress_interval=int(args.progress_interval),
                writer=writer,
            )
            saved_trajectories = writer.saved_count
    else:
        with EvaluationResultWriter(
            trajectory_path=trajectory_path,
            reports_dir=reports_dir,
            overwrite=bool(args.overwrite),
        ) as writer:
            results, errors = run_sequential(
                jobs=jobs,
                run_config=config,
                device=args.device,
                progress_interval=int(args.progress_interval),
                writer=writer,
            )
            saved_trajectories = writer.saved_count
    elapsed_seconds = time.monotonic() - started_at
    summary = build_run_summary(
        run_id=run_id,
        sample_path=sample_path,
        trajectory_path=trajectory_path,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
        mcts_simulations=int(args.mcts_simulations),
        n_workers=int(args.workers),
        total_jobs=len(jobs),
        results=results,
        errors=errors,
        elapsed_seconds=elapsed_seconds,
        saved_trajectories=saved_trajectories,
        input_mode=args.input_mode,
        tail_states=args.tail_states,
    )
    write_run_summary(summary=summary, reports_dir=reports_dir)
    print_summary(summary)


@dataclass(frozen=True)
class _RunConfig:
    sample_path: Path
    input_mode: str
    tail_states: int | None
    checkpoint_path: Path
    checkpoint_step: int
    run_id: str
    mcts_config: MCTSConfig


def resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    checkpoint_dir: str | Path,
) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el checkpoint: {path}")
        return path

    candidates = list(Path(checkpoint_dir).glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No se encontraron checkpoints .pt en {checkpoint_dir}.")

    return max(
        candidates,
        key=lambda path: (parse_checkpoint_step(path), path.stat().st_mtime),
    )


def parse_checkpoint_step(path: str | Path) -> int:
    match = re.search(r"_(\d+)\.pt$", Path(path).name)
    if match is None:
        return 0
    return int(match.group(1))


def build_evaluation_jobs(
    sample_path: str | Path,
    n_samples: int | None,
    shuffle: bool,
    seed: int | None,
    input_mode: str = "initial",
) -> list[EvaluationJob]:
    if input_mode == "partial":
        source_ids = TrajectoryBuffer(sample_path, mode="r").list_ids()
    else:
        sample_count = len(SampleBuffer(sample_path, mode="r"))
        source_ids = [None] * sample_count

    if not source_ids:
        raise ValueError(f"El test set esta vacio: {sample_path}")

    indices = list(range(len(source_ids)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    if n_samples is not None:
        indices = indices[: int(n_samples)]

    seed_base = (
        int(seed)
        if seed is not None
        else int(time.time_ns() % (2**31 - 1))
    )
    return [
        EvaluationJob(
            job_id=f"test_mcts_{job_index:06d}",
            sample_index=int(sample_index),
            seed=int(seed_base + job_index),
            source_trajectory_id=source_ids[sample_index],
        )
        for job_index, sample_index in enumerate(indices)
    ]


def run_sequential(
    jobs: list[EvaluationJob],
    run_config: _RunConfig,
    device: str,
    progress_interval: int,
    writer: EvaluationResultWriter,
) -> tuple[list[EvaluationResult], list[str]]:
    if not jobs:
        return [], []

    first_case = load_test_case(run_config, jobs[0])
    evaluator = ResNetStateEvaluator(
        setup=first_case["problem_setup"],
        checkpoint_path=run_config.checkpoint_path,
        device=device,
    )
    results: list[EvaluationResult] = []
    errors: list[str] = []
    started_at = time.monotonic()

    for job in jobs:
        try:
            result = run_job(job=job, run_config=run_config, evaluator=evaluator)
            writer.write(result)
            print_job_result(result)
            results.append(result)
        except Exception as exc:
            errors.append(f"{job.job_id}: {exc}")
        print_progress(
            processed=len(results) + len(errors),
            total=len(jobs),
            ok=len(results),
            failed=len(errors),
            started_at=started_at,
            progress_interval=progress_interval,
        )

    return results, errors


def run_multiprocess(
    jobs: list[EvaluationJob],
    run_config: _RunConfig,
    n_workers: int,
    device: str,
    evaluator_batch_size: int,
    evaluator_batch_wait: float,
    request_timeout: float | None,
    multiprocessing_start_method: str,
    progress_interval: int,
    writer: EvaluationResultWriter,
) -> tuple[list[EvaluationResult], list[str]]:
    ctx = mp.get_context(multiprocessing_start_method)
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()
    evaluator_request_queue = ctx.Queue()
    response_queues = {f"worker_{index}": ctx.Queue() for index in range(n_workers)}
    response_queues["orchestrator"] = ctx.Queue()

    evaluator_config = CentralizedEvaluatorConfig(
        checkpoint_path=run_config.checkpoint_path,
        device=device,
        max_batch_size=evaluator_batch_size,
        batch_wait_s=evaluator_batch_wait,
        request_timeout_s=request_timeout,
    )
    evaluator_process = ctx.Process(
        target=_run_evaluator_server_process,
        args=(evaluator_config, evaluator_request_queue, response_queues),
        daemon=True,
    )
    evaluator_process.start()

    worker_processes = []
    for worker_index in range(n_workers):
        client_id = f"worker_{worker_index}"
        process = ctx.Process(
            target=_run_worker_process,
            args=(
                client_id,
                run_config,
                evaluator_request_queue,
                response_queues[client_id],
                job_queue,
                result_queue,
                request_timeout,
            ),
            daemon=True,
        )
        process.start()
        worker_processes.append(process)

    try:
        return collect_multiprocess_results(
            jobs=jobs,
            job_queue=job_queue,
            result_queue=result_queue,
            worker_processes=worker_processes,
            progress_interval=progress_interval,
            writer=writer,
        )
    finally:
        for _ in worker_processes:
            job_queue.put(None)
        shutdown_evaluator_server(
            evaluator_request_queue=evaluator_request_queue,
            orchestrator_response_queue=response_queues["orchestrator"],
        )
        for process in worker_processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        evaluator_process.join(timeout=5)
        if evaluator_process.is_alive():
            evaluator_process.terminate()
            evaluator_process.join(timeout=5)


def collect_multiprocess_results(
    jobs: list[EvaluationJob],
    job_queue,
    result_queue,
    worker_processes: list[mp.Process],
    progress_interval: int,
    writer: EvaluationResultWriter,
) -> tuple[list[EvaluationResult], list[str]]:
    results: list[EvaluationResult] = []
    errors: list[str] = []
    started_at = time.monotonic()
    next_job_index = 0

    def submit_next() -> bool:
        nonlocal next_job_index
        if next_job_index >= len(jobs):
            return False
        job_queue.put(jobs[next_job_index])
        next_job_index += 1
        return True

    for _ in worker_processes:
        if not submit_next():
            break

    while len(results) + len(errors) < len(jobs):
        try:
            message = result_queue.get(timeout=1.0)
        except queue.Empty:
            if not any(process.is_alive() for process in worker_processes):
                errors.append("Todos los workers finalizaron antes de completar jobs.")
                remaining = len(jobs) - len(results) - len(errors)
                for _ in range(max(remaining, 0)):
                    errors.append("Job pendiente sin resultado porque finalizaron los workers.")
                break
            continue

        if isinstance(message, WorkerError):
            errors.append(f"{message.job_id}: {message.error}")
        else:
            try:
                writer.write(message)
                print_job_result(message)
                results.append(message)
            except Exception as exc:
                errors.append(f"{message.job_id}: save failed: {exc}")

        if next_job_index < len(jobs):
            submit_next()

        print_progress(
            processed=len(results) + len(errors),
            total=len(jobs),
            ok=len(results),
            failed=len(errors),
            started_at=started_at,
            progress_interval=progress_interval,
        )

    return results, errors


def run_job(
    job: EvaluationJob,
    run_config: _RunConfig,
    evaluator: Any,
) -> EvaluationResult:
    case = load_test_case(run_config, job)
    setup = case["problem_setup"]
    initial_state = case["state"]
    original_value = float(case["original_value"])

    if hasattr(evaluator, "setup"):
        evaluator.setup = setup

    engine = WorkforceEngine(setup)
    mcts_config = run_config.mcts_config.model_copy(
        update={"random_seed": int(job.seed)}
    )
    started_at = time.monotonic()

    if engine.check_terminality(initial_state):
        final_state = initial_state.copy_state()
        final_reward = float(engine.compute_reward(final_state))
        trajectory = []
    else:
        mcts = MCTS(engine=engine, evaluator=evaluator, config=mcts_config)
        trajectory, final_reward, final_state = generate_mcts_trajectory(
            initial_state=initial_state.copy_state(),
            engine=engine,
            mcts=mcts,
            debug=False,
        )

    trajectory = with_terminal_state(trajectory, final_state, final_reward)
    elapsed_seconds = time.monotonic() - started_at
    trajectory_prefix = "partial_mcts" if run_config.input_mode == "partial" else "test_mcts"
    trajectory_id = f"{trajectory_prefix}_{int(job.sample_index):06d}"
    value_error = float(final_reward - original_value)

    metrics = {
        "trajectory_id": trajectory_id,
        "source_sample_index": int(job.sample_index),
        "source_trajectory_id": str(case["source_trajectory_id"]),
        "evaluation_type": (
            "partial_tail" if run_config.input_mode == "partial" else "initial"
        ),
        "requested_tail_states": case.get("requested_tail_states"),
        "effective_tail_states": case.get("effective_tail_states"),
        "source_start_index": case.get("source_start_index"),
        "source_trajectory_length": case.get("source_trajectory_length"),
        "checkpoint_path": str(run_config.checkpoint_path),
        "checkpoint_step": int(run_config.checkpoint_step),
        "mcts_simulations": int(run_config.mcts_config.num_simulations),
        "elapsed_seconds": float(elapsed_seconds),
        "states_count": int(len(trajectory)),
        "final_reward": float(final_reward),
        "original_value": float(original_value),
        "value_error": value_error,
        "is_positive": bool(final_reward > 0),
        "is_better_than_original": bool(final_reward > original_value),
        "run_id": run_config.run_id,
    }
    metadata = {
        **metrics,
        "sample_source": "test_mcts_eval",
        "source_test_trajectory_id": str(case["trajectory_id"]),
    }
    return EvaluationResult(
        job_id=job.job_id,
        trajectory_id=trajectory_id,
        trajectory=trajectory,
        problem_setup=setup,
        metadata=metadata,
        metrics=metrics,
    )


def load_test_case(
    run_config: _RunConfig,
    job: EvaluationJob,
) -> dict[str, Any]:
    if run_config.input_mode == "partial":
        return load_partial_test_case(
            trajectory_path=run_config.sample_path,
            trajectory_id=str(job.source_trajectory_id),
            tail_states=int(run_config.tail_states),
        )
    return load_initial_test_case(run_config.sample_path, job.sample_index)


def load_initial_test_case(
    sample_path: str | Path,
    sample_index: int,
) -> dict[str, Any]:
    batch = SampleBuffer(sample_path, mode="r").load_batch([int(sample_index)])
    setup = ProblemSetup(
        mobile_days_off_count=int(batch.X["mobile_days_off_count"][0]),
        fixed_day_off=none_if_minus_one(batch.X["fixed_day_off"][0]),
        allowed_entry_hours=batch.X["allowed_entry_hours"][0],
        max_overcoverage_tolerance=float(batch.X["max_overcoverage_tolerance"][0]),
        closing_hour=none_if_minus_one(batch.X["closing_hour"][0]),
    )
    state = WorkforceState(
        residual_demand=batch.X["residual_demand"][0],
        remaining_stock=batch.X["remaining_stock"][0],
        expansion_mode=bool(batch.X["expansion_mode"][0]),
        current_modality=none_if_minus_one(batch.X["current_modality"][0]),
        current_entry_hour=none_if_minus_one(batch.X["current_entry_hour"][0]),
        assignment_week=int(batch.X["assignment_week"][0]),
        initial_demand_total=int(batch.X["initial_demand_total"][0]),
    )
    source_trajectory_id = str(batch.metadata["source_trajectory_id"][0])
    if source_trajectory_id == "":
        source_trajectory_id = str(batch.metadata["trajectory_id"][0])
    return {
        "problem_setup": setup,
        "state": state,
        "original_value": float(batch.Y["value"][0]),
        "trajectory_id": str(batch.metadata["trajectory_id"][0]),
        "source_trajectory_id": source_trajectory_id,
    }


def load_partial_test_case(
    trajectory_path: str | Path,
    trajectory_id: str,
    tail_states: int,
) -> dict[str, Any]:
    if int(tail_states) <= 0:
        raise ValueError("tail_states debe ser positivo.")

    record = TrajectoryBuffer(trajectory_path, mode="r").load(trajectory_id)
    source_length = len(record.samples)
    if source_length == 0:
        raise ValueError(f"La trayectoria {trajectory_id} esta vacia.")

    start_index = max(0, source_length - int(tail_states))
    sample = record.samples[start_index]
    state_data = sample["state"]
    setup = ProblemSetup(**record.problem_setup)
    state = WorkforceState(
        residual_demand=state_data["residual_demand"],
        remaining_stock=state_data["remaining_stock"],
        expansion_mode=bool(state_data["expansion_mode"]),
        current_modality=state_data["current_modality"],
        current_entry_hour=state_data["current_entry_hour"],
        assignment_week=int(state_data["assignment_week"]),
        initial_demand_total=int(state_data["initial_demand_total"]),
    )
    return {
        "problem_setup": setup,
        "state": state,
        "original_value": float(record.final_reward),
        "trajectory_id": str(record.trajectory_id),
        "source_trajectory_id": str(record.trajectory_id),
        "requested_tail_states": int(tail_states),
        "effective_tail_states": int(source_length - start_index),
        "source_start_index": int(start_index),
        "source_trajectory_length": int(source_length),
    }


def with_terminal_state(
    trajectory: list[dict[str, Any]],
    final_state: WorkforceState,
    final_reward: float,
) -> list[dict[str, Any]]:
    output = []
    for step_index, sample in enumerate(trajectory):
        output.append(
            {
                **sample,
                "value": float(final_reward),
                "reward": float(final_reward),
                "metadata": {
                    **sample.get("metadata", {}),
                    "is_terminal": False,
                    "step_index": int(step_index),
                },
            }
        )

    output.append(
        {
            "state": final_state,
            "policy": np.zeros((ACTION_SPACE_SIZE,), dtype=np.float32),
            "action_id": -1,
            "value": float(final_reward),
            "reward": float(final_reward),
            "metadata": {
                "is_terminal": True,
                "step_index": int(len(output)),
            },
        }
    )
    return output


def save_results(
    results: list[EvaluationResult],
    trajectory_path: Path,
    reports_dir: Path,
    overwrite: bool,
) -> int:
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    trajectory_buffer = TrajectoryBuffer(
        trajectory_path,
        mode="w" if overwrite else "a",
    )
    jsonl_mode = "w" if overwrite else "a"

    saved = 0
    with (reports_dir / "trajectories.jsonl").open(jsonl_mode, encoding="utf-8") as fh:
        for result in sorted(
            results,
            key=lambda item: item.metrics["source_sample_index"],
        ):
            trajectory_buffer.save(
                trajectory=result.trajectory,
                problem_setup=result.problem_setup,
                trajectory_id=result.trajectory_id,
                action_space_size=ACTION_SPACE_SIZE,
                metadata=result.metadata,
            )
            fh.write(json.dumps(to_jsonable(result.metrics), sort_keys=True) + "\n")
            saved += 1

    return saved


def build_run_summary(
    run_id: str,
    sample_path: Path,
    trajectory_path: Path,
    checkpoint_path: Path,
    checkpoint_step: int,
    mcts_simulations: int,
    n_workers: int,
    total_jobs: int,
    results: list[EvaluationResult],
    errors: list[str],
    elapsed_seconds: float,
    saved_trajectories: int,
    input_mode: str = "initial",
    tail_states: int | None = None,
) -> dict[str, Any]:
    metrics = [result.metrics for result in results]
    final_rewards = [float(item["final_reward"]) for item in metrics]
    original_values = [float(item["original_value"]) for item in metrics]
    value_errors = [float(item["value_error"]) for item in metrics]
    positive_count = sum(1 for reward in final_rewards if reward > 0)
    better_count = sum(1 for item in metrics if bool(item["is_better_than_original"]))
    worse_count = sum(1 for error in value_errors if error < 0)
    same_count = sum(1 for error in value_errors if error == 0)

    return {
        "run_id": run_id,
        "evaluation_type": (
            "partial_tail" if input_mode == "partial" else "initial"
        ),
        "requested_tail_states": (
            int(tail_states) if tail_states is not None else None
        ),
        "sample_path": str(sample_path),
        "trajectory_path": str(trajectory_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_step),
        "mcts_simulations": int(mcts_simulations),
        "n_workers": int(n_workers),
        "total_jobs": int(total_jobs),
        "completed_jobs": int(len(results)),
        "failed_jobs": int(len(errors)),
        "saved_trajectories": int(saved_trajectories),
        "positive_count": int(positive_count),
        "positive_rate": safe_ratio(positive_count, len(results)),
        "better_than_original_count": int(better_count),
        "better_than_original_rate": safe_ratio(better_count, len(results)),
        "worse_than_original_count": int(worse_count),
        "same_as_original_count": int(same_count),
        "mean_final_reward": safe_mean(final_rewards),
        "mean_original_value": safe_mean(original_values),
        "mean_value_error": safe_mean(value_errors),
        "elapsed_seconds": float(elapsed_seconds),
        "errors": list(errors),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_run_summary(summary: dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(summary), fh, indent=2, sort_keys=True)
    with (reports_dir / "runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_jsonable(summary), sort_keys=True) + "\n")


def print_summary(summary: dict[str, Any]) -> None:
    print("[test_mcts] done", flush=True)
    for key in (
        "completed_jobs",
        "failed_jobs",
        "saved_trajectories",
        "positive_count",
        "better_than_original_count",
        "worse_than_original_count",
        "mean_final_reward",
        "mean_original_value",
        "mean_value_error",
        "elapsed_seconds",
    ):
        print(f"{key}={summary[key]}", flush=True)


def print_job_result(result: EvaluationResult) -> None:
    metrics = result.metrics
    partial_details = ""
    if metrics.get("evaluation_type") == "partial_tail":
        partial_details = (
            f"tail={metrics['requested_tail_states']} "
            f"tail_effective={metrics['effective_tail_states']} "
            f"start={metrics['source_start_index']} "
        )
    print(
        "[test_mcts] saved "
        f"trajectory_id={metrics['trajectory_id']} "
        f"sample={metrics['source_sample_index']} "
        f"{partial_details}"
        f"states={metrics['states_count']} "
        f"reward={metrics['final_reward']:.6f} "
        f"original={metrics['original_value']:.6f} "
        f"error={metrics['value_error']:.6f} "
        f"elapsed={metrics['elapsed_seconds']:.2f}s",
        flush=True,
    )


def print_progress(
    processed: int,
    total: int,
    ok: int,
    failed: int,
    started_at: float,
    progress_interval: int,
) -> None:
    if processed == 0:
        return
    if processed != total and processed % int(progress_interval) != 0:
        return
    elapsed = max(time.monotonic() - started_at, 1e-9)
    rate = processed / elapsed
    print(
        "[test_mcts] "
        f"jobs={processed}/{total} ok={ok} failed={failed} rate={rate:.2f} jobs/s",
        flush=True,
    )


def shutdown_evaluator_server(
    evaluator_request_queue,
    orchestrator_response_queue,
) -> None:
    client = CentralizedEvaluatorClient(
        setup=None,
        client_id="orchestrator",
        request_queue=evaluator_request_queue,
        response_queue=orchestrator_response_queue,
        request_timeout_s=10,
    )
    try:
        client.shutdown_server()
    except Exception:
        pass


def _run_worker_process(
    client_id: str,
    run_config: _RunConfig,
    evaluator_request_queue,
    evaluator_response_queue,
    job_queue,
    result_queue,
    request_timeout_s: float | None,
) -> None:
    evaluator = CentralizedEvaluatorClient(
        setup=None,
        client_id=client_id,
        request_queue=evaluator_request_queue,
        response_queue=evaluator_response_queue,
        request_timeout_s=request_timeout_s,
    )
    while True:
        job = job_queue.get()
        if job is None:
            break
        try:
            result_queue.put(
                run_job(job=job, run_config=run_config, evaluator=evaluator)
            )
        except Exception as exc:
            result_queue.put(WorkerError(job_id=job.job_id, error=str(exc)))


def _run_evaluator_server_process(
    config: CentralizedEvaluatorConfig,
    request_queue,
    response_queues,
) -> None:
    server = CentralizedEvaluatorServer(
        config=config,
        request_queue=request_queue,
        response_queues=response_queues,
    )
    server.run_forever()


def none_if_minus_one(value: Any) -> int | None:
    value_int = int(value)
    if value_int < 0:
        return None
    return value_int


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()

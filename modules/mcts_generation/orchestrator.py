from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from modules.evaluators.centralized import (
    CentralizedEvaluatorClient,
    CentralizedEvaluatorConfig,
    CentralizedEvaluatorServer,
)
from modules.storage import SampleBuffer, TrajectoryBuffer

from .config import MCTSGenerationConfig
from .schemas import MCTSGenerationJob
from .worker import MCTSGenerationWorker


@dataclass(frozen=True)
class MCTSOrchestratorConfig:
    source_buffer_path: str | Path
    sample_buffer_path: str | Path
    n_workers: int = 1
    overwrite_samples: bool = False
    print_progress: bool = True
    progress_interval: int = 1
    centralized_evaluator_config: CentralizedEvaluatorConfig | None = None
    multiprocessing_start_method: str = "fork"
    sample_limit_per_cycle: int | None = None

    def __post_init__(self) -> None:
        if self.n_workers <= 0:
            raise ValueError("n_workers debe ser positivo.")
        if self.progress_interval <= 0:
            raise ValueError("progress_interval debe ser positivo.")
        if self.n_workers > 1 and self.centralized_evaluator_config is None:
            raise ValueError(
                "centralized_evaluator_config es requerido con n_workers > 1."
            )
        if self.sample_limit_per_cycle is not None and self.sample_limit_per_cycle <= 0:
            raise ValueError("sample_limit_per_cycle debe ser positivo o None.")


@dataclass
class MCTSCycleReport:
    cycle_index: int
    completed_jobs: int
    failed_jobs: int
    saved_samples: int
    generated_trajectories: int
    used_mcts_jobs: int
    reweighted_jobs: int
    sample_start_index: int = 0
    sample_end_index: int = 0


@dataclass
class MCTSOrchestratorReport:
    source_buffer_path: str
    sample_buffer_path: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    generated_trajectories: int
    saved_samples: int
    used_mcts_jobs: int
    reweighted_jobs: int
    cycle_reports: list[MCTSCycleReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def build_mcts_generation_jobs(
    source_trajectory_ids: list[str],
    seed: int | None = None,
    job_id_prefix: str = "mcts_",
) -> list[MCTSGenerationJob]:
    jobs = []
    for index, source_trajectory_id in enumerate(source_trajectory_ids):
        job_seed = int(seed) + index if seed is not None else index
        jobs.append(
            MCTSGenerationJob(
                job_id=f"{job_id_prefix}{index:06d}",
                source_trajectory_id=str(source_trajectory_id),
                seed=int(job_seed),
            )
        )
    return jobs


class MCTSGenerationOrchestrator:
    """
    Orquestador para conectar workers MCTS, evaluator y SampleBuffer.

    Los workers devuelven trayectorias finalizadas y el orquestador las aplana
    al SampleBuffer. Con multiproceso usa un evaluador centralizado compartido
    y puede pausar entre ciclos para que un learner externo actualice pesos.
    """

    def __init__(
        self,
        config: MCTSOrchestratorConfig,
        generation_config: MCTSGenerationConfig,
        evaluator,
        on_cycle_ready: Callable[[MCTSCycleReport], str | Path | None] | None = None,
    ) -> None:
        self.config = config
        self.generation_config = generation_config
        self.evaluator = evaluator
        self.on_cycle_ready = on_cycle_ready

    def run(
        self,
        jobs: Iterable[MCTSGenerationJob],
    ) -> MCTSOrchestratorReport:
        job_list = list(jobs)
        if int(self.config.n_workers) > 1:
            return self._run_multiprocess(job_list)
        return self._run_sequential(job_list)

    def _run_sequential(
        self,
        job_list: list[MCTSGenerationJob],
    ) -> MCTSOrchestratorReport:
        mode = "w" if self.config.overwrite_samples else "a"
        sample_buffer = SampleBuffer(self.config.sample_buffer_path, mode=mode)
        started_at = time.monotonic()

        worker = MCTSGenerationWorker(
            source_buffer_path=self.config.source_buffer_path,
            config=self.generation_config,
            evaluator=self.evaluator,
        )

        completed_jobs = 0
        failed_jobs = 0
        generated_trajectories = 0
        saved_samples = 0
        used_mcts_jobs = 0
        reweighted_jobs = 0
        errors: list[str] = []
        cycle_reports: list[MCTSCycleReport] = []
        cycle_start_index = sample_buffer.trained_until
        cycle_stats = _CycleStats(
            cycle_index=0,
            sample_start_index=cycle_start_index,
            saved_samples=len(sample_buffer) - cycle_start_index,
        )

        for job in job_list:
            try:
                result = worker.run(job)
                completed_jobs += 1
                generated_trajectories += len(result.trajectories)
                cycle_stats.completed_jobs += 1
                cycle_stats.generated_trajectories += len(result.trajectories)
                if result.used_mcts:
                    used_mcts_jobs += 1
                    cycle_stats.used_mcts_jobs += 1
                else:
                    reweighted_jobs += 1
                    cycle_stats.reweighted_jobs += 1

                appended = sample_buffer.append_trajectories(result.trajectories)
                saved_samples += appended
                cycle_stats.saved_samples += appended
            except Exception as exc:
                failed_jobs += 1
                cycle_stats.failed_jobs += 1
                errors.append(f"{job.job_id}: {exc}")

            self._print_progress(
                processed_jobs=completed_jobs + failed_jobs,
                total_jobs=len(job_list),
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                saved_samples=saved_samples,
                started_at=started_at,
                force=False,
            )

            if self._cycle_limit_reached(cycle_stats):
                cycle_reports.append(
                    self._close_cycle(cycle_stats, None, sample_buffer)
                )
                cycle_stats = _CycleStats(
                    cycle_index=cycle_stats.cycle_index + 1,
                    sample_start_index=len(sample_buffer),
                )

        if cycle_stats.has_activity:
            cycle_reports.append(self._close_cycle(cycle_stats, None, sample_buffer))

        self._print_progress(
            processed_jobs=completed_jobs + failed_jobs,
            total_jobs=len(job_list),
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            saved_samples=saved_samples,
            started_at=started_at,
            force=True,
        )

        return MCTSOrchestratorReport(
            source_buffer_path=str(self.config.source_buffer_path),
            sample_buffer_path=str(self.config.sample_buffer_path),
            total_jobs=len(job_list),
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            generated_trajectories=generated_trajectories,
            saved_samples=saved_samples,
            used_mcts_jobs=used_mcts_jobs,
            reweighted_jobs=reweighted_jobs,
            cycle_reports=cycle_reports,
            errors=errors,
        )

    def _run_multiprocess(
        self,
        job_list: list[MCTSGenerationJob],
    ) -> MCTSOrchestratorReport:
        if self.config.centralized_evaluator_config is None:
            raise ValueError("centralized_evaluator_config es requerido.")

        ctx = mp.get_context(self.config.multiprocessing_start_method)
        job_queue = ctx.Queue()
        result_queue = ctx.Queue()
        evaluator_request_queue = ctx.Queue()

        response_queues = {
            f"worker_{index}": ctx.Queue()
            for index in range(int(self.config.n_workers))
        }
        response_queues["orchestrator"] = ctx.Queue()

        evaluator_process = ctx.Process(
            target=_run_evaluator_server_process,
            args=(
                self.config.centralized_evaluator_config,
                evaluator_request_queue,
                response_queues,
            ),
            daemon=True,
        )
        evaluator_process.start()

        worker_processes = []
        for worker_index in range(int(self.config.n_workers)):
            client_id = f"worker_{worker_index}"
            process = ctx.Process(
                target=_run_worker_process,
                args=(
                    client_id,
                    self.config.source_buffer_path,
                    self.generation_config,
                    evaluator_request_queue,
                    response_queues[client_id],
                    job_queue,
                    result_queue,
                    self.config.centralized_evaluator_config.request_timeout_s,
                ),
                daemon=True,
            )
            process.start()
            worker_processes.append(process)

        try:
            return self._collect_multiprocess_results(
                job_list=job_list,
                job_queue=job_queue,
                result_queue=result_queue,
                evaluator_request_queue=evaluator_request_queue,
                orchestrator_response_queue=response_queues["orchestrator"],
                worker_processes=worker_processes,
                evaluator_process=evaluator_process,
            )
        finally:
            for process in worker_processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            if evaluator_process.is_alive():
                evaluator_process.terminate()
            evaluator_process.join(timeout=5)

    def _collect_multiprocess_results(
        self,
        job_list: list[MCTSGenerationJob],
        job_queue,
        result_queue,
        evaluator_request_queue,
        orchestrator_response_queue,
        worker_processes: list[mp.Process],
        evaluator_process: mp.Process,
    ) -> MCTSOrchestratorReport:
        mode = "w" if self.config.overwrite_samples else "a"
        sample_buffer = SampleBuffer(self.config.sample_buffer_path, mode=mode)
        started_at = time.monotonic()

        completed_jobs = 0
        failed_jobs = 0
        generated_trajectories = 0
        saved_samples = 0
        used_mcts_jobs = 0
        reweighted_jobs = 0
        errors: list[str] = []
        cycle_reports: list[MCTSCycleReport] = []
        cycle_start_index = sample_buffer.trained_until
        cycle_stats = _CycleStats(
            cycle_index=0,
            sample_start_index=cycle_start_index,
            saved_samples=len(sample_buffer) - cycle_start_index,
        )
        next_job_index = 0
        in_flight_jobs = 0

        def submit_next_job() -> bool:
            nonlocal next_job_index, in_flight_jobs
            if next_job_index >= len(job_list):
                return False
            job_queue.put(job_list[next_job_index])
            next_job_index += 1
            in_flight_jobs += 1
            return True

        for _ in worker_processes:
            if not submit_next_job():
                break

        while completed_jobs + failed_jobs < len(job_list):
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                if not any(process.is_alive() for process in worker_processes):
                    errors.append("Todos los workers finalizaron antes de completar jobs.")
                    failed_jobs = len(job_list) - completed_jobs
                    break
                continue
            in_flight_jobs -= 1

            if isinstance(message, _WorkerError):
                failed_jobs += 1
                cycle_stats.failed_jobs += 1
                errors.append(f"{message.job_id}: {message.error}")
            else:
                completed_jobs += 1
                generated_trajectories += len(message.trajectories)
                cycle_stats.completed_jobs += 1
                cycle_stats.generated_trajectories += len(message.trajectories)
                if message.used_mcts:
                    used_mcts_jobs += 1
                    cycle_stats.used_mcts_jobs += 1
                else:
                    reweighted_jobs += 1
                    cycle_stats.reweighted_jobs += 1
                appended = sample_buffer.append_trajectories(message.trajectories)
                saved_samples += appended
                cycle_stats.saved_samples += appended

            if self._cycle_limit_reached(cycle_stats):
                if in_flight_jobs == 0:
                    cycle_reports.append(
                        self._close_cycle(
                            cycle_stats,
                            (
                                evaluator_request_queue,
                                orchestrator_response_queue,
                            ),
                            sample_buffer,
                        )
                    )
                    cycle_stats = _CycleStats(
                        cycle_index=cycle_stats.cycle_index + 1,
                        sample_start_index=len(sample_buffer),
                    )
                    for _ in worker_processes:
                        if not submit_next_job():
                            break
            elif next_job_index < len(job_list):
                submit_next_job()

            self._print_progress(
                processed_jobs=completed_jobs + failed_jobs,
                total_jobs=len(job_list),
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                saved_samples=saved_samples,
                started_at=started_at,
                force=False,
            )

        if cycle_stats.has_activity:
            cycle_reports.append(
                self._close_cycle(
                    cycle_stats,
                    (
                        evaluator_request_queue,
                        orchestrator_response_queue,
                    ),
                    sample_buffer,
                )
            )

        for _ in worker_processes:
            job_queue.put(None)

        self._shutdown_evaluator_server(
            evaluator_request_queue=evaluator_request_queue,
            orchestrator_response_queue=orchestrator_response_queue,
        )
        evaluator_process.join(timeout=10)

        for process in worker_processes:
            process.join(timeout=10)

        self._print_progress(
            processed_jobs=completed_jobs + failed_jobs,
            total_jobs=len(job_list),
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            saved_samples=saved_samples,
            started_at=started_at,
            force=True,
        )

        return MCTSOrchestratorReport(
            source_buffer_path=str(self.config.source_buffer_path),
            sample_buffer_path=str(self.config.sample_buffer_path),
            total_jobs=len(job_list),
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            generated_trajectories=generated_trajectories,
            saved_samples=saved_samples,
            used_mcts_jobs=used_mcts_jobs,
            reweighted_jobs=reweighted_jobs,
            cycle_reports=cycle_reports,
            errors=errors,
        )

    def _shutdown_evaluator_server(
        self,
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
        client.shutdown_server()

    def _cycle_limit_reached(self, cycle_stats: "_CycleStats") -> bool:
        limit = self.config.sample_limit_per_cycle
        return limit is not None and cycle_stats.saved_samples >= int(limit)

    def _close_cycle(
        self,
        cycle_stats: "_CycleStats",
        evaluator_control: tuple | None,
        sample_buffer: SampleBuffer,
    ) -> MCTSCycleReport:
        report = cycle_stats.to_report(sample_end_index=len(sample_buffer))
        checkpoint_path = None
        if self.on_cycle_ready is not None:
            checkpoint_path = self.on_cycle_ready(report)

        if checkpoint_path is not None:
            self._reload_evaluator_weights(checkpoint_path, evaluator_control)
            sample_buffer.mark_trained_until(report.sample_end_index)

        return report

    def _reload_evaluator_weights(
        self,
        checkpoint_path: str | Path,
        evaluator_control: tuple | None,
    ) -> None:
        if evaluator_control is None:
            if not hasattr(self.evaluator, "reload_weights"):
                raise RuntimeError(
                    "El evaluator secuencial no soporta reload_weights."
                )
            self.evaluator.reload_weights(checkpoint_path)
            return

        evaluator_request_queue, orchestrator_response_queue = evaluator_control
        client = CentralizedEvaluatorClient(
            setup=None,
            client_id="orchestrator",
            request_queue=evaluator_request_queue,
            response_queue=orchestrator_response_queue,
            request_timeout_s=60,
        )
        client.reload_weights(checkpoint_path)

    @staticmethod
    def list_source_trajectory_ids(
        source_buffer_path: str | Path,
    ) -> list[str]:
        return TrajectoryBuffer(source_buffer_path, mode="r").list_ids()

    def _print_progress(
        self,
        processed_jobs: int,
        total_jobs: int,
        completed_jobs: int,
        failed_jobs: int,
        saved_samples: int,
        started_at: float,
        force: bool,
    ) -> None:
        if not self.config.print_progress:
            return
        if processed_jobs == 0:
            return

        should_print = (
            force
            or processed_jobs == total_jobs
            or processed_jobs % int(self.config.progress_interval) == 0
        )
        if not should_print:
            return

        elapsed = max(time.monotonic() - started_at, 1e-9)
        rate = processed_jobs / elapsed
        print(
            "[mcts_generation] "
            f"jobs={processed_jobs}/{total_jobs} "
            f"ok={completed_jobs} failed={failed_jobs} "
            f"samples={saved_samples} "
            f"rate={rate:.2f} jobs/s",
            flush=True,
        )


@dataclass(frozen=True)
class _WorkerError:
    job_id: str
    error: str


@dataclass
class _CycleStats:
    cycle_index: int
    sample_start_index: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    saved_samples: int = 0
    generated_trajectories: int = 0
    used_mcts_jobs: int = 0
    reweighted_jobs: int = 0

    @property
    def has_activity(self) -> bool:
        return self.completed_jobs > 0 or self.failed_jobs > 0

    def to_report(self, sample_end_index: int) -> MCTSCycleReport:
        return MCTSCycleReport(
            cycle_index=self.cycle_index,
            completed_jobs=self.completed_jobs,
            failed_jobs=self.failed_jobs,
            saved_samples=self.saved_samples,
            generated_trajectories=self.generated_trajectories,
            used_mcts_jobs=self.used_mcts_jobs,
            reweighted_jobs=self.reweighted_jobs,
            sample_start_index=self.sample_start_index,
            sample_end_index=int(sample_end_index),
        )


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


def _run_worker_process(
    client_id: str,
    source_buffer_path: str | Path,
    generation_config: MCTSGenerationConfig,
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
    worker = MCTSGenerationWorker(
        source_buffer_path=source_buffer_path,
        config=generation_config,
        evaluator=evaluator,
    )

    while True:
        job = job_queue.get()
        if job is None:
            break
        try:
            result_queue.put(worker.run(job))
        except Exception as exc:
            result_queue.put(_WorkerError(job_id=job.job_id, error=str(exc)))

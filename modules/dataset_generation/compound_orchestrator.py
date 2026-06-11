from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
from typing import Iterable

from modules.storage import CompoundTrajectoryBuffer

from .compound_schemas import (
    CompoundGenerationJob,
    CompoundOrchestratorConfig,
    CompoundOrchestratorReport,
    CompoundWorkerResult,
)
from .compound_worker import CompoundFullTrajectoryWorker


def build_compound_generation_jobs(
    n_jobs: int,
    job_id_prefix: str = "",
) -> list[CompoundGenerationJob]:
    if n_jobs < 0:
        raise ValueError("n_jobs debe ser >= 0.")
    return [
        CompoundGenerationJob(
            job_id=f"{job_id_prefix}{index:06d}",
        )
        for index in range(int(n_jobs))
    ]


def _run_compound_worker(
    worker: CompoundFullTrajectoryWorker,
    job: CompoundGenerationJob,
) -> CompoundWorkerResult:
    return worker.run(job)


class CompoundDatasetOrchestrator:
    """Ejecuta workers compuestos y centraliza la escritura Zarr."""

    def __init__(
        self,
        config: CompoundOrchestratorConfig,
        worker: CompoundFullTrajectoryWorker,
    ) -> None:
        self.config = config
        self.worker = worker

    def run(
        self,
        jobs: Iterable[CompoundGenerationJob],
    ) -> CompoundOrchestratorReport:
        job_list = list(jobs)
        buffer: CompoundTrajectoryBuffer | None = None
        started_at = time.monotonic()
        completed = 0
        failed = 0
        saved = 0
        trajectory_ids: list[str] = []
        errors: list[str] = []
        rewards: list[float] = []
        lengths: list[int] = []
        initial_demands: list[float] = []
        input_resource_counts: list[float] = []
        resource_counts: list[float] = []

        for job, result in self._iter_results(job_list):
            if isinstance(result, Exception):
                failed += 1
                errors.append(f"{job.job_id}: {result}")
            else:
                generated = result.generated
                if buffer is None:
                    buffer = self._open_buffer()
                buffer.save(
                    trajectory=generated.trajectory,
                    problem_setup=generated.problem_setup,
                    trajectory_id=generated.trajectory_id,
                    metadata=generated.metadata,
                    temporal_chunk_size=self.config.temporal_chunk_size,
                )
                completed += 1
                saved += 1
                trajectory_ids.append(generated.trajectory_id)
                rewards.append(float(generated.metadata["final_reward"]))
                lengths.append(int(generated.metadata["trajectory_length"]))
                initial_demands.append(
                    float(generated.metadata["initial_demand_total"])
                )
                input_resource_counts.append(
                    float(generated.metadata["n_resources"])
                )
                resource_counts.append(
                    float(sum(generated.metadata["output_stock"]))
                )

            self._print_progress(
                processed=completed + failed,
                total=len(job_list),
                completed=completed,
                failed=failed,
                saved=saved,
                started_at=started_at,
                force=False,
            )

        self._print_progress(
            processed=completed + failed,
            total=len(job_list),
            completed=completed,
            failed=failed,
            saved=saved,
            started_at=started_at,
            force=True,
        )
        if buffer is None:
            self._open_buffer()

        return CompoundOrchestratorReport(
            output_path=str(self.config.output_path),
            total_jobs=len(job_list),
            completed_jobs=completed,
            failed_jobs=failed,
            saved_trajectories=saved,
            trajectory_ids=trajectory_ids,
            stats={
                "mean_final_reward": (
                    sum(rewards) / len(rewards) if rewards else 0.0
                ),
                "mean_trajectory_length": (
                    sum(lengths) / len(lengths) if lengths else 0.0
                ),
                "mean_initial_demand_total": (
                    sum(initial_demands) / len(initial_demands)
                    if initial_demands
                    else 0.0
                ),
                "mean_input_resources": (
                    sum(input_resource_counts) / len(input_resource_counts)
                    if input_resource_counts
                    else 0.0
                ),
                "mean_output_resources": (
                    sum(resource_counts) / len(resource_counts)
                    if resource_counts
                    else 0.0
                ),
            },
            errors=errors,
        )

    def _open_buffer(self) -> CompoundTrajectoryBuffer:
        return CompoundTrajectoryBuffer(
            self.config.output_path,
            mode="w" if self.config.overwrite else "a",
        )

    def _iter_results(self, jobs: list[CompoundGenerationJob]):
        if self.config.n_workers == 1:
            for job in jobs:
                try:
                    yield job, self.worker.run(job)
                except Exception as exc:
                    yield job, exc
            return

        with ProcessPoolExecutor(
            max_workers=self.config.n_workers,
            mp_context=mp.get_context(
                self.config.multiprocessing_start_method
            ),
        ) as executor:
            futures = {
                executor.submit(
                    _run_compound_worker,
                    self.worker,
                    job,
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    yield job, future.result()
                except Exception as exc:
                    yield job, exc

    def _print_progress(
        self,
        processed: int,
        total: int,
        completed: int,
        failed: int,
        saved: int,
        started_at: float,
        force: bool,
    ) -> None:
        if not self.config.print_progress or processed == 0:
            return
        interval = self.config.progress_interval
        if not force and processed != total and processed % interval != 0:
            return
        elapsed = max(time.monotonic() - started_at, 1e-9)
        print(
            "[compound_dataset] "
            f"jobs={processed}/{total} ok={completed} failed={failed} "
            f"saved={saved} rate={processed / elapsed:.2f} jobs/s",
            flush=True,
        )

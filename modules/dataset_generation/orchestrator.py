from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import secrets
import time
from typing import Iterable

from modules.storage import TrajectoryBuffer

from .schemas import (
    DatasetGenerationConfig,
    DatasetGenerationReport,
    GenerationJob,
    GenerationWorkerResult,
)
from .worker_protocol import TrajectoryGenerationWorker


def build_generation_jobs(
    n_jobs: int,
    seed: int | None = None,
    job_id_prefix: str = "",
) -> list[GenerationJob]:
    jobs = []
    for index in range(int(n_jobs)):
        job_id = f"{job_id_prefix}{index:06d}"
        job_seed = secrets.randbits(31) if seed is None else int(seed) + index
        jobs.append(
            GenerationJob(
                job_id=job_id,
                seed=job_seed,
            )
        )
    return jobs


def _run_worker_job(
    worker: TrajectoryGenerationWorker,
    job: GenerationJob,
) -> GenerationWorkerResult:
    return worker.run(job)


class TrajectoryDatasetOrchestrator:
    """
    Ejecuta jobs de generación y guarda las trayectorias resultantes.

    El orquestador no conoce la lógica de generación. Esa responsabilidad vive
    en el worker recibido.
    """

    def __init__(
        self,
        config: DatasetGenerationConfig,
        worker: TrajectoryGenerationWorker,
    ) -> None:
        self.config = config
        self.worker = worker

    def run(self, jobs: Iterable[GenerationJob]) -> DatasetGenerationReport:
        job_list = list(jobs)
        mode = "w" if self.config.overwrite else "a"
        buffer = TrajectoryBuffer(self.config.output_path, mode=mode)
        started_at = time.monotonic()

        completed_jobs = 0
        failed_jobs = 0
        saved_trajectories = 0
        errors: list[str] = []
        stats = _StatsAccumulator(
            fields=[
                "initial_demand_total",
                "final_reward",
                "final_value",
            ]
        )
        resource_totals = {
            "mod_4": 0,
            "mod_6": 0,
            "mod_8": 0,
        }
        trajectory_ids: list[str] = []

        for result in self._iter_results(job_list):
            if isinstance(result, Exception):
                failed_jobs += 1
                errors.append(str(result))
                self._print_progress(
                    completed_jobs=completed_jobs,
                    failed_jobs=failed_jobs,
                    saved_trajectories=saved_trajectories,
                    total_jobs=len(job_list),
                    stats=stats,
                    started_at=started_at,
                    force=False,
                )
                continue

            completed_jobs += 1
            for generated in result.trajectories:
                buffer.save(
                    trajectory=generated.trajectory,
                    problem_setup=generated.problem_setup,
                    trajectory_id=generated.trajectory_id,
                    action_space_size=self.config.action_space_size,
                    temporal_chunk_size=self.config.temporal_chunk_size,
                    metadata=generated.metadata,
                )
                saved_trajectories += 1
                stats.add(generated.metadata)
                _add_resource_totals(resource_totals, generated.metadata)
                if generated.trajectory_id is not None:
                    trajectory_ids.append(str(generated.trajectory_id))

            self._print_progress(
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                saved_trajectories=saved_trajectories,
                total_jobs=len(job_list),
                stats=stats,
                started_at=started_at,
                force=False,
            )

        self._print_progress(
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            saved_trajectories=saved_trajectories,
            total_jobs=len(job_list),
            stats=stats,
            started_at=started_at,
            force=True,
        )

        return DatasetGenerationReport(
            output_path=self.config.output_path,
            total_jobs=len(job_list),
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            saved_trajectories=saved_trajectories,
            stats=stats.summary(),
            resource_totals=resource_totals,
            trajectory_ids=trajectory_ids,
            errors=errors,
        )

    def _iter_results(
        self,
        jobs: list[GenerationJob],
    ):
        if int(self.config.n_workers) <= 1:
            for job in jobs:
                try:
                    yield self.worker.run(job)
                except Exception as exc:
                    yield exc
            return

        with ProcessPoolExecutor(max_workers=int(self.config.n_workers)) as executor:
            futures = [
                executor.submit(_run_worker_job, self.worker, job)
                for job in jobs
            ]

            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as exc:
                    yield exc

    def _print_progress(
        self,
        completed_jobs: int,
        failed_jobs: int,
        saved_trajectories: int,
        total_jobs: int,
        stats: "_StatsAccumulator",
        started_at: float,
        force: bool,
    ) -> None:
        if not self.config.print_progress:
            return

        processed_jobs = completed_jobs + failed_jobs
        if processed_jobs == 0:
            return

        interval = max(1, int(self.config.progress_interval))
        should_print = force or processed_jobs == total_jobs or processed_jobs % interval == 0
        if not should_print:
            return

        elapsed = max(time.monotonic() - started_at, 1e-9)
        jobs_per_second = processed_jobs / elapsed
        message = (
            f"[dataset_generation] jobs={processed_jobs}/{total_jobs} "
            f"ok={completed_jobs} failed={failed_jobs} "
            f"saved={saved_trajectories} "
            f"rate={jobs_per_second:.2f} jobs/s"
        )

        print(message, flush=True)


class _StatsAccumulator:
    def __init__(self, fields: list[str]) -> None:
        self.fields = fields
        self.values = {field: [] for field in fields}

    def add(self, metadata: dict) -> None:
        for field in self.fields:
            if field in metadata:
                self.values[field].append(float(metadata[field]))

    def summary(self) -> dict[str, dict[str, float]]:
        output = {}
        for field, values in self.values.items():
            if not values:
                continue
            total = sum(values)
            output[field] = {
                "count": float(len(values)),
                "min": float(min(values)),
                "max": float(max(values)),
                "mean": float(total / len(values)),
            }
        return output


def _add_resource_totals(
    resource_totals: dict[str, int],
    metadata: dict,
) -> None:
    initial_stock = metadata.get("initial_stock")
    if initial_stock is None:
        return

    resource_totals["mod_4"] += int(initial_stock[0])
    resource_totals["mod_6"] += int(initial_stock[1])
    resource_totals["mod_8"] += int(initial_stock[2])

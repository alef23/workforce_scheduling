from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

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

    def __post_init__(self) -> None:
        if self.n_workers != 1:
            raise NotImplementedError(
                "La primera version del orquestador MCTS soporta n_workers=1."
            )
        if self.progress_interval <= 0:
            raise ValueError("progress_interval debe ser positivo.")


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
    Orquestador minimo para conectar worker MCTS y SampleBuffer.

    Esta primera version es secuencial. El objetivo es validar el contrato:
    worker devuelve trayectorias finalizadas y el orquestador las aplana al
    SampleBuffer. La version multiproceso/ciclos/learner vendra despues.
    """

    def __init__(
        self,
        config: MCTSOrchestratorConfig,
        generation_config: MCTSGenerationConfig,
        evaluator,
    ) -> None:
        self.config = config
        self.generation_config = generation_config
        self.evaluator = evaluator

    def run(
        self,
        jobs: Iterable[MCTSGenerationJob],
    ) -> MCTSOrchestratorReport:
        job_list = list(jobs)
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

        for job in job_list:
            try:
                result = worker.run(job)
                completed_jobs += 1
                generated_trajectories += len(result.trajectories)
                if result.used_mcts:
                    used_mcts_jobs += 1
                else:
                    reweighted_jobs += 1

                saved_samples += sample_buffer.append_trajectories(result.trajectories)
            except Exception as exc:
                failed_jobs += 1
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
            errors=errors,
        )

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

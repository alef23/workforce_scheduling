from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
from typing import Any


@dataclass(frozen=True)
class CompoundGenerationJob:
    job_id: str


@dataclass
class CompoundGeneratedTrajectory:
    trajectory: list[dict[str, Any]]
    problem_setup: Any
    trajectory_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompoundWorkerResult:
    job_id: str
    generated: CompoundGeneratedTrajectory


@dataclass(frozen=True)
class CompoundOrchestratorConfig:
    output_path: str
    n_workers: int = 1
    overwrite: bool = False
    print_progress: bool = True
    progress_interval: int = 1
    temporal_chunk_size: int = 128
    multiprocessing_start_method: str = "spawn"

    def __post_init__(self) -> None:
        if self.n_workers <= 0:
            raise ValueError("n_workers debe ser positivo.")
        if self.progress_interval <= 0:
            raise ValueError("progress_interval debe ser positivo.")
        if self.temporal_chunk_size <= 0:
            raise ValueError("temporal_chunk_size debe ser positivo.")
        if self.multiprocessing_start_method not in mp.get_all_start_methods():
            raise ValueError(
                "multiprocessing_start_method no está disponible: "
                f"{self.multiprocessing_start_method!r}."
            )


@dataclass
class CompoundOrchestratorReport:
    output_path: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    saved_trajectories: int
    trajectory_ids: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

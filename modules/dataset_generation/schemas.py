from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationJob:
    job_id: str
    seed: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedTrajectory:
    trajectory: list[dict[str, Any]]
    problem_setup: Any
    trajectory_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationWorkerResult:
    job_id: str
    worker_type: str
    trajectories: list[GeneratedTrajectory]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetGenerationConfig:
    output_path: str
    n_workers: int = 1
    action_space_size: int = 55
    temporal_chunk_size: int = 128
    overwrite: bool = False
    print_progress: bool = True
    progress_interval: int = 100


@dataclass
class DatasetGenerationReport:
    output_path: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    saved_trajectories: int
    stats: dict[str, dict[str, float]] = field(default_factory=dict)
    resource_totals: dict[str, int] = field(default_factory=dict)
    trajectory_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

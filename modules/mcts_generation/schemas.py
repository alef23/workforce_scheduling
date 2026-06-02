from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MCTSGenerationJob:
    job_id: str
    source_trajectory_id: str
    seed: int


@dataclass
class GeneratedSampleTrajectory:
    trajectory: list[dict[str, Any]]
    problem_setup: Any
    trajectory_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCTSWorkerResult:
    job_id: str
    source_trajectory_id: str
    trajectories: list[GeneratedSampleTrajectory]
    used_mcts: bool
    metadata: dict[str, Any] = field(default_factory=dict)

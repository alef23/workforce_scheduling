from __future__ import annotations

from typing import Protocol

from .schemas import GenerationJob, GenerationWorkerResult


class TrajectoryGenerationWorker(Protocol):
    worker_type: str

    def run(self, job: GenerationJob) -> GenerationWorkerResult:
        ...

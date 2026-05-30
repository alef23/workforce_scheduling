from __future__ import annotations

import random
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modules.storage import TrajectoryBuffer
from modules.trajectory_generation import (
    extract_actions_from_trajectory,
    flatten_action_chunks,
    reorder_chunks_for_expansion_mode,
    replay_actions_as_trajectory,
    split_actions_into_resource_chunks,
)
from modules.workforce_engine.engine import WorkforceEngine
from modules.workforce_engine.schemas import ProblemSetup

from .schemas import GeneratedTrajectory, GenerationJob, GenerationWorkerResult


@dataclass(frozen=True)
class StockAdjustmentConfig:
    p_stock: float = 0.2
    require_actual_reduction: bool = True
    max_reduction_attempts: int = 20

    def __post_init__(self) -> None:
        if not 0 <= self.p_stock <= 1:
            raise ValueError("p_stock debe estar entre 0 y 1.")
        if self.max_reduction_attempts <= 0:
            raise ValueError("max_reduction_attempts debe ser positivo.")


@dataclass
class StockAdjustmentTrajectoryWorker:
    source_buffer_path: str | Path
    config: StockAdjustmentConfig = StockAdjustmentConfig()
    trajectory_id_prefix: str = "stock"

    worker_type: str = "stock_adjustment"

    def run(self, job: GenerationJob) -> GenerationWorkerResult:
        source_trajectory_id = str(job.payload["source_trajectory_id"])
        rng = random.Random(int(job.seed))

        source_buffer = TrajectoryBuffer(self.source_buffer_path, mode="r")
        record = source_buffer.load(source_trajectory_id)

        problem_setup = ProblemSetup(**record.problem_setup)
        engine = WorkforceEngine(problem_setup)

        original_trajectory = self._samples_to_trajectory(record.samples)
        original_actions = extract_actions_from_trajectory(original_trajectory)
        original_stock = self._initial_stock(record.samples)
        initial_demand = self._initial_demand(record.samples)

        stock_was_reduced = rng.random() < float(self.config.p_stock)
        if stock_was_reduced:
            output_stock = self._sample_reduced_stock(
                original_stock=original_stock,
                rng=rng,
            )
        else:
            output_stock = list(original_stock)

        output_actions = self._actions_for_stock(
            actions=original_actions,
            output_stock=output_stock,
            stock_was_reduced=stock_was_reduced,
            rng=rng,
        )

        replayed = replay_actions_as_trajectory(
            initial_demand=initial_demand,
            initial_stock=output_stock,
            actions=output_actions,
            engine=engine,
            require_terminal=True,
        )

        trajectory = replayed["trajectory"]
        has_expansion_mode, first_expansion_step = self._expansion_metadata(trajectory)

        output_trajectory_id = f"{self.trajectory_id_prefix}_{source_trajectory_id}"
        metadata = {
            "stage": "stock_adjusted",
            "worker_type": self.worker_type,
            "job_id": job.job_id,
            "seed": int(job.seed),
            "source_trajectory_id": source_trajectory_id,
            "stock_was_reduced": bool(stock_was_reduced),
            "p_stock": float(self.config.p_stock),
            "original_stock": original_stock,
            "output_stock": output_stock,
            "has_expansion_mode": bool(has_expansion_mode),
            "first_expansion_step": first_expansion_step,
            "initial_demand_total": int(initial_demand.sum()),
            "final_reward": float(replayed["final_reward"]),
            "final_value": float(replayed["final_reward"]),
            "trajectory_length": int(len(trajectory)),
            "source_trajectory_length": int(len(record.samples)),
        }

        return GenerationWorkerResult(
            job_id=job.job_id,
            worker_type=self.worker_type,
            trajectories=[
                GeneratedTrajectory(
                    trajectory=trajectory,
                    problem_setup=problem_setup,
                    trajectory_id=output_trajectory_id,
                    metadata=metadata,
                )
            ],
            metadata=metadata,
        )

    @staticmethod
    def _samples_to_trajectory(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "state": sample["state"],
                "policy": sample["policy"],
                "action_id": sample["action_id"],
                "reward": sample["reward"],
            }
            for sample in samples
        ]

    @staticmethod
    def _initial_stock(samples: list[dict[str, Any]]) -> list[int]:
        return [
            int(value)
            for value in np.asarray(samples[0]["state"]["remaining_stock"], dtype=int)
        ]

    @staticmethod
    def _initial_demand(samples: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            samples[0]["state"]["residual_demand"],
            dtype=int,
        )

    def _sample_reduced_stock(
        self,
        original_stock: list[int],
        rng: random.Random,
    ) -> list[int]:
        for _ in range(int(self.config.max_reduction_attempts)):
            reduced = [
                rng.randint(0, int(original_stock[0])),
                rng.randint(0, int(original_stock[1])),
                rng.randint(0, int(original_stock[2])),
            ]
            if not self.config.require_actual_reduction or reduced != original_stock:
                return [int(value) for value in reduced]

        return self._force_reduction(original_stock=original_stock, rng=rng)

    @staticmethod
    def _force_reduction(
        original_stock: list[int],
        rng: random.Random,
    ) -> list[int]:
        reducible_indices = [
            index
            for index, value in enumerate(original_stock)
            if int(value) > 0
        ]
        if not reducible_indices:
            return [int(value) for value in original_stock]

        reduced = [int(value) for value in original_stock]
        index = rng.choice(reducible_indices)
        reduced[index] = rng.randint(0, reduced[index] - 1)
        return reduced

    @staticmethod
    def _actions_for_stock(
        actions: list[int],
        output_stock: list[int],
        stock_was_reduced: bool,
        rng: random.Random,
    ) -> list[int]:
        if not stock_was_reduced:
            return [int(action) for action in actions]

        chunks = split_actions_into_resource_chunks(actions)
        ordered_chunks = reorder_chunks_for_expansion_mode(
            resources_chunks=chunks,
            initial_stock=output_stock,
            rng=rng,
        )
        return flatten_action_chunks(ordered_chunks)

    @staticmethod
    def _expansion_metadata(
        trajectory: list[dict[str, Any]],
    ) -> tuple[bool, int | None]:
        for step_index, sample in enumerate(trajectory):
            state = sample["state"]
            if bool(getattr(state, "expansion_mode", False)):
                return True, int(step_index)
        return False, None


def build_stock_adjustment_jobs(
    source_trajectory_ids: list[str],
    seed: int | None = None,
    job_id_prefix: str = "stock_",
) -> list[GenerationJob]:
    jobs = []
    for index, source_trajectory_id in enumerate(source_trajectory_ids):
        job_seed = secrets.randbits(31) if seed is None else int(seed) + index
        jobs.append(
            GenerationJob(
                job_id=f"{job_id_prefix}{index:06d}",
                seed=job_seed,
                payload={
                    "source_trajectory_id": str(source_trajectory_id),
                },
            )
        )
    return jobs

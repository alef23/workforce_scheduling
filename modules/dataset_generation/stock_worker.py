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
    replay_actions_as_trajectory,
    split_actions_into_resource_chunks,
)
from modules.workforce_engine.engine import WorkforceEngine
from modules.workforce_engine.schemas import ProblemSetup

from .schemas import GeneratedTrajectory, GenerationJob, GenerationWorkerResult


@dataclass(frozen=True)
class StockAdjustmentConfig:
    p_stock: float = 0.2

    def __post_init__(self) -> None:
        if not 0 <= self.p_stock <= 1:
            raise ValueError("p_stock debe estar entre 0 y 1.")


@dataclass
class StockAdjustmentTrajectoryWorker:
    source_buffer_path: str | Path | None
    config: StockAdjustmentConfig = StockAdjustmentConfig()
    trajectory_id_prefix: str = "stock"

    worker_type: str = "stock_adjustment"

    def run(self, job: GenerationJob) -> GenerationWorkerResult:
        if self.source_buffer_path is None:
            raise ValueError("source_buffer_path es requerido para run().")

        source_trajectory_id = str(job.payload["source_trajectory_id"])
        rng = random.Random(int(job.seed))

        source_buffer = TrajectoryBuffer(self.source_buffer_path, mode="r")
        record = source_buffer.load(source_trajectory_id)

        problem_setup = ProblemSetup(**record.problem_setup)
        original_trajectory = self._samples_to_trajectory(record.samples)
        return self._adjust_trajectory(
            job=job,
            source_trajectory_id=source_trajectory_id,
            problem_setup=problem_setup,
            original_trajectory=original_trajectory,
            final_reward=float(record.final_reward),
            rng=rng,
        )

    def run_from_generated(
        self,
        job: GenerationJob,
        generated: GeneratedTrajectory,
    ) -> GenerationWorkerResult:
        source_trajectory_id = str(generated.trajectory_id)
        problem_setup = (
            generated.problem_setup
            if isinstance(generated.problem_setup, ProblemSetup)
            else ProblemSetup(**generated.problem_setup)
        )
        final_reward = float(
            generated.metadata.get(
                "final_reward",
                generated.trajectory[-1]["reward"],
            )
        )
        return self._adjust_trajectory(
            job=job,
            source_trajectory_id=source_trajectory_id,
            problem_setup=problem_setup,
            original_trajectory=generated.trajectory,
            final_reward=final_reward,
            rng=random.Random(int(job.seed)),
        )

    def _adjust_trajectory(
        self,
        job: GenerationJob,
        source_trajectory_id: str,
        problem_setup: ProblemSetup,
        original_trajectory: list[dict[str, Any]],
        final_reward: float,
        rng: random.Random,
    ) -> GenerationWorkerResult:
        original_actions = extract_actions_from_trajectory(original_trajectory)
        original_stock = self._initial_stock(original_trajectory)
        initial_demand = self._initial_demand(original_trajectory)
        output_trajectory_id = f"{self.trajectory_id_prefix}_{source_trajectory_id}"

        should_reduce_stock = rng.random() < float(self.config.p_stock)
        if not should_reduce_stock:
            has_expansion_mode, first_expansion_step = self._expansion_metadata(
                original_trajectory
            )
            metadata = self._build_metadata(
                job=job,
                source_trajectory_id=source_trajectory_id,
                stock_was_reduced=False,
                original_stock=original_stock,
                output_stock=list(original_stock),
                has_expansion_mode=has_expansion_mode,
                first_expansion_step=first_expansion_step,
                initial_demand_total=int(initial_demand.sum()),
                final_reward=float(final_reward),
                trajectory_length=len(original_trajectory),
                source_trajectory_length=len(original_trajectory),
                stock_cut_index=None,
            )
            return self._build_result(
                job=job,
                problem_setup=problem_setup,
                trajectory=original_trajectory,
                trajectory_id=output_trajectory_id,
                metadata=metadata,
            )

        chunks = split_actions_into_resource_chunks(original_actions)
        cut_index = self._sample_stock_cut_index(
            chunks=chunks,
            rng=rng,
        )
        if cut_index is None:
            has_expansion_mode, first_expansion_step = self._expansion_metadata(
                original_trajectory
            )
            metadata = self._build_metadata(
                job=job,
                source_trajectory_id=source_trajectory_id,
                stock_was_reduced=False,
                original_stock=original_stock,
                output_stock=list(original_stock),
                has_expansion_mode=has_expansion_mode,
                first_expansion_step=first_expansion_step,
                initial_demand_total=int(initial_demand.sum()),
                final_reward=float(final_reward),
                trajectory_length=len(original_trajectory),
                source_trajectory_length=len(original_trajectory),
                stock_cut_index=None,
            )
            return self._build_result(
                job=job,
                problem_setup=problem_setup,
                trajectory=original_trajectory,
                trajectory_id=output_trajectory_id,
                metadata=metadata,
            )

        output_stock = self._stock_from_pre_expansion_chunks(chunks[:cut_index])
        output_actions = flatten_action_chunks(chunks)
        engine = WorkforceEngine(problem_setup)
        replayed = replay_actions_as_trajectory(
            initial_demand=initial_demand,
            initial_stock=output_stock,
            actions=output_actions,
            engine=engine,
            require_terminal=True,
        )

        trajectory = replayed["trajectory"]
        has_expansion_mode, first_expansion_step = self._expansion_metadata(trajectory)

        metadata = self._build_metadata(
            job=job,
            source_trajectory_id=source_trajectory_id,
            stock_was_reduced=True,
            original_stock=original_stock,
            output_stock=output_stock,
            has_expansion_mode=has_expansion_mode,
            first_expansion_step=first_expansion_step,
            initial_demand_total=int(initial_demand.sum()),
            final_reward=float(replayed["final_reward"]),
            trajectory_length=len(trajectory),
            source_trajectory_length=len(original_trajectory),
            stock_cut_index=cut_index,
        )

        return self._build_result(
            job=job,
            problem_setup=problem_setup,
            trajectory=trajectory,
            trajectory_id=output_trajectory_id,
            metadata=metadata,
        )

    def _build_result(
        self,
        job: GenerationJob,
        problem_setup: ProblemSetup,
        trajectory: list[dict[str, Any]],
        trajectory_id: str,
        metadata: dict[str, Any],
    ) -> GenerationWorkerResult:
        return GenerationWorkerResult(
            job_id=job.job_id,
            worker_type=self.worker_type,
            trajectories=[
                GeneratedTrajectory(
                    trajectory=trajectory,
                    problem_setup=problem_setup,
                    trajectory_id=trajectory_id,
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
        state = samples[0]["state"]
        remaining_stock = (
            state["remaining_stock"]
            if isinstance(state, dict)
            else state.remaining_stock
        )
        return [
            int(value)
            for value in np.asarray(remaining_stock, dtype=int)
        ]

    @staticmethod
    def _initial_demand(samples: list[dict[str, Any]]) -> np.ndarray:
        state = samples[0]["state"]
        residual_demand = (
            state["residual_demand"]
            if isinstance(state, dict)
            else state.residual_demand
        )
        return np.asarray(
            residual_demand,
            dtype=int,
        )

    @staticmethod
    def _sample_stock_cut_index(
        chunks: list[list[int]],
        rng: random.Random,
    ) -> int | None:
        if len(chunks) < 2:
            return None
        return rng.randint(1, len(chunks) - 1)

    @staticmethod
    def _stock_from_pre_expansion_chunks(chunks: list[list[int]]) -> list[int]:
        stock = [0, 0, 0]
        for chunk in chunks:
            stock[int(chunk[0])] += 1
        return stock

    @staticmethod
    def _expansion_metadata(
        trajectory: list[dict[str, Any]],
    ) -> tuple[bool, int | None]:
        for step_index, sample in enumerate(trajectory):
            state = sample["state"]
            if isinstance(state, dict):
                expansion_mode = state.get("expansion_mode", False)
            else:
                expansion_mode = getattr(state, "expansion_mode", False)
            if bool(expansion_mode):
                return True, int(step_index)
        return False, None

    def _build_metadata(
        self,
        job: GenerationJob,
        source_trajectory_id: str,
        stock_was_reduced: bool,
        original_stock: list[int],
        output_stock: list[int],
        has_expansion_mode: bool,
        first_expansion_step: int | None,
        initial_demand_total: int,
        final_reward: float,
        trajectory_length: int,
        source_trajectory_length: int,
        stock_cut_index: int | None,
    ) -> dict[str, Any]:
        return {
            "stage": "stock_adjusted",
            "worker_type": self.worker_type,
            "job_id": job.job_id,
            "seed": int(job.seed),
            "source_trajectory_id": source_trajectory_id,
            "stock_was_reduced": bool(stock_was_reduced),
            "p_stock": float(self.config.p_stock),
            "original_stock": [int(value) for value in original_stock],
            "output_stock": [int(value) for value in output_stock],
            "stock_cut_index": stock_cut_index,
            "has_expansion_mode": bool(has_expansion_mode),
            "first_expansion_step": first_expansion_step,
            "initial_demand_total": int(initial_demand_total),
            "final_reward": float(final_reward),
            "final_value": float(final_reward),
            "trajectory_length": int(trajectory_length),
            "source_trajectory_length": int(source_trajectory_length),
        }


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

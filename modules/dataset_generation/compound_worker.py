from __future__ import annotations

import random
import secrets
from dataclasses import dataclass

from modules.demand_simulator import (
    CompoundDemandSimulator,
    DemandNoiseGenerator,
)
from modules.trajectory_generation import (
    CompoundStockAdjuster,
    CompoundTrajectoryReplayer,
)
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine
from modules.workforce_engine.schemas import ProblemSetup

from .compound_schemas import (
    CompoundGeneratedTrajectory,
    CompoundGenerationJob,
    CompoundWorkerResult,
)
from .raw_worker import NoiseGenerationConfig


@dataclass
class CompoundFullTrajectoryWorker:
    """Ejecuta raw, noise, replay y stock adjustment en memoria."""

    problem_setup: ProblemSetup
    # Límite superior; cada job samplea uniformemente entre 1 y este valor.
    n_resources: int = 20
    p_stock: float = 0.2
    noise_config: NoiseGenerationConfig = NoiseGenerationConfig()
    trajectory_id_prefix: str = "compound"

    def __post_init__(self) -> None:
        if self.n_resources <= 0 or self.n_resources > 20:
            raise ValueError("n_resources debe estar entre 1 y 20.")
        if not 0 <= self.p_stock <= 1:
            raise ValueError("p_stock debe estar entre 0 y 1.")
        if (
            self.noise_config.require_k_greater_than_max_overcoverage_tolerance
            and self.noise_config.k_max
            <= self.problem_setup.max_overcoverage_tolerance
        ):
            raise ValueError(
                "noise_config.k_max debe superar max_overcoverage_tolerance."
            )

    def run(self, job: CompoundGenerationJob) -> CompoundWorkerResult:
        system_rng = random.SystemRandom()
        sampled_n_resources = system_rng.randint(1, self.n_resources)
        coverage, base_trajectory = CompoundDemandSimulator(
            self.problem_setup
        ).compute_coverage(sampled_n_resources)
        noise_result = self._build_noise_generator(system_rng).generate(coverage)

        engine = CompoundWorkforceEngine(self.problem_setup)
        raw_result = CompoundTrajectoryReplayer(engine).replay_trajectory(
            initial_demand=noise_result.initial_demand,
            source_trajectory=base_trajectory,
        )
        stock_result = CompoundStockAdjuster(
            engine=engine,
            p_stock=self.p_stock,
        ).adjust(raw_result["trajectory"])

        trajectory_id = (
            f"{self.trajectory_id_prefix}_{job.job_id}_"
            f"{secrets.token_hex(4)}"
        )
        metadata = {
            "pipeline": "compound_raw_noise_stock",
            "job_id": job.job_id,
            "n_resources": int(sampled_n_resources),
            "max_n_resources": int(self.n_resources),
            "coverage_total": int(coverage.sum()),
            "coverage_max_cell": int(coverage.max()),
            "initial_demand_total": int(noise_result.initial_demand.sum()),
            "noise_k_max": float(self.noise_config.k_max),
            "noise_k_effective": float(noise_result.k_effective),
            "noise_discount_total": int(noise_result.discount_total),
            "raw_action_count": int(raw_result["consumed_action_count"]),
            "raw_stopped_early": bool(raw_result["stopped_early"]),
            "raw_final_reward": float(raw_result["final_reward"]),
            "original_stock": stock_result.original_stock.tolist(),
            "output_stock": stock_result.output_stock.tolist(),
            "stock_was_reduced": bool(stock_result.stock_was_reduced),
            "selected_chunk_indices": stock_result.selected_chunk_indices,
            "reordered_chunk_indices": stock_result.reordered_chunk_indices,
            "first_expansion_step": stock_result.first_expansion_step,
            "stopped_early": bool(stock_result.stopped_early),
            "source_action_count": int(stock_result.source_action_count),
            "consumed_action_count": int(stock_result.consumed_action_count),
            "trajectory_length": len(stock_result.trajectory),
            "final_reward": float(stock_result.final_reward),
            "final_value": float(stock_result.final_reward),
        }
        return CompoundWorkerResult(
            job_id=job.job_id,
            generated=CompoundGeneratedTrajectory(
                trajectory=stock_result.trajectory,
                problem_setup=self.problem_setup,
                trajectory_id=trajectory_id,
                metadata=metadata,
            ),
        )

    def _build_noise_generator(
        self,
        rng: random.SystemRandom,
    ) -> DemandNoiseGenerator:
        config = self.noise_config
        return DemandNoiseGenerator(
            k=float(config.k_max),
            max_daily_peaks=rng.randint(
                config.max_daily_peaks_min,
                config.max_daily_peaks_max,
            ),
            max_hourly_peaks=rng.randint(
                config.max_hourly_peaks_min,
                config.max_hourly_peaks_max,
            ),
            sigma_lambda=rng.uniform(
                config.sigma_lambda_min,
                config.sigma_lambda_max,
            ),
            sigma_alpha=rng.uniform(
                config.sigma_alpha_min,
                config.sigma_alpha_max,
            ),
            sigma_u=float(config.sigma_u),
            epsilon=float(config.epsilon),
            k_exponential_lambda=float(config.k_exponential_lambda),
            q_baseline=float(config.q_baseline),
            capacity_gamma=float(config.capacity_gamma),
            min_capacity_factor=float(config.min_capacity_factor),
        )

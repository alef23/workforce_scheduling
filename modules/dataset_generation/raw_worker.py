from __future__ import annotations

import random
from dataclasses import dataclass

from modules.demand_simulator import DemandNoiseGenerator, DemandSimulator
from modules.trajectory_generation import (
    ProblemSetupSampler,
    extract_actions_from_trajectory,
    replay_actions_as_trajectory,
)
from modules.workforce_engine.engine import WorkforceEngine

from .schemas import GeneratedTrajectory, GenerationJob, GenerationWorkerResult


@dataclass(frozen=True)
class ProblemSetupSamplingConfig:
    allowed_entry_hours: list[int] | None = None
    closing_hour: int | None = 22
    mobile_days_off_count: int | None = None
    fixed_day_off: int | None = None
    max_overcoverage_tolerance: float = 0.1
    random_entry_hours_count: int = 3
    random_entry_hours_pool: tuple[int, ...] = tuple(range(24))

    def __post_init__(self) -> None:
        if self.random_entry_hours_count <= 0:
            raise ValueError("random_entry_hours_count debe ser positivo.")
        if self.random_entry_hours_count > len(self.random_entry_hours_pool):
            raise ValueError(
                "random_entry_hours_count no puede superar el tamaño "
                "de random_entry_hours_pool."
            )

    def build_sampler(self, seed: int) -> ProblemSetupSampler:
        return ProblemSetupSampler(
            allowed_entry_hours=self.allowed_entry_hours,
            closing_hour=self.closing_hour,
            mobile_days_off_count=self.mobile_days_off_count,
            fixed_day_off=self.fixed_day_off,
            max_overcoverage_tolerance=self.max_overcoverage_tolerance,
            random_entry_hours_count=self.random_entry_hours_count,
            random_entry_hours_pool=self.random_entry_hours_pool,
            seed=seed,
        )


@dataclass(frozen=True)
class ResourceSamplingConfig:
    mod_4_max: int = 20
    mod_6_max: int = 20
    mod_8_max: int = 20

    def __post_init__(self) -> None:
        for name in ("mod_4_max", "mod_6_max", "mod_8_max"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} debe ser >= 0.")

        if self.mod_4_max + self.mod_6_max + self.mod_8_max <= 0:
            raise ValueError("Al menos una modalidad debe permitir stock mayor a cero.")


@dataclass(frozen=True)
class NoiseGenerationConfig:
    k_max: float = 0.8
    require_k_greater_than_max_overcoverage_tolerance: bool = True
    max_daily_peaks_min: int = 0
    max_daily_peaks_max: int = 4
    max_hourly_peaks_min: int = 0
    max_hourly_peaks_max: int = 2
    sigma_lambda_min: float = 0.0
    sigma_lambda_max: float = 3.0
    sigma_alpha_min: float = 0.0
    sigma_alpha_max: float = 3.0
    sigma_u: float = 0.1
    epsilon: float = 1e-9
    chi_square_c: float = 4.0
    q_baseline: float = 0.0
    capacity_gamma: float = 0.5
    min_capacity_factor: float = 0.30

    def __post_init__(self) -> None:
        if self.k_max < 0:
            raise ValueError("k_max debe ser >= 0.")

        for name in ("max_daily_peaks", "max_hourly_peaks"):
            min_value = getattr(self, f"{name}_min")
            max_value = getattr(self, f"{name}_max")
            if min_value < 0 or max_value < 0:
                raise ValueError(f"{name}_min y {name}_max deben ser >= 0.")
            if min_value > max_value:
                raise ValueError(f"{name}_min no puede ser mayor a {name}_max.")

        for name in ("sigma_lambda", "sigma_alpha"):
            min_value = getattr(self, f"{name}_min")
            max_value = getattr(self, f"{name}_max")
            if min_value < 0 or max_value < 0:
                raise ValueError(f"{name}_min y {name}_max deben ser >= 0.")
            if min_value > max_value:
                raise ValueError(f"{name}_min no puede ser mayor a {name}_max.")

        if self.sigma_u < 0:
            raise ValueError("sigma_u debe ser >= 0.")
        if self.epsilon <= 0:
            raise ValueError("epsilon debe ser positivo.")
        if self.chi_square_c <= 0:
            raise ValueError("chi_square_c debe ser positivo.")
        if self.q_baseline < 0:
            raise ValueError("q_baseline debe ser >= 0.")
        if self.capacity_gamma <= 0:
            raise ValueError("capacity_gamma debe ser positivo.")
        if not 0 <= self.min_capacity_factor <= 1:
            raise ValueError("min_capacity_factor debe estar entre 0 y 1.")


@dataclass
class RawDemandTrajectoryWorker:
    setup_config: ProblemSetupSamplingConfig
    resource_config: ResourceSamplingConfig
    noise_config: NoiseGenerationConfig
    trajectory_id_prefix: str = "raw"

    worker_type: str = "raw_demand"

    def run(self, job: GenerationJob) -> GenerationWorkerResult:
        rng = random.Random(int(job.seed))

        setup_seed = rng.randint(0, 2**31 - 1)
        simulator_seed = rng.randint(0, 2**31 - 1)
        noise_seed = rng.randint(0, 2**31 - 1)

        problem_setup = self.setup_config.build_sampler(seed=setup_seed).sample()
        initial_stock = self._sample_initial_stock(rng)

        self._validate_noise_config(problem_setup)

        simulator = DemandSimulator(
            problem_setup=problem_setup,
            seed=simulator_seed,
        )
        coverage_matrix, base_trajectory = simulator.compute_coverage(
            mod_4=initial_stock[0],
            mod_6=initial_stock[1],
            mod_8=initial_stock[2],
        )

        noise_generator = self._build_noise_generator(rng, noise_seed)
        noise_result = noise_generator.generate(coverage_matrix)

        engine = WorkforceEngine(problem_setup)
        base_actions = extract_actions_from_trajectory(base_trajectory)
        replayed = replay_actions_as_trajectory(
            initial_demand=noise_result.initial_demand,
            initial_stock=initial_stock,
            actions=base_actions,
            engine=engine,
            require_terminal=True,
        )

        trajectory_id = f"{self.trajectory_id_prefix}_{job.job_id}"
        metadata = {
            "stage": "raw",
            "worker_type": self.worker_type,
            "job_id": job.job_id,
            "seed": int(job.seed),
            "setup_seed": int(setup_seed),
            "simulator_seed": int(simulator_seed),
            "noise_seed": int(noise_seed),
            "initial_stock": initial_stock,
            "coverage_total": int(coverage_matrix.sum()),
            "initial_demand_total": int(noise_result.initial_demand.sum()),
            "noise_k_effective": float(noise_result.k_effective),
            "noise_discount_total": int(noise_result.discount_total),
            "final_reward": float(replayed["final_reward"]),
            "final_value": float(replayed["final_reward"]),
            "trajectory_length": int(len(replayed["trajectory"])),
        }

        return GenerationWorkerResult(
            job_id=job.job_id,
            worker_type=self.worker_type,
            trajectories=[
                GeneratedTrajectory(
                    trajectory=replayed["trajectory"],
                    problem_setup=problem_setup,
                    trajectory_id=trajectory_id,
                    metadata=metadata,
                )
            ],
            metadata=metadata,
        )

    def _sample_initial_stock(self, rng: random.Random) -> list[int]:
        config = self.resource_config

        while True:
            stock = [
                rng.randint(0, config.mod_4_max),
                rng.randint(0, config.mod_6_max),
                rng.randint(0, config.mod_8_max),
            ]
            if sum(stock) > 0:
                return [int(value) for value in stock]

    def _build_noise_generator(
        self,
        rng: random.Random,
        seed: int,
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
            chi_square_c=float(config.chi_square_c),
            q_baseline=float(config.q_baseline),
            capacity_gamma=float(config.capacity_gamma),
            min_capacity_factor=float(config.min_capacity_factor),
            seed=int(seed),
        )

    def _validate_noise_config(self, problem_setup) -> None:
        if not self.noise_config.require_k_greater_than_max_overcoverage_tolerance:
            return

        max_overcoverage_tolerance = float(problem_setup.max_overcoverage_tolerance)
        noise_k = float(self.noise_config.k_max)

        if noise_k <= max_overcoverage_tolerance:
            raise ValueError(
                "noise_config.k_max debe ser mayor que "
                "problem_setup.max_overcoverage_tolerance."
            )

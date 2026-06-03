# modules/trajectory_generation/scenario_generator.py

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .trajectory_replayer import (
    extract_actions_from_trajectory,
    replay_actions_as_trajectory,
)
from .stock_scenario_sampler import apply_stock_scenario
from .trajectory_augmentation import generate_augmented_action_sequences
from .mcts_expansion_sampler import generate_mcts_trajectories_from_states


@dataclass
class ScenarioGenerationConfig:
    mod_4_min: int = 0
    mod_4_max: int = 100
    mod_6_min: int = 0
    mod_6_max: int = 100
    mod_8_min: int = 0
    mod_8_max: int = 100

    augmented_samples_per_case: int = 10
    p_stock: float = 0.7

    p_mcts_from_state: float = 0.0
    max_mcts_trajectories_per_case: Optional[int] = None

    noise_k_max: float = 0.30
    require_noise_k_greater_than_scoring_k: bool = True

    noise_max_daily_peaks_min: int = 0
    noise_max_daily_peaks_max: int = 4
    noise_max_hourly_peaks_min: int = 0
    noise_max_hourly_peaks_max: int = 2

    sigma_lambda_min: float = 0.0
    sigma_lambda_max: float = 3.0
    sigma_alpha_min: float = 0.0
    sigma_alpha_max: float = 3.0

    sigma_u: float = 0.1
    epsilon: float = 1e-9
    k_exponential_lambda: float = 10.0
    q_baseline: float = 0.0

    seed: Optional[int] = None


@dataclass
class ScenarioGenerationResult:
    problem_setup: Any
    coverage_matrix: np.ndarray
    noise_result: Any

    corrected_trajectory: list[dict]
    trajectory_pre_mcts: list[dict]

    augmented_trajectories: list[list[dict]]
    mcts_trajectories: list[list[dict]]
    trajectories_to_save: list[list[dict]]

    stock_mode: bool
    initial_stock: list[int]
    stock_adjusted_initial_stock: list[int]


def _sample_resource_counts(
    config: ScenarioGenerationConfig,
    rng: random.Random,
) -> tuple[int, int, int]:

    while True:
        mod_4 = rng.randint(config.mod_4_min, config.mod_4_max)
        mod_6 = rng.randint(config.mod_6_min, config.mod_6_max)
        mod_8 = rng.randint(config.mod_8_min, config.mod_8_max)

        if mod_4 + mod_6 + mod_8 > 0:
            return mod_4, mod_6, mod_8


def _build_noise_generator(
    DemandNoiseGenerator,
    problem_setup: Any,
    config: ScenarioGenerationConfig,
    rng: random.Random,
):
    scoring_k = float(problem_setup.max_overcoverage_tolerance)
    noise_k_max = float(config.noise_k_max)

    if config.require_noise_k_greater_than_scoring_k:
        if noise_k_max <= scoring_k:
            raise ValueError(
                f"noise_k_max debe ser mayor que scoring_k. "
                f"noise_k_max={noise_k_max}, scoring_k={scoring_k}"
            )

    return DemandNoiseGenerator(
        k=noise_k_max,
        max_daily_peaks=rng.randint(
            config.noise_max_daily_peaks_min,
            config.noise_max_daily_peaks_max,
        ),
        max_hourly_peaks=rng.randint(
            config.noise_max_hourly_peaks_min,
            config.noise_max_hourly_peaks_max,
        ),
        sigma_lambda=rng.uniform(
            config.sigma_lambda_min,
            config.sigma_lambda_max,
        ),
        sigma_alpha=rng.uniform(
            config.sigma_alpha_min,
            config.sigma_alpha_max,
        ),
        sigma_u=config.sigma_u,
        epsilon=config.epsilon,
        k_exponential_lambda=config.k_exponential_lambda,
        q_baseline=config.q_baseline,
    )


def generate_one_scenario(
    *,
    problem_setup: Any,
    config: ScenarioGenerationConfig,
    DemandSimulator,
    DemandNoiseGenerator,
    WorkforceEngine,
    mcts_factory_builder: Callable[[Any], Callable[[], Any]] | None = None,
) -> ScenarioGenerationResult:
    """
    Genera un escenario completo en memoria.

    No guarda en Zarr.
    No importa módulos externos del proyecto.
    Recibe las clases existentes como dependencias.
    """

    rng = random.Random(config.seed)

    mod_4, mod_6, mod_8 = _sample_resource_counts(config, rng)
    initial_stock = [mod_4, mod_6, mod_8]

    simulator = DemandSimulator(
        problem_setup=problem_setup,
        seed=config.seed,
    )

    coverage_matrix, base_trajectory = simulator.compute_coverage(
        mod_4=mod_4,
        mod_6=mod_6,
        mod_8=mod_8,
    )

    noise_generator = _build_noise_generator(
        DemandNoiseGenerator=DemandNoiseGenerator,
        problem_setup=problem_setup,
        config=config,
        rng=rng,
    )

    noise_result = noise_generator.generate(coverage_matrix)

    engine = WorkforceEngine(problem_setup)

    base_actions = extract_actions_from_trajectory(base_trajectory)

    corrected_result = replay_actions_as_trajectory(
        initial_demand=noise_result.initial_demand,
        initial_stock=initial_stock,
        actions=base_actions,
        engine=engine,
        require_terminal=True,
    )

    corrected_trajectory = corrected_result["trajectory"]
    corrected_actions = extract_actions_from_trajectory(corrected_trajectory)

    stock_result = apply_stock_scenario(
        action_ids=corrected_actions,
        original_stock=initial_stock,
        p_stock=config.p_stock,
        seed=config.seed,
    )

    pre_mcts_result = replay_actions_as_trajectory(
        initial_demand=noise_result.initial_demand,
        initial_stock=stock_result.initial_stock,
        actions=stock_result.actions,
        engine=engine,
        require_terminal=True,
    )

    trajectory_pre_mcts = pre_mcts_result["trajectory"]

    mcts_trajectories = []

    if mcts_factory_builder is not None and config.p_mcts_from_state > 0:
        mcts_factory = mcts_factory_builder(engine)

        mcts_trajectories = generate_mcts_trajectories_from_states(
            trajectory=trajectory_pre_mcts,
            engine=engine,
            mcts_factory=mcts_factory,
            p_mcts_from_state=config.p_mcts_from_state,
            max_mcts_trajectories=config.max_mcts_trajectories_per_case,
            seed=config.seed,
            debug=False,
        )

    pre_mcts_actions = extract_actions_from_trajectory(trajectory_pre_mcts)

    augmented_action_sequences = generate_augmented_action_sequences(
        action_ids=pre_mcts_actions,
        n_samples=config.augmented_samples_per_case,
        initial_stock=stock_result.initial_stock,
        seed=config.seed,
    )

    augmented_trajectories = []

    for action_sequence in augmented_action_sequences:
        replayed_augmented = replay_actions_as_trajectory(
            initial_demand=noise_result.initial_demand,
            initial_stock=stock_result.initial_stock,
            actions=action_sequence,
            engine=engine,
            require_terminal=True,
        )

        augmented_trajectories.append(replayed_augmented["trajectory"])

    trajectories_to_save = (
        [trajectory_pre_mcts]
        + augmented_trajectories
        + mcts_trajectories
    )

    return ScenarioGenerationResult(
        problem_setup=problem_setup,
        coverage_matrix=coverage_matrix,
        noise_result=noise_result,
        corrected_trajectory=corrected_trajectory,
        trajectory_pre_mcts=trajectory_pre_mcts,
        augmented_trajectories=augmented_trajectories,
        mcts_trajectories=mcts_trajectories,
        trajectories_to_save=trajectories_to_save,
        stock_mode=stock_result.stock_mode,
        initial_stock=initial_stock,
        stock_adjusted_initial_stock=stock_result.initial_stock,
    )

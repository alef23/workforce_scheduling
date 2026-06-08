from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modules.mcts.mcts_schemas import MCTSConfig


class MCTSStartMode(str, Enum):
    INITIAL_ONLY = "initial_only"
    FORWARD_SAMPLED = "forward_sampled"
    BACKWARD_SAMPLED = "backward_sampled"
    TAIL_FORWARD_SAMPLED = "tail_forward_sampled"


@dataclass(frozen=True)
class ReweightedPolicyConfig:
    policy_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.policy_weight < 0:
            raise ValueError("policy_weight debe ser >= 0.")


@dataclass(frozen=True)
class MCTSGenerationConfig:
    p_mcts: float
    start_mode: MCTSStartMode
    max_seed_states: int
    seed_state_probability: float
    mcts_config: MCTSConfig
    tail_window_size: int | None = None
    mcts_policy_weight: float = 1.0
    reweighted_policy_config: ReweightedPolicyConfig = ReweightedPolicyConfig()

    def __post_init__(self) -> None:
        if not 0 <= self.p_mcts <= 1:
            raise ValueError("p_mcts debe estar entre 0 y 1.")
        if self.max_seed_states < 0:
            raise ValueError("max_seed_states debe ser >= 0.")
        if not 0 <= self.seed_state_probability <= 1:
            raise ValueError("seed_state_probability debe estar entre 0 y 1.")
        if self.tail_window_size is not None and self.tail_window_size <= 0:
            raise ValueError("tail_window_size debe ser positivo o None.")
        if (
            self.start_mode == MCTSStartMode.TAIL_FORWARD_SAMPLED
            and self.tail_window_size is None
        ):
            raise ValueError(
                "tail_window_size es requerido para tail_forward_sampled."
            )
        if self.mcts_policy_weight < 0:
            raise ValueError("mcts_policy_weight debe ser >= 0.")

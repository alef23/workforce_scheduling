"""Pydantic schemas for the MCTS module.

This file intentionally contains only MCTS-specific structures. It does not
import the Workforce Engine or any workforce-specific schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MCTSMode(str, Enum):
    """Final action selection mode."""

    TRAINING = "training"
    INFERENCE = "inference"


class MCTSConfig(BaseModel):
    """Configuration parameters for Monte Carlo Tree Search."""

    num_simulations: int = Field(..., gt=0)
    c_puct: float = Field(..., gt=0)
    temperature: float = Field(default=1.0, gt=0)
    mode: MCTSMode = MCTSMode.INFERENCE
    random_seed: Optional[int] = None
    debug: bool = False


class MCTSAction(BaseModel):
    """Statistics for one outgoing edge/action from an MCTS node."""

    action_id: int = Field(..., ge=0)
    prior: float = Field(..., ge=0)
    visit_count: int = Field(default=0, ge=0)
    value_sum: float = 0.0
    q_value: float = 0.0
    child_node_id: Optional[int] = None

    def update(self, value: float) -> None:
        """Update N(s,a), W(s,a) and Q(s,a)."""

        self.visit_count += 1
        self.value_sum += float(value)
        self.q_value = self.value_sum / self.visit_count


class MCTSNode(BaseModel):
    """Node of the MCTS tree.

    The state is intentionally typed as Any to keep this module independent
    from the specific environment state class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_id: int = Field(..., ge=0)
    parent_id: Optional[int] = None
    parent_action_id: Optional[int] = None
    state: Any
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_reward: Optional[float] = None
    value_estimate: Optional[float] = None
    actions: Dict[int, MCTSAction] = Field(default_factory=dict)

    @property
    def visit_count(self) -> int:
        """Derived node visit count N(s) = sum_a N(s,a)."""

        return sum(action.visit_count for action in self.actions.values())


class MCTSResult(BaseModel):
    """Result returned by one MCTS search from a root state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root_node_id: int
    selected_action_id: int
    policy: np.ndarray
    root_stats: Dict[int, Dict[str, Any]]
    num_simulations: int
    diagnostics: Optional[Dict[str, Any]] = None

    @field_validator("policy", mode="before")
    @classmethod
    def coerce_policy(cls, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=float)

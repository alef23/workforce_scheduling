"""Domain-agnostic Monte Carlo Tree Search implementation.

The MCTS module is intentionally independent from the Workforce Engine
implementation. It only requires an engine-like object that implements:

- action_space_size: int
- step(state, action_id) -> StepResult-like object with next_state, is_terminal and reward
- legal_mask(state, policy) -> np.ndarray
- check_terminality(state) -> bool
- compute_reward(state) -> float

And an evaluator-like object that implements:

- predict(state) -> tuple[policy, value]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from .mcts_schemas import MCTSAction, MCTSConfig, MCTSMode, MCTSNode, MCTSResult
from modules.evaluators.base import EvaluatorProtocol


class StepResultProtocol(Protocol):
    """Minimal step result interface required by MCTS."""

    next_state: Any
    is_terminal: bool
    reward: float


class EngineProtocol(Protocol):
    """Minimal environment interface required by MCTS."""

    action_space_size: int

    def step(self, state: Any, action_id: int) -> StepResultProtocol:
        """Apply one action and return a StepResult-like object."""

    def legal_mask(self, state: Any, policy: np.ndarray) -> np.ndarray:
        """Mask illegal actions and return a normalized probability vector."""

    def check_terminality(self, state: Any) -> bool:
        """Return whether the state is terminal."""

    def compute_reward(self, state: Any) -> float:
        """Return terminal reward for a terminal state."""


class MCTS:
    """Monte Carlo Tree Search.

    This class does not know domain rules. It delegates transitions and action
    legality to the engine, and obtains policy/value from the evaluator.
    """

    def __init__(
        self,
        engine: EngineProtocol,
        evaluator: EvaluatorProtocol,
        config: MCTSConfig,
    ) -> None:
        self.engine = engine
        self.evaluator = evaluator
        self.config = config
        self.nodes: Dict[int, MCTSNode] = {}
        self.root_node_id: Optional[int] = None
        self.next_node_id: int = 0
        self.rng = np.random.default_rng(config.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(self, root_state: Any) -> MCTSResult:
        """Run MCTS simulations from the current root and return a decision."""

        if self.root_node_id is None:
            is_terminal = self.engine.check_terminality(root_state)
            self.root_node_id = self.create_node(
                state=root_state,
                parent_id=None,
                parent_action_id=None,
                is_terminal=is_terminal,
                terminal_reward=self.engine.compute_reward(root_state) if is_terminal else None,
            )

        root = self.nodes[self.root_node_id]
        if root.is_terminal:
            raise ValueError("Cannot search from a terminal root node")

        if not root.is_expanded:
            self.expand_node(self.root_node_id)

        for _ in range(self.config.num_simulations):
            self.run_simulation(self.root_node_id)

        policy = self.compute_root_policy(self.root_node_id)
        selected_action_id = self.select_final_action(policy)
        root_stats = self.get_root_stats()

        diagnostics = None
        if self.config.debug:
            diagnostics = {
                "num_nodes": len(self.nodes),
                "root_visit_count": self.nodes[self.root_node_id].visit_count,
            }

        return MCTSResult(
            root_node_id=self.root_node_id,
            selected_action_id=selected_action_id,
            policy=policy,
            root_stats=root_stats,
            num_simulations=self.config.num_simulations,
            diagnostics=diagnostics,
        )

    def advance_root(self, action_id: int) -> int:
        """Advance the root to the child reached by the executed action."""

        if self.root_node_id is None:
            raise ValueError("Cannot advance root because root_node_id is None")

        root = self.nodes[self.root_node_id]
        if action_id not in root.actions:
            raise ValueError(f"Action {action_id} is not available in the current root")

        child_node_id = self.get_or_create_child(self.root_node_id, action_id)
        child = self.nodes[child_node_id]
        child.parent_id = None
        child.parent_action_id = None
        self.root_node_id = child_node_id
        return child_node_id

    def reset_tree(self) -> None:
        """Clear the entire search tree."""

        self.nodes = {}
        self.root_node_id = None
        self.next_node_id = 0

    def get_root_policy(self) -> np.ndarray:
        """Return the current root policy computed from visit counts."""

        if self.root_node_id is None:
            raise ValueError("root_node_id is None")
        return self.compute_root_policy(self.root_node_id)

    def get_root_stats(self) -> Dict[int, Dict[str, Any]]:
        """Return MCTS statistics for the current root actions."""

        if self.root_node_id is None:
            raise ValueError("root_node_id is None")
        root = self.nodes[self.root_node_id]
        return {
            action_id: {
                "prior": action.prior,
                "visit_count": action.visit_count,
                "value_sum": action.value_sum,
                "q_value": action.q_value,
                "child_node_id": action.child_node_id,
            }
            for action_id, action in root.actions.items()
        }

    # ------------------------------------------------------------------
    # Node lifecycle
    # ------------------------------------------------------------------
    def create_node(
        self,
        state: Any,
        parent_id: Optional[int],
        parent_action_id: Optional[int],
        is_terminal: bool = False,
        terminal_reward: Optional[float] = None,
    ) -> int:
        """Create and register a node without expanding it."""

        node_id = self.next_node_id
        self.next_node_id += 1

        node = MCTSNode(
            node_id=node_id,
            parent_id=parent_id,
            parent_action_id=parent_action_id,
            state=state,
            is_expanded=False,
            is_terminal=is_terminal,
            terminal_reward=terminal_reward,
            value_estimate=None,
            actions={},
        )
        self.nodes[node_id] = node
        return node_id

    def expand_node(self, node_id: int) -> float:
        """Expand a non-terminal node using evaluator policy/value."""

        node = self.nodes[node_id]
        if node.is_terminal:
            if node.terminal_reward is None:
                raise ValueError("Terminal node has no terminal_reward")
            return node.terminal_reward

        policy, value = self.evaluator.predict(node.state)
        policy = np.asarray(policy, dtype=float)
        if policy.shape != (self.engine.action_space_size,):
            raise ValueError(
                f"Evaluator policy must have shape ({self.engine.action_space_size},)"
            )

        masked_probs = np.asarray(self.engine.legal_mask(node.state, policy), dtype=float)
        if masked_probs.shape != (self.engine.action_space_size,):
            raise ValueError(
                f"Masked probs must have shape ({self.engine.action_space_size},)"
            )

        node.actions = {
            int(action_id): MCTSAction(action_id=int(action_id), prior=float(prob))
            for action_id, prob in enumerate(masked_probs)
            if prob > 0
        }
        if not node.actions:
            raise ValueError("Expanded node has no legal actions")

        node.value_estimate = float(value)
        node.is_expanded = True
        return float(value)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def run_simulation(self, root_node_id: int) -> float:
        """Run one full simulation from root and backpropagate its value."""

        current_node_id = root_node_id
        path: List[Tuple[int, int]] = []

        while True:
            node = self.nodes[current_node_id]

            if node.is_terminal:
                if node.terminal_reward is None:
                    raise ValueError("Terminal node has no terminal_reward")
                v_sim = node.terminal_reward
                break

            if not node.is_expanded:
                v_sim = self.expand_node(current_node_id)
                break

            action_id = self.select_action_puct(current_node_id)
            path.append((current_node_id, action_id))
            current_node_id = self.get_or_create_child(current_node_id, action_id)

        self.backpropagate(path, v_sim)
        return float(v_sim)

    def select_action_puct(self, node_id: int) -> int:
        """Select the best action from a node according to PUCT."""

        scores = self.compute_puct_scores(node_id)
        return max(scores.items(), key=lambda item: (item[1], -item[0]))[0]

    def compute_puct_scores(self, node_id: int) -> Dict[int, float]:
        """Compute Q(s,a) + U(s,a) for each legal action in a node."""

        node = self.nodes[node_id]
        if not node.actions:
            raise ValueError("Cannot compute PUCT scores for a node without actions")

        total_visits = sum(action.visit_count for action in node.actions.values())
        sqrt_total = np.sqrt(max(1, total_visits))

        scores: Dict[int, float] = {}
        for action_id, action in node.actions.items():
            exploration = (
                self.config.c_puct
                * action.prior
                * sqrt_total
                / (1 + action.visit_count)
            )
            scores[action_id] = action.q_value + exploration
        return scores

    def get_or_create_child(self, node_id: int, action_id: int) -> int:
        """Return an existing child or generate it through engine.step."""

        node = self.nodes[node_id]
        action = node.actions[action_id]

        if action.child_node_id is not None:
            return action.child_node_id

        step_result = self.engine.step(node.state, action_id)
        child_node_id = self.create_node(
            state=step_result.next_state,
            parent_id=node_id,
            parent_action_id=action_id,
            is_terminal=step_result.is_terminal,
            terminal_reward=step_result.reward if step_result.is_terminal else None,
        )
        action.child_node_id = child_node_id
        return child_node_id

    def backpropagate(self, path: List[Tuple[int, int]], v_sim: float) -> None:
        """Update edge statistics along a simulation path."""

        for node_id, action_id in path:
            action = self.nodes[node_id].actions[action_id]
            action.update(v_sim)

    # ------------------------------------------------------------------
    # Root policy and final action
    # ------------------------------------------------------------------
    def compute_root_policy(self, root_node_id: int) -> np.ndarray:
        """Compute root policy from visits, using priors as fallback."""

        policy = np.zeros(self.engine.action_space_size, dtype=float)
        root = self.nodes[root_node_id]
        if not root.actions:
            return policy

        total_visits = sum(action.visit_count for action in root.actions.values())
        if total_visits > 0:
            for action_id, action in root.actions.items():
                policy[action_id] = action.visit_count / total_visits
            return policy

        prior_sum = sum(action.prior for action in root.actions.values())
        if prior_sum <= 0:
            raise ValueError("Cannot compute root policy: no visits and zero prior sum")
        for action_id, action in root.actions.items():
            policy[action_id] = action.prior / prior_sum
        return policy

    def select_final_action(self, policy: np.ndarray) -> int:
        """Select the real action from the root policy."""

        if self.config.mode == MCTSMode.INFERENCE:
            return int(np.argmax(policy))

        probs = np.asarray(policy, dtype=float)
        total = probs.sum()
        if total <= 0:
            raise ValueError("Cannot sample from an empty policy")
        probs = probs / total
        return int(self.rng.choice(np.arange(len(probs)), p=probs))

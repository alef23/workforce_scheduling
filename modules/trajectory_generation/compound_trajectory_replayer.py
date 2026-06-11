from __future__ import annotations

from typing import Any

import numpy as np

from modules.workforce_engine.compound_actions import ACTION_SPACE_SIZE
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine
from modules.workforce_engine.compound_schemas import CompoundWorkforceState


class CompoundTrajectoryReplayer:
    """Reconstruye trayectorias compuestas mediante el engine canónico."""

    def __init__(self, engine: CompoundWorkforceEngine) -> None:
        self.engine = engine

    def replay_trajectory(
        self,
        initial_demand: np.ndarray,
        source_trajectory: list[dict[str, Any]],
        initial_stock: list[int] | np.ndarray | None = None,
        require_terminal: bool = True,
    ) -> dict[str, Any]:
        """Reproduce los action_id de una trayectoria base."""
        if not source_trajectory:
            raise ValueError("source_trajectory no puede estar vacía.")

        stock = (
            initial_stock
            if initial_stock is not None
            else self._state_value(
                source_trajectory[0]["state"],
                "remaining_stock",
            )
        )
        actions = [
            int(sample["action_id"])
            for sample in source_trajectory
        ]
        return self.replay_actions(
            initial_demand=initial_demand,
            initial_stock=stock,
            actions=actions,
            require_terminal=require_terminal,
        )

    def replay_actions(
        self,
        initial_demand: np.ndarray,
        initial_stock: list[int] | np.ndarray,
        actions: list[int],
        require_terminal: bool = True,
    ) -> dict[str, Any]:
        """
        Aplica acciones hasta terminalidad o hasta agotar la secuencia.

        El reward terminal se copia a todos los samples generados. Si la
        terminalidad ocurre antes, las acciones restantes no se ejecutan.
        """
        if not actions:
            raise ValueError("actions no puede estar vacía.")

        current_state = self._build_initial_state(
            initial_demand=initial_demand,
            initial_stock=initial_stock,
        )
        if self.engine.check_terminality(current_state):
            raise ValueError("El estado inicial ya es terminal.")

        trajectory: list[dict[str, Any]] = []
        final_reward: float | None = None
        is_terminal = False

        for source_step_index, raw_action_id in enumerate(actions):
            action_id = int(raw_action_id)
            legal_actions = self.engine.get_legal_actions(current_state)
            if (
                action_id < 0
                or action_id >= ACTION_SPACE_SIZE
                or not legal_actions[action_id]
            ):
                legal_ids = np.flatnonzero(legal_actions).tolist()
                raise ValueError(
                    f"Acción ilegal durante replay: action_id={action_id}. "
                    f"Acciones legales={legal_ids}"
                )

            trajectory.append(
                {
                    "state": current_state,
                    "policy": self._uniform_policy(legal_actions),
                    "action_id": action_id,
                    "reward": None,
                    "metadata": {
                        "source_step_index": int(source_step_index),
                    },
                }
            )
            step_result = self.engine.step(current_state, action_id)
            current_state = step_result.next_state
            is_terminal = bool(step_result.is_terminal)

            if is_terminal:
                final_reward = float(step_result.reward)
                break

        if final_reward is None:
            if require_terminal:
                raise RuntimeError("La secuencia de acciones no alcanzó terminalidad.")
            final_reward = 0.0

        for sample in trajectory:
            sample["reward"] = final_reward

        consumed_action_count = len(trajectory)
        source_action_count = len(actions)
        return {
            "trajectory": trajectory,
            "final_state": current_state,
            "final_reward": final_reward,
            "is_terminal": is_terminal,
            "source_action_count": source_action_count,
            "consumed_action_count": consumed_action_count,
            "stopped_early": consumed_action_count < source_action_count,
        }

    @staticmethod
    def _build_initial_state(
        initial_demand: np.ndarray,
        initial_stock: list[int] | np.ndarray,
    ) -> CompoundWorkforceState:
        demand = np.asarray(initial_demand)
        stock = np.asarray(initial_stock)
        initial_demand_total = int(demand.sum())
        if initial_demand_total <= 0:
            raise ValueError("initial_demand debe tener suma positiva.")

        return CompoundWorkforceState(
            residual_demand=demand,
            remaining_stock=stock,
            expansion_mode=bool(np.all(stock == 0)),
            current_modality=None,
            assignment_week=0,
            initial_demand_total=initial_demand_total,
        )

    @staticmethod
    def _uniform_policy(legal_actions: np.ndarray) -> np.ndarray:
        legal_count = int(legal_actions.sum())
        if legal_count <= 0:
            raise RuntimeError("No hay acciones legales para construir la policy.")

        policy = np.zeros(ACTION_SPACE_SIZE, dtype=float)
        policy[legal_actions] = 1.0 / legal_count
        return policy

    @staticmethod
    def _state_value(state: Any, field: str) -> Any:
        if isinstance(state, dict):
            return state[field]
        return getattr(state, field)

from __future__ import annotations

from typing import Any

import numpy as np

from modules.workforce_engine.schemas import WorkforceState


ACTION_SPACE_SIZE = 55


def extract_actions_from_trajectory(trajectory: list[dict]) -> list[int]:
    """Extrae action_id de una trayectoria."""

    return [int(sample["action_id"]) for sample in trajectory]


def build_uniform_policy_from_legal_actions(
    legal_actions: Any,
    action_space_size: int = ACTION_SPACE_SIZE,
) -> np.ndarray:
    """
    Construye una policy uniforme sobre acciones legales.

    Acepta:
    - máscara booleana shape (55,)
    - lista/array de ids legales
    """

    policy = np.zeros(action_space_size, dtype=float)
    legal = np.asarray(legal_actions)

    if legal.dtype == bool:
        legal_ids = np.where(legal)[0]
    else:
        legal_ids = legal.astype(int)

    if len(legal_ids) == 0:
        raise ValueError("No hay acciones legales para construir policy.")

    policy[legal_ids] = 1.0 / len(legal_ids)
    return policy


def _legal_action_ids(legal_actions: Any) -> set[int]:
    legal = np.asarray(legal_actions)

    if legal.dtype == bool:
        return set(int(i) for i in np.where(legal)[0])

    return set(int(i) for i in legal.astype(int))


def build_initial_state(
    initial_demand: np.ndarray,
    initial_stock: list[int] | np.ndarray,
) -> WorkforceState:
    """Construye el estado inicial para replay."""

    demand = np.asarray(initial_demand, dtype=int)
    stock = np.asarray(initial_stock, dtype=int)

    return WorkforceState(
        residual_demand=demand,
        remaining_stock=stock,
        expansion_mode=bool(np.all(stock == 0)),
        current_modality=None,
        current_entry_hour=None,
        assignment_week=0,
        initial_demand_total=float(demand.sum()),
    )


def replay_actions_as_trajectory(
    initial_demand: np.ndarray,
    initial_stock: list[int] | np.ndarray,
    actions: list[int],
    engine: Any,
    require_terminal: bool = True,
) -> dict[str, Any]:
    """
    Reconstruye una trayectoria completa aplicando una secuencia de acciones.

    Este método centraliza la reconstrucción de:
    - state
    - policy uniforme legal
    - action_id
    - reward final
    """

    current_state = build_initial_state(
        initial_demand=initial_demand,
        initial_stock=initial_stock,
    )

    trajectory: list[dict[str, Any]] = []
    is_terminal = False
    final_reward: float | None = None

    for action_id in actions:
        action_id = int(action_id)

        legal_actions = engine.get_legal_actions(current_state)
        legal_ids = _legal_action_ids(legal_actions)

        if action_id not in legal_ids:
            raise ValueError(
                f"Acción ilegal durante replay: action_id={action_id}. "
                f"Acciones legales={sorted(legal_ids)}"
            )

        policy = build_uniform_policy_from_legal_actions(legal_actions)

        trajectory.append(
            {
                "state": current_state,
                "policy": policy,
                "action_id": action_id,
                "reward": None,
            }
        )

        step_result = engine.step(current_state, action_id)

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

    return {
        "trajectory": trajectory,
        "final_state": current_state,
        "final_reward": final_reward,
        "is_terminal": is_terminal,
    }

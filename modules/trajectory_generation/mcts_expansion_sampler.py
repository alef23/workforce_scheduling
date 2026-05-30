from __future__ import annotations

from typing import Any, Callable
import random


def generate_mcts_trajectory(
    initial_state: Any,
    engine: Any,
    mcts: Any,
    debug: bool = False,
) -> tuple[list[dict[str, Any]], float, Any]:
    """Genera una trayectoria completa usando MCTS desde un estado inicial."""

    trajectory: list[dict[str, Any]] = []

    current_state = initial_state
    is_terminal = False
    final_reward = 0.0
    step_count = 0

    while not is_terminal:
        mcts_result = mcts.search(current_state)

        selected_action_id = int(mcts_result.selected_action_id)
        policy = mcts_result.policy

        trajectory.append(
            {
                "state": current_state,
                "policy": policy,
                "action_id": selected_action_id,
                "reward": None,
            }
        )

        step_result = engine.step(current_state, selected_action_id)

        current_state = step_result.next_state
        is_terminal = bool(step_result.is_terminal)

        mcts.advance_root(selected_action_id)
        step_count += 1

        if debug:
            print(
                f"[MCTS] step={step_count} | "
                f"action={selected_action_id} | "
                f"terminal={is_terminal} | "
                f"reward={step_result.reward}"
            )

        if is_terminal:
            final_reward = float(step_result.reward)

    for sample in trajectory:
        sample["reward"] = final_reward

    return trajectory, final_reward, current_state


def generate_mcts_trajectories_from_states(
    trajectory: list[dict[str, Any]],
    engine: Any,
    mcts_factory: Callable[[], Any],
    p_mcts_from_state: float,
    max_mcts_trajectories: int | None = None,
    seed: int | None = None,
    debug: bool = False,
) -> list[list[dict[str, Any]]]:
    """
    Genera trayectorias MCTS desde estados de una trayectoria base.

    Recorre cada estado y, con probabilidad p_mcts_from_state,
    continúa desde ese estado usando MCTS.

    Las trayectorias resultantes NO se augmentan posteriormente.
    """

    rng = random.Random(seed)
    mcts_trajectories: list[list[dict[str, Any]]] = []

    if p_mcts_from_state <= 0:
        return mcts_trajectories

    for state_idx, sample in enumerate(trajectory):
        if max_mcts_trajectories is not None:
            if len(mcts_trajectories) >= int(max_mcts_trajectories):
                return mcts_trajectories

        if rng.random() > float(p_mcts_from_state):
            continue

        state = sample["state"]

        if engine.check_terminality(state):
            continue

        initial_state = state.copy_state() if hasattr(state, "copy_state") else state
        mcts = mcts_factory()

        mcts_trajectory, final_reward, final_state = generate_mcts_trajectory(
            initial_state=initial_state,
            engine=engine,
            mcts=mcts,
            debug=False,
        )

        mcts_trajectories.append(mcts_trajectory)

        if debug:
            print(
                f"[MCTS from state] "
                f"state_idx={state_idx} | "
                f"len={len(mcts_trajectory)} | "
                f"reward={final_reward:.6f}"
            )

    return mcts_trajectories

from __future__ import annotations

import random

from .config import MCTSStartMode


def select_seed_step_indices(
    trajectory_length: int,
    start_mode: MCTSStartMode,
    max_seed_states: int,
    seed_state_probability: float,
    rng: random.Random,
    tail_window_size: int | None = None,
) -> list[int]:
    """
    Selecciona indices de estados para iniciar trayectorias MCTS.

    INITIAL_ONLY devuelve solo el estado 0.
    Los modos sampleados incluyen siempre el estado 0 y luego agregan hasta
    max_seed_states estados sampleados desde la trayectoria fuente.
    Las trayectorias MCTS generadas desde esos estados deben continuar hasta
    terminalidad; este helper solo elige puntos de partida.
    """
    if trajectory_length <= 0:
        raise ValueError("trajectory_length debe ser positivo.")
    if max_seed_states < 0:
        raise ValueError("max_seed_states debe ser >= 0.")
    if not 0 <= seed_state_probability <= 1:
        raise ValueError("seed_state_probability debe estar entre 0 y 1.")
    if tail_window_size is not None and tail_window_size <= 0:
        raise ValueError("tail_window_size debe ser positivo o None.")

    mode = MCTSStartMode(start_mode)
    if mode == MCTSStartMode.INITIAL_ONLY:
        return [0]
    if mode == MCTSStartMode.TAIL_FORWARD_SAMPLED and tail_window_size is None:
        raise ValueError(
            "tail_window_size es requerido para tail_forward_sampled."
        )

    sampled = _sample_additional_seed_indices(
        trajectory_length=trajectory_length,
        mode=mode,
        max_seed_states=max_seed_states,
        seed_state_probability=seed_state_probability,
        rng=rng,
        tail_window_size=tail_window_size,
    )
    return [0] + sampled


def _sample_additional_seed_indices(
    trajectory_length: int,
    mode: MCTSStartMode,
    max_seed_states: int,
    seed_state_probability: float,
    rng: random.Random,
    tail_window_size: int | None,
) -> list[int]:
    if max_seed_states == 0:
        return []

    if mode == MCTSStartMode.FORWARD_SAMPLED:
        candidate_indices = range(1, trajectory_length)
    elif mode == MCTSStartMode.BACKWARD_SAMPLED:
        candidate_indices = range(max(trajectory_length - 2, 0), 0, -1)
    elif mode == MCTSStartMode.TAIL_FORWARD_SAMPLED:
        terminal_index = trajectory_length - 1
        first_candidate = max(1, terminal_index - int(tail_window_size))
        candidate_indices = range(first_candidate, terminal_index)
    else:
        raise ValueError(f"Modo no soportado: {mode}")

    selected: list[int] = []
    for step_index in candidate_indices:
        if rng.random() > seed_state_probability:
            continue
        selected.append(int(step_index))
        if len(selected) >= max_seed_states:
            break

    return selected

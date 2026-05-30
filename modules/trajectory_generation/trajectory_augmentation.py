from __future__ import annotations

import random


MODALITY_ACTION_IDS = {0, 1, 2}


def split_actions_into_resource_chunks(action_ids: list[int]) -> list[list[int]]:
    """
    Corta una secuencia de action_id en chunks de recursos completos.

    Cada chunk comienza con una acción de modalidad:
        0 -> modalidad 4h
        1 -> modalidad 6h
        2 -> modalidad 8h
    """

    chunks: list[list[int]] = []
    current_chunk: list[int] = []

    for action_id in action_ids:
        action_id = int(action_id)

        if action_id in MODALITY_ACTION_IDS:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [action_id]
        else:
            current_chunk.append(action_id)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def flatten_action_chunks(chunks: list[list[int]]) -> list[int]:
    """Convierte una lista de chunks en una lista plana de acciones."""

    return [int(action_id) for chunk in chunks for action_id in chunk]


def reorder_chunks_for_expansion_mode(
    resources_chunks: list[list[int]],
    initial_stock: list[int],
    rng: random.Random | None = None,
) -> list[list[int]]:
    """
    Reordena chunks para respetar el stock inicial.

    Criterio:
    - Primero ubica chunks que entran dentro del stock inicial.
    - Luego ubica chunks excedentes, que se ejecutarán en expansion_mode.
    - Mezcla aleatoriamente cada grupo.
    """

    rng = rng or random

    used_stock = [0, 0, 0]
    pre_expansion_chunks: list[list[int]] = []
    post_expansion_chunks: list[list[int]] = []

    for chunk in resources_chunks:
        modality_idx = int(chunk[0])

        if used_stock[modality_idx] < int(initial_stock[modality_idx]):
            pre_expansion_chunks.append(chunk)
            used_stock[modality_idx] += 1
        else:
            post_expansion_chunks.append(chunk)

    rng.shuffle(pre_expansion_chunks)
    rng.shuffle(post_expansion_chunks)

    return pre_expansion_chunks + post_expansion_chunks


def generate_augmented_action_sequences(
    action_ids: list[int],
    n_samples: int,
    initial_stock: list[int] | None = None,
    seed: int | None = None,
) -> list[list[int]]:
    """
    Genera nuevas secuencias por reordenamiento de chunks.

    Si initial_stock se informa, respeta la lógica de expansion_mode:
    primero chunks cubiertos por stock, luego chunks excedentes.

    Devuelve únicamente secuencias de acciones. No reconstruye estados.
    """

    rng = random.Random(seed)
    resources_chunks = split_actions_into_resource_chunks(action_ids)

    augmented_sequences: list[list[int]] = []

    for _ in range(int(n_samples)):
        chunks = [chunk.copy() for chunk in resources_chunks]
        rng.shuffle(chunks)

        if initial_stock is None:
            ordered_chunks = chunks
        else:
            ordered_chunks = reorder_chunks_for_expansion_mode(
                resources_chunks=chunks,
                initial_stock=initial_stock,
                rng=rng,
            )

        augmented_sequences.append(flatten_action_chunks(ordered_chunks))

    return augmented_sequences

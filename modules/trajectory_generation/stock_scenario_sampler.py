from __future__ import annotations

import random
from dataclasses import dataclass

from .trajectory_augmentation import (
    split_actions_into_resource_chunks,
    flatten_action_chunks,
    reorder_chunks_for_expansion_mode,
)


@dataclass
class StockScenarioResult:
    """Resultado de aplicar o no una reducción de stock."""

    actions: list[int]
    initial_stock: list[int]
    stock_mode: bool


def count_resource_chunks_by_modality(action_ids: list[int]) -> list[int]:
    """
    Cuenta cuántos chunks de recursos hay por modalidad.

    Devuelve:
        [cantidad_mod4, cantidad_mod6, cantidad_mod8]
    """

    chunks = split_actions_into_resource_chunks(action_ids)
    counts = [0, 0, 0]

    for chunk in chunks:
        modality_idx = int(chunk[0])
        counts[modality_idx] += 1

    return counts


def sample_reduced_stock(
    modality_counts: list[int],
    rng: random.Random | None = None,
) -> list[int]:
    """
    Samplea un stock reducido usando cantidad real de chunks disponibles.

    Garantiza:
        reduced_stock[m] <= cantidad de chunks de modalidad m.
    """

    rng = rng or random

    return [
        rng.randint(0, int(modality_counts[0])),
        rng.randint(0, int(modality_counts[1])),
        rng.randint(0, int(modality_counts[2])),
    ]


def apply_stock_scenario(
    action_ids: list[int],
    original_stock: list[int],
    p_stock: float,
    seed: int | None = None,
) -> StockScenarioResult:
    """
    Con probabilidad p_stock, reduce el stock inicial y reordena chunks
    para inducir expansion_mode.

    Si no se activa stock mode, devuelve acciones y stock originales.
    """

    rng = random.Random(seed)

    if rng.random() > float(p_stock):
        return StockScenarioResult(
            actions=[int(a) for a in action_ids],
            initial_stock=[int(s) for s in original_stock],
            stock_mode=False,
        )

    chunks = split_actions_into_resource_chunks(action_ids)
    modality_counts = count_resource_chunks_by_modality(action_ids)

    reduced_stock = sample_reduced_stock(
        modality_counts=modality_counts,
        rng=rng,
    )

    ordered_chunks = reorder_chunks_for_expansion_mode(
        resources_chunks=chunks,
        initial_stock=reduced_stock,
        rng=rng,
    )

    return StockScenarioResult(
        actions=flatten_action_chunks(ordered_chunks),
        initial_stock=reduced_stock,
        stock_mode=True,
    )

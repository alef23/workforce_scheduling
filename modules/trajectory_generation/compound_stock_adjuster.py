from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from modules.workforce_engine.compound_actions import decode_action
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine

from .compound_trajectory_replayer import CompoundTrajectoryReplayer


WEEKS_PER_RESOURCE = 4


@dataclass(frozen=True)
class CompoundStockAdjustmentResult:
    trajectory: list[dict[str, Any]]
    original_stock: np.ndarray
    output_stock: np.ndarray
    stock_was_reduced: bool
    selected_chunk_indices: list[int]
    reordered_chunk_indices: list[int]
    first_expansion_step: int | None
    final_reward: float
    stopped_early: bool
    source_action_count: int
    consumed_action_count: int


class CompoundStockAdjuster:
    """Reduce stock seleccionando chunks completos de recursos."""

    def __init__(
        self,
        engine: CompoundWorkforceEngine,
        p_stock: float = 0.2,
        seed: int | None = None,
    ) -> None:
        if not 0 <= p_stock <= 1:
            raise ValueError("p_stock debe estar entre 0 y 1.")

        self.engine = engine
        self.p_stock = float(p_stock)
        self.rng = random.Random(seed)
        self.replayer = CompoundTrajectoryReplayer(engine)

    def adjust(
        self,
        trajectory: list[dict[str, Any]],
    ) -> CompoundStockAdjustmentResult:
        if not trajectory:
            raise ValueError("trajectory no puede estar vacía.")

        chunks = self._split_resource_chunks(trajectory)
        original_stock = self._initial_stock(trajectory)
        source_action_count = len(trajectory)

        if self.rng.random() >= self.p_stock:
            return CompoundStockAdjustmentResult(
                trajectory=trajectory,
                original_stock=original_stock.copy(),
                output_stock=original_stock.copy(),
                stock_was_reduced=False,
                selected_chunk_indices=list(range(len(chunks))),
                reordered_chunk_indices=list(range(len(chunks))),
                first_expansion_step=self._first_expansion_step(trajectory),
                final_reward=float(trajectory[-1]["reward"]),
                stopped_early=False,
                source_action_count=source_action_count,
                consumed_action_count=source_action_count,
            )

        selected_count = self.rng.randint(0, len(chunks) - 1)
        selected_indices = self.rng.sample(
            range(len(chunks)),
            selected_count,
        )
        selected_set = set(selected_indices)
        remaining_indices = [
            index
            for index in range(len(chunks))
            if index not in selected_set
        ]
        self.rng.shuffle(remaining_indices)
        reordered_indices = selected_indices + remaining_indices
        output_stock = self._stock_from_chunks(
            chunks,
            selected_indices,
        )
        reordered_actions = [
            int(sample["action_id"])
            for chunk_index in reordered_indices
            for sample in chunks[chunk_index]
        ]

        replay_result = self.replayer.replay_actions(
            initial_demand=self._initial_demand(trajectory),
            initial_stock=output_stock,
            actions=reordered_actions,
        )
        adjusted_trajectory = replay_result["trajectory"]

        return CompoundStockAdjustmentResult(
            trajectory=adjusted_trajectory,
            original_stock=original_stock.copy(),
            output_stock=output_stock,
            stock_was_reduced=True,
            selected_chunk_indices=[
                int(index)
                for index in selected_indices
            ],
            reordered_chunk_indices=[
                int(index)
                for index in reordered_indices
            ],
            first_expansion_step=self._first_expansion_step(
                adjusted_trajectory
            ),
            final_reward=float(replay_result["final_reward"]),
            stopped_early=bool(replay_result["stopped_early"]),
            source_action_count=int(replay_result["source_action_count"]),
            consumed_action_count=int(replay_result["consumed_action_count"]),
        )

    def _split_resource_chunks(
        self,
        trajectory: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        if len(trajectory) % WEEKS_PER_RESOURCE != 0:
            raise ValueError(
                "La trayectoria compuesta debe contener chunks completos "
                "de cuatro acciones."
            )

        chunks = [
            trajectory[start:start + WEEKS_PER_RESOURCE]
            for start in range(0, len(trajectory), WEEKS_PER_RESOURCE)
        ]
        for chunk_index, chunk in enumerate(chunks):
            weeks = [
                int(self._state_value(sample["state"], "assignment_week"))
                for sample in chunk
            ]
            if weeks != [0, 1, 2, 3]:
                raise ValueError(
                    f"Chunk {chunk_index} no contiene semanas [0, 1, 2, 3]."
                )

            modalities = {
                decode_action(int(sample["action_id"])).modality_index
                for sample in chunk
            }
            if len(modalities) != 1:
                raise ValueError(
                    f"Chunk {chunk_index} mezcla modalidades."
                )

        return chunks

    @staticmethod
    def _stock_from_chunks(
        chunks: list[list[dict[str, Any]]],
        selected_indices: list[int],
    ) -> np.ndarray:
        stock = np.zeros(3, dtype=int)
        for chunk_index in selected_indices:
            first_action_id = int(chunks[chunk_index][0]["action_id"])
            modality_index = decode_action(first_action_id).modality_index
            stock[modality_index] += 1
        return stock

    @classmethod
    def _initial_stock(
        cls,
        trajectory: list[dict[str, Any]],
    ) -> np.ndarray:
        return np.asarray(
            cls._state_value(
                trajectory[0]["state"],
                "remaining_stock",
            ),
            dtype=int,
        ).copy()

    @classmethod
    def _initial_demand(
        cls,
        trajectory: list[dict[str, Any]],
    ) -> np.ndarray:
        return np.asarray(
            cls._state_value(
                trajectory[0]["state"],
                "residual_demand",
            ),
            dtype=int,
        ).copy()

    @classmethod
    def _first_expansion_step(
        cls,
        trajectory: list[dict[str, Any]],
    ) -> int | None:
        for step_index, sample in enumerate(trajectory):
            if bool(
                cls._state_value(
                    sample["state"],
                    "expansion_mode",
                )
            ):
                return int(step_index)
        return None

    @staticmethod
    def _state_value(state: Any, field: str) -> Any:
        if isinstance(state, dict):
            return state[field]
        return getattr(state, field)

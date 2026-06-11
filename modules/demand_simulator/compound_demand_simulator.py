from __future__ import annotations

import random
from typing import Any

import numpy as np

from modules.workforce_engine.compound_actions import (
    ACTION_SPACE_SIZE,
    ENTRY_HOURS,
    FIXED_DAY_OFF,
    MODALITIES,
    decode_action,
)
from modules.workforce_engine.compound_schemas import CompoundWorkforceState
from modules.workforce_engine.schemas import ProblemSetup


WEEKS = 4
DAYS_PER_WEEK = 7
MAX_RESOURCES = 20
MODALITY_TO_INDEX = {
    modality: index
    for index, modality in enumerate(MODALITIES)
}


class CompoundDemandSimulator:
    """Genera cobertura y trayectoria positiva con acciones semanales."""

    def __init__(
        self,
        problem_setup: ProblemSetup,
        seed: int | None = None,
    ) -> None:
        self._validate_fixed_setup(problem_setup)
        self.problem_setup = problem_setup
        self.seed = seed
        self.rng = random.Random(seed)
        self._legal_actions_by_modality = self._build_legal_actions_by_modality()
        self._initial_legal_actions = tuple(
            action_id
            for action_ids in self._legal_actions_by_modality
            for action_id in action_ids
        )

    def compute_coverage(
        self,
        n_resources: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Genera una matriz de cobertura y su trayectoria positiva.

        Cada recurso aporta cuatro acciones consecutivas. La primera acción
        selecciona implícitamente la modalidad y las tres restantes se samplean
        dentro del bloque legal de esa misma modalidad.
        """
        self._validate_n_resources(n_resources)
        actions, generated_stock, coverage = self._generate_plan(n_resources)
        trajectory = self._build_trajectory(
            coverage=coverage,
            generated_stock=generated_stock,
            actions=actions,
        )
        return coverage, trajectory

    def _generate_plan(
        self,
        n_resources: int,
    ) -> tuple[list[int], np.ndarray, np.ndarray]:
        actions: list[int] = []
        generated_stock = np.zeros(len(MODALITIES), dtype=int)
        coverage = np.zeros((24, 28), dtype=int)

        for _resource_index in range(n_resources):
            first_action_id = int(self.rng.choice(self._initial_legal_actions))
            first_action = decode_action(first_action_id)
            modality_index = first_action.modality_index
            generated_stock[modality_index] += 1

            resource_actions = [first_action_id]
            resource_actions.extend(
                int(self.rng.choice(self._legal_actions_by_modality[modality_index]))
                for _ in range(WEEKS - 1)
            )

            for week, action_id in enumerate(resource_actions):
                actions.append(action_id)
                self._add_weekly_coverage(
                    coverage=coverage,
                    action_id=action_id,
                    week=week,
                )

        return actions, generated_stock, coverage

    def _build_trajectory(
        self,
        coverage: np.ndarray,
        generated_stock: np.ndarray,
        actions: list[int],
    ) -> list[dict[str, Any]]:
        accumulated_coverage = np.zeros_like(coverage)
        remaining_stock = generated_stock.copy()
        initial_demand_total = int(coverage.sum())
        reward = float(np.tanh(2.0))
        trajectory: list[dict[str, Any]] = []

        for step_index, action_id in enumerate(actions):
            assignment_week = step_index % WEEKS
            action = decode_action(action_id)
            current_modality = (
                None if assignment_week == 0 else action.modality
            )
            state = CompoundWorkforceState(
                residual_demand=coverage - accumulated_coverage,
                remaining_stock=remaining_stock,
                expansion_mode=False,
                current_modality=current_modality,
                assignment_week=assignment_week,
                initial_demand_total=initial_demand_total,
            )
            legal_actions = self._legal_actions_for_state(state)
            if not legal_actions[action_id]:
                raise RuntimeError(
                    f"El plan generado contiene una acción ilegal: {action_id}."
                )

            trajectory.append(
                {
                    "state": state,
                    "policy": self._uniform_policy(legal_actions),
                    "action_id": int(action_id),
                    "reward": reward,
                }
            )
            self._add_weekly_coverage(
                coverage=accumulated_coverage,
                action_id=action_id,
                week=assignment_week,
            )

            if assignment_week == WEEKS - 1:
                remaining_stock = remaining_stock.copy()
                remaining_stock[action.modality_index] -= 1

        if not np.array_equal(accumulated_coverage, coverage):
            raise RuntimeError(
                "La trayectoria generada no reconstruye la matriz de cobertura."
            )
        if np.any(remaining_stock != 0):
            raise RuntimeError("La trayectoria no consumió todo el stock generado.")

        return trajectory

    def _legal_actions_for_state(
        self,
        state: CompoundWorkforceState,
    ) -> np.ndarray:
        legal = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

        if state.current_modality is None:
            for modality_index, action_ids in enumerate(
                self._legal_actions_by_modality
            ):
                if state.remaining_stock[modality_index] > 0:
                    legal[list(action_ids)] = True
            return legal

        modality_index = MODALITY_TO_INDEX[state.current_modality]
        legal[list(self._legal_actions_by_modality[modality_index])] = True
        return legal

    def _build_legal_actions_by_modality(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        closing_hour = int(self.problem_setup.closing_hour)
        legal_by_modality: list[tuple[int, ...]] = []

        for modality_index, modality in enumerate(MODALITIES):
            start = modality_index * 18
            stop = start + 18
            legal_by_modality.append(
                tuple(
                    action_id
                    for action_id in range(start, stop)
                    if decode_action(action_id).entry_hour + modality
                    <= closing_hour
                )
            )

        return tuple(legal_by_modality)

    @staticmethod
    def _add_weekly_coverage(
        coverage: np.ndarray,
        action_id: int,
        week: int,
    ) -> None:
        action = decode_action(action_id)
        week_start = week * DAYS_PER_WEEK

        for relative_day in range(DAYS_PER_WEEK):
            if relative_day in action.days_off:
                continue
            absolute_day = week_start + relative_day
            coverage[
                action.entry_hour:action.entry_hour + action.modality,
                absolute_day,
            ] += 1

    @staticmethod
    def _uniform_policy(legal_actions: np.ndarray) -> np.ndarray:
        legal_count = int(legal_actions.sum())
        if legal_count <= 0:
            raise RuntimeError("No hay acciones legales para construir la policy.")

        policy = np.zeros(ACTION_SPACE_SIZE, dtype=float)
        policy[legal_actions] = 1.0 / legal_count
        return policy

    @staticmethod
    def _validate_n_resources(n_resources: int) -> None:
        if not isinstance(n_resources, int):
            raise TypeError("n_resources debe ser un entero.")
        if n_resources <= 0 or n_resources > MAX_RESOURCES:
            raise ValueError(
                f"n_resources debe estar entre 1 y {MAX_RESOURCES}."
            )

    @staticmethod
    def _validate_fixed_setup(setup: ProblemSetup) -> None:
        if setup.mobile_days_off_count != 1:
            raise ValueError("El simulador compuesto requiere un franco móvil.")
        if setup.fixed_day_off != FIXED_DAY_OFF:
            raise ValueError("El simulador compuesto requiere fixed_day_off=6.")
        if setup.allowed_entry_hours != list(ENTRY_HOURS):
            raise ValueError(
                "El simulador compuesto requiere allowed_entry_hours=[6, 12, 18]."
            )
        if setup.closing_hour != 22:
            raise ValueError("El simulador compuesto requiere closing_hour=22.")

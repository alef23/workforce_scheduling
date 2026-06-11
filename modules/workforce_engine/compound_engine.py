from __future__ import annotations

from typing import Optional

import numpy as np

from .compound_actions import (
    ACTION_SPACE_SIZE,
    ENTRY_HOURS,
    FIXED_DAY_OFF,
    MODALITIES,
    CompoundAction,
    decode_action,
    encode_action,
)
from .compound_schemas import CompoundStepResult, CompoundWorkforceState
from .schemas import ProblemSetup


MODALITY_TO_INDEX = {modality: index for index, modality in enumerate(MODALITIES)}
DAYS_PER_WEEK = 7
WEEKS = 4


class CompoundWorkforceEngine:
    """
    Engine experimental donde cada acción aplica una cobertura semanal completa.

    Cumple el mismo contrato operativo que WorkforceEngine para MCTS, pero usa
    un estado y un espacio de acciones propios.
    """

    def __init__(self, setup: ProblemSetup) -> None:
        self._validate_fixed_setup(setup)
        self.setup = setup
        self.action_space_size = ACTION_SPACE_SIZE

    def step(
        self,
        state: CompoundWorkforceState,
        action_id: int,
    ) -> CompoundStepResult:
        if self._is_terminal_state_context(state):
            raise ValueError("No se puede aplicar step sobre un estado terminal.")

        is_valid, reason = self.validate_action(state, action_id)
        if not is_valid:
            raise ValueError(f"Acción ilegal {action_id}: {reason}")

        action = decode_action(action_id)
        modality = state.current_modality or action.modality
        residual = self._apply_weekly_coverage(
            state=state,
            modality=modality,
            entry_hour=action.entry_hour,
            days_off=action.days_off,
        )

        if state.assignment_week < WEEKS - 1:
            next_state = state.copy_state(
                residual_demand=residual,
                current_modality=modality,
                assignment_week=state.assignment_week + 1,
            )
            return CompoundStepResult(
                next_state=next_state,
                is_terminal=False,
                reward=0.0,
            )

        next_state = state.copy_state(
            residual_demand=residual,
            current_modality=None,
            assignment_week=0,
        )
        next_state = self._discount_stock_after_resource_completion(
            next_state,
            modality,
        )
        is_terminal = self.check_terminality(next_state)
        reward = self.compute_reward(next_state) if is_terminal else 0.0
        return CompoundStepResult(
            next_state=next_state,
            is_terminal=is_terminal,
            reward=reward,
        )

    def legal_mask(
        self,
        state: CompoundWorkforceState,
        priors: np.ndarray,
    ) -> np.ndarray:
        probabilities = np.asarray(priors, dtype=float)
        if probabilities.shape != (ACTION_SPACE_SIZE,):
            raise ValueError(
                f"priors debe tener shape ({ACTION_SPACE_SIZE},), "
                f"pero tiene {probabilities.shape}."
            )
        if np.any(probabilities < 0):
            raise ValueError("priors no puede contener valores negativos.")

        legal = self.get_legal_actions(state)
        if not np.any(legal):
            raise ValueError("No existen acciones legales para el estado recibido.")

        output = np.zeros(ACTION_SPACE_SIZE, dtype=float)
        legal_mass = probabilities[legal].sum()
        if legal_mass <= 0:
            output[legal] = 1.0 / int(legal.sum())
        else:
            output[legal] = probabilities[legal] / legal_mass
        return output

    def get_legal_actions(
        self,
        state: CompoundWorkforceState,
    ) -> np.ndarray:
        legal = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        if self._is_terminal_state_context(state):
            return legal

        for action_id in range(ACTION_SPACE_SIZE):
            legal[action_id] = self.validate_action(state, action_id)[0]
        return legal

    def validate_action(
        self,
        state: CompoundWorkforceState,
        action_id: int,
    ) -> tuple[bool, Optional[str]]:
        try:
            action = decode_action(action_id)
        except (TypeError, ValueError) as exc:
            return False, str(exc)

        if (
            state.current_modality is not None
            and action.modality != state.current_modality
        ):
            return False, "La acción no respeta la modalidad activa."

        modality_index = MODALITY_TO_INDEX[action.modality]
        if (
            state.current_modality is None
            and not state.expansion_mode
            and state.remaining_stock[modality_index] <= 0
        ):
            return False, f"No hay stock disponible para modalidad {action.modality}."

        if action.entry_hour + action.modality > int(self.setup.closing_hour):
            return False, "La combinación de horario y modalidad supera el cierre."

        return True, None

    @staticmethod
    def encode_action(
        modality_index: int,
        entry_hour_index: int,
        mobile_day_off: int,
    ) -> int:
        return encode_action(
            modality_index=modality_index,
            entry_hour_index=entry_hour_index,
            mobile_day_off=mobile_day_off,
        )

    @staticmethod
    def decode_action(action_id: int) -> CompoundAction:
        return decode_action(action_id)

    def check_terminality(self, state: CompoundWorkforceState) -> bool:
        demand_covered = bool(np.all(state.residual_demand <= 0))
        overcoverage_exceeded = (
            self.compute_overcoverage_index(state)
            <= -2 * self.setup.max_overcoverage_tolerance
        )
        return demand_covered or overcoverage_exceeded

    def compute_overcoverage_index(
        self,
        state: CompoundWorkforceState,
    ) -> float:
        negative_residual_sum = np.minimum(state.residual_demand, 0).sum()
        return float(negative_residual_sum / state.initial_demand_total)

    def compute_reward(self, state: CompoundWorkforceState) -> float:
        rho = self.compute_overcoverage_index(state)
        tolerance = self.setup.max_overcoverage_tolerance
        return float(np.tanh(2 * (1 - abs(rho) / tolerance)))

    def _apply_weekly_coverage(
        self,
        state: CompoundWorkforceState,
        modality: int,
        entry_hour: int,
        days_off: frozenset[int],
    ) -> np.ndarray:
        residual = state.residual_demand.copy()
        week_start_day = state.assignment_week * DAYS_PER_WEEK

        for relative_day in range(DAYS_PER_WEEK):
            if relative_day in days_off:
                continue
            absolute_day = week_start_day + relative_day
            residual[
                entry_hour:entry_hour + modality,
                absolute_day,
            ] -= 1

        return residual

    def _discount_stock_after_resource_completion(
        self,
        state: CompoundWorkforceState,
        modality: int,
    ) -> CompoundWorkforceState:
        if state.expansion_mode:
            return state

        modality_index = MODALITY_TO_INDEX[modality]
        stock = state.remaining_stock.copy()
        if stock[modality_index] <= 0:
            raise ValueError(
                f"No hay stock disponible para descontar modalidad {modality}."
            )

        stock[modality_index] -= 1
        return state.copy_state(
            remaining_stock=stock,
            expansion_mode=bool(np.all(stock == 0)),
        )

    def _is_terminal_state_context(
        self,
        state: CompoundWorkforceState,
    ) -> bool:
        no_resource_in_progress = (
            state.current_modality is None
            and state.assignment_week == 0
        )
        return no_resource_in_progress and self.check_terminality(state)

    @staticmethod
    def _validate_fixed_setup(setup: ProblemSetup) -> None:
        expected_hours = list(ENTRY_HOURS)
        if setup.mobile_days_off_count != 1:
            raise ValueError("El engine compuesto requiere un franco móvil.")
        if setup.fixed_day_off != FIXED_DAY_OFF:
            raise ValueError("El engine compuesto requiere fixed_day_off=6.")
        if setup.allowed_entry_hours != expected_hours:
            raise ValueError(
                "El engine compuesto requiere allowed_entry_hours=[6, 12, 18]."
            )
        if setup.closing_hour != 22:
            raise ValueError("El engine compuesto requiere closing_hour=22.")

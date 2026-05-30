from __future__ import annotations

from typing import Optional

import numpy as np

from .schemas import ActionType, ProblemSetup, StepResult, WorkforceState
# from schemas import ActionType, ProblemSetup, StepResult, WorkforceState
# from schemas import ActionType, ProblemSetup, WorkforceState


ACTION_SPACE_SIZE = 55

MODALITY_ACTION_START = 0
MODALITY_ACTION_END = 2

ENTRY_HOUR_ACTION_START = 3
ENTRY_HOUR_ACTION_END = 26

DAY_OFF_ACTION_START = 27
DAY_OFF_ACTION_END = 54

MODALITIES = [4, 6, 8]
MODALITY_TO_INDEX = {4: 0, 6: 1, 8: 2}

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
WEEKS = 4
DAYS_IN_HORIZON = 28

DAY_OFF_ACTION_MATRIX = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6],
        [1, 7, 8, 9, 10, 11, 12],
        [2, 8, 13, 14, 15, 16, 17],
        [3, 9, 14, 18, 19, 20, 21],
        [4, 10, 15, 19, 22, 23, 24],
        [5, 11, 16, 20, 23, 25, 26],
        [6, 12, 17, 21, 24, 26, 27],
    ],
    dtype=int,
)


def _build_day_off_action_to_pair() -> dict[int, tuple[int, int]]:
    """
    Construye un mapeo desde id interno de franco hacia el par canónico de días.

    Para acciones fuera de la diagonal, se conserva el par con fila <= columna.
    """
    mapping: dict[int, tuple[int, int]] = {}
    for row in range(DAYS_PER_WEEK):
        for col in range(row, DAYS_PER_WEEK):
            internal_id = int(DAY_OFF_ACTION_MATRIX[row, col])
            mapping[internal_id] = (row, col)
    return mapping


DAY_OFF_ACTION_TO_PAIR = _build_day_off_action_to_pair()


class WorkforceEngine:
    """
    Motor determinístico de transición, legalidad, cobertura, terminalidad y scoring.

    El engine no conserva estado interno de una trayectoria. Todos los métodos reciben
    explícitamente el WorkforceState sobre el cual operar.
    """

    def __init__(self, setup: ProblemSetup):
        self.setup = setup
        self.action_space_size = ACTION_SPACE_SIZE

    # -------------------------------------------------------------------------
    # API pública
    # -------------------------------------------------------------------------

    def step(self, state: WorkforceState, action_id: int) -> StepResult:
        """
        Aplica una acción legal al estado actual y devuelve el resultado.

        Si la acción es ilegal, lanza ValueError.
        """
        if self._is_terminal_state_context(state):
            raise ValueError("No se puede aplicar step sobre un estado terminal.")

        is_valid, reason = self.validate_action(state, action_id)
        if not is_valid:
            raise ValueError(f"Acción ilegal {action_id}: {reason}")

        action_type = self.get_action_type(action_id)

        if action_type == ActionType.MODALITY:
            next_state = self.apply_modality_selection(state, action_id)
            return StepResult(next_state=next_state, is_terminal=False, reward=0.0)

        if action_type == ActionType.ENTRY_HOUR:
            next_state, resource_completed = self.apply_entry_hour_selection(
                state, action_id
            )
            return self._build_step_result(next_state, resource_completed)

        if action_type == ActionType.DAY_OFFS:
            next_state, resource_completed = self.apply_days_off_selection(
                state, action_id
            )
            return self._build_step_result(next_state, resource_completed)

        raise ValueError(f"Tipo de acción no soportado: {action_type}")

    def legal_mask(self, state: WorkforceState, priors: np.ndarray) -> np.ndarray:
        """
        Enmascara acciones ilegales y renormaliza probabilidades legales.

        priors debe ser un vector de probabilidades o scores no negativos de tamaño
        ACTION_SPACE_SIZE.
        """
        probabilities = np.asarray(priors, dtype=float)

        if probabilities.shape != (ACTION_SPACE_SIZE,):
            raise ValueError(
                f"priors debe tener shape ({ACTION_SPACE_SIZE},), "
                f"pero tiene shape {probabilities.shape}."
            )

        if np.any(probabilities < 0):
            raise ValueError("priors no puede contener valores negativos.")

        legal = self.get_legal_actions(state)

        if not np.any(legal):
            raise ValueError("No existen acciones legales para el estado recibido.")

        output = np.zeros(ACTION_SPACE_SIZE, dtype=float)
        legal_mass = probabilities[legal].sum()

        if legal_mass <= 0:
            output[legal] = 1.0 / legal.sum()
            return output

        output[legal] = probabilities[legal] / legal_mass
        return output

    def get_legal_actions(self, state: WorkforceState) -> np.ndarray:
        """
        Devuelve un vector booleano de tamaño 55 indicando qué acciones son legales.
        """
        legal = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

        if self._is_terminal_state_context(state):
            return legal

        for action_id in range(ACTION_SPACE_SIZE):
            is_valid, _ = self.validate_action(state, action_id)
            legal[action_id] = is_valid

        return legal

    def encode_action(self, action_type: ActionType, value) -> int:
        """
        Traduce una acción conceptual a action_id.

        Ejemplos:
            encode_action(ActionType.MODALITY, 4) -> 0
            encode_action(ActionType.ENTRY_HOUR, 16) -> 19
            encode_action(ActionType.DAY_OFFS, (2, 5)) -> 43
        """
        if action_type == ActionType.MODALITY:
            if value not in MODALITY_TO_INDEX:
                raise ValueError("La modalidad debe ser 4, 6 u 8.")
            return MODALITY_TO_INDEX[value]

        if action_type == ActionType.ENTRY_HOUR:
            if not isinstance(value, int) or value < 0 or value > 23:
                raise ValueError("La hora de entrada debe estar entre 0 y 23.")
            return ENTRY_HOUR_ACTION_START + value

        if action_type == ActionType.DAY_OFFS:
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or not all(isinstance(day, int) and 0 <= day <= 6 for day in value)
            ):
                raise ValueError("El valor de DAY_OFFS debe ser una tupla (d1, d2).")
            d1, d2 = value
            internal_id = int(DAY_OFF_ACTION_MATRIX[d1, d2])
            return DAY_OFF_ACTION_START + internal_id

        raise ValueError(f"Tipo de acción no soportado: {action_type}")

    def decode_action(self, action_id: int) -> dict:
        """
        Traduce un action_id a una representación interpretable.
        """
        action_type = self.get_action_type(action_id)

        if action_type == ActionType.MODALITY:
            return {
                "action_type": action_type,
                "modality": MODALITIES[action_id],
            }

        if action_type == ActionType.ENTRY_HOUR:
            return {
                "action_type": action_type,
                "entry_hour": action_id - ENTRY_HOUR_ACTION_START,
            }

        internal_id = action_id - DAY_OFF_ACTION_START
        return {
            "action_type": action_type,
            "internal_day_off_action_id": internal_id,
            "day_pair": DAY_OFF_ACTION_TO_PAIR[internal_id],
        }

    def get_action_type(self, action_id: int) -> ActionType:
        """
        Determina el tipo de acción según el rango del action_id.
        """
        if not isinstance(action_id, int):
            raise TypeError("action_id debe ser entero.")

        if MODALITY_ACTION_START <= action_id <= MODALITY_ACTION_END:
            return ActionType.MODALITY

        if ENTRY_HOUR_ACTION_START <= action_id <= ENTRY_HOUR_ACTION_END:
            return ActionType.ENTRY_HOUR

        if DAY_OFF_ACTION_START <= action_id <= DAY_OFF_ACTION_END:
            return ActionType.DAY_OFFS

        raise ValueError(f"action_id fuera de rango: {action_id}")

    # -------------------------------------------------------------------------
    # Validación de acciones
    # -------------------------------------------------------------------------

    def validate_action(
        self, state: WorkforceState, action_id: int
    ) -> tuple[bool, Optional[str]]:
        """
        Valida si una acción es legal para el estado recibido.

        Devuelve:
            (True, None) si la acción es legal.
            (False, motivo) si la acción es ilegal.
        """
        try:
            action_type = self.get_action_type(action_id)
        except (TypeError, ValueError) as exc:
            return False, str(exc)

        expected_action_type = self._get_expected_action_type(state)

        if expected_action_type is None:
            return False, "El estado no espera nuevas acciones."

        if action_type != expected_action_type:
            return (
                False,
                f"Se esperaba acción {expected_action_type.value}, "
                f"pero se recibió {action_type.value}.",
            )

        if action_type == ActionType.MODALITY:
            return self._validate_modality_action(state, action_id)

        if action_type == ActionType.ENTRY_HOUR:
            return self._validate_entry_hour_action(state, action_id)

        if action_type == ActionType.DAY_OFFS:
            return self._validate_day_off_action(state, action_id)

        return False, "Tipo de acción no soportado."

    def _get_expected_action_type(self, state: WorkforceState) -> Optional[ActionType]:
        if state.current_modality is None:
            return ActionType.MODALITY

        if state.current_entry_hour is None:
            return ActionType.ENTRY_HOUR

        if self.setup.mobile_days_off_count > 0:
            return ActionType.DAY_OFFS

        return None

    def _validate_modality_action(
        self, state: WorkforceState, action_id: int
    ) -> tuple[bool, Optional[str]]:
        modality = MODALITIES[action_id]
        modality_index = MODALITY_TO_INDEX[modality]

        if state.current_modality is not None:
            return False, "Ya existe una modalidad seleccionada."

        if not state.expansion_mode and state.remaining_stock[modality_index] <= 0:
            return False, f"No hay stock disponible para modalidad {modality}."

        return True, None

    def _validate_entry_hour_action(
        self, state: WorkforceState, action_id: int
    ) -> tuple[bool, Optional[str]]:
        if state.current_modality is None:
            return False, "No se puede elegir horario sin modalidad seleccionada."

        if state.current_entry_hour is not None:
            return False, "Ya existe una hora de entrada seleccionada."

        entry_hour = action_id - ENTRY_HOUR_ACTION_START

        if entry_hour not in self.setup.get_allowed_entry_hours():
            return False, f"La hora {entry_hour} no está permitida."

        if not self._is_entry_hour_compatible_with_closing(
            entry_hour, state.current_modality
        ):
            return (
                False,
                "La combinación de hora de entrada y modalidad viola el cierre "
                "operativo.",
            )

        # Si no hay francos móviles, la hora dispara la cobertura semanal. En ese
        # caso ya se puede validar que la cobertura no exceda el horizonte.
        if self.setup.mobile_days_off_count == 0:
            days_off = self._get_days_off_without_mobile_action()
            if not self._weekly_coverage_fits_horizon(
                state, entry_hour, state.current_modality, days_off
            ):
                return False, "La cobertura semanal excede el horizonte de 28 días."

        # Si hay francos móviles, evitamos elegir una hora que deje sin ninguna
        # acción de franco factible.
        if self.setup.mobile_days_off_count > 0:
            has_feasible_day_off_action = False
            for day_off_action_id in self.get_legal_day_off_action_ids():
                internal_state = state.copy_state(current_entry_hour=entry_hour)
                if self._validate_day_off_action(internal_state, day_off_action_id)[0]:
                    has_feasible_day_off_action = True
                    break

            if not has_feasible_day_off_action:
                return (
                    False,
                    "La hora seleccionada no deja ninguna acción de franco factible.",
                )

        return True, None

    def _validate_day_off_action(
        self, state: WorkforceState, action_id: int
    ) -> tuple[bool, Optional[str]]:
        if self.setup.mobile_days_off_count == 0:
            return False, "No corresponde acción de francos si no hay francos móviles."

        if state.current_modality is None:
            return False, "No se puede elegir franco sin modalidad seleccionada."

        if state.current_entry_hour is None:
            return False, "No se puede elegir franco sin hora de entrada seleccionada."

        if action_id not in self.get_legal_day_off_action_ids():
            return False, "La acción de francos no es legal para el ProblemSetup."

        days_off = self.decode_day_off_action(action_id)

        if not self._weekly_coverage_fits_horizon(
            state, state.current_entry_hour, state.current_modality, days_off
        ):
            return False, "La cobertura semanal excede el horizonte de 28 días."

        return True, None

    # -------------------------------------------------------------------------
    # Aplicación de acciones
    # -------------------------------------------------------------------------

    def apply_modality_selection(
        self, state: WorkforceState, action_id: int
    ) -> WorkforceState:
        modality = MODALITIES[action_id]
        return state.copy_state(current_modality=modality)

    def apply_entry_hour_selection(
        self, state: WorkforceState, action_id: int
    ) -> tuple[WorkforceState, bool]:
        entry_hour = action_id - ENTRY_HOUR_ACTION_START
        next_state = state.copy_state(current_entry_hour=entry_hour)

        if self.setup.mobile_days_off_count == 0:
            days_off = self._get_days_off_without_mobile_action()
            return self.apply_weekly_coverage(next_state, days_off)

        return next_state, False

    def apply_days_off_selection(
        self, state: WorkforceState, action_id: int
    ) -> tuple[WorkforceState, bool]:
        days_off = self.decode_day_off_action(action_id)
        return self.apply_weekly_coverage(state, days_off)

    def apply_weekly_coverage(
        self, state: WorkforceState, days_off: set[int]
    ) -> tuple[WorkforceState, bool]:
        """
        Aplica cobertura semanal al residual_demand y avanza el ciclo del recurso.
        """
        if state.current_modality is None or state.current_entry_hour is None:
            raise ValueError(
                "Para aplicar cobertura debe existir modalidad y hora de entrada."
            )

        next_state = state.copy_state()
        residual = next_state.residual_demand.copy()

        week_start_day = state.assignment_week * DAYS_PER_WEEK
        working_days = set(range(DAYS_PER_WEEK)) - set(days_off)

        for relative_day in sorted(working_days):
            absolute_day = week_start_day + relative_day
            covered_cells = self._get_shift_cells(
                absolute_day=absolute_day,
                entry_hour=state.current_entry_hour,
                modality=state.current_modality,
            )

            for hour, day in covered_cells:
                residual[hour, day] -= 1

        next_state = next_state.copy_state(residual_demand=residual)
        return self._advance_week_or_close_resource(next_state)

    # -------------------------------------------------------------------------
    # Francos
    # -------------------------------------------------------------------------

    def get_legal_day_off_action_ids(self) -> list[int]:
        """
        Devuelve las acciones reales de franco legales para el ProblemSetup.
        """
        if self.setup.mobile_days_off_count == 0:
            return []

        legal_internal_ids: set[int] = set()

        if self.setup.fixed_day_off is None and self.setup.mobile_days_off_count == 1:
            # Un solo franco móvil: diagonal.
            for day in range(DAYS_PER_WEEK):
                legal_internal_ids.add(int(DAY_OFF_ACTION_MATRIX[day, day]))

        elif self.setup.fixed_day_off is not None and self.setup.mobile_days_off_count == 1:
            # Un franco fijo + uno móvil: fila del fijo, excluyendo diagonal.
            fixed_day = self.setup.fixed_day_off
            for mobile_day in range(DAYS_PER_WEEK):
                if mobile_day == fixed_day:
                    continue
                legal_internal_ids.add(
                    int(DAY_OFF_ACTION_MATRIX[fixed_day, mobile_day])
                )

        elif self.setup.fixed_day_off is None and self.setup.mobile_days_off_count == 2:
            # Dos francos móviles: todos los pares no diagonales.
            for d1 in range(DAYS_PER_WEEK):
                for d2 in range(d1 + 1, DAYS_PER_WEEK):
                    legal_internal_ids.add(int(DAY_OFF_ACTION_MATRIX[d1, d2]))

        else:
            raise ValueError("Combinación de francos no soportada por ProblemSetup.")

        return sorted(DAY_OFF_ACTION_START + internal_id for internal_id in legal_internal_ids)

    def decode_day_off_action(self, action_id: int) -> set[int]:
        """
        Devuelve el conjunto de días de franco semanal representado por action_id.

        Incluye el franco fijo si existe.
        """
        if action_id < DAY_OFF_ACTION_START or action_id > DAY_OFF_ACTION_END:
            raise ValueError("action_id no pertenece al bloque de francos.")

        internal_id = action_id - DAY_OFF_ACTION_START
        if internal_id not in DAY_OFF_ACTION_TO_PAIR:
            raise ValueError(f"ID interno de franco inválido: {internal_id}")

        d1, d2 = DAY_OFF_ACTION_TO_PAIR[internal_id]

        if self.setup.fixed_day_off is None:
            if self.setup.mobile_days_off_count == 1:
                return {d1}
            if self.setup.mobile_days_off_count == 2:
                return {d1, d2}

        if self.setup.fixed_day_off is not None and self.setup.mobile_days_off_count == 1:
            return {d1, d2}

        raise ValueError("No corresponde decodificar acción de francos para este setup.")

    def _get_days_off_without_mobile_action(self) -> set[int]:
        """
        Días de franco cuando mobile_days_off_count = 0.
        """
        if self.setup.fixed_day_off is None:
            return set()
        return {self.setup.fixed_day_off}

    # -------------------------------------------------------------------------
    # Cobertura y cierre
    # -------------------------------------------------------------------------

    def _is_entry_hour_compatible_with_closing(
        self, entry_hour: int, modality: int
    ) -> bool:
        if self.setup.closing_hour is None:
            return True
        return entry_hour + modality <= self.setup.closing_hour

    def _weekly_coverage_fits_horizon(
        self,
        state: WorkforceState,
        entry_hour: int,
        modality: int,
        days_off: set[int],
    ) -> bool:
        week_start_day = state.assignment_week * DAYS_PER_WEEK
        working_days = set(range(DAYS_PER_WEEK)) - set(days_off)

        for relative_day in working_days:
            absolute_day = week_start_day + relative_day
            cells = self._get_shift_cells(
                absolute_day=absolute_day,
                entry_hour=entry_hour,
                modality=modality,
                validate_only=True,
            )
            if cells is None:
                return False

        return True

    def _get_shift_cells(
        self,
        absolute_day: int,
        entry_hour: int,
        modality: int,
        validate_only: bool = False,
    ) -> Optional[list[tuple[int, int]]]:
        """
        Devuelve las celdas (hour, day) cubiertas por un turno.

        Si no hay closing_hour, el turno puede cruzar medianoche.
        Si la cobertura excede el horizonte de 28 días, devuelve None cuando
        validate_only=True o lanza ValueError en modo normal.
        """
        cells: list[tuple[int, int]] = []

        for offset in range(modality):
            raw_hour = entry_hour + offset
            day_increment = raw_hour // HOURS_PER_DAY
            hour = raw_hour % HOURS_PER_DAY
            day = absolute_day + day_increment

            if day >= DAYS_IN_HORIZON:
                if validate_only:
                    return None
                raise ValueError("La cobertura excede el horizonte de 28 días.")

            cells.append((hour, day))

        return cells

    # -------------------------------------------------------------------------
    # Avance, stock, terminalidad y scoring
    # -------------------------------------------------------------------------

    def _advance_week_or_close_resource(
        self, state: WorkforceState
    ) -> tuple[WorkforceState, bool]:
        """
        Avanza la semana del recurso o cierra el ciclo mensual si se completó semana 4.
        """
        modality_used = state.current_modality

        if state.assignment_week < 3:
            next_state = state.copy_state(
                assignment_week=state.assignment_week + 1,
                current_entry_hour=None,
            )
            return next_state, False

        next_state = state.copy_state(
            assignment_week=0,
            current_entry_hour=None,
            current_modality=None,
        )

        next_state = self._discount_stock_after_resource_completion(
            next_state, modality_used
        )

        return next_state, True

    def _discount_stock_after_resource_completion(
        self, state: WorkforceState, modality_used: int
    ) -> WorkforceState:
        if state.expansion_mode:
            return state

        modality_index = MODALITY_TO_INDEX[modality_used]
        stock = state.remaining_stock.copy()

        if stock[modality_index] <= 0:
            raise ValueError(
                f"No hay stock disponible para descontar modalidad {modality_used}."
            )

        stock[modality_index] -= 1
        expansion_mode = bool(np.all(stock == 0))

        return state.copy_state(
            remaining_stock=stock,
            expansion_mode=expansion_mode,
        )

    def check_terminality(self, state: WorkforceState) -> bool:
        """
        Evalúa condiciones terminales sobre un estado.

        Este método es puramente evaluativo. El engine lo invoca como terminalidad
        efectiva únicamente después de completar la semana 4 de un recurso.
        """
        demand_covered = bool(np.all(state.residual_demand <= 0))
        overcoverage_exceeded = (
            self.compute_overcoverage_index(state)
            <= -2 * self.setup.max_overcoverage_tolerance
        )
        return demand_covered or overcoverage_exceeded

    def compute_overcoverage_index(self, state: WorkforceState) -> float:
        negative_residual_sum = np.minimum(state.residual_demand, 0).sum()
        return float(negative_residual_sum / state.initial_demand_total)

    def compute_reward(self, state: WorkforceState) -> float:
        rho = self.compute_overcoverage_index(state)
        k = self.setup.max_overcoverage_tolerance
        return float(np.tanh(2 * (1 - abs(rho) / k)))

    def _build_step_result(
        self, next_state: WorkforceState, resource_completed: bool
    ) -> StepResult:
        if not resource_completed:
            return StepResult(
                next_state=next_state,
                is_terminal=False,
                reward=0.0,
            )

        is_terminal = self.check_terminality(next_state)
        reward = self.compute_reward(next_state) if is_terminal else 0.0

        return StepResult(
            next_state=next_state,
            is_terminal=is_terminal,
            reward=reward,
        )

    def _is_terminal_state_context(self, state: WorkforceState) -> bool:
        """
        Determina si el estado parece ser un estado terminal ya cerrado.

        La terminalidad real solo se produce al cerrar un recurso mensual. Para no
        bloquear estados intermedios que accidentalmente ya cubrieron toda la demanda,
        se considera contexto terminal únicamente si no hay recurso en curso.
        """
        no_resource_in_progress = (
            state.current_modality is None
            and state.current_entry_hour is None
            and state.assignment_week == 0
        )
        return no_resource_in_progress and self.check_terminality(state)

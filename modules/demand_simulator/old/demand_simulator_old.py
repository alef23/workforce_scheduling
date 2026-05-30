import random
from itertools import combinations
from typing import Optional, Any
import numpy as np

class DemandSimulator:
    """
    Simulador de cobertura para una matriz de 24 x 28.

    Convenciones:
    - Filas: horas del día, 0..23
    - Columnas: días del horizonte, 0..27
    - Semanas: 0..3
    - Día de semana: 0..6
        0 = domingo
        1 = lunes
        2 = martes
        3 = miércoles
        4 = jueves
        5 = viernes
        6 = sábado

    En esta etapa, el simulador genera:
    - R: matriz de cobertura acumulada
    - T: trayectoria de acciones y matrices Rt
    """

    def __init__(
        self,
        entry_hours: list[int],
        close_hour: Optional[int],
        fixed_holidays: Optional[int],
        var_holidays: int,
        seed: Optional[int] = None,
    ):
        self.entry_hours = entry_hours
        self.close_hour = close_hour
        self.fixed_holidays = fixed_holidays
        self.var_holidays = var_holidays
        self.seed = seed

        self.rng = random.Random(seed)

        self._validate_init_params()

        self.holiday_options = self._build_holiday_options()
        self.holiday_action_map = self._build_holiday_action_map()

    # ============================================================
    # MÉTODO PRINCIPAL
    # ============================================================

    def compute_coverage(
        self,
        mod_4: int,
        mod_6: int,
        mod_8: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Genera la matriz de cobertura acumulada con todos los recursos.

        Inputs
        ------
        mod_4:
            Cantidad de recursos de 4 horas.

        mod_6:
            Cantidad de recursos de 6 horas.

        mod_8:
            Cantidad de recursos de 8 horas.

        Outputs
        -------
        R:
            Matriz de cobertura acumulada, shape (24, 28).

        T:
            Trayectoria de acciones.
        """

        self._validate_resource_inputs(mod_4, mod_6, mod_8)

        # 1. Generar recursos y mezclarlos aleatoriamente.
        resources_list = self._build_resources_list(mod_4, mod_6, mod_8)

        # 2. Expandir cada recurso a sus 4 semanas.
        resource_id_list, modality_list, week_list = self._expand_resources_by_week(
            resources_list
        )

        # 3. Elegir horario de ingreso legal para cada recurso-semana.
        entry_hour_list = self._sample_entry_hours(modality_list)

        # 4. Elegir francos para cada recurso-semana.
        holidays_list = self._sample_holidays_list(len(modality_list))

        # 5. Combinar todo en asignaciones semanales.
        assignments = list(
            zip(
                resource_id_list,
                modality_list,
                week_list,
                entry_hour_list,
                holidays_list,
            )
        )

        # 6. Construir matriz de cobertura y trayectoria.
        R, T = self._build_coverage_and_trajectory(assignments)

        return R, T

    # ============================================================
    # VALIDACIONES
    # ============================================================

    def _validate_init_params(self) -> None:
        if not isinstance(self.entry_hours, list) or len(self.entry_hours) == 0:
            raise ValueError("entry_hours debe ser una lista no vacía.")

        if any(not isinstance(h, int) or h < 0 or h > 23 for h in self.entry_hours):
            raise ValueError("entry_hours debe contener enteros entre 0 y 23.")

        if len(set(self.entry_hours)) != len(self.entry_hours):
            raise ValueError("entry_hours no debe contener valores repetidos.")

        if self.close_hour is not None:
            if not isinstance(self.close_hour, int) or self.close_hour < 0 or self.close_hour > 23:
                raise ValueError("close_hour debe ser un entero entre 0 y 23 o None.")

        if self.fixed_holidays is not None:
            if (
                not isinstance(self.fixed_holidays, int)
                or self.fixed_holidays < 0
                or self.fixed_holidays > 6
            ):
                raise ValueError("fixed_holidays debe ser un entero entre 0 y 6 o None.")

        if self.var_holidays not in (0, 1, 2):
            raise ValueError("var_holidays debe ser 0, 1 o 2.")

        fixed_count = 1 if self.fixed_holidays is not None else 0
        total_holidays = fixed_count + self.var_holidays

        if total_holidays > 2:
            raise ValueError(
                "No puede haber más de 2 días de franco entre fixed_holidays "
                "y var_holidays."
            )

    @staticmethod
    def _validate_resource_inputs(mod_4: int, mod_6: int, mod_8: int) -> None:
        for name, value in {
            "mod_4": mod_4,
            "mod_6": mod_6,
            "mod_8": mod_8,
        }.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} debe ser un entero mayor o igual a cero.")

    # ============================================================
    # GENERACIÓN DE LISTAS BASE
    # ============================================================

    def _build_resources_list(
        self,
        mod_4: int,
        mod_6: int,
        mod_8: int,
    ) -> list[int]:
        """
        Genera una lista aleatoria de modalidades de recursos.

        Ejemplo:
        mod_4 = 2, mod_6 = 1, mod_8 = 1

        Antes del shuffle:
        [4, 4, 6, 8]

        Después del shuffle:
        [6, 4, 8, 4]
        """

        resources_list = (
            [4 for _ in range(mod_4)]
            + [6 for _ in range(mod_6)]
            + [8 for _ in range(mod_8)]
        )

        self.rng.shuffle(resources_list)

        return resources_list

    @staticmethod
    def _expand_resources_by_week(
        resources_list: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Repite cada recurso por las 4 semanas.

        Si resources_list = [8, 4, 6]

        Entonces:
        resource_id_list = [0,0,0,0, 1,1,1,1, 2,2,2,2]
        modality_list    = [8,8,8,8, 4,4,4,4, 6,6,6,6]
        week_list        = [0,1,2,3, 0,1,2,3, 0,1,2,3]
        """

        amount_of_resources = len(resources_list)

        resource_id_list = [
            resource_id
            for resource_id in range(amount_of_resources)
            for _ in range(4)
        ]

        modality_list = [
            modality
            for modality in resources_list
            for _ in range(4)
        ]

        week_list = [
            week
            for _ in range(amount_of_resources)
            for week in range(4)
        ]

        return resource_id_list, modality_list, week_list

    # ============================================================
    # HORARIOS DE INGRESO
    # ============================================================

    def _sample_entry_hours(self, modality_list: list[int]) -> list[int]:
        """
        Para cada modalidad semanal, elige un horario de ingreso legal.
        """

        entry_hour_list = []

        for modality in modality_list:
            legal_hours = self._get_legal_entry_hours(modality)

            if len(legal_hours) == 0:
                raise ValueError(
                    f"No hay horarios legales para modalidad {modality} "
                    f"con close_hour={self.close_hour}."
                )

            selected_hour = self.rng.choice(legal_hours)
            entry_hour_list.append(selected_hour)

        return entry_hour_list

    def _get_legal_entry_hours(self, modality: int) -> list[int]:
        """
        Devuelve los horarios de ingreso legales para una modalidad.

        Si close_hour es None:
            se permite overflow al día siguiente.

        Si close_hour está definido:
            la jornada no puede exceder el cierre.

        Ejemplo:
            close_hour = 22
            modality = 8
            entry_hour = 15 -> cubre 15..22, legal
            entry_hour = 16 -> cubre 16..23, ilegal
        """

        if modality not in (4, 6, 8):
            raise ValueError(f"Modalidad inválida: {modality}")

        if self.close_hour is None:
            return list(self.entry_hours)

        legal_hours = []

        for hour in self.entry_hours:
            last_covered_hour = hour + modality - 1

            if last_covered_hour <= self.close_hour:
                legal_hours.append(hour)

        return legal_hours

    # ============================================================
    # FRANCOS
    # ============================================================

    def _build_holiday_options(self) -> list[tuple[int, ...]]:
        """
        Construye todas las combinaciones de francos permitidas según:
        - fixed_holidays
        - var_holidays

        Retorna una lista de tuplas.

        Ejemplos:
        fixed_holidays=None, var_holidays=1
            [(0,), (1,), ..., (6,)]

        fixed_holidays=None, var_holidays=2
            [(0,1), (0,2), ..., (5,6)]

        fixed_holidays=0, var_holidays=1
            [(0,1), (0,2), ..., (0,6)]

        fixed_holidays=0, var_holidays=0
            [(0,)]
        """

        all_days = list(range(7))

        if self.fixed_holidays is None:
            if self.var_holidays == 0:
                return [tuple()]

            return [
                tuple(days)
                for days in combinations(all_days, self.var_holidays)
            ]

        fixed = self.fixed_holidays

        if self.var_holidays == 0:
            return [(fixed,)]

        available_days = [d for d in all_days if d != fixed]

        return [
            tuple(sorted((fixed,) + days))
            for days in combinations(available_days, self.var_holidays)
        ]

    def _sample_holidays_list(self, amount: int) -> list[tuple[int, ...]]:
        """
        Genera una lista de francos, una por cada asignación semanal.
        """

        return [
            self.rng.choice(self.holiday_options)
            for _ in range(amount)
        ]

    # ============================================================
    # COBERTURA
    # ============================================================

    def _build_coverage_and_trajectory(
        self,
        assignments: list[tuple[int, int, int, int, tuple[int, ...]]],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Recorre las asignaciones semanales y construye:

        R:
            matriz de cobertura acumulada.

        T:
            trayectoria de acciones.

        Cada asignación semanal tiene:
            resource_id, modality, week, entry_hour, holidays

        Para cada asignación semanal guardamos 3 acciones:
            1. modalidad
            2. hora de ingreso
            3. francos

        La cobertura se actualiza luego de la acción de francos,
        porque recién ahí queda definido el bloque semanal completo.
        """

        R = np.zeros((24, 28), dtype=int)
        T = []

        step = 0

        for assignment in assignments:
            resource_id, modality, week, entry_hour, holidays = assignment

            # 1. Acción de modalidad
            T.append(
                self._make_trajectory_step(
                    step=step,
                    resource_id=resource_id,
                    week=week,
                    action_type="modality",
                    action_id=self._modality_action_id(modality),
                    action_value=modality,
                    Rt=R,
                )
            )
            step += 1

            # 2. Acción de hora de ingreso
            T.append(
                self._make_trajectory_step(
                    step=step,
                    resource_id=resource_id,
                    week=week,
                    action_type="entry_hour",
                    action_id=self._entry_hour_action_id(entry_hour),
                    action_value=entry_hour,
                    Rt=R,
                )
            )
            step += 1

            # 3. Acción de francos
            weekly_coverage = self._build_weekly_coverage(
                modality=modality,
                week=week,
                entry_hour=entry_hour,
                holidays=holidays,
            )

            R = R + weekly_coverage

            T.append(
                self._make_trajectory_step(
                    step=step,
                    resource_id=resource_id,
                    week=week,
                    action_type="holidays",
                    action_id=self._holiday_action_id(holidays),
                    action_value=holidays,
                    Rt=R,
                )
            )
            step += 1

        return R, T

    def _build_weekly_coverage(
        self,
        modality: int,
        week: int,
        entry_hour: int,
        holidays: tuple[int, ...],
    ) -> np.ndarray:
        """
        Construye la matriz de cobertura de una asignación semanal.

        week:
            0, 1, 2, 3

        global_day:
            week * 7 + day_of_week
        """

        coverage = np.zeros((24, 28), dtype=int)
        holidays_set = set(holidays)

        start_day = week * 7
        end_day = start_day + 6

        for global_day in range(start_day, end_day + 1):
            day_of_week = global_day % 7

            if day_of_week in holidays_set:
                continue

            self._apply_daily_shift(
                coverage=coverage,
                global_day=global_day,
                entry_hour=entry_hour,
                modality=modality,
            )

        return coverage

    def _apply_daily_shift(
        self,
        coverage: np.ndarray,
        global_day: int,
        entry_hour: int,
        modality: int,
    ) -> None:
        """
        Aplica la cobertura de una jornada diaria.

        Si close_hour es None, se permite overflow al día siguiente.
        Si close_hour está definido, no debería haber overflow porque
        el horario ya fue validado como legal.
        """

        for offset in range(modality):
            raw_hour = entry_hour + offset
            target_day = global_day

            if raw_hour >= 24:
                if self.close_hour is not None:
                    continue

                target_day += raw_hour // 24
                target_hour = raw_hour % 24
            else:
                target_hour = raw_hour

            if 0 <= target_day <= 27:
                coverage[target_hour, target_day] += 1

    # ============================================================
    # TRAYECTORIA
    # ============================================================

    @staticmethod
    def _make_trajectory_step(
        step: int,
        resource_id: int,
        week: int,
        action_type: str,
        action_id: int,
        action_value: Any,
        Rt: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "step": step,
            "resource_id": resource_id,
            "week": week,
            "action_type": action_type,
            "action_id": action_id,
            "action_value": action_value,
            "Rt": Rt.copy(),
        }

    # ============================================================
    # CODIFICACIÓN DE ACCIONES
    # ============================================================

    @staticmethod
    def _modality_action_id(modality: int) -> int:
        """
        0 -> modalidad 4h
        1 -> modalidad 6h
        2 -> modalidad 8h
        """

        mapping = {
            4: 0,
            6: 1,
            8: 2,
        }

        return mapping[modality]

    @staticmethod
    def _entry_hour_action_id(entry_hour: int) -> int:
        """
        3  -> hora 0
        4  -> hora 1
        ...
        26 -> hora 23
        """

        return 3 + entry_hour

    def _build_holiday_action_map(self) -> dict[tuple[int, ...], int]:
        """
        Construye el dominio de acciones para francos.

        Modalidades:
            0, 1, 2

        Horas:
            3..26

        Francos:
            desde 27 en adelante.

        Incluye todo el dominio posible:
            - sin francos: ()
            - un franco: (0,), ..., (6,)
            - dos francos: (0,1), ..., (5,6)

        Luego, en la generación real, solo se usan las combinaciones válidas
        según fixed_holidays y var_holidays.
        """

        action_map = {}
        action_id = 27

        # Sin francos
        action_map[tuple()] = action_id
        action_id += 1

        # Un franco
        for day in range(7):
            action_map[(day,)] = action_id
            action_id += 1

        # Dos francos
        for pair in combinations(range(7), 2):
            action_map[tuple(pair)] = action_id
            action_id += 1

        return action_map

    def _holiday_action_id(self, holidays: tuple[int, ...]) -> int:
        holidays = tuple(sorted(holidays))

        if holidays not in self.holiday_action_map:
            raise ValueError(f"Combinación de francos no codificada: {holidays}")

        return self.holiday_action_map[holidays]

    def get_holiday_action_table(self) -> dict[int, tuple[int, ...]]:
        """
        Devuelve la tabla action_id -> holidays.
        """

        return {
            action_id: holidays
            for holidays, action_id in self.holiday_action_map.items()
        }
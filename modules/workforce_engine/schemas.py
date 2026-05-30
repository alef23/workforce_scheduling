from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ActionType(str, Enum):
    """Tipos de acción soportados por el Workforce Engine."""

    MODALITY = "MODALITY"
    ENTRY_HOUR = "ENTRY_HOUR"
    DAY_OFFS = "DAY_OFFS"


class ProblemSetup(BaseModel):
    """
    Configuración fija del problema de planificación.

    Este objeto contiene reglas estructurales del entorno, no el estado dinámico
    de una trayectoria.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mobile_days_off_count: int
    fixed_day_off: Optional[int] = None
    allowed_entry_hours: Optional[list[int]] = None
    max_overcoverage_tolerance: float
    closing_hour: Optional[int] = None

    @field_validator("mobile_days_off_count")
    @classmethod
    def validate_mobile_days_off_count(cls, value: int) -> int:
        if not isinstance(value, int):
            raise TypeError("mobile_days_off_count debe ser un entero.")
        if value not in (0, 1, 2):
            raise ValueError("mobile_days_off_count debe ser 0, 1 o 2.")
        return value

    @field_validator("fixed_day_off")
    @classmethod
    def validate_fixed_day_off(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if not isinstance(value, int):
            raise TypeError("fixed_day_off debe ser None o un entero entre 0 y 6.")
        if value < 0 or value > 6:
            raise ValueError("fixed_day_off debe estar entre 0 y 6.")
        return value

    @field_validator("allowed_entry_hours")
    @classmethod
    def validate_allowed_entry_hours(
        cls, value: Optional[list[int]]
    ) -> Optional[list[int]]:
        if value is None:
            return value
        if not isinstance(value, list):
            raise TypeError("allowed_entry_hours debe ser None o una lista de enteros.")
        if len(value) == 0:
            raise ValueError("allowed_entry_hours no puede ser una lista vacía.")

        for hour in value:
            if not isinstance(hour, int):
                raise TypeError("Todos los valores de allowed_entry_hours deben ser enteros.")
            if hour < 0 or hour > 23:
                raise ValueError("Todos los horarios permitidos deben estar entre 0 y 23.")

        if len(value) != len(set(value)):
            raise ValueError("allowed_entry_hours no debe contener duplicados.")

        return sorted(value)

    @field_validator("max_overcoverage_tolerance")
    @classmethod
    def validate_max_overcoverage_tolerance(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("max_overcoverage_tolerance debe cumplir 0 < k <= 1.")
        return float(value)

    @field_validator("closing_hour")
    @classmethod
    def validate_closing_hour(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if not isinstance(value, int):
            raise TypeError("closing_hour debe ser None o un entero entre 0 y 23.")
        if value < 0 or value > 23:
            raise ValueError("closing_hour debe estar entre 0 y 23.")
        return value

    @model_validator(mode="after")
    def validate_combined_rules(self) -> "ProblemSetup":
        fixed_count = 1 if self.fixed_day_off is not None else 0
        total_days_off = self.mobile_days_off_count + fixed_count

        if total_days_off > 2:
            raise ValueError(
                "La suma de francos móviles y franco fijo no puede superar 2."
            )

        if self.closing_hour is not None and self.allowed_entry_hours is not None:
            invalid_hours = [
                hour for hour in self.allowed_entry_hours if hour >= self.closing_hour
            ]
            if invalid_hours:
                raise ValueError(
                    "Si existe closing_hour, ningún allowed_entry_hour puede ser "
                    f"mayor o igual al cierre. Valores inválidos: {invalid_hours}"
                )

        return self

    def get_allowed_entry_hours(self) -> list[int]:
        """
        Devuelve los horarios de entrada permitidos.

        Si allowed_entry_hours es None, se consideran permitidas todas las horas 0-23.
        La compatibilidad fina con modalidad y cierre se valida dentro del engine.
        """
        if self.allowed_entry_hours is None:
            return list(range(24))
        return list(self.allowed_entry_hours)


class WorkforceState(BaseModel):
    """
    Estado dinámico de una trayectoria de planificación.

    No almacena historia completa ni decisiones previas; solamente la información
    necesaria para continuar desde el estado actual.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    residual_demand: np.ndarray
    remaining_stock: np.ndarray
    expansion_mode: bool = False
    current_modality: Optional[int] = None
    current_entry_hour: Optional[int] = None
    assignment_week: int = 0
    initial_demand_total: int

    @field_validator("residual_demand", mode="before")
    @classmethod
    def coerce_residual_demand(cls, value) -> np.ndarray:
        return np.asarray(value)

    @field_validator("remaining_stock", mode="before")
    @classmethod
    def coerce_remaining_stock(cls, value) -> np.ndarray:
        return np.asarray(value)

    @field_validator("residual_demand")
    @classmethod
    def validate_residual_demand(cls, value: np.ndarray) -> np.ndarray:
        if value.shape != (24, 28):
            raise ValueError("residual_demand debe tener shape (24, 28).")
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError("residual_demand debe contener valores enteros.")
        return value.astype(int, copy=True)

    @field_validator("remaining_stock")
    @classmethod
    def validate_remaining_stock(cls, value: np.ndarray) -> np.ndarray:
        if value.shape != (3,):
            raise ValueError("remaining_stock debe tener shape (3,).")
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError("remaining_stock debe contener valores enteros.")
        if np.any(value < 0):
            raise ValueError("remaining_stock no puede contener valores negativos.")
        return value.astype(int, copy=True)

    @field_validator("expansion_mode")
    @classmethod
    def validate_expansion_mode(cls, value: bool) -> bool:
        if not isinstance(value, bool):
            raise TypeError("expansion_mode debe ser booleano.")
        return value

    @field_validator("current_modality")
    @classmethod
    def validate_current_modality(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if value not in (4, 6, 8):
            raise ValueError("current_modality debe ser None, 4, 6 u 8.")
        return value

    @field_validator("current_entry_hour")
    @classmethod
    def validate_current_entry_hour(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if not isinstance(value, int):
            raise TypeError("current_entry_hour debe ser None o un entero entre 0 y 23.")
        if value < 0 or value > 23:
            raise ValueError("current_entry_hour debe estar entre 0 y 23.")
        return value

    @field_validator("assignment_week")
    @classmethod
    def validate_assignment_week(cls, value: int) -> int:
        if not isinstance(value, int):
            raise TypeError("assignment_week debe ser un entero.")
        if value < 0 or value > 3:
            raise ValueError("assignment_week debe estar entre 0 y 3.")
        return value

    @field_validator("initial_demand_total")
    @classmethod
    def validate_initial_demand_total(cls, value: int) -> int:
        if not isinstance(value, int):
            raise TypeError("initial_demand_total debe ser un entero.")
        if value <= 0:
            raise ValueError("initial_demand_total debe ser mayor que 0.")
        return value

    def copy_state(self, **updates) -> "WorkforceState":
        """
        Crea una copia validada del estado.

        Se copian explícitamente los arrays numpy para evitar contaminación entre ramas
        de búsqueda en MCTS.
        """
        data = self.model_dump()
        data["residual_demand"] = self.residual_demand.copy()
        data["remaining_stock"] = self.remaining_stock.copy()
        data.update(updates)
        return WorkforceState(**data)


class StepResult(BaseModel):
    """Resultado de una transición del Workforce Engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    next_state: WorkforceState
    is_terminal: bool
    reward: float

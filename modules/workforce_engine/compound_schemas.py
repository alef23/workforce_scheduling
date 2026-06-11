from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class CompoundWorkforceState(BaseModel):
    """Estado dinámico para acciones semanales compuestas."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    residual_demand: np.ndarray
    remaining_stock: np.ndarray
    expansion_mode: bool = False
    current_modality: Optional[int] = None
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
            raise TypeError("residual_demand debe contener enteros.")
        return value.astype(int, copy=True)

    @field_validator("remaining_stock")
    @classmethod
    def validate_remaining_stock(cls, value: np.ndarray) -> np.ndarray:
        if value.shape != (3,):
            raise ValueError("remaining_stock debe tener shape (3,).")
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError("remaining_stock debe contener enteros.")
        if np.any(value < 0):
            raise ValueError("remaining_stock no puede contener valores negativos.")
        if int(value.sum()) > 20:
            raise ValueError("El stock total no puede superar 20 recursos.")
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
        if value is not None and value not in (4, 6, 8):
            raise ValueError("current_modality debe ser None, 4, 6 u 8.")
        return value

    @field_validator("assignment_week")
    @classmethod
    def validate_assignment_week(cls, value: int) -> int:
        if not isinstance(value, int) or value not in range(4):
            raise ValueError("assignment_week debe ser 0, 1, 2 o 3.")
        return value

    @field_validator("initial_demand_total")
    @classmethod
    def validate_initial_demand_total(cls, value: int) -> int:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("initial_demand_total debe ser un entero positivo.")
        return value

    def copy_state(self, **updates) -> "CompoundWorkforceState":
        data = self.model_dump()
        data["residual_demand"] = self.residual_demand.copy()
        data["remaining_stock"] = self.remaining_stock.copy()
        data.update(updates)
        return CompoundWorkforceState(**data)


class CompoundStepResult(BaseModel):
    """Resultado individual compatible con el contrato de transición MCTS."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    next_state: CompoundWorkforceState
    is_terminal: bool
    reward: float

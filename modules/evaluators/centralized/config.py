from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CentralizedEvaluatorConfig:
    """
    Configuracion del proceso unico de evaluacion ResNet.

    El server es el unico componente que debe cargar el modelo en GPU.
    """

    checkpoint_path: str | Path
    device: str = "auto"
    max_batch_size: int = 32
    batch_wait_s: float = 0.01
    request_timeout_s: float | None = None
    demand_ref: float = 300.0
    stock_ref: float = 100.0

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size debe ser positivo.")
        if self.batch_wait_s < 0:
            raise ValueError("batch_wait_s debe ser >= 0.")
        if self.request_timeout_s is not None and self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s debe ser positivo o None.")
        if self.demand_ref <= 0:
            raise ValueError("demand_ref debe ser positivo.")
        if self.stock_ref <= 0:
            raise ValueError("stock_ref debe ser positivo.")

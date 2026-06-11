from __future__ import annotations

from typing import Any

import torch


class CompoundActionStateEncoder:
    """
    Encoder experimental para el dominio fijo de acciones compuestas.

    Convierte los inputs crudos en un tensor shape (B, 11, 28, 28):

    - residual_demand: 1 canal
    - initial_demand_total: 1 canal
    - remaining_stock: 3 canales
    - current_modality: 3 canales one-hot
    - assignment_week: 3 canales one-hot reducidos
    """

    CHANNELS = 11
    HEIGHT = 28
    WIDTH = 28
    REAL_HOURS = 24
    TOP_PADDING = 2

    DEMAND_REF = 20.0
    STOCK_REF = 20.0
    INITIAL_DEMAND_TOTAL_REF = DEMAND_REF * REAL_HOURS * WIDTH

    def __init__(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype

    def __call__(self, X: dict[str, Any]) -> torch.Tensor:
        residual_demand = self._as_tensor(
            X["residual_demand"],
            dtype=torch.float32,
        )
        if residual_demand.ndim == 2:
            residual_demand = residual_demand.unsqueeze(0)
        if residual_demand.shape[1:] != (24, 28):
            raise ValueError(
                "residual_demand debe tener shape (B, 24, 28) o (24, 28)."
            )

        batch_size = int(residual_demand.shape[0])
        encoded = torch.zeros(
            (batch_size, self.CHANNELS, self.HEIGHT, self.WIDTH),
            dtype=self.dtype,
            device=self.device,
        )
        encoded[
            :,
            0,
            self.TOP_PADDING:self.TOP_PADDING + self.REAL_HOURS,
            :,
        ] = residual_demand / self.DEMAND_REF

        initial_demand_total = self._vector(
            X["initial_demand_total"],
            batch_size=batch_size,
            dtype=torch.float32,
        )
        encoded[:, 1] = (
            initial_demand_total / self.INITIAL_DEMAND_TOTAL_REF
        )[:, None, None]

        remaining_stock = self._matrix(
            X["remaining_stock"],
            batch_size=batch_size,
            width=3,
            dtype=torch.float32,
        )
        normalized_stock = remaining_stock / self.STOCK_REF
        encoded[:, 2] = normalized_stock[:, 0, None, None]
        encoded[:, 3] = normalized_stock[:, 1, None, None]
        encoded[:, 4] = normalized_stock[:, 2, None, None]

        current_modality = self._vector(
            X["current_modality"],
            batch_size=batch_size,
            dtype=torch.int64,
        )
        for index, modality in enumerate((4, 6, 8)):
            encoded[:, 5 + index] = (
                current_modality == modality
            )[:, None, None].to(self.dtype)

        assignment_week = self._vector(
            X["assignment_week"],
            batch_size=batch_size,
            dtype=torch.int64,
        )
        for week in (1, 2, 3):
            encoded[:, 7 + week] = (
                assignment_week == week
            )[:, None, None].to(self.dtype)

        return encoded

    def _as_tensor(self, value: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _vector(
        self,
        value: Any,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = self._as_tensor(value, dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.repeat(batch_size)
        if tensor.ndim == 1 and tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.repeat(batch_size)
        if tensor.shape != (batch_size,):
            raise ValueError(
                f"Vector esperado shape ({batch_size},), "
                f"recibido {tuple(tensor.shape)}."
            )
        return tensor

    def _matrix(
        self,
        value: Any,
        batch_size: int,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = self._as_tensor(value, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.repeat(batch_size, 1)
        if tensor.shape != (batch_size, width):
            raise ValueError(
                f"Matriz esperada shape ({batch_size}, {width}), "
                f"recibida {tuple(tensor.shape)}."
            )
        return tensor

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class StateEncoder:
    """
    Convierte un estado crudo en un tensor PyTorch de entrada para la red.

    Input esperado:
        X: dict con estado + ProblemSetup

    Output:
        torch.Tensor shape (B, 93, 28, 28)

    Convenciones:
        - residual_demand: (B, 24, 28) o (24, 28)
        - initial_demand_total: (B,) o escalar
        - remaining_stock: (B, 3) o (3,)
        - variables escalares: (B,) o escalar
        - allowed_entry_hours: (B, 24), (24,), lista de horas permitidas
          o lista por sample para batches
        - valores None provenientes de Zarr se representan como -1
    """

    CHANNELS = 93
    HEIGHT = 28
    WIDTH = 28

    REAL_HOURS = 24
    TOP_PADDING = 2
    BOTTOM_PADDING = 2

    DEMAND_CHANNEL = 0
    INITIAL_DEMAND_TOTAL_CHANNEL = 1

    STOCK_CHANNELS = {
        4: 2,
        6: 3,
        8: 4,
    }

    MODALITY_CHANNELS = {
        4: 5,
        6: 6,
        8: 7,
    }

    WEEK_CHANNEL_OFFSET = 8             # canales 8, 9, 10 para semanas 1, 2, 3
    MOBILE_DAYS_OFF_OFFSET = 11         # canales 11, 12, 13
    FIXED_DAY_OFF_OFFSET = 14           # canales 14..20
    ALLOWED_ENTRY_HOURS_OFFSET = 21     # canales 21..44
    CLOSING_HOUR_OFFSET = 45            # canales 45..68
    CURRENT_ENTRY_HOUR_OFFSET = 69      # canales 69..92

    def __init__(
        self,
        demand_ref: float = 300.0,
        stock_ref: float = 100.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.demand_ref = float(demand_ref)
        self.stock_ref = float(stock_ref)
        self.initial_demand_total_ref = float(demand_ref * 24 * 28)

        self.device = torch.device(device)
        self.dtype = dtype

    def __call__(self, X: dict[str, Any]) -> torch.Tensor:
        return self.transform(X)

    def transform(self, X: dict[str, Any]) -> torch.Tensor:
        residual_demand = self._get_residual_demand(X["residual_demand"])
        batch_size = residual_demand.shape[0]

        encoded = torch.zeros(
            (batch_size, self.CHANNELS, self.HEIGHT, self.WIDTH),
            dtype=self.dtype,
            device=self.device,
        )

        self._encode_residual_demand(
            encoded=encoded,
            residual_demand=residual_demand,
        )

        self._encode_initial_demand_total(
            encoded=encoded,
            initial_demand_total=self._get_vector(
                X["initial_demand_total"],
                batch_size=batch_size,
                dtype=torch.float32,
            ),
        )

        self._encode_remaining_stock(
            encoded=encoded,
            remaining_stock=self._get_matrix(
                X["remaining_stock"],
                batch_size=batch_size,
                width=3,
                dtype=torch.float32,
            ),
        )

        self._encode_current_modality(
            encoded=encoded,
            current_modality=self._get_vector(
                X["current_modality"],
                batch_size=batch_size,
                dtype=torch.int64,
            ),
        )

        self._encode_assignment_week(
            encoded=encoded,
            assignment_week=self._get_vector(
                X["assignment_week"],
                batch_size=batch_size,
                dtype=torch.int64,
            ),
        )

        self._encode_mobile_days_off_count(
            encoded=encoded,
            mobile_days_off_count=self._get_vector(
                X["mobile_days_off_count"],
                batch_size=batch_size,
                dtype=torch.int64,
            ),
        )

        self._encode_fixed_day_off(
            encoded=encoded,
            fixed_day_off=self._get_vector(
                X["fixed_day_off"],
                batch_size=batch_size,
                dtype=torch.int64,
            ),
        )

        self._encode_allowed_entry_hours(
            encoded=encoded,
            allowed_entry_hours=self._get_allowed_entry_hours_mask(
                X["allowed_entry_hours"],
                batch_size=batch_size,
            ),
        )

        self._encode_closing_hour(
            encoded=encoded,
            closing_hour=self._get_vector(
                X["closing_hour"],
                batch_size=batch_size,
                dtype=torch.int64,
            ),
        )

        self._encode_current_entry_hour(
            encoded=encoded,
            current_entry_hour=self._get_vector(
                X["current_entry_hour"],
                batch_size=batch_size,
                dtype=torch.int64,
            ),
        )

        return encoded

    # ============================================================
    # Normalización de inputs
    # ============================================================

    def _to_tensor(
        self,
        value: Any,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.as_tensor(
            value,
            dtype=dtype,
            device=self.device,
        )

    def _get_residual_demand(self, value: Any) -> torch.Tensor:
        tensor = self._to_tensor(value, dtype=torch.float32)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        if tensor.shape[1:] != (24, 28):
            raise ValueError(
                f"residual_demand debe tener shape (B, 24, 28) o (24, 28), "
                f"pero tiene {tuple(tensor.shape)}."
            )

        return tensor

    def _get_vector(
        self,
        value: Any,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if value is None:
            return torch.full(
                (batch_size,),
                fill_value=-1,
                dtype=dtype,
                device=self.device,
            )

        tensor = self._to_tensor(value, dtype=dtype)

        if tensor.ndim == 0:
            tensor = tensor.repeat(batch_size)

        if tensor.ndim == 1 and tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.repeat(batch_size)

        if tensor.shape != (batch_size,):
            raise ValueError(
                f"Vector esperado shape ({batch_size},), "
                f"pero tiene {tuple(tensor.shape)}."
            )

        return tensor

    def _get_matrix(
        self,
        value: Any,
        batch_size: int,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = self._to_tensor(value, dtype=dtype)

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.repeat(batch_size, 1)

        if tensor.shape != (batch_size, width):
            raise ValueError(
                f"Matriz esperada shape ({batch_size}, {width}), "
                f"pero tiene {tuple(tensor.shape)}."
            )

        return tensor

    def _get_allowed_entry_hours_mask(
        self,
        value: Any,
        batch_size: int,
    ) -> torch.Tensor:
        if value is None:
            return torch.ones(
                (batch_size, 24),
                dtype=self.dtype,
                device=self.device,
            )

        if isinstance(value, (list, tuple)):
            return self._allowed_entry_hours_from_sequence(value, batch_size)

        arr = np.asarray(value)

        if arr.ndim == 2:
            tensor = self._to_tensor(arr, dtype=self.dtype)

            if tensor.shape[0] == 1 and batch_size > 1:
                tensor = tensor.repeat(batch_size, 1)

            if tensor.shape != (batch_size, 24):
                raise ValueError(
                    f"allowed_entry_hours debe tener shape ({batch_size}, 24), "
                    f"pero tiene {tuple(tensor.shape)}."
                )

            return tensor

        if arr.ndim == 1:
            if arr.shape[0] == 24 and np.all(np.isin(arr, [0, 1, False, True])):
                tensor = self._to_tensor(arr, dtype=self.dtype).unsqueeze(0)

                if batch_size > 1:
                    tensor = tensor.repeat(batch_size, 1)

                return tensor

            if batch_size != 1:
                raise ValueError(
                    "Una lista simple de horas permitidas solo es válida "
                    "para batch_size=1. Para batch usar máscara (B, 24)."
                )

            mask = torch.zeros(
                (1, 24),
                dtype=self.dtype,
                device=self.device,
            )

            for hour in arr:
                mask[0, int(hour)] = 1.0

            return mask

        raise ValueError("Formato inválido para allowed_entry_hours.")

    def _allowed_entry_hours_from_sequence(
        self,
        value: list[Any] | tuple[Any, ...],
        batch_size: int,
    ) -> torch.Tensor:
        if self._is_mask_like(value):
            tensor = self._to_tensor(np.asarray(value), dtype=self.dtype).unsqueeze(0)

            if batch_size > 1:
                tensor = tensor.repeat(batch_size, 1)

            return tensor

        if batch_size == 1 and self._is_hour_sequence(value):
            return self._hours_to_mask([value])

        if len(value) != batch_size:
            raise ValueError(
                "allowed_entry_hours debe tener un elemento por sample "
                f"para batch_size={batch_size}."
            )

        return self._hours_to_mask(value)

    def _hours_to_mask(
        self,
        values: list[Any] | tuple[Any, ...],
    ) -> torch.Tensor:
        mask = torch.zeros(
            (len(values), 24),
            dtype=self.dtype,
            device=self.device,
        )

        for row, hours in enumerate(values):
            if hours is None:
                mask[row, :] = 1.0
                continue

            if not self._is_hour_sequence(hours):
                raise ValueError(
                    "Cada entrada de allowed_entry_hours debe ser None "
                    "o una secuencia de horas."
                )

            for hour in hours:
                hour_int = int(hour)
                if hour_int < 0 or hour_int > 23:
                    raise ValueError(
                        "allowed_entry_hours solo acepta horas entre 0 y 23."
                    )
                mask[row, hour_int] = 1.0

        return mask

    @staticmethod
    def _is_mask_like(value: Any) -> bool:
        try:
            arr = np.asarray(value)
        except ValueError:
            return False
        return (
            arr.ndim == 1
            and arr.shape[0] == 24
            and np.all(np.isin(arr, [0, 1, False, True]))
        )

    @staticmethod
    def _is_hour_sequence(value: Any) -> bool:
        if value is None or not isinstance(value, (list, tuple, np.ndarray)):
            return False
        try:
            arr = np.asarray(value)
        except ValueError:
            return False
        return arr.ndim == 1

    # ============================================================
    # Escritura de canales
    # ============================================================

    def _fill_channel(
        self,
        encoded: torch.Tensor,
        channel: int,
        values: torch.Tensor,
    ) -> None:
        encoded[:, channel, :, :] = values[:, None, None].to(self.dtype)

    def _fill_binary_channel(
        self,
        encoded: torch.Tensor,
        channel: int,
        mask: torch.Tensor,
    ) -> None:
        encoded[:, channel, :, :] = mask[:, None, None].to(self.dtype)

    def _encode_residual_demand(
        self,
        encoded: torch.Tensor,
        residual_demand: torch.Tensor,
    ) -> None:
        """
        Canal 0:
            demanda residual normalizada por D_ref,
            ubicada en filas 2:26.
        """

        demand_normalized = residual_demand / self.demand_ref

        encoded[
            :,
            self.DEMAND_CHANNEL,
            self.TOP_PADDING:self.TOP_PADDING + self.REAL_HOURS,
            :,
        ] = demand_normalized.to(self.dtype)

    def _encode_initial_demand_total(
        self,
        encoded: torch.Tensor,
        initial_demand_total: torch.Tensor,
    ) -> None:
        """
        Canal 1:
            demanda inicial total normalizada por D_ref * 24 * 28,
            replicada en toda la grilla.
        """

        value = initial_demand_total / self.initial_demand_total_ref

        self._fill_channel(
            encoded=encoded,
            channel=self.INITIAL_DEMAND_TOTAL_CHANNEL,
            values=value,
        )

    def _encode_remaining_stock(
        self,
        encoded: torch.Tensor,
        remaining_stock: torch.Tensor,
    ) -> None:
        """
        Canales 2-4:
            stock 4h, 6h, 8h normalizado por N_ref.
        """

        stock_normalized = remaining_stock / self.stock_ref

        encoded[:, 2, :, :] = stock_normalized[:, 0, None, None]
        encoded[:, 3, :, :] = stock_normalized[:, 1, None, None]
        encoded[:, 4, :, :] = stock_normalized[:, 2, None, None]

    def _encode_current_modality(
        self,
        encoded: torch.Tensor,
        current_modality: torch.Tensor,
    ) -> None:
        """
        Canales 5-7:
            one-hot modalidad activa 4, 6, 8.
        """

        for modality, channel in self.MODALITY_CHANNELS.items():
            self._fill_binary_channel(
                encoded=encoded,
                channel=channel,
                mask=current_modality == modality,
            )

    def _encode_assignment_week(
        self,
        encoded: torch.Tensor,
        assignment_week: torch.Tensor,
    ) -> None:
        """
        Canales 8-10:
            one-hot reducido de semana.

        week=0 -> implícita, canales 8-10 en cero
        week=1 -> canal 8
        week=2 -> canal 9
        week=3 -> canal 10
        """

        for week in (1, 2, 3):
            channel = self.WEEK_CHANNEL_OFFSET + (week - 1)

            self._fill_binary_channel(
                encoded=encoded,
                channel=channel,
                mask=assignment_week == week,
            )

    def _encode_mobile_days_off_count(
        self,
        encoded: torch.Tensor,
        mobile_days_off_count: torch.Tensor,
    ) -> None:
        """
        Canales 11-13:
            one-hot cantidad de francos móviles 0, 1, 2.
        """

        for count in (0, 1, 2):
            channel = self.MOBILE_DAYS_OFF_OFFSET + count

            self._fill_binary_channel(
                encoded=encoded,
                channel=channel,
                mask=mobile_days_off_count == count,
            )

    def _encode_fixed_day_off(
        self,
        encoded: torch.Tensor,
        fixed_day_off: torch.Tensor,
    ) -> None:
        """
        Canales 14-20:
            one-hot día de franco fijo.
            -1 significa sin franco fijo.
        """

        for day in range(7):
            channel = self.FIXED_DAY_OFF_OFFSET + day

            self._fill_binary_channel(
                encoded=encoded,
                channel=channel,
                mask=fixed_day_off == day,
            )

    def _encode_allowed_entry_hours(
        self,
        encoded: torch.Tensor,
        allowed_entry_hours: torch.Tensor,
    ) -> None:
        """
        Canales 21-44:
            multi-hot horarios de ingreso permitidos.
        """

        values = allowed_entry_hours[:, :, None, None].to(self.dtype)

        encoded[
            :,
            self.ALLOWED_ENTRY_HOURS_OFFSET:self.ALLOWED_ENTRY_HOURS_OFFSET + 24,
            :,
            :,
        ] = values.expand(-1, -1, self.HEIGHT, self.WIDTH)

    def _encode_closing_hour(
        self,
        encoded: torch.Tensor,
        closing_hour: torch.Tensor,
    ) -> None:
        """
        Canales 45-68:
            one-hot horario de cierre.
            -1 significa sin cierre operativo.
        """

        for hour in range(24):
            channel = self.CLOSING_HOUR_OFFSET + hour

            self._fill_binary_channel(
                encoded=encoded,
                channel=channel,
                mask=closing_hour == hour,
            )

    def _encode_current_entry_hour(
        self,
        encoded: torch.Tensor,
        current_entry_hour: torch.Tensor,
    ) -> None:
        """
        Canales 69-92:
            one-hot horario de ingreso activo.
            -1 significa que todavía no hay horario seleccionado.
        """

        for hour in range(24):
            channel = self.CURRENT_ENTRY_HOUR_OFFSET + hour

            self._fill_binary_channel(
                encoded=encoded,
                channel=channel,
                mask=current_entry_hour == hour,
            )

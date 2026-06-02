from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from modules.evaluators.resnet.encoder import StateEncoder
from modules.evaluators.resnet.resnet_evaluator import WorkforceResNet

from .config import CentralizedEvaluatorConfig
from .messages import (
    ControlResponse,
    PredictRequest,
    PredictResponse,
    ReloadWeightsRequest,
    ShutdownRequest,
)


class CentralizedEvaluatorServer:
    """
    Proceso unico de inferencia ResNet para multiples workers MCTS.

    El server es sincrono desde el punto de vista del cliente: cada worker espera
    su respuesta. Internamente agrupa requests pendientes para aprovechar la GPU.
    """

    def __init__(
        self,
        config: CentralizedEvaluatorConfig,
        request_queue: mp.Queue,
        response_queues: Mapping[str, mp.Queue],
    ) -> None:
        self.config = config
        self.request_queue = request_queue
        self.response_queues = dict(response_queues)
        self.device = self._resolve_device(config.device)
        self.model: WorkforceResNet | None = None
        self.encoder = StateEncoder(
            demand_ref=config.demand_ref,
            stock_ref=config.stock_ref,
            device=self.device,
        )
        self.model_version = 0
        self.running = False

    def run_forever(self) -> None:
        self._load_checkpoint(self.config.checkpoint_path)
        self.running = True

        while self.running:
            request = self.request_queue.get()

            if isinstance(request, PredictRequest):
                batch = self._collect_predict_batch(first_request=request)
                self._handle_predict_batch(batch)
                continue

            if isinstance(request, ReloadWeightsRequest):
                self._handle_reload(request)
                continue

            if isinstance(request, ShutdownRequest):
                self._handle_shutdown(request)
                continue

            self._send_control_error(
                client_id=getattr(request, "client_id", "unknown"),
                request_id=getattr(request, "request_id", "unknown"),
                error=f"Request no soportado: {type(request).__name__}",
            )

    def _collect_predict_batch(
        self,
        first_request: PredictRequest,
    ) -> list[PredictRequest]:
        batch = [first_request]
        max_batch_size = int(self.config.max_batch_size)

        while len(batch) < max_batch_size:
            try:
                request = self.request_queue.get(timeout=self.config.batch_wait_s)
            except queue.Empty:
                break

            if isinstance(request, PredictRequest):
                batch.append(request)
                continue

            self._handle_control_during_predict_collection(request)
            break

        return batch

    def _handle_control_during_predict_collection(self, request: Any) -> None:
        if isinstance(request, ReloadWeightsRequest):
            self._handle_reload(request)
            return
        if isinstance(request, ShutdownRequest):
            self._handle_shutdown(request)
            return
        self._send_control_error(
            client_id=getattr(request, "client_id", "unknown"),
            request_id=getattr(request, "request_id", "unknown"),
            error=f"Request no soportado: {type(request).__name__}",
        )

    def _handle_predict_batch(self, batch: list[PredictRequest]) -> None:
        if not batch:
            return

        try:
            X = self._build_batch_input(batch)
            with torch.no_grad():
                encoded = self.encoder(X)
                policy_logits, values = self._model(encoded)
                policies = F.softmax(policy_logits, dim=1).detach().cpu().numpy()
                values_np = values.detach().cpu().numpy()

            for index, request in enumerate(batch):
                self._response_queue(request.client_id).put(
                    PredictResponse(
                        request_id=request.request_id,
                        policy=np.asarray(policies[index], dtype=np.float32),
                        value=float(values_np[index]),
                        model_version=self.model_version,
                    )
                )
        except Exception as exc:
            for request in batch:
                self._response_queue(request.client_id).put(
                    PredictResponse(
                        request_id=request.request_id,
                        policy=np.zeros((55,), dtype=np.float32),
                        value=0.0,
                        model_version=self.model_version,
                        error=str(exc),
                    )
                )

    def _handle_reload(self, request: ReloadWeightsRequest) -> None:
        try:
            self._load_checkpoint(request.checkpoint_path)
            self._response_queue(request.client_id).put(
                ControlResponse(
                    request_id=request.request_id,
                    model_version=self.model_version,
                )
            )
        except Exception as exc:
            self._send_control_error(request.client_id, request.request_id, str(exc))

    def _handle_shutdown(self, request: ShutdownRequest) -> None:
        self.running = False
        self._response_queue(request.client_id).put(
            ControlResponse(
                request_id=request.request_id,
                model_version=self.model_version,
            )
        )

    def _send_control_error(
        self,
        client_id: str,
        request_id: str,
        error: str,
    ) -> None:
        self._response_queue(client_id).put(
            ControlResponse(
                request_id=request_id,
                model_version=self.model_version,
                error=error,
            )
        )

    def _response_queue(self, client_id: str) -> mp.Queue:
        try:
            return self.response_queues[str(client_id)]
        except KeyError as exc:
            raise KeyError(
                f"No existe response_queue registrada para client_id={client_id!r}."
            ) from exc

    def _build_batch_input(self, batch: list[PredictRequest]) -> dict[str, Any]:
        return {
            "residual_demand": np.stack(
                [np.asarray(request.state.residual_demand) for request in batch],
                axis=0,
            ),
            "initial_demand_total": np.asarray(
                [int(request.state.initial_demand_total) for request in batch],
                dtype=np.int64,
            ),
            "remaining_stock": np.stack(
                [np.asarray(request.state.remaining_stock) for request in batch],
                axis=0,
            ),
            "current_modality": [
                self._none_to_minus_one(request.state.current_modality)
                for request in batch
            ],
            "assignment_week": [
                int(request.state.assignment_week)
                for request in batch
            ],
            "mobile_days_off_count": [
                int(request.setup.mobile_days_off_count)
                for request in batch
            ],
            "fixed_day_off": [
                self._none_to_minus_one(request.setup.fixed_day_off)
                for request in batch
            ],
            "allowed_entry_hours": [
                request.setup.allowed_entry_hours
                for request in batch
            ],
            "closing_hour": [
                self._none_to_minus_one(request.setup.closing_hour)
                for request in batch
            ],
            "current_entry_hour": [
                self._none_to_minus_one(request.state.current_entry_hour)
                for request in batch
            ],
        }

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el checkpoint: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        model_config = checkpoint.get("model_config", {})

        model = WorkforceResNet(**model_config).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model
        self.config = replace(self.config, checkpoint_path=path)
        self.model_version += 1

    @property
    def _model(self) -> WorkforceResNet:
        if self.model is None:
            raise RuntimeError("El modelo todavia no fue cargado.")
        return self.model

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if str(device) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _none_to_minus_one(value: int | None) -> int:
        return -1 if value is None else int(value)

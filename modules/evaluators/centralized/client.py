from __future__ import annotations

import multiprocessing as mp
import queue
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from modules.evaluators.base import EvaluatorProtocol
from modules.workforce_engine.schemas import ProblemSetup, WorkforceState

from .config import CentralizedEvaluatorConfig
from .messages import (
    ControlResponse,
    PredictRequest,
    PredictResponse,
    ReloadWeightsRequest,
    ShutdownRequest,
)
from .server import CentralizedEvaluatorServer


class CentralizedEvaluatorClient(EvaluatorProtocol):
    """
    Cliente liviano compatible con MCTS.

    Cada llamada a predict es sincrona: envia un estado al server centralizado y
    espera una respuesta. El server es el unico que carga la ResNet en GPU.
    """

    action_space_size = 55

    def __init__(
        self,
        setup: ProblemSetup,
        client_id: str,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        request_timeout_s: float | None = None,
    ) -> None:
        self.setup = setup
        self.client_id = str(client_id)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.request_timeout_s = request_timeout_s
        self._pending_responses: dict[str, PredictResponse | ControlResponse] = {}
        self.model_version: int | None = None

    def predict(self, state: WorkforceState) -> tuple[np.ndarray, float]:
        request_id = self._request_id()
        self.request_queue.put(
            PredictRequest(
                request_id=request_id,
                client_id=self.client_id,
                setup=self.setup,
                state=state,
            )
        )

        response = self._wait_for_response(request_id)
        if not isinstance(response, PredictResponse):
            raise RuntimeError(
                f"Respuesta inesperada para predict: {type(response).__name__}"
            )
        if response.error is not None:
            raise RuntimeError(response.error)

        self.model_version = int(response.model_version)
        return response.policy.astype(float), float(response.value)

    def reload_weights(self, checkpoint_path: str | Path) -> int:
        request_id = self._request_id()
        self.request_queue.put(
            ReloadWeightsRequest(
                request_id=request_id,
                client_id=self.client_id,
                checkpoint_path=checkpoint_path,
            )
        )
        response = self._wait_for_response(request_id)
        if not isinstance(response, ControlResponse):
            raise RuntimeError(
                f"Respuesta inesperada para reload: {type(response).__name__}"
            )
        if response.error is not None:
            raise RuntimeError(response.error)

        self.model_version = int(response.model_version)
        return self.model_version

    def shutdown_server(self) -> int:
        request_id = self._request_id()
        self.request_queue.put(
            ShutdownRequest(
                request_id=request_id,
                client_id=self.client_id,
            )
        )
        response = self._wait_for_response(request_id)
        if not isinstance(response, ControlResponse):
            raise RuntimeError(
                f"Respuesta inesperada para shutdown: {type(response).__name__}"
            )
        if response.error is not None:
            raise RuntimeError(response.error)

        self.model_version = int(response.model_version)
        return self.model_version

    def _wait_for_response(
        self,
        request_id: str,
    ) -> PredictResponse | ControlResponse:
        if request_id in self._pending_responses:
            return self._pending_responses.pop(request_id)

        while True:
            try:
                response = self.response_queue.get(timeout=self.request_timeout_s)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"Timeout esperando respuesta del evaluador: {request_id}"
                ) from exc

            response_id = getattr(response, "request_id", None)
            if response_id == request_id:
                return response

            if response_id is not None:
                self._pending_responses[str(response_id)] = response

    @staticmethod
    def _request_id() -> str:
        return uuid.uuid4().hex

    @classmethod
    def start_server(
        cls,
        config: CentralizedEvaluatorConfig,
        setup: ProblemSetup,
        context: mp.context.BaseContext | None = None,
    ) -> tuple["CentralizedEvaluatorClient", mp.Process]:
        """
        Helper para pruebas y scripts locales.

        En un orquestador real, un unico proceso debe crear el server y repartir
        las colas a los workers.
        """
        ctx = context or mp.get_context("spawn")
        request_queue = ctx.Queue()
        response_queue = ctx.Queue()
        client_id = "main"

        process = ctx.Process(
            target=_run_server_process,
            args=(config, request_queue, {client_id: response_queue}),
            daemon=True,
        )
        process.start()

        client = cls(
            setup=setup,
            client_id=client_id,
            request_queue=request_queue,
            response_queue=response_queue,
            request_timeout_s=config.request_timeout_s,
        )
        return client, process


def _run_server_process(
    config: CentralizedEvaluatorConfig,
    request_queue: mp.Queue,
    response_queues: dict[str, mp.Queue],
) -> None:
    server = CentralizedEvaluatorServer(
        config=config,
        request_queue=request_queue,
        response_queues=response_queues,
    )
    server.run_forever()

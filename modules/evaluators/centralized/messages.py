from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


RequestKind = Literal["predict", "reload", "shutdown"]


@dataclass(frozen=True)
class PredictRequest:
    request_id: str
    client_id: str
    setup: Any
    state: Any
    kind: RequestKind = "predict"


@dataclass(frozen=True)
class ReloadWeightsRequest:
    request_id: str
    client_id: str
    checkpoint_path: str | Path
    kind: RequestKind = "reload"


@dataclass(frozen=True)
class ShutdownRequest:
    request_id: str
    client_id: str
    kind: RequestKind = "shutdown"


@dataclass(frozen=True)
class PredictResponse:
    request_id: str
    policy: np.ndarray
    value: float
    model_version: int
    error: str | None = None


@dataclass(frozen=True)
class ControlResponse:
    request_id: str
    model_version: int
    error: str | None = None

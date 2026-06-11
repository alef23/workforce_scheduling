from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from modules.evaluators.base import EvaluatorProtocol
from modules.evaluators.resnet.compound_encoder import (
    CompoundActionStateEncoder,
)
from modules.evaluators.resnet.resnet_evaluator import WorkforceResNet
from modules.workforce_engine.compound_actions import ACTION_SPACE_SIZE
from modules.workforce_engine.compound_schemas import CompoundWorkforceState


class CompoundResNetEvaluator(EvaluatorProtocol):
    """Wrapper de inferencia para la ResNet de acciones compuestas."""

    INPUT_CHANNELS = CompoundActionStateEncoder.CHANNELS

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        self.device = self._resolve_device(device)
        self.encoder = CompoundActionStateEncoder(device=self.device)
        self.model: WorkforceResNet | None = None
        self.action_space_size = ACTION_SPACE_SIZE
        self.checkpoint_path = self._resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
        )
        self.reload_weights(self.checkpoint_path)

    def predict(
        self,
        state: CompoundWorkforceState,
    ) -> tuple[np.ndarray, float]:
        """Devuelve probabilidades para 54 acciones y el value del estado."""
        policies, values = self.predict_batch([state])
        return policies[0], float(values[0])

    def predict_batch(
        self,
        states: Sequence[CompoundWorkforceState],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evalúa varios estados en una única pasada por la red."""
        if len(states) == 0:
            raise ValueError("states debe contener al menos un estado.")

        with torch.inference_mode():
            encoded = self.encoder(self._build_encoder_input(states))
            policy_logits, value_tensor = self._model(encoded)
            policy_tensor = F.softmax(policy_logits, dim=1)

        policies = policy_tensor.cpu().numpy().astype(float)
        values = value_tensor.cpu().numpy().astype(float)
        return policies, values

    def reload_weights(self, checkpoint_path: str | Path) -> int:
        """Recarga un checkpoint compatible y devuelve su global_step."""
        path = self._resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
            checkpoint_dir=None,
        )
        checkpoint = torch.load(path, map_location=self.device)
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError(
                f"El checkpoint {path} no contiene un model_config válido."
            )

        self._validate_model_config(model_config, path)
        model = WorkforceResNet(**model_config).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model
        self.checkpoint_path = path
        self.action_space_size = int(model.action_space_size)
        training_state = checkpoint.get("training_state", {})
        return int(training_state.get("global_step", 0))

    @property
    def _model(self) -> WorkforceResNet:
        if self.model is None:
            raise RuntimeError("El modelo compuesto todavía no fue cargado.")
        return self.model

    @classmethod
    def _validate_model_config(
        cls,
        model_config: dict,
        checkpoint_path: Path,
    ) -> None:
        input_channels = int(model_config.get("input_channels", 93))
        action_space_size = int(model_config.get("action_space_size", 55))

        if input_channels != cls.INPUT_CHANNELS:
            raise ValueError(
                f"Checkpoint incompatible {checkpoint_path}: "
                f"input_channels={input_channels}, esperado {cls.INPUT_CHANNELS}."
            )
        if action_space_size != ACTION_SPACE_SIZE:
            raise ValueError(
                f"Checkpoint incompatible {checkpoint_path}: "
                f"action_space_size={action_space_size}, "
                f"esperado {ACTION_SPACE_SIZE}."
            )

    @staticmethod
    def _build_encoder_input(
        states: Sequence[CompoundWorkforceState],
    ) -> dict[str, np.ndarray]:
        return {
            "residual_demand": np.stack(
                [state.residual_demand for state in states]
            ),
            "initial_demand_total": np.asarray(
                [state.initial_demand_total for state in states],
                dtype=np.int64,
            ),
            "remaining_stock": np.stack(
                [state.remaining_stock for state in states]
            ),
            "current_modality": np.asarray(
                [
                    -1 if state.current_modality is None else state.current_modality
                    for state in states
                ],
                dtype=np.int64,
            ),
            "assignment_week": np.asarray(
                [state.assignment_week for state in states],
                dtype=np.int64,
            ),
        }

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if str(device) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @classmethod
    def _resolve_checkpoint_path(
        cls,
        checkpoint_path: str | Path | None,
        checkpoint_dir: str | Path | None,
    ) -> Path:
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(f"No existe el checkpoint: {path}")
            return path

        directory = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else Path(__file__).resolve().parent / "checkpoints_compound_actions"
        )
        checkpoint_paths = list(directory.glob("*.pt"))
        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No se encontraron checkpoints .pt en {directory}."
            )

        return max(checkpoint_paths, key=cls._checkpoint_sort_key)

    @staticmethod
    def _checkpoint_sort_key(path: Path) -> tuple[int, int]:
        match = re.search(r"(\d+)(?=\.pt$)", path.name)
        step = int(match.group(1)) if match else -1
        return step, path.stat().st_mtime_ns

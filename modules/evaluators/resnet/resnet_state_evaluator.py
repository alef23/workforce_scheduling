from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from modules.evaluators.base import EvaluatorProtocol
from modules.evaluators.resnet.encoder import StateEncoder
from modules.evaluators.resnet.resnet_evaluator import WorkforceResNet
from modules.workforce_engine.schemas import ProblemSetup, WorkforceState


class ResNetStateEvaluator(EvaluatorProtocol):
    """
    Wrapper de inferencia para usar WorkforceResNet como evaluador del engine.

    Recibe un WorkforceState, lo combina con un ProblemSetup fijo, encodea el
    estado y devuelve una policy de 55 acciones más un value escalar.
    """

    def __init__(
        self,
        setup: ProblemSetup,
        checkpoint_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        device: str | torch.device = "auto",
        demand_ref: float = 300.0,
        stock_ref: float = 100.0,
    ) -> None:
        self.setup = setup
        self.device = self._resolve_device(device)
        self.checkpoint_path = self._resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
        )

        self.model: WorkforceResNet | None = None
        self.action_space_size = 55
        self.reload_weights(self.checkpoint_path)

        self.encoder = StateEncoder(
            demand_ref=demand_ref,
            stock_ref=stock_ref,
            device=self.device,
        )

    def reload_weights(self, checkpoint_path: str | Path) -> int:
        """
        Recarga pesos desde un checkpoint y devuelve el global_step cargado.
        """
        path = self._resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
            checkpoint_dir=None,
        )
        checkpoint = torch.load(path, map_location=self.device)
        model_config = checkpoint.get("model_config", {})

        self.model = WorkforceResNet(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_path = path
        self.action_space_size = int(self.model.action_space_size)
        training_state = checkpoint.get("training_state", {})
        return int(training_state.get("global_step", 0))

    def predict(self, state: WorkforceState) -> tuple[np.ndarray, float]:
        """
        Devuelve policy y value para un WorkforceState.

        Returns
        -------
        policy:
            np.ndarray shape (55,) con probabilidades sobre todo el espacio de
            acciones. La legalidad se aplica luego en WorkforceEngine.legal_mask.
        value:
            float en [-1, 1].
        """
        X = self._build_encoder_input(state)

        with torch.no_grad():
            encoded = self.encoder(X)
            policy_logits, value_tensor = self._model(encoded)
            policy_tensor = F.softmax(policy_logits, dim=1)

        policy = policy_tensor.squeeze(0).detach().cpu().numpy().astype(float)
        value = float(value_tensor.squeeze(0).detach().cpu().item())

        return policy, value

    @property
    def _model(self) -> WorkforceResNet:
        if self.model is None:
            raise RuntimeError("El modelo todavia no fue cargado.")
        return self.model

    def _build_encoder_input(self, state: WorkforceState) -> dict[str, Any]:
        return {
            "residual_demand": state.residual_demand,
            "initial_demand_total": state.initial_demand_total,
            "remaining_stock": state.remaining_stock,
            "current_modality": state.current_modality,
            "assignment_week": state.assignment_week,
            "mobile_days_off_count": self.setup.mobile_days_off_count,
            "fixed_day_off": self.setup.fixed_day_off,
            "allowed_entry_hours": self.setup.allowed_entry_hours,
            "closing_hour": self.setup.closing_hour,
            "current_entry_hour": state.current_entry_hour,
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

        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"

        checkpoint_paths = sorted(
            Path(checkpoint_dir).glob("*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No se encontraron checkpoints .pt en {checkpoint_dir}."
            )

        return checkpoint_paths[0]

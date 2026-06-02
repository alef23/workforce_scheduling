from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from modules.evaluators.resnet.encoder import StateEncoder
from modules.evaluators.resnet.resnet_evaluator import WorkforceResNet
from modules.storage import SampleBuffer


@dataclass(frozen=True)
class ResNetLearnerConfig:
    sample_buffer_path: str | Path
    checkpoint_path: str | Path | None = None
    checkpoint_dir: str | Path = "modules/evaluators/resnet/checkpoints"
    device: str | torch.device = "auto"
    batch_size: int = 64
    train_steps: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    seed: int | None = None
    demand_ref: float = 300.0
    stock_ref: float = 100.0
    model_config: dict[str, Any] | None = None
    save_every_steps: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size debe ser positivo.")
        if self.train_steps <= 0:
            raise ValueError("train_steps debe ser positivo.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay debe ser >= 0.")
        if self.value_loss_weight < 0:
            raise ValueError("value_loss_weight debe ser >= 0.")
        if self.policy_loss_weight < 0:
            raise ValueError("policy_loss_weight debe ser >= 0.")
        if self.save_every_steps is not None and self.save_every_steps <= 0:
            raise ValueError("save_every_steps debe ser positivo o None.")


@dataclass(frozen=True)
class ResNetTrainStepMetrics:
    step: int
    global_step: int
    loss: float
    policy_loss: float
    value_loss: float
    mean_policy_weight: float


@dataclass(frozen=True)
class ResNetLearnerReport:
    checkpoint_path: str
    global_step: int
    trained_steps: int
    sample_count: int
    metrics: list[ResNetTrainStepMetrics] = field(default_factory=list)


class ResNetSampleLearner:
    """
    Entrena WorkforceResNet desde SampleBuffer.

    El buffer entrega X/Y crudos, el learner los encodea en batches aleatorios y
    guarda un checkpoint que el evaluador centralizado puede recargar.
    """

    def __init__(self, config: ResNetLearnerConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.rng = np.random.default_rng(config.seed)
        self.encoder = StateEncoder(
            demand_ref=config.demand_ref,
            stock_ref=config.stock_ref,
            device=self.device,
        )

        checkpoint = self._load_checkpoint(config.checkpoint_path)
        model_config = self._resolve_model_config(checkpoint)
        self.model = WorkforceResNet(**model_config).to(self.device)
        self.global_step = 0

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            training_state = checkpoint.get("training_state", {})
            self.global_step = int(training_state.get("global_step", 0))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        if checkpoint is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model_config = model_config

    def train(self) -> ResNetLearnerReport:
        sample_buffer = SampleBuffer(self.config.sample_buffer_path, mode="r")
        sample_count = len(sample_buffer)
        if sample_count == 0:
            raise ValueError("El SampleBuffer no contiene samples.")

        self.model.train()
        metrics: list[ResNetTrainStepMetrics] = []
        checkpoint_path: Path | None = None

        for local_step in range(1, int(self.config.train_steps) + 1):
            batch_indices = self._sample_indices(sample_count)
            batch = sample_buffer.load_batch(batch_indices)
            X = self.encoder(batch.X)
            target_policy = torch.as_tensor(
                batch.Y["policy"],
                dtype=torch.float32,
                device=self.device,
            )
            target_value = torch.as_tensor(
                batch.Y["value"],
                dtype=torch.float32,
                device=self.device,
            )
            policy_weight = torch.as_tensor(
                batch.Y["policy_weight"],
                dtype=torch.float32,
                device=self.device,
            )

            self.optimizer.zero_grad(set_to_none=True)
            policy_logits, value = self.model(X)
            policy_loss = self._weighted_soft_cross_entropy(
                logits=policy_logits,
                target_policy=target_policy,
                policy_weight=policy_weight,
            )
            value_loss = F.mse_loss(value, target_value)
            loss = (
                float(self.config.policy_loss_weight) * policy_loss
                + float(self.config.value_loss_weight) * value_loss
            )
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
            metrics.append(
                ResNetTrainStepMetrics(
                    step=local_step,
                    global_step=self.global_step,
                    loss=float(loss.detach().cpu().item()),
                    policy_loss=float(policy_loss.detach().cpu().item()),
                    value_loss=float(value_loss.detach().cpu().item()),
                    mean_policy_weight=float(policy_weight.mean().detach().cpu().item()),
                )
            )

            if self._should_save_intermediate(local_step):
                checkpoint_path = self.save_checkpoint()

        checkpoint_path = self.save_checkpoint()
        return ResNetLearnerReport(
            checkpoint_path=str(checkpoint_path),
            global_step=self.global_step,
            trained_steps=int(self.config.train_steps),
            sample_count=sample_count,
            metrics=metrics,
        )

    def save_checkpoint(self) -> Path:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"workforce_resnet_{self.global_step:06d}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": self.model_config,
                "training_state": {
                    "global_step": self.global_step,
                    "trained": True,
                },
            },
            checkpoint_path,
        )
        return checkpoint_path

    def _sample_indices(self, sample_count: int) -> np.ndarray:
        replace = int(self.config.batch_size) > sample_count
        return self.rng.choice(
            sample_count,
            size=int(self.config.batch_size),
            replace=replace,
        ).astype(np.int64)

    def _should_save_intermediate(self, local_step: int) -> bool:
        interval = self.config.save_every_steps
        if interval is None:
            return False
        return local_step < int(self.config.train_steps) and local_step % interval == 0

    def _resolve_model_config(self, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
        if self.config.model_config is not None:
            return dict(self.config.model_config)
        if checkpoint is not None:
            return dict(checkpoint.get("model_config", {}))
        return {}

    def _load_checkpoint(self, checkpoint_path: str | Path | None) -> dict[str, Any] | None:
        if checkpoint_path is None:
            return None
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el checkpoint: {path}")
        return torch.load(path, map_location=self.device)

    @staticmethod
    def _weighted_soft_cross_entropy(
        logits: torch.Tensor,
        target_policy: torch.Tensor,
        policy_weight: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        per_sample_loss = -(target_policy * log_probs).sum(dim=1)
        return (per_sample_loss * policy_weight).mean()

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if str(device) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

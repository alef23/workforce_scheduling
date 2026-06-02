from pathlib import Path

import numpy as np
import torch

from modules.evaluators.resnet.resnet_evaluator import WorkforceResNet
from modules.learning import ResNetLearnerConfig, ResNetSampleLearner
from modules.storage.sample_buffer import SampleBatch


class _FakeSampleBuffer:
    def __init__(self, *_args, **_kwargs) -> None:
        self.batch = _build_batch()

    def __len__(self) -> int:
        return 4

    def load_batch(self, indices):
        idx = np.asarray(indices, dtype=int)
        return SampleBatch(
            actions=self.batch.actions[idx],
            X={key: _take(value, idx) for key, value in self.batch.X.items()},
            Y={key: value[idx] for key, value in self.batch.Y.items()},
            metadata={
                key: value[idx]
                for key, value in self.batch.metadata.items()
            },
        )


def test_resnet_sample_learner_trains_and_saves_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    from modules.learning import resnet_sample_learner as learner_module

    monkeypatch.setattr(learner_module, "SampleBuffer", _FakeSampleBuffer)

    model_config = {
        "hidden_channels": 4,
        "num_res_blocks": 1,
        "policy_channels": 1,
        "value_channels": 1,
        "value_hidden_dim": 8,
    }
    initial_checkpoint = tmp_path / "initial.pt"
    torch.save(
        {
            "model_state_dict": WorkforceResNet(**model_config).state_dict(),
            "model_config": model_config,
            "training_state": {"global_step": 0, "trained": False},
        },
        initial_checkpoint,
    )

    learner = ResNetSampleLearner(
        ResNetLearnerConfig(
            sample_buffer_path="unused.zarr",
            checkpoint_path=initial_checkpoint,
            checkpoint_dir=tmp_path / "checkpoints",
            device="cpu",
            batch_size=2,
            train_steps=2,
            learning_rate=1e-3,
            seed=123,
        )
    )

    report = learner.train()

    assert report.global_step == 2
    assert report.trained_steps == 2
    assert report.sample_count == 4
    assert len(report.metrics) == 2
    assert Path(report.checkpoint_path).exists()

    checkpoint = torch.load(report.checkpoint_path, map_location="cpu")
    assert checkpoint["training_state"]["global_step"] == 2
    assert checkpoint["training_state"]["trained"] is True
    assert checkpoint["model_config"] == model_config


def test_policy_loss_uses_policy_weight() -> None:
    logits = torch.zeros((2, 2), dtype=torch.float32)
    target_policy = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    policy_weight = torch.tensor([1.0, 0.5], dtype=torch.float32)

    loss = ResNetSampleLearner._weighted_soft_cross_entropy(
        logits=logits,
        target_policy=target_policy,
        policy_weight=policy_weight,
    )

    assert torch.isclose(loss, torch.tensor(0.75 * np.log(2), dtype=torch.float32))


def _build_batch() -> SampleBatch:
    n = 4
    policy = np.zeros((n, 55), dtype=np.float32)
    policy[:, 0] = 1.0
    return SampleBatch(
        actions=np.zeros((n,), dtype=np.int32),
        X={
            "residual_demand": np.ones((n, 24, 28), dtype=np.int32),
            "remaining_stock": np.ones((n, 3), dtype=np.int32),
            "expansion_mode": np.zeros((n,), dtype=bool),
            "current_modality": np.full((n,), 4, dtype=np.int32),
            "current_entry_hour": np.full((n,), 6, dtype=np.int32),
            "assignment_week": np.zeros((n,), dtype=np.int32),
            "initial_demand_total": np.full((n,), 672, dtype=np.int64),
            "mobile_days_off_count": np.zeros((n,), dtype=np.int32),
            "fixed_day_off": np.full((n,), 5, dtype=np.int32),
            "allowed_entry_hours": [[6, 12] for _ in range(n)],
            "max_overcoverage_tolerance": np.full((n,), 0.1, dtype=np.float32),
            "closing_hour": np.full((n,), 22, dtype=np.int32),
        },
        Y={
            "policy": policy,
            "value": np.linspace(0.0, 0.3, n, dtype=np.float32),
            "policy_weight": np.array([1.0, 0.5, 1.0, 0.5], dtype=np.float32),
        },
        metadata={
            "trajectory_id": np.array(["t0", "t1", "t2", "t3"]),
            "step_index": np.arange(n, dtype=np.int32),
            "sample_source": np.array(["mcts", "stock", "mcts", "stock"]),
            "source_trajectory_id": np.array(["s0", "s1", "s2", "s3"]),
            "model_version": np.zeros((n,), dtype=np.int32),
            "sample_index": np.arange(n, dtype=np.int64),
        },
    )


def _take(value, idx):
    if isinstance(value, list):
        return [value[int(i)] for i in idx]
    return value[idx]

import numpy as np
import torch

from modules.evaluators.resnet.compound_encoder import (
    CompoundActionStateEncoder,
)
from modules.evaluators.resnet.resnet_evaluator import WorkforceResNet
from scripts.benchmark_compound_resnet import (
    MODEL_CONFIG,
    measure_replay_memory,
)


def test_compound_encoder_shape_and_normalization() -> None:
    encoder = CompoundActionStateEncoder(device="cpu")
    encoded = encoder(
        {
            "residual_demand": np.full((2, 24, 28), 20, dtype=np.int32),
            "initial_demand_total": np.full(
                (2,),
                20 * 24 * 28,
                dtype=np.int64,
            ),
            "remaining_stock": np.array(
                [[20, 10, 0], [0, 5, 20]],
                dtype=np.int32,
            ),
            "current_modality": np.array([-1, 6], dtype=np.int32),
            "assignment_week": np.array([0, 3], dtype=np.int32),
        }
    )

    assert encoded.shape == (2, 11, 28, 28)
    assert torch.all(encoded[:, 0, 2:26, :] == 1.0)
    assert torch.all(encoded[:, 1] == 1.0)
    assert torch.all(encoded[0, 2] == 1.0)
    assert torch.all(encoded[0, 3] == 0.5)
    assert torch.all(encoded[0, 4] == 0.0)
    assert torch.all(encoded[0, 5:8] == 0.0)
    assert torch.all(encoded[1, 6] == 1.0)
    assert torch.all(encoded[0, 8:11] == 0.0)
    assert torch.all(encoded[1, 10] == 1.0)


def test_compound_resnet_outputs_54_actions() -> None:
    model = WorkforceResNet(**MODEL_CONFIG)
    policy_logits, values = model(torch.zeros((2, 11, 28, 28)))

    assert policy_logits.shape == (2, 54)
    assert values.shape == (2,)


def test_replay_memory_uses_raw_inputs() -> None:
    report = measure_replay_memory(1000)

    assert report["sample_count"] == 1000
    assert report["raw_total_bytes"] < report["encoded_total_bytes"]
    assert report["raw_bytes_per_sample"] > 0
    assert report["deque_total_bytes"] > report["raw_total_bytes"]

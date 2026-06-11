from __future__ import annotations

import argparse
from collections import deque
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.evaluators.resnet.compound_encoder import CompoundActionStateEncoder
from modules.evaluators.resnet.resnet_evaluator import WorkforceResNet


DEFAULT_CHECKPOINT_DIR = (
    "modules/evaluators/resnet/checkpoints_compound_actions"
)
DEFAULT_BATCH_SIZES = (256, 512, 1024, 2048, 3072, 4056)
MODEL_CONFIG = {
    "input_channels": CompoundActionStateEncoder.CHANNELS,
    "board_height": 28,
    "board_width": 28,
    "hidden_channels": 128,
    "num_res_blocks": 8,
    "action_space_size": 54,
    "policy_channels": 8,
    "value_channels": 4,
    "value_hidden_dim": 256,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inicializa y mide la ResNet experimental de 54 acciones."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--ram-samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
    )
    return parser.parse_args()


def initialize_checkpoint(
    checkpoint_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "workforce_resnet_compound_000000.pt"
    if path.exists() and not overwrite:
        return path

    model = WorkforceResNet(**MODEL_CONFIG)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": dict(MODEL_CONFIG),
            "encoder_config": {
                "name": "CompoundActionStateEncoder",
                "channels": CompoundActionStateEncoder.CHANNELS,
                "demand_ref": CompoundActionStateEncoder.DEMAND_REF,
                "initial_demand_total_ref": (
                    CompoundActionStateEncoder.INITIAL_DEMAND_TOTAL_REF
                ),
                "stock_ref": CompoundActionStateEncoder.STOCK_REF,
            },
            "action_space": {
                "size": 54,
                "modalities": [4, 6, 8],
                "entry_hours": [6, 12, 18],
                "fixed_day_off": 6,
                "mobile_days_off": [0, 1, 2, 3, 4, 5],
            },
            "training_state": {
                "global_step": 0,
                "trained": False,
            },
        },
        path,
    )
    return path


def build_dummy_inputs(
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    residual_demand = rng.integers(
        low=-4,
        high=21,
        size=(batch_size, 24, 28),
        dtype=np.int32,
    )
    initial_demand_total = np.maximum(
        residual_demand.clip(min=0).sum(axis=(1, 2)),
        1,
    ).astype(np.int64)
    return {
        "residual_demand": residual_demand,
        "initial_demand_total": initial_demand_total,
        "remaining_stock": rng.integers(
            low=0,
            high=21,
            size=(batch_size, 3),
            dtype=np.int32,
        ),
        "current_modality": rng.choice(
            np.array([-1, 4, 6, 8], dtype=np.int32),
            size=batch_size,
        ),
        "assignment_week": rng.integers(
            low=0,
            high=4,
            size=batch_size,
            dtype=np.int32,
        ),
    }


def benchmark_gpu(
    checkpoint_path: Path,
    device: torch.device,
    batch_sizes: list[int],
    warmups: int,
    repeats: int,
) -> list[dict[str, float | int | str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = WorkforceResNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    encoder = CompoundActionStateEncoder(device=device)
    rng = np.random.default_rng(20260611)
    results: list[dict[str, float | int | str]] = []

    for batch_size in batch_sizes:
        X = build_dummy_inputs(batch_size, rng)
        try:
            with torch.inference_mode():
                for _ in range(int(warmups)):
                    encoded = encoder(X)
                    model(encoded)
                _synchronize(device)

                elapsed_ms = []
                torch.cuda.reset_peak_memory_stats(device)
                for _ in range(int(repeats)):
                    _synchronize(device)
                    started_at = time.perf_counter()
                    encoded = encoder(X)
                    policy_logits, values = model(encoded)
                    _synchronize(device)
                    elapsed_ms.append(
                        (time.perf_counter() - started_at) * 1000
                    )

                median_ms = float(np.median(elapsed_ms))
                results.append(
                    {
                        "batch_size": int(batch_size),
                        "status": "ok",
                        "median_ms": median_ms,
                        "p95_ms": float(np.percentile(elapsed_ms, 95)),
                        "states_per_second": float(
                            batch_size / (median_ms / 1000)
                        ),
                        "peak_allocated_mb": float(
                            torch.cuda.max_memory_allocated(device) / 1024**2
                        ),
                        "peak_reserved_mb": float(
                            torch.cuda.max_memory_reserved(device) / 1024**2
                        ),
                    }
                )
                del encoded, policy_logits, values
        except torch.OutOfMemoryError:
            results.append(
                {
                    "batch_size": int(batch_size),
                    "status": "oom",
                }
            )
        finally:
            del X
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def measure_replay_memory(sample_count: int) -> dict[str, int | float]:
    rng = np.random.default_rng(20260611)
    X = build_dummy_inputs(sample_count, rng)
    Y = {
        "policy": np.zeros((sample_count, 54), dtype=np.float32),
        "value": np.zeros((sample_count,), dtype=np.float32),
        "policy_weight": np.ones((sample_count,), dtype=np.float32),
    }
    actions = np.zeros((sample_count,), dtype=np.int32)

    raw_x_bytes = sum(value.nbytes for value in X.values())
    raw_y_bytes = sum(value.nbytes for value in Y.values())
    action_bytes = actions.nbytes

    encoded_elements = (
        sample_count
        * CompoundActionStateEncoder.CHANNELS
        * CompoundActionStateEncoder.HEIGHT
        * CompoundActionStateEncoder.WIDTH
    )
    encoded_x_bytes = encoded_elements * np.dtype(np.float32).itemsize
    sample_deque = deque(
        (
            {
                key: value[index].copy()
                if isinstance(value[index], np.ndarray)
                else value[index].item()
                for key, value in X.items()
            },
            {
                key: value[index].copy()
                if isinstance(value[index], np.ndarray)
                else value[index].item()
                for key, value in Y.items()
            },
            int(actions[index]),
        )
        for index in range(sample_count)
    )
    deque_total_bytes = _deep_size(sample_deque)

    return {
        "sample_count": int(sample_count),
        "raw_x_bytes": int(raw_x_bytes),
        "raw_y_bytes": int(raw_y_bytes),
        "action_bytes": int(action_bytes),
        "raw_total_bytes": int(raw_x_bytes + raw_y_bytes + action_bytes),
        "raw_bytes_per_sample": float(
            (raw_x_bytes + raw_y_bytes + action_bytes) / sample_count
        ),
        "encoded_x_bytes": int(encoded_x_bytes),
        "encoded_total_bytes": int(
            encoded_x_bytes + raw_y_bytes + action_bytes
        ),
        "deque_total_bytes": int(deque_total_bytes),
        "deque_bytes_per_sample": float(
            deque_total_bytes / sample_count
        ),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _deep_size(value: object, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)

    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            _deep_size(key, seen) + _deep_size(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return size + sum(_deep_size(item, seen) for item in value)
    return size


def main() -> None:
    args = parse_args()
    checkpoint_path = initialize_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        overwrite=args.overwrite_checkpoint,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible para el benchmark.")

    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"model_config={json.dumps(MODEL_CONFIG, sort_keys=True)}")

    if not args.skip_gpu:
        gpu_results = benchmark_gpu(
            checkpoint_path=checkpoint_path,
            device=device,
            batch_sizes=[int(value) for value in args.batch_sizes],
            warmups=args.warmups,
            repeats=args.repeats,
        )
        print("gpu_results=")
        print(json.dumps(gpu_results, indent=2, sort_keys=True))

    ram_results = measure_replay_memory(args.ram_samples)
    print("ram_results=")
    print(json.dumps(ram_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

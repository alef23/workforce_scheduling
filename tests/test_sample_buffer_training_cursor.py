import numpy as np

from modules.storage import SampleBuffer


def test_sample_buffer_training_cursor_is_monotonic(tmp_path) -> None:
    buffer = SampleBuffer(tmp_path / "samples.zarr", mode="w")

    assert len(buffer) == 0
    assert buffer.trained_until == 0

    buffer.append_samples([_sample("t0"), _sample("t1")])
    assert buffer.trained_until == 0

    buffer.mark_trained_until(2)
    assert buffer.trained_until == 2


def test_legacy_sample_buffer_defaults_to_already_trained(tmp_path) -> None:
    path = tmp_path / "legacy.zarr"
    buffer = SampleBuffer(path, mode="w")
    buffer.append_samples([_sample("t0")])
    del buffer.samples_group.attrs["trained_until"]

    reopened = SampleBuffer(path, mode="a")

    assert reopened.trained_until == 1


def _sample(trajectory_id: str) -> dict:
    policy = np.zeros(55, dtype=np.float32)
    policy[0] = 1.0
    return {
        "trajectory_id": trajectory_id,
        "step_index": 0,
        "state": {
            "residual_demand": np.ones((24, 28), dtype=np.int32),
            "remaining_stock": np.ones(3, dtype=np.int32),
            "expansion_mode": False,
            "current_modality": 4,
            "current_entry_hour": None,
            "assignment_week": 0,
            "initial_demand_total": 672,
        },
        "problem_setup": {
            "mobile_days_off_count": 0,
            "fixed_day_off": 5,
            "allowed_entry_hours": [6, 12],
            "max_overcoverage_tolerance": 0.1,
            "closing_hour": 22,
        },
        "policy": policy,
        "action_id": 0,
        "value": 0.0,
        "policy_weight": 1.0,
        "metadata": {},
    }

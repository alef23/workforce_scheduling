from types import SimpleNamespace

import numpy as np

from scripts.generate_initial_state_test_set import build_initial_state_sample


def test_build_initial_state_sample_uses_initial_state_and_final_value() -> None:
    state = {
        "residual_demand": np.zeros((24, 28), dtype=np.int32),
        "remaining_stock": np.array([1, 2, 3], dtype=np.int32),
        "expansion_mode": False,
        "current_modality": None,
        "current_entry_hour": None,
        "assignment_week": 0,
        "initial_demand_total": 10,
    }
    setup = {
        "mobile_days_off_count": 1,
        "fixed_day_off": 3,
        "allowed_entry_hours": [6, 12, 18],
        "max_overcoverage_tolerance": 0.1,
        "closing_hour": 22,
    }
    generated = SimpleNamespace(
        trajectory_id="test_raw_000000",
        problem_setup=setup,
        metadata={"final_value": -0.25, "seed": 123},
        trajectory=[
            {
                "state": state,
                "policy": np.ones(55, dtype=np.float32) / 55,
                "action_id": 0,
                "reward": 0.0,
            },
            {
                "state": state,
                "policy": np.ones(55, dtype=np.float32) / 55,
                "action_id": 9,
                "reward": -0.25,
            },
        ],
    )

    sample = build_initial_state_sample(generated)

    assert sample["trajectory_id"] == "test_raw_000000"
    assert sample["step_index"] == 0
    assert sample["state"] is state
    assert sample["problem_setup"] is setup
    assert sample["action_id"] == 0
    assert sample["value"] == -0.25
    assert sample["policy_weight"] == 1.0
    assert sample["metadata"]["sample_source"] == "test_initial_raw"
    assert sample["metadata"]["source_trajectory_id"] == "test_raw_000000"

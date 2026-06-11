from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.workforce_engine.compound_engine import CompoundWorkforceEngine
from modules.workforce_engine.compound_schemas import CompoundWorkforceState
from modules.workforce_engine.schemas import ProblemSetup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mide transiciones individuales del engine compuesto."
    )
    parser.add_argument("--repeats", type=int, default=500)
    return parser.parse_args()


def build_engine() -> CompoundWorkforceEngine:
    return CompoundWorkforceEngine(
        ProblemSetup(
            mobile_days_off_count=1,
            fixed_day_off=6,
            allowed_entry_hours=[6, 12, 18],
            max_overcoverage_tolerance=0.1,
            closing_hour=22,
        )
    )


def build_state(
    current_modality: int | None,
    assignment_week: int,
) -> CompoundWorkforceState:
    residual = np.full((24, 28), 10, dtype=int)
    return CompoundWorkforceState(
        residual_demand=residual,
        remaining_stock=np.array([7, 7, 6], dtype=int),
        expansion_mode=False,
        current_modality=current_modality,
        assignment_week=assignment_week,
        initial_demand_total=int(residual.sum()),
    )


def benchmark_state(
    engine: CompoundWorkforceEngine,
    state: CompoundWorkforceState,
    repeats: int,
) -> dict[str, float | int | str]:
    legal_action_ids = np.flatnonzero(engine.get_legal_actions(state))
    selected_action_id = int(legal_action_ids[0])

    single_times = []
    expansion_times = []
    for _ in range(repeats):
        started_at = time.perf_counter()
        engine.step(state, selected_action_id)
        single_times.append((time.perf_counter() - started_at) * 1000)

        started_at = time.perf_counter()
        for action_id in legal_action_ids:
            engine.step(state, int(action_id))
        expansion_times.append((time.perf_counter() - started_at) * 1000)

    return {
        "current_modality": (
            "none" if state.current_modality is None else state.current_modality
        ),
        "assignment_week": state.assignment_week,
        "legal_actions": int(len(legal_action_ids)),
        "single_step_median_ms": float(np.median(single_times)),
        "single_step_p95_ms": float(np.percentile(single_times, 95)),
        "all_successors_median_ms": float(np.median(expansion_times)),
        "all_successors_p95_ms": float(np.percentile(expansion_times, 95)),
    }


def main() -> None:
    args = parse_args()
    engine = build_engine()
    cases = [
        build_state(current_modality=None, assignment_week=0),
        build_state(current_modality=4, assignment_week=1),
        build_state(current_modality=6, assignment_week=1),
        build_state(current_modality=8, assignment_week=1),
    ]
    results = [
        benchmark_state(engine, state, repeats=args.repeats)
        for state in cases
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

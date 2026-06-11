from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.evaluators.resnet.compound_evaluator import CompoundResNetEvaluator
from modules.mcts.mcts import MCTS
from modules.mcts.mcts_schemas import MCTSConfig, MCTSMode
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine
from modules.workforce_engine.compound_schemas import CompoundWorkforceState
from modules.workforce_engine.schemas import ProblemSetup


class TimedEvaluator:
    def __init__(self, evaluator: CompoundResNetEvaluator) -> None:
        self.evaluator = evaluator
        self.action_space_size = evaluator.action_space_size
        self.predict_calls = 0
        self.predict_seconds = 0.0

    def predict(
        self,
        state: CompoundWorkforceState,
    ) -> tuple[np.ndarray, float]:
        started_at = time.perf_counter()
        output = self.evaluator.predict(state)
        self.predict_seconds += time.perf_counter() - started_at
        self.predict_calls += 1
        return output


class TimedEngine:
    def __init__(self, engine: CompoundWorkforceEngine) -> None:
        self.engine = engine
        self.action_space_size = engine.action_space_size
        self.step_calls = 0
        self.step_seconds = 0.0

    def step(self, state: CompoundWorkforceState, action_id: int):
        started_at = time.perf_counter()
        output = self.engine.step(state, action_id)
        self.step_seconds += time.perf_counter() - started_at
        self.step_calls += 1
        return output

    def legal_mask(
        self,
        state: CompoundWorkforceState,
        policy: np.ndarray,
    ) -> np.ndarray:
        return self.engine.legal_mask(state, policy)

    def check_terminality(self, state: CompoundWorkforceState) -> bool:
        return self.engine.check_terminality(state)

    def compute_reward(self, state: CompoundWorkforceState) -> float:
        return self.engine.compute_reward(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mide trayectorias MCTS con el dominio de acciones compuestas."
    )
    parser.add_argument("--simulations", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-path")
    return parser.parse_args()


def build_setup() -> ProblemSetup:
    return ProblemSetup(
        mobile_days_off_count=1,
        fixed_day_off=6,
        allowed_entry_hours=[6, 12, 18],
        max_overcoverage_tolerance=0.1,
        closing_hour=22,
    )


def build_initial_state() -> CompoundWorkforceState:
    residual = np.ones((24, 28), dtype=int)
    return CompoundWorkforceState(
        residual_demand=residual,
        remaining_stock=np.array([7, 7, 6], dtype=int),
        expansion_mode=False,
        current_modality=None,
        assignment_week=0,
        initial_demand_total=int(residual.sum()),
    )


def tree_depth_from_root(mcts: MCTS) -> int:
    if mcts.root_node_id is None:
        return 0

    root_id = mcts.root_node_id
    maximum = 0
    for node in mcts.nodes.values():
        depth = 0
        current = node
        visited: set[int] = set()
        while current.node_id != root_id and current.parent_id is not None:
            if current.node_id in visited:
                break
            visited.add(current.node_id)
            depth += 1
            current = mcts.nodes[current.parent_id]
        if current.node_id == root_id:
            maximum = max(maximum, depth)
    return maximum


def run_trajectory(
    evaluator: CompoundResNetEvaluator,
    simulations: int,
    max_actions: int,
    c_puct: float,
) -> dict[str, Any]:
    timed_evaluator = TimedEvaluator(evaluator)
    timed_engine = TimedEngine(CompoundWorkforceEngine(build_setup()))
    mcts = MCTS(
        engine=timed_engine,
        evaluator=timed_evaluator,
        config=MCTSConfig(
            num_simulations=simulations,
            c_puct=c_puct,
            mode=MCTSMode.INFERENCE,
        ),
    )
    state = build_initial_state()
    policies: list[np.ndarray] = []
    maximum_depth = 0
    completed_resources = 0
    terminal = False
    reward = 0.0

    started_at = time.perf_counter()
    for _ in range(max_actions):
        result = mcts.search(state)
        if result.policy.shape != (54,):
            raise RuntimeError(
                f"Policy inválida: se esperaba (54,), se obtuvo {result.policy.shape}."
            )
        policies.append(result.policy.copy())
        maximum_depth = max(maximum_depth, tree_depth_from_root(mcts))

        step_result = timed_engine.step(state, int(result.selected_action_id))
        mcts.advance_root(int(result.selected_action_id))
        state = step_result.next_state

        if state.current_modality is None and state.assignment_week == 0:
            completed_resources += 1
        if step_result.is_terminal:
            terminal = True
            reward = float(step_result.reward)
            break

    elapsed_seconds = time.perf_counter() - started_at
    action_count = len(policies)
    return {
        "simulations": simulations,
        "terminal": terminal,
        "reward": reward if terminal else None,
        "actions": action_count,
        "states": action_count + 1,
        "completed_resources": completed_resources,
        "maximum_search_depth": maximum_depth,
        "tree_nodes": len(mcts.nodes),
        "evaluator_calls": timed_evaluator.predict_calls,
        "evaluator_seconds": timed_evaluator.predict_seconds,
        "engine_step_calls": timed_engine.step_calls,
        "engine_step_seconds": timed_engine.step_seconds,
        "elapsed_seconds": elapsed_seconds,
        "mean_search_seconds": (
            elapsed_seconds / action_count if action_count else 0.0
        ),
        "policy_shape": [54],
        "final_positive_demand": int(np.maximum(state.residual_demand, 0).sum()),
        "final_negative_demand": int(np.minimum(state.residual_demand, 0).sum()),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = (
        "actions",
        "states",
        "completed_resources",
        "maximum_search_depth",
        "tree_nodes",
        "evaluator_calls",
        "evaluator_seconds",
        "engine_step_calls",
        "engine_step_seconds",
        "elapsed_seconds",
        "mean_search_seconds",
    )
    summary: dict[str, Any] = {
        "simulations": results[0]["simulations"],
        "repeats": len(results),
        "terminal_runs": sum(bool(result["terminal"]) for result in results),
    }
    for field in numeric_fields:
        values = [float(result[field]) for result in results]
        summary[f"{field}_median"] = float(np.median(values))
    rewards = [
        float(result["reward"])
        for result in results
        if result["reward"] is not None
    ]
    summary["reward_median"] = (
        float(np.median(rewards)) if rewards else None
    )
    return summary


def main() -> None:
    args = parse_args()
    evaluator = CompoundResNetEvaluator(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    evaluator.predict(build_initial_state())

    runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for simulations in args.simulations:
        current_runs = [
            run_trajectory(
                evaluator=evaluator,
                simulations=int(simulations),
                max_actions=int(args.max_actions),
                c_puct=float(args.c_puct),
            )
            for _ in range(int(args.repeats))
        ]
        runs.extend(current_runs)
        summaries.append(summarize(current_runs))

    print(
        json.dumps(
            {
                "checkpoint": str(evaluator.checkpoint_path),
                "device": str(evaluator.device),
                "summaries": summaries,
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

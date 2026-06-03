from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye un dashboard HTML estatico desde logs y buffers."
    )
    parser.add_argument(
        "--reports-dir",
        default="datasets/reports",
        help="Directorio con JSONL de mcts_generation. Default: datasets/reports.",
    )
    parser.add_argument(
        "--raw-path",
        default="datasets/raw/trajectories.zarr",
        help="TrajectoryBuffer raw. Default: datasets/raw/trajectories.zarr.",
    )
    parser.add_argument(
        "--stock-path",
        default="datasets/derived/stock_adjusted/trajectories.zarr",
        help="TrajectoryBuffer stock_adjusted.",
    )
    parser.add_argument(
        "--sample-path",
        default="datasets/samples/samples.zarr",
        help="SampleBuffer actual. Default: datasets/samples/samples.zarr.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="modules/evaluators/resnet/checkpoints",
        help="Directorio de checkpoints ResNet.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="HTML destino. Default: <reports-dir>/training_dashboard.html.",
    )
    parser.add_argument(
        "--explorer-output",
        default=None,
        help="HTML destino del explorador. Default: <reports-dir>/trajectory_explorer.html.",
    )
    parser.add_argument(
        "--max-sample-scan",
        type=int,
        default=200_000,
        help="Maximo de samples a leer para resumen. Default: 200000.",
    )
    parser.add_argument(
        "--max-trajectory-preview",
        type=int,
        default=20,
        help="Cantidad de trayectorias a listar como preview. Default: 20.",
    )
    parser.add_argument(
        "--explorer-trajectory-count",
        type=int,
        default=10,
        help="Cantidad de ultimas trayectorias por buffer para el explorador. Default: 10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output) if args.output else reports_dir / "training_dashboard.html"
    explorer_output_path = (
        Path(args.explorer_output)
        if args.explorer_output
        else reports_dir / "trajectory_explorer.html"
    )

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "reports_dir": str(reports_dir),
            "raw_path": str(args.raw_path),
            "stock_path": str(args.stock_path),
            "sample_path": str(args.sample_path),
            "checkpoint_dir": str(args.checkpoint_dir),
        },
        "logs": read_training_logs(reports_dir),
        "buffers": {
            "raw": summarize_trajectory_buffer(
                args.raw_path,
                max_preview=int(args.max_trajectory_preview),
            ),
            "stock": summarize_trajectory_buffer(
                args.stock_path,
                max_preview=int(args.max_trajectory_preview),
            ),
            "samples": summarize_sample_buffer(
                args.sample_path,
                max_scan=int(args.max_sample_scan),
            ),
        },
        "analysis": build_dataset_analysis(
            raw_path=args.raw_path,
            stock_path=args.stock_path,
            sample_path=args.sample_path,
            max_sample_scan=int(args.max_sample_scan),
        ),
        "checkpoints": summarize_checkpoints(args.checkpoint_dir),
        "explorer": {
            "raw": load_trajectory_explorer(
                args.raw_path,
                limit=int(args.explorer_trajectory_count),
            ),
            "stock": load_trajectory_explorer(
                args.stock_path,
                limit=int(args.explorer_trajectory_count),
            ),
            "samples": load_sample_explorer(
                args.sample_path,
                limit=int(args.explorer_trajectory_count),
                max_scan=int(args.max_sample_scan),
            ),
        },
    }
    data["derived"] = build_derived_summary(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    explorer_output_path.parent.mkdir(parents=True, exist_ok=True)

    overview_data = {key: value for key, value in data.items() if key != "explorer"}
    overview_data["paths"]["explorer_output"] = str(explorer_output_path)

    output_path.write_text(render_overview_dashboard(overview_data), encoding="utf-8")
    explorer_output_path.write_text(render_dashboard(data), encoding="utf-8")
    print(f"[dashboard] output={output_path}", flush=True)
    print(f"[dashboard] explorer_output={explorer_output_path}", flush=True)


def read_training_logs(reports_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "runs": read_jsonl(reports_dir / "mcts_generation_runs.jsonl"),
        "cycles": read_jsonl(reports_dir / "mcts_generation_cycles.jsonl"),
        "learner_steps": read_jsonl(reports_dir / "mcts_generation_learner_steps.jsonl"),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "event": "decode_error",
                        "path": str(path),
                        "line_number": line_number,
                        "error": str(exc),
                    }
                )
    return rows


def summarize_trajectory_buffer(
    store_path: str | Path,
    max_preview: int,
) -> dict[str, Any]:
    path = Path(store_path)
    if not path.exists():
        return {"exists": False, "path": str(path)}

    try:
        root = zarr.open_group(store=str(path), mode="r")
        group = root["trajectories"]
        ids = sorted(str(key) for key in group.group_keys())
        lengths = []
        rewards = []
        expansion_count = 0
        stock_reduced_count = 0
        preview = []

        for index, trajectory_id in enumerate(ids):
            trajectory_group = group[trajectory_id]
            attrs = dict(trajectory_group.attrs)
            length = _safe_int(attrs.get("length"))
            reward = _safe_float(attrs.get("final_reward"))
            has_expansion = _attr_bool(attrs.get("metadata.has_expansion_mode"))
            stock_reduced = _attr_bool(attrs.get("metadata.stock_was_reduced"))

            if length is not None:
                lengths.append(length)
            if reward is not None:
                rewards.append(reward)
            if has_expansion:
                expansion_count += 1
            if stock_reduced:
                stock_reduced_count += 1

            if index < max_preview:
                preview.append(
                    {
                        "trajectory_id": trajectory_id,
                        "length": length,
                        "final_reward": reward,
                        "stage": attrs.get("metadata.stage"),
                        "source_trajectory_id": attrs.get("metadata.source_trajectory_id"),
                        "stock_was_reduced": stock_reduced,
                        "output_stock": attrs.get("metadata.output_stock"),
                        "has_expansion_mode": has_expansion,
                        "first_expansion_step": attrs.get("metadata.first_expansion_step"),
                    }
                )

        return {
            "exists": True,
            "path": str(path),
            "count": len(ids),
            "length": summary_stats(lengths),
            "final_reward": summary_stats(rewards),
            "has_expansion_count": expansion_count,
            "stock_reduced_count": stock_reduced_count,
            "preview": preview,
        }
    except Exception as exc:
        return {"exists": True, "path": str(path), "error": str(exc)}


def summarize_sample_buffer(
    store_path: str | Path,
    max_scan: int,
) -> dict[str, Any]:
    path = Path(store_path)
    if not path.exists():
        return {"exists": False, "path": str(path)}

    try:
        root = zarr.open_group(store=str(path), mode="r")
        group = root["samples"]
        length = int(group.attrs.get("length", 0))
        if length <= 0:
            return {
                "exists": True,
                "path": str(path),
                "length": 0,
            }

        scan_count = min(length, int(max_scan))
        indices = np.arange(scan_count, dtype=int)
        scan_limited = scan_count < length

        sample_source_counts = read_counter(group, "sample_source", indices, str)
        trajectory_count = len(set(read_array(group, "trajectory_id", indices, str)))
        source_trajectory_count = len(
            set(read_array(group, "source_trajectory_id", indices, str))
        )
        policy_weights = read_array(group, "policy_weight", indices, float)
        values = read_array(group, "value", indices, float)
        expansion_mode = read_array(group, "expansion_mode", indices, bool)

        preview = []
        preview_indices = np.arange(min(25, scan_count), dtype=int)
        if len(preview_indices) > 0:
            trajectory_ids = read_array(group, "trajectory_id", preview_indices, str)
            step_indices = read_array(group, "step_index", preview_indices, int)
            sources = read_array(group, "sample_source", preview_indices, str)
            actions = read_array(group, "action_id", preview_indices, int)
            pweights = read_array(group, "policy_weight", preview_indices, float)
            vals = read_array(group, "value", preview_indices, float)
            for i in range(len(preview_indices)):
                preview.append(
                    {
                        "sample_index": int(preview_indices[i]),
                        "trajectory_id": trajectory_ids[i],
                        "step_index": int(step_indices[i]),
                        "sample_source": sources[i],
                        "action_id": int(actions[i]),
                        "policy_weight": float(pweights[i]),
                        "value": float(vals[i]),
                    }
                )

        return {
            "exists": True,
            "path": str(path),
            "length": length,
            "scan_count": scan_count,
            "scan_limited": scan_limited,
            "sample_source_counts": dict(sample_source_counts),
            "trajectory_count_scanned": trajectory_count,
            "source_trajectory_count_scanned": source_trajectory_count,
            "policy_weight": summary_stats(policy_weights),
            "value": summary_stats(values),
            "expansion_mode_count_scanned": int(np.asarray(expansion_mode, dtype=bool).sum()),
            "preview": preview,
        }
    except Exception as exc:
        return {"exists": True, "path": str(path), "error": str(exc)}


def load_trajectory_explorer(
    store_path: str | Path,
    limit: int,
) -> dict[str, Any]:
    path = Path(store_path)
    if not path.exists():
        return {"exists": False, "path": str(path), "trajectories": []}

    try:
        root = zarr.open_group(store=str(path), mode="r")
        group = root["trajectories"]
        ids = sorted(str(key) for key in group.group_keys())[-int(limit):]
        trajectories = [
            read_trajectory_group(group[trajectory_id], trajectory_id)
            for trajectory_id in ids
        ]
        return {
            "exists": True,
            "path": str(path),
            "trajectory_ids": ids,
            "trajectories": trajectories,
        }
    except Exception as exc:
        return {"exists": True, "path": str(path), "error": str(exc), "trajectories": []}


def read_trajectory_group(group, trajectory_id: str) -> dict[str, Any]:
    length = int(group.attrs.get("length", group["action_id"].shape[0]))
    policy = group["policy"][:].astype(np.float32)
    actions = group["action_id"][:].astype(np.int32)
    rewards = group["reward"][:].astype(np.float32)

    steps = []
    for step_index in range(length):
        policy_vector = policy[step_index]
        action_id = int(actions[step_index])
        steps.append(
            {
                "step_index": int(step_index),
                "action_id": action_id,
                "action": decode_action(action_id),
                "reward": float(rewards[step_index]),
                "value": float(rewards[step_index]),
                "policy": policy_vector.astype(float).tolist(),
                "policy_top": top_policy_entries(policy_vector, action_id),
                "policy_sum": float(policy_vector.sum()),
                "legal_count": int(np.count_nonzero(policy_vector > 0)),
                "selected_policy_prob": float(policy_vector[action_id]),
                "state": read_state_from_group(group, step_index),
            }
        )

    return {
        "trajectory_id": str(trajectory_id),
        "length": length,
        "final_reward": _safe_float(group.attrs.get("final_reward")),
        "attrs": attrs_to_json(dict(group.attrs)),
        "problem_setup": read_problem_setup_attrs(group),
        "steps": steps,
    }


def load_sample_explorer(
    store_path: str | Path,
    limit: int,
    max_scan: int,
) -> dict[str, Any]:
    path = Path(store_path)
    if not path.exists():
        return {"exists": False, "path": str(path), "trajectories": []}

    try:
        root = zarr.open_group(store=str(path), mode="r")
        group = root["samples"]
        length = int(group.attrs.get("length", 0))
        if length <= 0:
            return {"exists": True, "path": str(path), "length": 0, "trajectories": []}

        scan_count = min(length, int(max_scan))
        start = length - scan_count
        indices = np.arange(start, length, dtype=int)
        trajectory_ids = read_array(group, "trajectory_id", indices, str)
        ordered_ids = []
        seen = set()
        for trajectory_id in reversed(trajectory_ids.tolist()):
            trajectory_id = str(trajectory_id)
            if trajectory_id in seen:
                continue
            seen.add(trajectory_id)
            ordered_ids.append(trajectory_id)
            if len(ordered_ids) >= int(limit):
                break
        selected_ids = list(reversed(ordered_ids))

        trajectories = []
        for trajectory_id in selected_ids:
            local_positions = np.flatnonzero(trajectory_ids == trajectory_id)
            sample_indices = indices[local_positions]
            trajectories.append(read_sample_trajectory(group, trajectory_id, sample_indices))

        return {
            "exists": True,
            "path": str(path),
            "length": length,
            "scan_count": scan_count,
            "trajectory_ids": selected_ids,
            "trajectories": trajectories,
        }
    except Exception as exc:
        return {"exists": True, "path": str(path), "error": str(exc), "trajectories": []}


def read_sample_trajectory(group, trajectory_id: str, indices: np.ndarray) -> dict[str, Any]:
    order = np.argsort(group["step_index"][indices].astype(np.int32))
    sample_indices = indices[order]

    actions = group["action_id"][sample_indices].astype(np.int32)
    policies = group["policy"][sample_indices].astype(np.float32)
    values = group["value"][sample_indices].astype(np.float32)
    policy_weights = group["policy_weight"][sample_indices].astype(np.float32)
    step_indices = group["step_index"][sample_indices].astype(np.int32)
    sample_sources = read_array(group, "sample_source", sample_indices, str)
    source_trajectory_ids = read_array(group, "source_trajectory_id", sample_indices, str)
    model_versions = read_array(group, "model_version", sample_indices, int)

    steps = []
    for local_index, sample_index in enumerate(sample_indices):
        policy_vector = policies[local_index]
        action_id = int(actions[local_index])
        steps.append(
            {
                "sample_index": int(sample_index),
                "step_index": int(step_indices[local_index]),
                "action_id": action_id,
                "action": decode_action(action_id),
                "value": float(values[local_index]),
                "reward": float(values[local_index]),
                "policy_weight": float(policy_weights[local_index]),
                "sample_source": str(sample_sources[local_index]) if len(sample_sources) else "",
                "source_trajectory_id": (
                    str(source_trajectory_ids[local_index])
                    if len(source_trajectory_ids)
                    else ""
                ),
                "model_version": (
                    int(model_versions[local_index])
                    if len(model_versions)
                    else None
                ),
                "policy": policy_vector.astype(float).tolist(),
                "policy_top": top_policy_entries(policy_vector, action_id),
                "policy_sum": float(policy_vector.sum()),
                "legal_count": int(np.count_nonzero(policy_vector > 0)),
                "selected_policy_prob": float(policy_vector[action_id]),
                "state": read_state_from_group(group, int(sample_index)),
            }
        )

    sources = sorted({step.get("sample_source", "") for step in steps if step.get("sample_source")})
    source_ids = sorted(
        {
            step.get("source_trajectory_id", "")
            for step in steps
            if step.get("source_trajectory_id")
        }
    )
    return {
        "trajectory_id": str(trajectory_id),
        "length": len(steps),
        "final_reward": steps[-1]["value"] if steps else None,
        "sample_sources": sources,
        "source_trajectory_ids": source_ids,
        "steps": steps,
    }


def read_state_from_group(group, index: int) -> dict[str, Any]:
    state = {
        "residual_demand": group["residual_demand"][index].astype(int).tolist(),
        "remaining_stock": group["remaining_stock"][index].astype(int).tolist(),
        "expansion_mode": bool(group["expansion_mode"][index]),
        "current_modality": none_if_minus_one(group["current_modality"][index]),
        "current_entry_hour": none_if_minus_one(group["current_entry_hour"][index]),
        "assignment_week": int(group["assignment_week"][index]),
        "initial_demand_total": int(group["initial_demand_total"][index]),
    }
    if "mobile_days_off_count" in group:
        state["mobile_days_off_count"] = int(group["mobile_days_off_count"][index])
    if "fixed_day_off" in group:
        state["fixed_day_off"] = none_if_minus_one(group["fixed_day_off"][index])
    if "allowed_entry_hours_mask" in group:
        mask = group["allowed_entry_hours_mask"][index].astype(int)
        state["allowed_entry_hours"] = None if np.all(mask == 1) else [
            int(hour) for hour in np.flatnonzero(mask)
        ]
    if "max_overcoverage_tolerance" in group:
        state["max_overcoverage_tolerance"] = float(
            group["max_overcoverage_tolerance"][index]
        )
    if "closing_hour" in group:
        state["closing_hour"] = none_if_minus_one(group["closing_hour"][index])
    return state


def summarize_checkpoints(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(checkpoint_dir)
    if not path.exists():
        return []

    rows = []
    for checkpoint in sorted(path.glob("*.pt")):
        rows.append(
            {
                "path": str(checkpoint),
                "name": checkpoint.name,
                "size_mb": round(checkpoint.stat().st_size / (1024 * 1024), 2),
                "modified_at": datetime.fromtimestamp(
                    checkpoint.stat().st_mtime
                ).isoformat(timespec="seconds"),
                "global_step_from_name": parse_checkpoint_step(checkpoint.name),
            }
        )
    return rows


def build_dataset_analysis(
    raw_path: str | Path,
    stock_path: str | Path,
    sample_path: str | Path,
    max_sample_scan: int,
) -> dict[str, Any]:
    return {
        "raw": analyze_trajectory_buffer(raw_path, stock_stage=False),
        "stock": analyze_trajectory_buffer(stock_path, stock_stage=True),
        "samples": analyze_sample_buffer(sample_path, max_scan=max_sample_scan),
    }


def analyze_trajectory_buffer(
    store_path: str | Path,
    stock_stage: bool,
) -> dict[str, Any]:
    path = Path(store_path)
    if not path.exists():
        return {"exists": False, "path": str(path)}

    try:
        root = zarr.open_group(store=str(path), mode="r")
        group = root["trajectories"]
        ids = sorted(str(key) for key in group.group_keys())

        initial_demand = []
        final_rewards = []
        lengths = []
        initial_stock_total = []
        initial_stock_by_modality = [[], [], []]
        output_stock_total = []
        output_stock_by_modality = [[], [], []]
        mobile_days_off = Counter()
        fixed_day_off = Counter()
        allowed_hours = Counter()
        stock_was_reduced = Counter()
        expansion = Counter()

        for trajectory_id in ids:
            attrs = dict(group[trajectory_id].attrs)
            append_number(initial_demand, attrs.get("metadata.initial_demand_total"))
            append_number(final_rewards, attrs.get("final_reward"))
            append_number(lengths, attrs.get("length"))

            initial_stock = attrs.get("metadata.initial_stock")
            if initial_stock is None:
                initial_stock = attrs.get("metadata.original_stock")
            append_stock_stats(initial_stock, initial_stock_total, initial_stock_by_modality)

            output_stock = attrs.get("metadata.output_stock")
            append_stock_stats(output_stock, output_stock_total, output_stock_by_modality)

            mobile_days_off[str(attrs.get("problem_setup.mobile_days_off_count"))] += 1
            fixed_day_off[str(attrs.get("problem_setup.fixed_day_off"))] += 1
            allowed_hours[str(attrs.get("problem_setup.allowed_entry_hours"))] += 1

            if stock_stage:
                stock_was_reduced[str(_attr_bool(attrs.get("metadata.stock_was_reduced")))] += 1
                expansion[str(_attr_bool(attrs.get("metadata.has_expansion_mode")))] += 1

        return {
            "exists": True,
            "path": str(path),
            "count": len(ids),
            "initial_demand_total": distribution(initial_demand),
            "final_reward": distribution(final_rewards),
            "trajectory_length": distribution(lengths),
            "initial_stock_total": distribution(initial_stock_total),
            "initial_stock_by_modality": {
                "mod_4": distribution(initial_stock_by_modality[0]),
                "mod_6": distribution(initial_stock_by_modality[1]),
                "mod_8": distribution(initial_stock_by_modality[2]),
            },
            "output_stock_total": distribution(output_stock_total),
            "output_stock_by_modality": {
                "mod_4": distribution(output_stock_by_modality[0]),
                "mod_6": distribution(output_stock_by_modality[1]),
                "mod_8": distribution(output_stock_by_modality[2]),
            },
            "mobile_days_off_count": dict(mobile_days_off),
            "fixed_day_off": dict(fixed_day_off),
            "allowed_entry_hours": dict(allowed_hours),
            "stock_was_reduced": dict(stock_was_reduced),
            "has_expansion_mode": dict(expansion),
        }
    except Exception as exc:
        return {"exists": True, "path": str(path), "error": str(exc)}


def analyze_sample_buffer(
    store_path: str | Path,
    max_scan: int,
) -> dict[str, Any]:
    path = Path(store_path)
    if not path.exists():
        return {"exists": False, "path": str(path)}

    try:
        root = zarr.open_group(store=str(path), mode="r")
        group = root["samples"]
        length = int(group.attrs.get("length", 0))
        scan_count = min(length, int(max_scan))
        if scan_count <= 0:
            return {"exists": True, "path": str(path), "length": length}

        indices = np.arange(scan_count, dtype=int)
        values = read_array(group, "value", indices, float)
        policy_weights = read_array(group, "policy_weight", indices, float)
        initial_demand = read_array(group, "initial_demand_total", indices, float)
        expansion_mode = read_array(group, "expansion_mode", indices, bool)
        action_ids = read_array(group, "action_id", indices, int)
        sources = read_array(group, "sample_source", indices, str)
        trajectory_ids = read_array(group, "trajectory_id", indices, str)

        action_type_counts = Counter(decode_action(int(action_id))["type"] for action_id in action_ids)
        action_id_counts = Counter(int(action_id) for action_id in action_ids)
        samples_per_trajectory = Counter(str(trajectory_id) for trajectory_id in trajectory_ids)

        return {
            "exists": True,
            "path": str(path),
            "length": length,
            "scan_count": scan_count,
            "scan_limited": scan_count < length,
            "sample_source": dict(Counter(str(source) for source in sources)),
            "action_type": dict(action_type_counts),
            "top_action_id": dict(action_id_counts.most_common(20)),
            "samples_per_trajectory": distribution(samples_per_trajectory.values()),
            "initial_demand_total": distribution(initial_demand),
            "value": distribution(values),
            "policy_weight": distribution(policy_weights),
            "expansion_mode_count": int(np.asarray(expansion_mode, dtype=bool).sum()),
        }
    except Exception as exc:
        return {"exists": True, "path": str(path), "error": str(exc)}


def append_number(output: list[float], value: Any) -> None:
    parsed = _safe_float(value)
    if parsed is not None:
        output.append(float(parsed))


def append_stock_stats(
    stock: Any,
    total_output: list[float],
    modality_outputs: list[list[float]],
) -> None:
    if stock is None or stock == "__NONE__":
        return
    values = [float(value) for value in stock]
    if len(values) != 3:
        return
    total_output.append(sum(values))
    for index, value in enumerate(values):
        modality_outputs[index].append(value)


def distribution(values: Any, bins: int = 12) -> dict[str, Any]:
    values = [float(value) for value in values if value is not None]
    stats = summary_stats(values)
    if not values:
        return {"stats": stats, "histogram": []}

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return {
            "stats": stats,
            "histogram": [
                {"min": min_value, "max": max_value, "count": len(values)}
            ],
        }

    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=int(bins))
    histogram = [
        {
            "min": float(edges[index]),
            "max": float(edges[index + 1]),
            "count": int(counts[index]),
        }
        for index in range(len(counts))
    ]
    return {"stats": stats, "histogram": histogram}


def build_derived_summary(data: dict[str, Any]) -> dict[str, Any]:
    logs = data["logs"]
    runs = logs["runs"]
    cycles = logs["cycles"]
    learner_steps = logs["learner_steps"]

    run_ids = sorted({row.get("run_id") for row in runs + cycles + learner_steps if row.get("run_id")})
    latest_run = runs[-1] if runs else None
    latest_cycle = cycles[-1] if cycles else None
    latest_step = learner_steps[-1] if learner_steps else None
    latest_checkpoint = data["checkpoints"][-1] if data["checkpoints"] else None

    total_samples_logged = sum(
        int((row.get("report") or {}).get("saved_samples", 0))
        for row in runs
    )
    total_mcts_jobs = sum(
        int((row.get("report") or {}).get("used_mcts_jobs", 0))
        for row in runs
    )
    total_reweighted_jobs = sum(
        int((row.get("report") or {}).get("reweighted_jobs", 0))
        for row in runs
    )

    cycles_by_run = defaultdict(list)
    for cycle in cycles:
        cycles_by_run[cycle.get("run_id")].append(cycle)

    return {
        "run_ids": run_ids,
        "latest_run": latest_run,
        "latest_cycle": latest_cycle,
        "latest_learner_step": latest_step,
        "latest_checkpoint": latest_checkpoint,
        "total_runs": len(runs),
        "total_cycles": len(cycles),
        "total_learner_steps": len(learner_steps),
        "total_samples_logged": total_samples_logged,
        "total_mcts_jobs": total_mcts_jobs,
        "total_reweighted_jobs": total_reweighted_jobs,
        "cycles_by_run": {key: value for key, value in cycles_by_run.items()},
    }


def read_array(group, name: str, indices: np.ndarray, dtype):
    if name not in group:
        return np.asarray([], dtype=dtype)
    return group[name][indices].astype(dtype)


def read_counter(group, name: str, indices: np.ndarray, dtype) -> Counter:
    if name not in group:
        return Counter()
    return Counter(group[name][indices].astype(dtype))


def summary_stats(values) -> dict[str, Any]:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0}
    output = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }
    if len(values) > 1:
        output["median"] = statistics.median(values)
    return output


def top_policy_entries(
    policy: np.ndarray,
    selected_action_id: int,
    limit: int = 12,
) -> list[dict[str, Any]]:
    policy = np.asarray(policy, dtype=np.float32)
    selected = int(selected_action_id)
    candidate_ids = set(np.argsort(policy)[-int(limit):].astype(int).tolist())
    candidate_ids.add(selected)
    ordered_ids = sorted(candidate_ids, key=lambda action_id: float(policy[action_id]), reverse=True)
    return [
        {
            "action_id": int(action_id),
            "action": decode_action(int(action_id)),
            "prob": float(policy[action_id]),
            "selected": int(action_id) == selected,
            "legal": bool(policy[action_id] > 0),
        }
        for action_id in ordered_ids
    ]


def decode_action(action_id: int) -> dict[str, Any]:
    action_id = int(action_id)
    if 0 <= action_id <= 2:
        modality = [4, 6, 8][action_id]
        return {
            "type": "MODALITY",
            "label": f"MOD {modality}h",
            "modality": modality,
        }
    if 3 <= action_id <= 26:
        hour = action_id - 3
        return {
            "type": "ENTRY_HOUR",
            "label": f"ENTRY {hour:02d}:00",
            "entry_hour": hour,
        }
    if 27 <= action_id <= 54:
        internal_id = action_id - 27
        pair = day_off_pair(internal_id)
        return {
            "type": "DAY_OFFS",
            "label": f"OFF {pair[0]}-{pair[1]}",
            "day_pair": pair,
        }
    return {"type": "UNKNOWN", "label": str(action_id)}


def day_off_pair(internal_id: int) -> list[int]:
    current = 0
    for row in range(7):
        for col in range(row, 7):
            if current == int(internal_id):
                return [int(row), int(col)]
            current += 1
    return [-1, -1]


def none_if_minus_one(value: Any) -> int | None:
    value = int(value)
    return None if value == -1 else value


def attrs_to_json(attrs: dict[str, Any]) -> dict[str, Any]:
    return {key: json_value(value) for key, value in attrs.items()}


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def read_problem_setup_attrs(group) -> dict[str, Any]:
    prefix = "problem_setup."
    return {
        key[len(prefix):]: json_value(value)
        for key, value in dict(group.attrs).items()
        if str(key).startswith(prefix)
    }


def parse_checkpoint_step(name: str) -> int | None:
    stem = Path(name).stem
    suffix = stem.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _attr_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def render_overview_dashboard(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Workforce Training Dashboard</title>
  <style>
    :root {{
      --bg: #f6f7f8;
      --panel: #ffffff;
      --text: #20242a;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #116a5b;
      --accent-2: #b54708;
      --accent-3: #344054;
      --danger: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    main {{ padding: 20px 28px 40px; }}
    h1 {{ margin: 0 0 4px; font-size: 24px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 14px 0 8px; font-size: 14px; }}
    a.button {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--accent);
      text-decoration: none;
      background: #fff;
      font-weight: 700;
    }}
    .subtle {{ color: var(--muted); }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    select, button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px 10px;
      color: var(--text);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .panel, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .panel {{ margin-bottom: 16px; }}
    .card {{ min-height: 92px; }}
    .metric {{ font-size: 26px; font-weight: 700; letter-spacing: 0; }}
    .label {{ color: var(--muted); margin-top: 4px; }}
    .chart {{
      min-height: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}
    svg {{ display: block; width: 100%; height: 240px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .pill {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef4f2;
      color: var(--accent);
      font-size: 12px;
      font-weight: 600;
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .two-col {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 700px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Workforce Training Dashboard</h1>
    <div class="subtle">Snapshot generado: <span id="generatedAt"></span></div>
    <div class="toolbar">
      <label>Run <select id="runSelect"></select></label>
      <button id="showAll">Mostrar todo</button>
      <a class="button" href="trajectory_explorer.html">Abrir explorador</a>
      <span class="subtle">Para actualizar, volver a ejecutar <code>scripts/build_training_dashboard.py</code>.</span>
    </div>
  </header>
  <main>
    <section class="grid" id="cards"></section>

    <section class="two-col">
      <div class="panel">
        <h2>Loss del learner</h2>
        <div class="chart" id="lossChart"></div>
      </div>
      <div class="panel">
        <h2>Samples por ciclo</h2>
        <div class="chart" id="cycleChart"></div>
      </div>
    </section>

    <section class="two-col">
      <div class="panel">
        <h2>Runs</h2>
        <div id="runsTable"></div>
      </div>
      <div class="panel">
        <h2>Ciclos</h2>
        <div id="cyclesTable"></div>
      </div>
    </section>

    <section class="panel">
      <h2>Analisis de trayectorias</h2>
      <div class="two-col">
        <div>
          <h3>Raw</h3>
          <div id="rawAnalysis"></div>
        </div>
        <div>
          <h3>Stock</h3>
          <div id="stockAnalysis"></div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Analisis de samples</h2>
      <div id="sampleAnalysis"></div>
    </section>

    <section class="two-col">
      <div class="panel">
        <h2>Buffers Zarr</h2>
        <div id="buffers"></div>
      </div>
      <div class="panel">
        <h2>Checkpoints</h2>
        <div id="checkpoints"></div>
      </div>
    </section>
  </main>
  <script id="dashboard-data" type="application/json">{encoded}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
    let selectedRun = 'ALL';

    function fmt(value, digits = 3) {{
      if (value === null || value === undefined) return '-';
      if (typeof value === 'number') {{
        if (Math.abs(value) >= 1000) return value.toLocaleString();
        return Number.isInteger(value) ? String(value) : value.toFixed(digits);
      }}
      return String(value);
    }}

    function rowsForRun(rows) {{
      if (selectedRun === 'ALL') return rows;
      return rows.filter(row => row.run_id === selectedRun);
    }}

    function initRunSelect() {{
      const select = document.getElementById('runSelect');
      const runIds = DATA.derived.run_ids || [];
      select.innerHTML = '<option value="ALL">Todos</option>' + runIds.map(id => `<option value="${{id}}">${{id}}</option>`).join('');
      if (runIds.length) {{
        selectedRun = runIds[runIds.length - 1];
        select.value = selectedRun;
      }}
      select.addEventListener('change', () => {{
        selectedRun = select.value;
        render();
      }});
      document.getElementById('showAll').addEventListener('click', () => {{
        selectedRun = 'ALL';
        select.value = 'ALL';
        render();
      }});
    }}

    function table(rows, columns) {{
      if (!rows.length) return '<p class="subtle">Sin datos.</p>';
      return `<table><thead><tr>${{columns.map(c => `<th>${{c.label}}</th>`).join('')}}</tr></thead><tbody>` +
        rows.map(row => `<tr>${{columns.map(c => `<td>${{c.render ? c.render(row) : fmt(row[c.key])}}</td>`).join('')}}</tr>`).join('') +
        '</tbody></table>';
    }}

    function lineChart(containerId, rows, series) {{
      const el = document.getElementById(containerId);
      if (!rows.length) {{
        el.innerHTML = '<div class="subtle">Sin datos.</div>';
        return;
      }}
      const width = 900, height = 240, pad = 36;
      const values = rows.flatMap(row => series.map(s => row.metric?.[s.key] ?? row[s.key]).filter(v => typeof v === 'number'));
      const minY = Math.min(...values);
      const maxY = Math.max(...values);
      const spanY = maxY - minY || 1;
      const x = i => pad + (i / Math.max(1, rows.length - 1)) * (width - 2 * pad);
      const y = v => height - pad - ((v - minY) / spanY) * (height - 2 * pad);
      const paths = series.map(s => {{
        const points = rows.map((row, i) => `${{x(i)}},${{y(row.metric?.[s.key] ?? row[s.key])}}`).join(' ');
        return `<polyline points="${{points}}" fill="none" stroke="${{s.color}}" stroke-width="2"/>`;
      }}).join('');
      const legend = series.map((s, i) => `<text x="${{pad + i * 130}}" y="18" fill="${{s.color}}" font-size="12">${{s.label}}</text>`).join('');
      el.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
        ${{legend}}
        <line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        <text x="4" y="${{pad}}" font-size="11" fill="#667085">${{fmt(maxY)}}</text>
        <text x="4" y="${{height-pad}}" font-size="11" fill="#667085">${{fmt(minY)}}</text>
        ${{paths}}
      </svg>`;
    }}

    function barChart(containerId, cycles) {{
      const el = document.getElementById(containerId);
      if (!cycles.length) {{
        el.innerHTML = '<div class="subtle">Sin datos.</div>';
        return;
      }}
      const width = 900, height = 240, pad = 36;
      const maxV = Math.max(...cycles.map(row => row.cycle?.saved_samples || 0), 1);
      const barW = (width - 2 * pad) / cycles.length;
      const bars = cycles.map((row, i) => {{
        const v = row.cycle?.saved_samples || 0;
        const h = (v / maxV) * (height - 2 * pad);
        return `<rect x="${{pad + i * barW + 2}}" y="${{height - pad - h}}" width="${{Math.max(2, barW - 4)}}" height="${{h}}" fill="#116a5b">
          <title>${{row.run_id}} ciclo ${{row.cycle?.cycle_index}}: ${{v}} samples</title>
        </rect>`;
      }}).join('');
      el.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
        <line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        ${{bars}}
      </svg>`;
    }}

    function miniHistogram(title, dist) {{
      if (!dist?.histogram?.length) return `<p class="subtle">${{title}}: sin datos.</p>`;
      const maxCount = Math.max(...dist.histogram.map(bin => bin.count), 1);
      const bars = dist.histogram.map(bin => {{
        const width = Math.round((bin.count / maxCount) * 100);
        return `<tr><td>${{fmt(bin.min)}} - ${{fmt(bin.max)}}</td><td><div style="height:10px;background:#edf0f5;border-radius:999px;overflow:hidden"><div style="height:100%;width:${{width}}%;background:#116a5b"></div></div></td><td>${{fmt(bin.count)}}</td></tr>`;
      }}).join('');
      return `<h3>${{title}}</h3><div class="subtle">mean=${{fmt(dist.stats?.mean)}} median=${{fmt(dist.stats?.median)}} min=${{fmt(dist.stats?.min)}} max=${{fmt(dist.stats?.max)}}</div><table><tbody>${{bars}}</tbody></table>`;
    }}

    function histogramPanel(title, dist, color = '#116a5b') {{
      if (!dist?.histogram?.length) return `<div class="panel"><h3>${{title}}</h3><p class="subtle">Sin datos.</p></div>`;
      const width = 760, height = 250, pad = 38;
      const bins = dist.histogram;
      const maxCount = Math.max(...bins.map(bin => bin.count), 1);
      const barW = (width - 2 * pad) / bins.length;
      const bars = bins.map((bin, index) => {{
        const h = (bin.count / maxCount) * (height - 2 * pad);
        const x = pad + index * barW + 2;
        const y = height - pad - h;
        return `<rect x="${{x}}" y="${{y}}" width="${{Math.max(2, barW - 4)}}" height="${{h}}" fill="${{color}}">
          <title>${{fmt(bin.min)}} - ${{fmt(bin.max)}}: ${{fmt(bin.count)}}</title>
        </rect>`;
      }}).join('');

      let cumulative = 0;
      const total = bins.reduce((acc, bin) => acc + bin.count, 0) || 1;
      const cdfPoints = bins.map((bin, index) => {{
        cumulative += bin.count;
        const x = pad + index * barW + barW / 2;
        const y = height - pad - (cumulative / total) * (height - 2 * pad);
        return `${{x}},${{y}}`;
      }}).join(' ');

      const labels = bins
        .filter((_, index) => index === 0 || index === bins.length - 1)
        .map((bin, index) => {{
          const x = index === 0 ? pad : width - pad - 38;
          return `<text x="${{x}}" y="${{height - 9}}" font-size="11" fill="#667085">${{fmt(index === 0 ? bin.min : bin.max)}}</text>`;
        }}).join('');

      return `<div class="panel">
        <h3>${{title}}</h3>
        <div class="subtle">mean=${{fmt(dist.stats?.mean)}} · median=${{fmt(dist.stats?.median)}} · min=${{fmt(dist.stats?.min)}} · max=${{fmt(dist.stats?.max)}} · n=${{fmt(dist.stats?.count)}}</div>
        <div class="chart">
          <svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
            <line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
            <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
            <text x="4" y="${{pad}}" font-size="11" fill="#667085">${{fmt(maxCount)}}</text>
            <text x="${{width - pad - 18}}" y="${{pad + 4}}" font-size="11" fill="#b54708">CDF</text>
            ${{bars}}
            <polyline points="${{cdfPoints}}" fill="none" stroke="#b54708" stroke-width="2.5"/>
            ${{labels}}
          </svg>
        </div>
      </div>`;
    }}

    function categoricalChart(title, counts, color = '#344054') {{
      const rows = Object.entries(counts || {{}})
        .sort((a, b) => Number(b[1]) - Number(a[1]))
        .slice(0, 20);
      if (!rows.length) return `<div class="panel"><h3>${{title}}</h3><p class="subtle">Sin datos.</p></div>`;
      const width = 760, height = Math.max(180, rows.length * 26 + 44), padLeft = 138, padRight = 36, padY = 24;
      const maxCount = Math.max(...rows.map(row => Number(row[1])), 1);
      const rowH = (height - 2 * padY) / rows.length;
      const bars = rows.map(([label, count], index) => {{
        const y = padY + index * rowH + 4;
        const w = (Number(count) / maxCount) * (width - padLeft - padRight);
        return `<g>
          <text x="8" y="${{y + rowH * 0.55}}" font-size="12" fill="#20242a">${{String(label).slice(0, 22)}}</text>
          <rect x="${{padLeft}}" y="${{y}}" width="${{w}}" height="${{Math.max(8, rowH - 8)}}" rx="3" fill="${{color}}"/>
          <text x="${{padLeft + w + 6}}" y="${{y + rowH * 0.55}}" font-size="12" fill="#667085">${{fmt(Number(count))}}</text>
        </g>`;
      }}).join('');
      return `<div class="panel">
        <h3>${{title}}</h3>
        <div class="chart">
          <svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">${{bars}}</svg>
        </div>
      </div>`;
    }}

    function countsTable(title, counts) {{
      const rows = Object.entries(counts || {{}}).sort((a, b) => String(a[0]).localeCompare(String(b[0])));
      return `<h3>${{title}}</h3>` + table(rows, [
        {{label: 'Valor', render: r => r[0]}},
        {{label: 'Count', render: r => fmt(r[1])}},
      ]);
    }}

    function renderCards() {{
      const runs = rowsForRun(DATA.logs.runs);
      const cycles = rowsForRun(DATA.logs.cycles);
      const steps = rowsForRun(DATA.logs.learner_steps);
      const latestStep = steps[steps.length - 1]?.metric;
      const savedSamples = runs.reduce((acc, row) => acc + (row.report?.saved_samples || 0), 0);
      const cards = [
        ['Runs', runs.length],
        ['Ciclos', cycles.length],
        ['Learner steps', steps.length],
        ['Samples logueados', savedSamples],
        ['Jobs MCTS', runs.reduce((acc, row) => acc + (row.report?.used_mcts_jobs || 0), 0)],
        ['Jobs reweighted', runs.reduce((acc, row) => acc + (row.report?.reweighted_jobs || 0), 0)],
        ['Failed jobs', runs.reduce((acc, row) => acc + (row.report?.failed_jobs || 0), 0)],
        ['Ultima loss', latestStep?.loss],
      ];
      document.getElementById('cards').innerHTML = cards.map(([label, value]) => `
        <div class="card"><div class="metric">${{fmt(value)}}</div><div class="label">${{label}}</div></div>
      `).join('');
    }}

    function renderTables() {{
      const runs = rowsForRun(DATA.logs.runs).slice(-20).reverse();
      const cycles = rowsForRun(DATA.logs.cycles).slice(-30).reverse();
      document.getElementById('runsTable').innerHTML = table(runs, [
        {{label: 'Run', render: r => `<span class="pill">${{r.run_id}}</span>`}},
        {{label: 'Workers', render: r => fmt(r.args?.workers)}},
        {{label: 'Device', render: r => fmt(r.args?.device)}},
        {{label: 'Jobs', render: r => fmt(r.report?.completed_jobs)}},
        {{label: 'Failed', render: r => fmt(r.report?.failed_jobs)}},
        {{label: 'Samples', render: r => fmt(r.report?.saved_samples)}},
        {{label: 'MCTS', render: r => fmt(r.report?.used_mcts_jobs)}},
        {{label: 'Reweighted', render: r => fmt(r.report?.reweighted_jobs)}},
      ]);
      document.getElementById('cyclesTable').innerHTML = table(cycles, [
        {{label: 'Run', render: r => `<span class="pill">${{r.run_id}}</span>`}},
        {{label: 'Ciclo', render: r => fmt(r.cycle?.cycle_index)}},
        {{label: 'Jobs', render: r => fmt(r.cycle?.completed_jobs)}},
        {{label: 'Samples', render: r => fmt(r.cycle?.saved_samples)}},
        {{label: 'MCTS', render: r => fmt(r.cycle?.used_mcts_jobs)}},
        {{label: 'Reweighted', render: r => fmt(r.cycle?.reweighted_jobs)}},
        {{label: 'Step', render: r => fmt(r.learner?.global_step)}},
        {{label: 'Loss', render: r => fmt(r.learner?.last_metric?.loss)}},
      ]);
    }}

    function renderTrajectoryAnalysis(targetId, analysis) {{
      if (!analysis?.exists || analysis.error) {{
        document.getElementById(targetId).innerHTML = `<p class="subtle">${{analysis?.error || 'Sin datos.'}}</p>`;
        return;
      }}
      document.getElementById(targetId).innerHTML = `
        <div class="grid">
          <div class="card"><div class="metric">${{fmt(analysis.count)}}</div><div class="label">Trayectorias</div></div>
          <div class="card"><div class="metric">${{fmt(analysis.initial_demand_total?.stats?.mean)}}</div><div class="label">Demanda media</div></div>
          <div class="card"><div class="metric">${{fmt(analysis.initial_stock_total?.stats?.mean)}}</div><div class="label">Stock inicial medio</div></div>
          <div class="card"><div class="metric">${{fmt(analysis.final_reward?.stats?.mean)}}</div><div class="label">Reward medio</div></div>
        </div>
        ${{miniHistogram('Demanda total inicial', analysis.initial_demand_total)}}
        ${{miniHistogram('Recursos totales iniciales', analysis.initial_stock_total)}}
        ${{miniHistogram('Mod 4h', analysis.initial_stock_by_modality?.mod_4)}}
        ${{miniHistogram('Mod 6h', analysis.initial_stock_by_modality?.mod_6)}}
        ${{miniHistogram('Mod 8h', analysis.initial_stock_by_modality?.mod_8)}}
        ${{miniHistogram('Reward final', analysis.final_reward)}}
        ${{countsTable('Mobile days off', analysis.mobile_days_off_count)}}
        ${{countsTable('Fixed day off', analysis.fixed_day_off)}}
        ${{countsTable('Stock reduced', analysis.stock_was_reduced)}}
        ${{countsTable('Expansion mode', analysis.has_expansion_mode)}}
      `;
    }}

    function renderSampleAnalysis() {{
      const analysis = DATA.analysis.samples;
      if (!analysis?.exists || analysis.error) {{
        document.getElementById('sampleAnalysis').innerHTML = `<p class="subtle">${{analysis?.error || 'Sin datos.'}}</p>`;
        return;
      }}
      document.getElementById('sampleAnalysis').innerHTML = `
        <div class="grid">
          <div class="card"><div class="metric">${{fmt(analysis.length)}}</div><div class="label">Samples total</div></div>
          <div class="card"><div class="metric">${{fmt(analysis.scan_count)}}</div><div class="label">Samples analizados</div></div>
          <div class="card"><div class="metric">${{fmt(analysis.expansion_mode_count)}}</div><div class="label">Expansion samples</div></div>
          <div class="card"><div class="metric">${{fmt(analysis.policy_weight?.stats?.mean)}}</div><div class="label">Policy weight medio</div></div>
        </div>
        <div class="two-col">
          <div>
            ${{categoricalChart('Sample source', analysis.sample_source, '#116a5b')}}
            ${{categoricalChart('Action type', analysis.action_type, '#344054')}}
            ${{categoricalChart('Top action_id', analysis.top_action_id, '#667085')}}
          </div>
          <div>
            ${{histogramPanel('Value', analysis.value, '#116a5b')}}
            ${{histogramPanel('Policy weight', analysis.policy_weight, '#344054')}}
            ${{histogramPanel('Demanda total inicial', analysis.initial_demand_total, '#116a5b')}}
            ${{histogramPanel('Samples por trayectoria', analysis.samples_per_trajectory, '#667085')}}
          </div>
        </div>
      `;
    }}

    function renderBuffersAndCheckpoints() {{
      const buffers = DATA.buffers;
      document.getElementById('buffers').innerHTML = table([
        ['Raw', buffers.raw],
        ['Stock', buffers.stock],
        ['Samples', buffers.samples],
      ], [
        {{label: 'Buffer', render: r => r[0]}},
        {{label: 'Existe', render: r => r[1].exists ? 'si' : 'no'}},
        {{label: 'Count', render: r => fmt(r[1].count ?? r[1].length)}},
        {{label: 'Path', render: r => r[1].path}},
      ]);
      document.getElementById('checkpoints').innerHTML = table((DATA.checkpoints || []).slice().reverse(), [
        {{label: 'Nombre', key: 'name'}},
        {{label: 'Step', key: 'global_step_from_name'}},
        {{label: 'MB', key: 'size_mb'}},
        {{label: 'Modificado', key: 'modified_at'}},
      ]);
    }}

    function render() {{
      document.getElementById('generatedAt').textContent = DATA.generated_at;
      renderCards();
      lineChart('lossChart', rowsForRun(DATA.logs.learner_steps), [
        {{key: 'loss', label: 'loss', color: '#116a5b'}},
        {{key: 'policy_loss', label: 'policy', color: '#b54708'}},
        {{key: 'value_loss', label: 'value', color: '#344054'}},
      ]);
      barChart('cycleChart', rowsForRun(DATA.logs.cycles));
      renderTables();
      renderTrajectoryAnalysis('rawAnalysis', DATA.analysis.raw);
      renderTrajectoryAnalysis('stockAnalysis', DATA.analysis.stock);
      renderSampleAnalysis();
      renderBuffersAndCheckpoints();
    }}

    initRunSelect();
    render();
  </script>
</body>
</html>
"""


def render_dashboard(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Workforce Trajectory Explorer</title>
  <style>
    :root {{
      --bg: #f6f7f8;
      --panel: #ffffff;
      --text: #20242a;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #116a5b;
      --accent-2: #b54708;
      --accent-3: #344054;
      --danger: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{ margin: 0 0 4px; font-size: 24px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 14px; }}
    main {{ padding: 20px 28px 40px; }}
    .subtle {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 92px;
    }}
    .metric {{ font-size: 26px; font-weight: 700; letter-spacing: 0; }}
    .label {{ color: var(--muted); margin-top: 4px; }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    select, button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px 10px;
      color: var(--text);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .chart {{
      width: 100%;
      min-height: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}
    svg {{ display: block; width: 100%; height: 240px; }}
    .pill {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef4f2;
      color: var(--accent);
      font-size: 12px;
      font-weight: 600;
    }}
    .warn {{ color: var(--danger); }}
    .tabs {{
      display: flex;
      gap: 8px;
      margin: 0 0 14px;
      border-bottom: 1px solid var(--line);
    }}
    .tab {{
      border: 0;
      border-bottom: 2px solid transparent;
      border-radius: 0;
      padding: 9px 10px;
      background: transparent;
      cursor: pointer;
    }}
    .tab.active {{
      border-bottom-color: var(--accent);
      color: var(--accent);
      font-weight: 700;
    }}
    .explorer-layout {{
      display: grid;
      grid-template-columns: minmax(360px, 1.4fr) minmax(300px, .9fr);
      gap: 16px;
      align-items: start;
    }}
    .stepbar {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin: 10px 0 12px;
    }}
    .stepbar input[type="range"] {{
      min-width: 260px;
      flex: 1;
    }}
    .heatmap-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      max-height: 620px;
    }}
    .heatmap-table {{
      border-collapse: collapse;
      min-width: 760px;
      font-size: 11px;
    }}
    .heatmap-table th,
    .heatmap-table td {{
      border: 1px solid #edf0f5;
      padding: 2px 4px;
      text-align: center;
      min-width: 24px;
      height: 20px;
    }}
    .heatmap-table th {{
      position: sticky;
      top: 0;
      background: #f8fafc;
      z-index: 1;
    }}
    .heatmap-table th:first-child {{
      left: 0;
      z-index: 2;
    }}
    .heatmap-table td:first-child {{
      position: sticky;
      left: 0;
      background: #f8fafc;
      color: var(--muted);
      font-weight: 600;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 145px 1fr;
      gap: 6px 10px;
      font-size: 13px;
    }}
    .kv div:nth-child(odd) {{
      color: var(--muted);
    }}
    .policy-row {{
      display: grid;
      grid-template-columns: 74px 115px 1fr 66px;
      gap: 8px;
      align-items: center;
      margin: 6px 0;
      font-size: 12px;
    }}
    .policy-bar {{
      height: 9px;
      background: #edf0f5;
      border-radius: 999px;
      overflow: hidden;
    }}
    .policy-fill {{
      height: 100%;
      background: var(--accent);
    }}
    .policy-row.selected .policy-fill {{
      background: var(--accent-2);
    }}
    .policy-row.selected {{
      font-weight: 700;
    }}
    pre {{
      white-space: pre-wrap;
      overflow: auto;
      background: #f2f4f7;
      padding: 10px;
      border-radius: 6px;
      border: 1px solid var(--line);
      max-height: 280px;
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .two-col {{ grid-template-columns: 1fr; }}
      .explorer-layout {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 700px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Workforce Trajectory Explorer</h1>
    <div class="subtle">Snapshot generado: <span id="generatedAt"></span></div>
    <div class="toolbar">
      <label>Run <select id="runSelect"></select></label>
      <button id="showAll">Mostrar todo</button>
      <a class="button" href="training_dashboard.html">Volver al dashboard</a>
      <span class="subtle">Para actualizar, volver a ejecutar <code>scripts/build_training_dashboard.py</code>.</span>
    </div>
  </header>
  <main>
    <section class="grid" id="cards"></section>

    <section class="two-col">
      <div class="panel">
        <h2>Loss del learner</h2>
        <div class="chart" id="lossChart"></div>
      </div>
      <div class="panel">
        <h2>Samples por ciclo</h2>
        <div class="chart" id="cycleChart"></div>
      </div>
    </section>

    <section class="two-col">
      <div class="panel">
        <h2>Runs</h2>
        <div id="runsTable"></div>
      </div>
      <div class="panel">
        <h2>Ciclos</h2>
        <div id="cyclesTable"></div>
      </div>
    </section>

    <section class="panel">
      <h2>Buffers Zarr</h2>
      <div id="buffers"></div>
    </section>

    <section class="two-col">
      <div class="panel">
        <h2>SampleBuffer actual</h2>
        <div id="sampleBuffer"></div>
      </div>
      <div class="panel">
        <h2>Checkpoints</h2>
        <div id="checkpoints"></div>
      </div>
    </section>

    <section class="panel">
      <h2>Explorador de trayectorias</h2>
      <div class="tabs">
        <button class="tab active" data-explorer-source="raw">Raw</button>
        <button class="tab" data-explorer-source="stock">Stock</button>
        <button class="tab" data-explorer-source="samples">SampleBuffer</button>
      </div>
      <div class="toolbar">
        <label>Trayectoria <select id="trajectorySelect"></select></label>
        <span class="subtle" id="trajectorySummary"></span>
      </div>
      <div class="stepbar">
        <button id="prevStep">Anterior</button>
        <input id="stepRange" type="range" min="0" max="0" value="0">
        <button id="nextStep">Siguiente</button>
        <span class="pill" id="stepLabel">step 0</span>
      </div>
      <div class="explorer-layout">
        <div>
          <h3>Demanda residual</h3>
          <div class="heatmap-wrap" id="residualHeatmap"></div>
        </div>
        <div>
          <h3>Estado y targets</h3>
          <div id="stateDetails"></div>
          <h3 style="margin-top:14px">Policy</h3>
          <div id="policyDetails"></div>
          <h3 style="margin-top:14px">Metadata</h3>
          <pre id="trajectoryMetadata"></pre>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Detalles crudos</h2>
      <pre id="rawDetails"></pre>
    </section>
  </main>
  <script id="dashboard-data" type="application/json">{encoded}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
    let selectedRun = 'ALL';
    let explorerSource = 'raw';
    let explorerTrajectoryIndex = 0;
    let explorerStepIndex = 0;

    function fmt(value, digits = 3) {{
      if (value === null || value === undefined) return '-';
      if (typeof value === 'number') {{
        if (Math.abs(value) >= 1000) return value.toLocaleString();
        return Number.isInteger(value) ? String(value) : value.toFixed(digits);
      }}
      return String(value);
    }}

    function rowsForRun(rows) {{
      if (selectedRun === 'ALL') return rows;
      return rows.filter(row => row.run_id === selectedRun);
    }}

    function explorerTrajectories() {{
      return DATA.explorer?.[explorerSource]?.trajectories || [];
    }}

    function currentTrajectory() {{
      return explorerTrajectories()[explorerTrajectoryIndex] || null;
    }}

    function currentStep() {{
      const trajectory = currentTrajectory();
      if (!trajectory || !trajectory.steps?.length) return null;
      return trajectory.steps[Math.min(explorerStepIndex, trajectory.steps.length - 1)];
    }}

    function initRunSelect() {{
      const select = document.getElementById('runSelect');
      const runIds = DATA.derived.run_ids || [];
      select.innerHTML = '<option value="ALL">Todos</option>' + runIds.map(id => `<option value="${{id}}">${{id}}</option>`).join('');
      if (runIds.length) {{
        selectedRun = runIds[runIds.length - 1];
        select.value = selectedRun;
      }}
      select.addEventListener('change', () => {{
        selectedRun = select.value;
        render();
      }});
      document.getElementById('showAll').addEventListener('click', () => {{
        selectedRun = 'ALL';
        select.value = 'ALL';
        render();
      }});
    }}

    function renderCards() {{
      const runs = rowsForRun(DATA.logs.runs);
      const cycles = rowsForRun(DATA.logs.cycles);
      const steps = rowsForRun(DATA.logs.learner_steps);
      const latestStep = steps[steps.length - 1]?.metric;
      const savedSamples = runs.reduce((acc, row) => acc + (row.report?.saved_samples || 0), 0);
      const mctsJobs = runs.reduce((acc, row) => acc + (row.report?.used_mcts_jobs || 0), 0);
      const reweightedJobs = runs.reduce((acc, row) => acc + (row.report?.reweighted_jobs || 0), 0);
      const failedJobs = runs.reduce((acc, row) => acc + (row.report?.failed_jobs || 0), 0);

      const cards = [
        ['Runs', runs.length],
        ['Ciclos', cycles.length],
        ['Learner steps', steps.length],
        ['Samples logueados', savedSamples],
        ['Jobs MCTS', mctsJobs],
        ['Jobs reweighted', reweightedJobs],
        ['Failed jobs', failedJobs],
        ['Ultima loss', latestStep?.loss],
      ];

      document.getElementById('cards').innerHTML = cards.map(([label, value]) => `
        <div class="card">
          <div class="metric">${{fmt(value)}}</div>
          <div class="label">${{label}}</div>
        </div>
      `).join('');
    }}

    function lineChart(containerId, rows, series) {{
      const el = document.getElementById(containerId);
      if (!rows.length) {{
        el.innerHTML = '<div class="panel subtle">Sin datos.</div>';
        return;
      }}
      const width = 900, height = 240, pad = 36;
      const xs = rows.map((_, i) => i);
      const values = rows.flatMap(row => series.map(s => row.metric?.[s.key] ?? row[s.key]).filter(v => typeof v === 'number'));
      const minY = Math.min(...values);
      const maxY = Math.max(...values);
      const spanY = maxY - minY || 1;
      const x = i => pad + (i / Math.max(1, rows.length - 1)) * (width - 2 * pad);
      const y = v => height - pad - ((v - minY) / spanY) * (height - 2 * pad);

      const paths = series.map(s => {{
        const points = rows.map((row, i) => {{
          const value = row.metric?.[s.key] ?? row[s.key];
          return `${{x(i)}},${{y(value)}}`;
        }}).join(' ');
        return `<polyline points="${{points}}" fill="none" stroke="${{s.color}}" stroke-width="2"/>`;
      }}).join('');
      const legend = series.map((s, i) => `<text x="${{pad + i * 130}}" y="18" fill="${{s.color}}" font-size="12">${{s.label}}</text>`).join('');
      el.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
        ${{legend}}
        <line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        <text x="4" y="${{pad}}" font-size="11" fill="#667085">${{fmt(maxY)}}</text>
        <text x="4" y="${{height-pad}}" font-size="11" fill="#667085">${{fmt(minY)}}</text>
        ${{paths}}
      </svg>`;
    }}

    function barChart(containerId, cycles) {{
      const el = document.getElementById(containerId);
      if (!cycles.length) {{
        el.innerHTML = '<div class="panel subtle">Sin datos.</div>';
        return;
      }}
      const width = 900, height = 240, pad = 36;
      const maxV = Math.max(...cycles.map(row => row.cycle?.saved_samples || 0), 1);
      const barW = (width - 2 * pad) / cycles.length;
      const bars = cycles.map((row, i) => {{
        const v = row.cycle?.saved_samples || 0;
        const h = (v / maxV) * (height - 2 * pad);
        const x = pad + i * barW + 2;
        const y = height - pad - h;
        return `<rect x="${{x}}" y="${{y}}" width="${{Math.max(2, barW - 4)}}" height="${{h}}" fill="#116a5b">
          <title>${{row.run_id}} ciclo ${{row.cycle?.cycle_index}}: ${{v}} samples</title>
        </rect>`;
      }}).join('');
      el.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
        <line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}" stroke="#d9dee7"/>
        <text x="4" y="${{pad}}" font-size="11" fill="#667085">${{fmt(maxV, 0)}}</text>
        ${{bars}}
      </svg>`;
    }}

    function table(rows, columns) {{
      if (!rows.length) return '<p class="subtle">Sin datos.</p>';
      return `<table><thead><tr>${{columns.map(c => `<th>${{c.label}}</th>`).join('')}}</tr></thead><tbody>` +
        rows.map(row => `<tr>${{columns.map(c => `<td>${{c.render ? c.render(row) : fmt(row[c.key])}}</td>`).join('')}}</tr>`).join('') +
        '</tbody></table>';
    }}

    function renderTables() {{
      const runs = rowsForRun(DATA.logs.runs).slice(-20).reverse();
      const cycles = rowsForRun(DATA.logs.cycles).slice(-30).reverse();
      document.getElementById('runsTable').innerHTML = table(runs, [
        {{label: 'Run', render: r => `<span class="pill">${{r.run_id}}</span>`}},
        {{label: 'Workers', render: r => fmt(r.args?.workers)}},
        {{label: 'Device', render: r => fmt(r.args?.device)}},
        {{label: 'Jobs', render: r => fmt(r.report?.completed_jobs)}},
        {{label: 'Failed', render: r => fmt(r.report?.failed_jobs)}},
        {{label: 'Samples', render: r => fmt(r.report?.saved_samples)}},
        {{label: 'MCTS', render: r => fmt(r.report?.used_mcts_jobs)}},
        {{label: 'Reweighted', render: r => fmt(r.report?.reweighted_jobs)}},
        {{label: 'Estado', key: 'status'}},
      ]);
      document.getElementById('cyclesTable').innerHTML = table(cycles, [
        {{label: 'Run', render: r => `<span class="pill">${{r.run_id}}</span>`}},
        {{label: 'Ciclo', render: r => fmt(r.cycle?.cycle_index)}},
        {{label: 'Jobs', render: r => fmt(r.cycle?.completed_jobs)}},
        {{label: 'Samples', render: r => fmt(r.cycle?.saved_samples)}},
        {{label: 'MCTS', render: r => fmt(r.cycle?.used_mcts_jobs)}},
        {{label: 'Reweighted', render: r => fmt(r.cycle?.reweighted_jobs)}},
        {{label: 'Step', render: r => fmt(r.learner?.global_step)}},
        {{label: 'Loss', render: r => fmt(r.learner?.last_metric?.loss)}},
      ]);
    }}

    function renderBuffers() {{
      const buffers = DATA.buffers;
      document.getElementById('buffers').innerHTML = table([
        ['Raw', buffers.raw],
        ['Stock', buffers.stock],
      ], [
        {{label: 'Buffer', render: r => r[0]}},
        {{label: 'Existe', render: r => r[1].exists ? 'si' : 'no'}},
        {{label: 'Trayectorias', render: r => fmt(r[1].count)}},
        {{label: 'Len media', render: r => fmt(r[1].length?.mean)}},
        {{label: 'Reward medio', render: r => fmt(r[1].final_reward?.mean)}},
        {{label: 'Expansion', render: r => fmt(r[1].has_expansion_count)}},
        {{label: 'Stock reduced', render: r => fmt(r[1].stock_reduced_count)}},
      ]);
    }}

    function renderSampleBuffer() {{
      const s = DATA.buffers.samples;
      const counts = s.sample_source_counts || {{}};
      document.getElementById('sampleBuffer').innerHTML = `
        <div class="grid">
          <div class="card"><div class="metric">${{fmt(s.length)}}</div><div class="label">Samples</div></div>
          <div class="card"><div class="metric">${{fmt(s.trajectory_count_scanned)}}</div><div class="label">Trajectories escaneadas</div></div>
          <div class="card"><div class="metric">${{fmt(s.expansion_mode_count_scanned)}}</div><div class="label">Expansion samples</div></div>
          <div class="card"><div class="metric">${{fmt(s.policy_weight?.mean)}}</div><div class="label">Policy weight medio</div></div>
        </div>
        <h3>Fuentes</h3>
        ${{table(Object.entries(counts), [
          {{label: 'Fuente', render: r => r[0]}},
          {{label: 'Count', render: r => fmt(r[1])}},
        ])}}
        <h3>Preview</h3>
        ${{table(s.preview || [], [
          {{label: '#', key: 'sample_index'}},
          {{label: 'Trajectory', key: 'trajectory_id'}},
          {{label: 'Step', key: 'step_index'}},
          {{label: 'Source', key: 'sample_source'}},
          {{label: 'Action', key: 'action_id'}},
          {{label: 'Weight', key: 'policy_weight'}},
          {{label: 'Value', key: 'value'}},
        ])}}
      `;
    }}

    function renderCheckpoints() {{
      document.getElementById('checkpoints').innerHTML = table((DATA.checkpoints || []).slice().reverse(), [
        {{label: 'Nombre', key: 'name'}},
        {{label: 'Step', key: 'global_step_from_name'}},
        {{label: 'MB', key: 'size_mb'}},
        {{label: 'Modificado', key: 'modified_at'}},
      ]);
    }}

    function initExplorer() {{
      document.querySelectorAll('[data-explorer-source]').forEach(button => {{
        button.addEventListener('click', () => {{
          explorerSource = button.dataset.explorerSource;
          explorerTrajectoryIndex = 0;
          explorerStepIndex = 0;
          document.querySelectorAll('[data-explorer-source]').forEach(tab => tab.classList.remove('active'));
          button.classList.add('active');
          renderExplorer();
        }});
      }});

      document.getElementById('trajectorySelect').addEventListener('change', event => {{
        explorerTrajectoryIndex = Number(event.target.value || 0);
        explorerStepIndex = 0;
        renderExplorer();
      }});
      document.getElementById('stepRange').addEventListener('input', event => {{
        explorerStepIndex = Number(event.target.value || 0);
        renderExplorerStep();
      }});
      document.getElementById('prevStep').addEventListener('click', () => {{
        explorerStepIndex = Math.max(0, explorerStepIndex - 1);
        renderExplorerStep();
      }});
      document.getElementById('nextStep').addEventListener('click', () => {{
        const trajectory = currentTrajectory();
        const last = Math.max(0, (trajectory?.steps?.length || 1) - 1);
        explorerStepIndex = Math.min(last, explorerStepIndex + 1);
        renderExplorerStep();
      }});
    }}

    function renderExplorer() {{
      const trajectories = explorerTrajectories();
      const select = document.getElementById('trajectorySelect');
      if (!trajectories.length) {{
        select.innerHTML = '';
        document.getElementById('trajectorySummary').textContent = 'Sin trayectorias para este origen.';
        document.getElementById('residualHeatmap').innerHTML = '<p class="subtle">Sin datos.</p>';
        document.getElementById('stateDetails').innerHTML = '';
        document.getElementById('policyDetails').innerHTML = '';
        document.getElementById('trajectoryMetadata').textContent = JSON.stringify(DATA.explorer?.[explorerSource] || {{}}, null, 2);
        return;
      }}

      explorerTrajectoryIndex = Math.min(explorerTrajectoryIndex, trajectories.length - 1);
      select.innerHTML = trajectories.map((trajectory, index) => {{
        const label = `${{trajectory.trajectory_id}} · len=${{trajectory.length}} · reward=${{fmt(trajectory.final_reward)}}`;
        return `<option value="${{index}}">${{label}}</option>`;
      }}).join('');
      select.value = String(explorerTrajectoryIndex);

      const trajectory = currentTrajectory();
      const stepRange = document.getElementById('stepRange');
      stepRange.max = String(Math.max(0, trajectory.steps.length - 1));
      explorerStepIndex = Math.min(explorerStepIndex, trajectory.steps.length - 1);
      stepRange.value = String(explorerStepIndex);
      document.getElementById('trajectorySummary').textContent =
        `${{explorerSource}} · ${{trajectory.trajectory_id}} · steps=${{trajectory.length}}`;
      renderExplorerStep();
    }}

    function renderExplorerStep() {{
      const trajectory = currentTrajectory();
      const step = currentStep();
      if (!trajectory || !step) return;

      const stepRange = document.getElementById('stepRange');
      stepRange.value = String(explorerStepIndex);
      document.getElementById('stepLabel').textContent =
        `step ${{explorerStepIndex + 1}}/${{trajectory.steps.length}} · idx=${{step.step_index}}`;
      renderHeatmap(step.state?.residual_demand || []);
      renderStateDetails(step);
      renderPolicyDetails(step);
      renderTrajectoryMetadata(trajectory, step);
    }}

    function renderHeatmap(matrix) {{
      const el = document.getElementById('residualHeatmap');
      if (!matrix.length) {{
        el.innerHTML = '<p class="subtle">Sin matriz residual.</p>';
        return;
      }}
      const scale = heatmapScaleForTrajectory(currentTrajectory());
      const maxPositive = scale.maxPositive;
      const maxNegative = scale.maxNegative;
      const colorForValue = value => {{
        value = Number(value);
        if (value === 0) return {{background: '#ffffff', color: '#20242a'}};
        if (value < 0) {{
          const t = Math.min(1, Math.abs(value) / Math.max(1, maxNegative));
          const light = 98 - Math.round(t * 42);
          return {{
            background: `hsl(5, 68%, ${{light}}%)`,
            color: t > 0.62 ? '#ffffff' : '#20242a',
          }};
        }}
        const t = Math.min(1, value / Math.max(1, maxPositive));
        const light = 96 - Math.round(t * 42);
        return {{
          background: `hsl(173, 52%, ${{light}}%)`,
          color: t > 0.65 ? '#ffffff' : '#20242a',
        }};
      }};
      const header = '<tr><th>h/d</th>' + matrix[0].map((_, day) => `<th>${{day}}</th>`).join('') + '</tr>';
      const rows = matrix.map((row, hour) => {{
        const cells = row.map(value => {{
          const colors = colorForValue(value);
          return `<td style="background:${{colors.background}};color:${{colors.color}}" title="${{value}}">${{value}}</td>`;
        }}).join('');
        return `<tr><td>${{hour}}</td>${{cells}}</tr>`;
      }}).join('');
      el.innerHTML = `<table class="heatmap-table"><thead>${{header}}</thead><tbody>${{rows}}</tbody></table>`;
    }}

    function heatmapScaleForTrajectory(trajectory) {{
      if (!trajectory?.steps?.length) return {{maxPositive: 1, maxNegative: 1}};
      if (trajectory._heatmapScale) return trajectory._heatmapScale;

      let maxPositive = 0;
      let maxNegative = 0;
      trajectory.steps.forEach(step => {{
        const matrix = step.state?.residual_demand || [];
        matrix.forEach(row => {{
          row.forEach(rawValue => {{
            const value = Number(rawValue);
            if (value > maxPositive) maxPositive = value;
            if (value < 0) maxNegative = Math.max(maxNegative, Math.abs(value));
          }});
        }});
      }});

      trajectory._heatmapScale = {{
        maxPositive: Math.max(1, maxPositive),
        maxNegative: Math.max(1, maxNegative),
      }};
      return trajectory._heatmapScale;
    }}

    function renderStateDetails(step) {{
      const state = step.state || {{}};
      const action = step.action || {{}};
      const trajectory = currentTrajectory() || {{}};
      const setup = trajectory.problem_setup || {{}};
      const attrs = trajectory.attrs || {{}};
      const valueOrSetup = (stateValue, setupValue) => {{
        if (stateValue !== undefined && stateValue !== null) return stateValue;
        if (setupValue !== undefined && setupValue !== null && setupValue !== '__NONE__') return setupValue;
        return undefined;
      }};
      const metadataValue = (stepValue, attrValue) => {{
        if (stepValue !== undefined && stepValue !== null && stepValue !== '') return stepValue;
        if (attrValue !== undefined && attrValue !== null && attrValue !== '__NONE__') return attrValue;
        return undefined;
      }};
      const rows = [
        ['Action', `${{step.action_id}} · ${{action.label || ''}}`],
        ['Action type', action.type],
        ['Reward/value', fmt(step.value ?? step.reward)],
        ['Policy weight', step.policy_weight === undefined ? '-' : fmt(step.policy_weight)],
        ['Selected prob', fmt(step.selected_policy_prob, 6)],
        ['Policy sum', fmt(step.policy_sum, 6)],
        ['Legal count', step.legal_count],
        ['Remaining stock', JSON.stringify(state.remaining_stock)],
        ['Expansion', String(state.expansion_mode)],
        ['Current modality', fmt(state.current_modality)],
        ['Current entry hour', fmt(state.current_entry_hour)],
        ['Assignment week', fmt(state.assignment_week)],
        ['Initial demand total', fmt(state.initial_demand_total)],
        ['Mobile days off', fmt(valueOrSetup(state.mobile_days_off_count, setup.mobile_days_off_count))],
        ['Fixed day off', fmt(valueOrSetup(state.fixed_day_off, setup.fixed_day_off))],
        [
          'Allowed hours',
          valueOrSetup(state.allowed_entry_hours, setup.allowed_entry_hours) === undefined
            ? '-'
            : JSON.stringify(valueOrSetup(state.allowed_entry_hours, setup.allowed_entry_hours)),
        ],
        ['Closing hour', fmt(valueOrSetup(state.closing_hour, setup.closing_hour))],
        ['Sample source', metadataValue(step.sample_source, attrs['metadata.stage']) || '-'],
        [
          'Source trajectory',
          metadataValue(step.source_trajectory_id, attrs['metadata.source_trajectory_id']) || '-',
        ],
      ];
      document.getElementById('stateDetails').innerHTML =
        '<div class="kv">' + rows.map(([key, value]) => `<div>${{key}}</div><div>${{value}}</div>`).join('') + '</div>';
    }}

    function renderPolicyDetails(step) {{
      const entries = step.policy_top || [];
      const maxProb = Math.max(...entries.map(entry => entry.prob), 1e-9);
      document.getElementById('policyDetails').innerHTML = entries.map(entry => {{
        const width = Math.max(1, Math.round((entry.prob / maxProb) * 100));
        const klass = entry.selected ? 'policy-row selected' : 'policy-row';
        return `<div class="${{klass}}">
          <div>#${{entry.action_id}}</div>
          <div>${{entry.action?.label || ''}}</div>
          <div class="policy-bar"><div class="policy-fill" style="width:${{width}}%"></div></div>
          <div>${{fmt(entry.prob, 6)}}</div>
        </div>`;
      }}).join('');
    }}

    function renderTrajectoryMetadata(trajectory, step) {{
      const metadata = {{
        trajectory_id: trajectory.trajectory_id,
        source: explorerSource,
        length: trajectory.length,
        final_reward: trajectory.final_reward,
        sample_sources: trajectory.sample_sources,
        source_trajectory_ids: trajectory.source_trajectory_ids,
        problem_setup: trajectory.problem_setup,
        attrs: trajectory.attrs,
        current_step: {{
          sample_index: step.sample_index,
          step_index: step.step_index,
          model_version: step.model_version,
        }},
      }};
      document.getElementById('trajectoryMetadata').textContent = JSON.stringify(metadata, null, 2);
    }}

    function render() {{
      document.getElementById('generatedAt').textContent = DATA.generated_at;
      renderCards();
      const steps = rowsForRun(DATA.logs.learner_steps);
      lineChart('lossChart', steps, [
        {{key: 'loss', label: 'loss', color: '#116a5b'}},
        {{key: 'policy_loss', label: 'policy', color: '#b54708'}},
        {{key: 'value_loss', label: 'value', color: '#344054'}},
      ]);
      barChart('cycleChart', rowsForRun(DATA.logs.cycles));
      renderTables();
      renderBuffers();
      renderSampleBuffer();
      renderCheckpoints();
      renderExplorer();
      document.getElementById('rawDetails').textContent = JSON.stringify({{
        paths: DATA.paths,
        derived: DATA.derived,
        latest_run: DATA.derived.latest_run,
        latest_cycle: DATA.derived.latest_cycle,
      }}, null, 2);
    }}

    initRunSelect();
    initExplorer();
    render();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

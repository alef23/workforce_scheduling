from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from build_training_dashboard import (
    build_derived_summary,
    read_test_evaluation_reports,
    read_training_logs,
    render_overview_dashboard,
    summarize_checkpoints,
)
from partial_evaluation_dashboard import (
    merge_partial_evaluation_metadata,
    read_partial_test_dataset,
    render_partial_evaluation_dashboard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye el dashboard liviano de entrenamiento y evaluacion."
    )
    parser.add_argument(
        "--reports-dir",
        default="datasets/reports",
        help="Directorio con logs JSONL de entrenamiento.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="modules/evaluators/resnet/checkpoints",
        help="Directorio de checkpoints ResNet.",
    )
    parser.add_argument(
        "--partial-eval-reports-dir",
        default="datasets/evaluation/mcts_partial/reports",
        help="Directorio con reportes de evaluacion MCTS parcial.",
    )
    parser.add_argument(
        "--partial-test-dataset-path",
        default="datasets/test/partial_trajectories.zarr",
        help="Buffer fijo de trayectorias completas del test parcial.",
    )
    parser.add_argument(
        "--partial-eval-trajectory-path",
        default="datasets/evaluation/mcts_partial/trajectories.zarr",
        help="Ruta informativa al buffer de trayectorias de evaluacion parcial.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="HTML destino. Default: <reports-dir>/model_dashboard.html.",
    )
    parser.add_argument(
        "--partial-output",
        default=None,
        help=(
            "HTML de evaluacion parcial. "
            "Default: <reports-dir>/partial_evaluation_dashboard.html."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    output_path = (
        Path(args.output) if args.output else reports_dir / "model_dashboard.html"
    )
    partial_output_path = (
        Path(args.partial_output)
        if args.partial_output
        else reports_dir / "partial_evaluation_dashboard.html"
    )
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "reports_dir": str(reports_dir),
            "checkpoint_dir": str(args.checkpoint_dir),
            "partial_eval_reports_dir": str(args.partial_eval_reports_dir),
            "partial_eval_trajectory_path": str(args.partial_eval_trajectory_path),
            "partial_dashboard_output": str(partial_output_path),
        },
        "logs": read_training_logs(reports_dir),
        "checkpoints": summarize_checkpoints(args.checkpoint_dir),
    }
    data["derived"] = build_derived_summary(data)
    partial_dataset = read_partial_test_dataset(args.partial_test_dataset_path)
    partial_evaluation = read_test_evaluation_reports(
        reports_dir=Path(args.partial_eval_reports_dir),
        trajectory_path=args.partial_eval_trajectory_path,
    )
    partial_data = {
        "generated_at": data["generated_at"],
        "dataset": partial_dataset,
        "evaluation": merge_partial_evaluation_metadata(
            dataset=partial_dataset,
            evaluation=partial_evaluation,
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_overview_dashboard(data), encoding="utf-8")
    partial_output_path.write_text(
        render_partial_evaluation_dashboard(partial_data),
        encoding="utf-8",
    )
    print(f"[model_dashboard] output={output_path}", flush=True)
    print(f"[model_dashboard] partial_output={partial_output_path}", flush=True)


if __name__ == "__main__":
    main()

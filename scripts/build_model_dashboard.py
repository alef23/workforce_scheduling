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
        "--test-eval-reports-dir",
        default="datasets/evaluation/mcts_test/reports",
        help="Directorio con reportes JSON de evaluacion MCTS.",
    )
    parser.add_argument(
        "--test-eval-trajectory-path",
        default="datasets/evaluation/mcts_test/trajectories.zarr",
        help="Ruta informativa al buffer de trayectorias evaluadas; no se abre.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="HTML destino. Default: <reports-dir>/model_dashboard.html.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    output_path = (
        Path(args.output) if args.output else reports_dir / "model_dashboard.html"
    )
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "reports_dir": str(reports_dir),
            "checkpoint_dir": str(args.checkpoint_dir),
            "test_eval_reports_dir": str(args.test_eval_reports_dir),
            "test_eval_trajectory_path": str(args.test_eval_trajectory_path),
        },
        "logs": read_training_logs(reports_dir),
        "test_evaluation": read_test_evaluation_reports(
            reports_dir=Path(args.test_eval_reports_dir),
            trajectory_path=args.test_eval_trajectory_path,
        ),
        "checkpoints": summarize_checkpoints(args.checkpoint_dir),
    }
    data["derived"] = build_derived_summary(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_overview_dashboard(data), encoding="utf-8")
    print(f"[model_dashboard] output={output_path}", flush=True)


if __name__ == "__main__":
    main()

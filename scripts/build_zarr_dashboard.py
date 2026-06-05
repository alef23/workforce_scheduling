from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from build_training_dashboard import (
    load_sample_explorer,
    load_trajectory_explorer,
    render_dashboard,
    summarize_sample_buffer,
    summarize_trajectory_buffer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye el dashboard de inspeccion de buffers Zarr."
    )
    parser.add_argument(
        "--reports-dir",
        default="datasets/reports",
        help="Directorio de salida por defecto.",
    )
    parser.add_argument(
        "--raw-path",
        default="datasets/raw/trajectories.zarr",
        help="TrajectoryBuffer raw.",
    )
    parser.add_argument(
        "--stock-path",
        default="datasets/derived/stock_adjusted/trajectories.zarr",
        help="TrajectoryBuffer stock_adjusted.",
    )
    parser.add_argument(
        "--sample-path",
        default="datasets/samples/samples.zarr",
        help="SampleBuffer actual.",
    )
    parser.add_argument(
        "--test-eval-trajectory-path",
        default="datasets/evaluation/mcts_test/trajectories.zarr",
        help="TrajectoryBuffer de evaluacion MCTS.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="HTML destino. Default: <reports-dir>/zarr_dashboard.html.",
    )
    parser.add_argument(
        "--max-sample-scan",
        type=int,
        default=200_000,
        help="Maximo de samples a leer para resumen.",
    )
    parser.add_argument(
        "--max-trajectory-preview",
        type=int,
        default=20,
        help="Cantidad de trayectorias a listar como preview.",
    )
    parser.add_argument(
        "--explorer-trajectory-count",
        type=int,
        default=10,
        help="Cantidad de ultimas trayectorias por buffer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    output_path = (
        Path(args.output) if args.output else reports_dir / "zarr_dashboard.html"
    )
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "raw_path": str(args.raw_path),
            "stock_path": str(args.stock_path),
            "sample_path": str(args.sample_path),
            "test_eval_trajectory_path": str(args.test_eval_trajectory_path),
        },
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
            "test_mcts": summarize_trajectory_buffer(
                args.test_eval_trajectory_path,
                max_preview=int(args.max_trajectory_preview),
            ),
        },
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
            "test_mcts": load_trajectory_explorer(
                args.test_eval_trajectory_path,
                limit=int(args.explorer_trajectory_count),
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dashboard(data), encoding="utf-8")
    print(f"[zarr_dashboard] output={output_path}", flush=True)


if __name__ == "__main__":
    main()

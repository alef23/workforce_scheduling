from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_generation import (
    DatasetGenerationConfig,
    NoiseGenerationConfig,
    ProblemSetupSamplingConfig,
    RawDemandTrajectoryWorker,
    RawStockTrajectoryWorker,
    ResourceSamplingConfig,
    StockAdjustmentConfig,
    TrajectoryDatasetOrchestrator,
    build_generation_jobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera el buffer fijo de trayectorias completas para evaluacion parcial."
        )
    )
    parser.add_argument(
        "n_samples",
        type=int,
        help="Cantidad de trayectorias completas a generar.",
    )
    parser.add_argument(
        "--output-path",
        default="datasets/test/partial_trajectories.zarr",
        help=(
            "TrajectoryBuffer destino. "
            "Default: datasets/test/partial_trajectories.zarr."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Cantidad de procesos workers en paralelo. Default: 4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recrea el buffer fijo si ya existe.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Frecuencia de impresion de avance. Default: 25 jobs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed base para un test set reproducible. Default: 12345.",
    )
    parser.add_argument(
        "--p-stock",
        type=float,
        default=0.2,
        help="Probabilidad de reducir stock en cada trayectoria. Default: 0.2.",
    )
    parser.add_argument(
        "--allowed-entry-hours",
        type=int,
        nargs="*",
        default=[6, 12, 18],
        help="Horas permitidas. Default: 6 12 18.",
    )
    parser.add_argument(
        "--closing-hour",
        type=int,
        default=22,
        help="Hora de cierre. Default: 22.",
    )
    parser.add_argument(
        "--max-overcoverage-tolerance",
        type=float,
        default=0.1,
        help="Tolerancia maxima de sobrecobertura. Default: 0.1.",
    )
    parser.add_argument(
        "--noise-k-max",
        type=float,
        default=0.8,
        help="K maximo del generador de ruido. Default: 0.8.",
    )
    parser.add_argument(
        "--noise-k-lambda",
        type=float,
        default=10.0,
        help="Lambda de la exponencial truncada para samplear k. Default: 10.0.",
    )
    parser.add_argument("--mod-4-max", type=int, default=20)
    parser.add_argument("--mod-6-max", type=int, default=20)
    parser.add_argument("--mod-8-max", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.n_samples) <= 0:
        raise ValueError("n_samples debe ser positivo.")

    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"El test set ya existe: {output_path}. Usa --overwrite para recrearlo."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    worker = RawStockTrajectoryWorker(
        raw_worker=RawDemandTrajectoryWorker(
            setup_config=ProblemSetupSamplingConfig(
                allowed_entry_hours=args.allowed_entry_hours,
                closing_hour=args.closing_hour,
                max_overcoverage_tolerance=args.max_overcoverage_tolerance,
            ),
            resource_config=ResourceSamplingConfig(
                mod_4_max=args.mod_4_max,
                mod_6_max=args.mod_6_max,
                mod_8_max=args.mod_8_max,
            ),
            noise_config=NoiseGenerationConfig(
                k_max=args.noise_k_max,
                k_exponential_lambda=args.noise_k_lambda,
            ),
            trajectory_id_prefix="partial_raw",
        ),
        stock_config=StockAdjustmentConfig(p_stock=args.p_stock),
        trajectory_id_prefix="partial",
    )
    orchestrator = TrajectoryDatasetOrchestrator(
        config=DatasetGenerationConfig(
            output_path=str(output_path),
            n_workers=args.workers,
            overwrite=args.overwrite,
            progress_interval=args.progress_interval,
        ),
        worker=worker,
    )
    jobs = build_generation_jobs(n_jobs=args.n_samples, seed=args.seed)

    print(f"[partial_test_set] output={output_path}", flush=True)
    print(f"[partial_test_set] n_samples={args.n_samples}", flush=True)
    print(f"[partial_test_set] seed={args.seed}", flush=True)
    print(f"[partial_test_set] workers={args.workers}", flush=True)
    print(f"[partial_test_set] p_stock={args.p_stock}", flush=True)

    report = orchestrator.run(jobs)

    print("[partial_test_set] done", flush=True)
    print(f"completed_jobs={report.completed_jobs}", flush=True)
    print(f"failed_jobs={report.failed_jobs}", flush=True)
    print(f"saved_trajectories={report.saved_trajectories}", flush=True)
    print(f"stats={report.stats}", flush=True)
    if report.errors:
        print("errors:", flush=True)
        for error in report.errors[:10]:
            print(f"- {error}", flush=True)


if __name__ == "__main__":
    main()

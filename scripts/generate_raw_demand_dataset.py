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
    ResourceSamplingConfig,
    TrajectoryDatasetOrchestrator,
    build_generation_jobs,
    create_dataset_buffer_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera trayectorias raw resueltas con RawDemandTrajectoryWorker."
    )
    parser.add_argument(
        "n_samples",
        type=int,
        help="Cantidad de trayectorias raw a generar.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Cantidad de procesos workers en paralelo. Default: 4.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets",
        help="Directorio raiz de buffers. Default: datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recrea el buffer raw si ya existe.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Frecuencia de impresion de avance. Default: 100 jobs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opcional para reproducibilidad. Default: no reproducible.",
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
        help="Tolerancia maxima de sobrecobertura del problema. Default: 0.1.",
    )
    parser.add_argument(
        "--noise-k-max",
        type=float,
        default=0.8,
        help="K maximo del generador de ruido. Default: 0.8.",
    )
    parser.add_argument(
        "--mod-4-max",
        type=int,
        default=20,
        help="Stock maximo para recursos de 4 horas. Default: 20.",
    )
    parser.add_argument(
        "--mod-6-max",
        type=int,
        default=20,
        help="Stock maximo para recursos de 6 horas. Default: 20.",
    )
    parser.add_argument(
        "--mod-8-max",
        type=int,
        default=20,
        help="Stock maximo para recursos de 8 horas. Default: 20.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = create_dataset_buffer_layout(args.output_root)

    worker = RawDemandTrajectoryWorker(
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
        ),
        trajectory_id_prefix="raw",
    )

    orchestrator = TrajectoryDatasetOrchestrator(
        config=DatasetGenerationConfig(
            output_path=str(paths.raw_trajectories),
            n_workers=args.workers,
            overwrite=args.overwrite,
            progress_interval=args.progress_interval,
        ),
        worker=worker,
    )

    jobs = build_generation_jobs(
        n_jobs=args.n_samples,
        seed=args.seed,
    )

    print(f"[dataset_generation] output={paths.raw_trajectories}", flush=True)
    print(f"[dataset_generation] workers={args.workers}", flush=True)
    print(f"[dataset_generation] n_samples={args.n_samples}", flush=True)

    report = orchestrator.run(jobs)

    print("[dataset_generation] done", flush=True)
    print(f"completed_jobs={report.completed_jobs}", flush=True)
    print(f"failed_jobs={report.failed_jobs}", flush=True)
    print(f"saved_trajectories={report.saved_trajectories}", flush=True)
    print(f"resource_totals={report.resource_totals}", flush=True)
    print(f"stats={report.stats}", flush=True)

    if report.errors:
        print("errors:", flush=True)
        for error in report.errors[:10]:
            print(f"- {error}", flush=True)


if __name__ == "__main__":
    main()

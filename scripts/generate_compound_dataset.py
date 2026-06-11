from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_generation import (
    CompoundDatasetOrchestrator,
    CompoundFullTrajectoryWorker,
    CompoundOrchestratorConfig,
    NoiseGenerationConfig,
    build_compound_generation_jobs,
)
from modules.workforce_engine.schemas import ProblemSetup


DEFAULT_OUTPUT_PATH = "datasets/compound/trajectories.zarr"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Debe ser un entero positivo.")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("Debe estar entre 0 y 1.")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera trayectorias del dominio compuesto mediante el circuito "
            "raw -> noise -> replay -> stock."
        )
    )
    parser.add_argument(
        "n_samples",
        type=positive_int,
        help="Cantidad de trayectorias finales a generar.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=4,
        help="Cantidad de procesos workers en paralelo. Default: 4.",
    )
    parser.add_argument(
        "--n-resources",
        type=positive_int,
        default=20,
        help=(
            "Máximo de recursos por problema. Cada job samplea uniformemente "
            "entre 1 y este valor. Default: 20."
        ),
    )
    parser.add_argument(
        "--p-stock",
        type=probability,
        default=0.2,
        help="Probabilidad de reducir el stock de una trayectoria. Default: 0.2.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Ruta del buffer Zarr final. "
            f"Default: {DEFAULT_OUTPUT_PATH}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recrea el buffer de trayectorias compuestas si ya existe.",
    )
    parser.add_argument(
        "--progress-interval",
        type=positive_int,
        default=100,
        help="Frecuencia de impresión del avance. Default: 100 jobs.",
    )
    parser.add_argument(
        "--temporal-chunk-size",
        type=positive_int,
        default=128,
        help="Tamaño temporal de los chunks Zarr. Default: 128.",
    )
    parser.add_argument(
        "--noise-k-max",
        type=probability,
        default=0.8,
        help="K máximo del generador de ruido. Default: 0.8.",
    )
    parser.add_argument(
        "--noise-k-lambda",
        type=float,
        default=10.0,
        help="Lambda de la exponencial truncada para samplear k. Default: 10.0.",
    )
    parser.add_argument(
        "--max-overcoverage-tolerance",
        type=probability,
        default=0.1,
        help="Tolerancia máxima de sobrecobertura. Default: 0.1.",
    )
    parser.add_argument(
        "--run-prefix",
        default="compound",
        help="Prefijo de los trajectory_id generados. Default: compound.",
    )
    parser.add_argument(
        "--multiprocessing-start-method",
        default="spawn",
        choices=("spawn", "fork", "forkserver"),
        help="Método de inicio de los workers. Default: spawn.",
    )
    return parser.parse_args(argv)


def run_generation(args: argparse.Namespace):
    if args.n_resources > 20:
        raise ValueError("--n-resources debe estar entre 1 y 20.")
    if args.noise_k_lambda <= 0:
        raise ValueError("--noise-k-lambda debe ser positivo.")
    if args.noise_k_max <= args.max_overcoverage_tolerance:
        raise ValueError(
            "--noise-k-max debe superar --max-overcoverage-tolerance."
        )
    if not args.run_prefix.strip():
        raise ValueError("--run-prefix no puede estar vacío.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    problem_setup = ProblemSetup(
        mobile_days_off_count=1,
        fixed_day_off=6,
        allowed_entry_hours=[6, 12, 18],
        max_overcoverage_tolerance=args.max_overcoverage_tolerance,
        closing_hour=22,
    )
    worker = CompoundFullTrajectoryWorker(
        problem_setup=problem_setup,
        n_resources=args.n_resources,
        p_stock=args.p_stock,
        noise_config=NoiseGenerationConfig(
            k_max=args.noise_k_max,
            k_exponential_lambda=args.noise_k_lambda,
        ),
        trajectory_id_prefix=args.run_prefix.strip(),
    )
    orchestrator = CompoundDatasetOrchestrator(
        config=CompoundOrchestratorConfig(
            output_path=str(output_path),
            n_workers=args.workers,
            overwrite=args.overwrite,
            progress_interval=args.progress_interval,
            temporal_chunk_size=args.temporal_chunk_size,
            multiprocessing_start_method=args.multiprocessing_start_method,
        ),
        worker=worker,
    )

    jobs = build_compound_generation_jobs(args.n_samples)

    print(f"[compound_dataset] output={output_path}", flush=True)
    print(f"[compound_dataset] workers={args.workers}", flush=True)
    print(f"[compound_dataset] n_samples={args.n_samples}", flush=True)
    print(
        f"[compound_dataset] n_resources_range=1..{args.n_resources}",
        flush=True,
    )
    print(f"[compound_dataset] p_stock={args.p_stock}", flush=True)
    print(
        "[compound_dataset] setup="
        "entry_hours=[6, 12, 18] closing_hour=22 "
        "fixed_day_off=6 mobile_days_off_count=1",
        flush=True,
    )

    return orchestrator.run(jobs)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        report = run_generation(args)
    except (OSError, ValueError) as exc:
        print(f"[compound_dataset] error: {exc}", file=sys.stderr, flush=True)
        return 1

    print("[compound_dataset] done", flush=True)
    print(f"completed_jobs={report.completed_jobs}", flush=True)
    print(f"failed_jobs={report.failed_jobs}", flush=True)
    print(f"saved_trajectories={report.saved_trajectories}", flush=True)
    print(f"stats={report.stats}", flush=True)

    if report.errors:
        print("errors:", flush=True)
        for error in report.errors[:10]:
            print(f"- {error}", flush=True)

    return 0 if report.failed_jobs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

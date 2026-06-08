from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_generation import (
    GenerationJob,
    NoiseGenerationConfig,
    ProblemSetupSamplingConfig,
    RawDemandTrajectoryWorker,
    RawStockTrajectoryWorker,
    ResourceSamplingConfig,
    StockAdjustmentConfig,
    build_generation_jobs,
)
from modules.storage import SampleBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un test set fijo con el estado inicial posterior a raw+noise->stock."
        )
    )
    parser.add_argument(
        "n_samples",
        type=int,
        help="Cantidad de estados iniciales a generar.",
    )
    parser.add_argument(
        "--output-path",
        default="datasets/test/initial_states.zarr",
        help="SampleBuffer destino. Default: datasets/test/initial_states.zarr.",
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
        help="Recrea el test set si ya existe.",
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
        help="Seed base para test set reproducible. Default: 12345.",
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
        "--noise-k-lambda",
        type=float,
        default=10.0,
        help="Lambda de la exponencial truncada para samplear k. Default: 10.0.",
    )
    parser.add_argument(
        "--p-stock",
        type=float,
        default=0.2,
        help="Probabilidad de reducir stock en cada trayectoria. Default: 0.2.",
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
            trajectory_id_prefix="test_raw",
        ),
        stock_config=StockAdjustmentConfig(p_stock=args.p_stock),
        trajectory_id_prefix="test_stock",
    )

    jobs = build_generation_jobs(
        n_jobs=args.n_samples,
        seed=args.seed,
    )

    print(f"[initial_state_test_set] output={output_path}", flush=True)
    print(f"[initial_state_test_set] n_samples={args.n_samples}", flush=True)
    print(f"[initial_state_test_set] seed={args.seed}", flush=True)
    print(f"[initial_state_test_set] workers={args.workers}", flush=True)
    print(f"[initial_state_test_set] p_stock={args.p_stock}", flush=True)

    samples = run_generation_jobs(
        worker=worker,
        jobs=jobs,
        n_workers=args.workers,
        progress_interval=args.progress_interval,
    )

    buffer = SampleBuffer(output_path, mode="w")
    saved_samples = buffer.append_samples(samples)

    print("[initial_state_test_set] done", flush=True)
    print(f"saved_samples={saved_samples}", flush=True)
    print(f"buffer_length={len(buffer)}", flush=True)


def run_generation_jobs(
    worker,
    jobs: Iterable[GenerationJob],
    n_workers: int,
    progress_interval: int,
) -> list[dict]:
    job_list = list(jobs)
    started_at = time.monotonic()
    completed = 0
    failed = 0
    samples = []
    errors = []

    for result in iter_worker_results(worker, job_list, n_workers):
        if isinstance(result, Exception):
            failed += 1
            errors.append(str(result))
        else:
            completed += 1
            generated = result.trajectories[0]
            samples.append(build_initial_state_sample(generated))

        if should_print_progress(completed + failed, len(job_list), progress_interval):
            elapsed = max(1e-9, time.monotonic() - started_at)
            rate = (completed + failed) / elapsed
            print(
                "[initial_state_test_set] "
                f"jobs={completed + failed}/{len(job_list)} "
                f"ok={completed} failed={failed} "
                f"samples={len(samples)} rate={rate:.2f} jobs/s",
                flush=True,
            )

    if errors:
        print("errors:", flush=True)
        for error in errors[:10]:
            print(f"- {error}", flush=True)

    return samples


def iter_worker_results(
    worker: RawDemandTrajectoryWorker,
    jobs: list[GenerationJob],
    n_workers: int,
):
    if int(n_workers) <= 1:
        for job in jobs:
            try:
                yield worker.run(job)
            except Exception as exc:  # pragma: no cover - defensive boundary
                yield exc
        return

    with ProcessPoolExecutor(max_workers=int(n_workers)) as executor:
        futures = [executor.submit(_run_worker_job, worker, job) for job in jobs]
        for future in as_completed(futures):
            try:
                yield future.result()
            except Exception as exc:  # pragma: no cover - defensive boundary
                yield exc


def _run_worker_job(worker: RawDemandTrajectoryWorker, job: GenerationJob):
    return worker.run(job)


def build_initial_state_sample(generated) -> dict:
    if not generated.trajectory:
        raise ValueError("generated.trajectory no puede estar vacia.")

    initial_sample = generated.trajectory[0]
    metadata = dict(generated.metadata)
    trajectory_id = str(generated.trajectory_id)

    metadata.update(
        {
            "sample_source": "test_initial_stock",
            "pipeline": "raw_noise_stock",
            "source_trajectory_id": trajectory_id,
            "stage": "test_initial",
        }
    )

    return {
        "trajectory_id": trajectory_id,
        "step_index": 0,
        "state": initial_sample["state"],
        "problem_setup": generated.problem_setup,
        "policy": initial_sample["policy"],
        "action_id": initial_sample["action_id"],
        "value": float(metadata["final_value"]),
        "policy_weight": 1.0,
        "metadata": metadata,
    }


def should_print_progress(done: int, total: int, progress_interval: int) -> bool:
    if done >= total:
        return True
    if progress_interval <= 0:
        return False
    return done % int(progress_interval) == 0


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_generation import (
    DatasetGenerationConfig,
    StockAdjustmentConfig,
    StockAdjustmentTrajectoryWorker,
    TrajectoryDatasetOrchestrator,
    build_stock_adjustment_jobs,
    create_dataset_buffer_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera trayectorias derivadas ajustando stock desde el buffer raw."
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
        "--source-path",
        default=None,
        help="Buffer raw fuente. Default: <output-root>/raw/trajectories.zarr.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recrea el buffer stock_adjusted si ya existe.",
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
        "--p-stock",
        type=float,
        default=0.2,
        help="Probabilidad de reducir stock en cada trayectoria. Default: 0.2.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Cantidad opcional de trayectorias raw a procesar. Default: todas.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Mezcla IDs fuente antes de aplicar --n-samples.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Saltea raw IDs que ya tengan una trayectoria stock_<raw_id> en output.",
    )
    return parser.parse_args()


def list_source_trajectory_ids(source_path: str | Path) -> list[str]:
    trajectories_dir = Path(source_path) / "trajectories"
    if not trajectories_dir.exists():
        raise FileNotFoundError(
            f"No existe el grupo de trayectorias raw: {trajectories_dir}"
        )

    return sorted(
        path.name
        for path in trajectories_dir.iterdir()
        if path.is_dir() and path.name.startswith("raw_")
    )


def select_source_ids(
    source_ids: list[str],
    n_samples: int | None,
    shuffle: bool,
    seed: int | None,
) -> list[str]:
    ids = list(source_ids)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ids)

    if n_samples is not None:
        ids = ids[: int(n_samples)]

    return ids


def list_existing_stock_trajectory_ids(output_path: str | Path) -> set[str]:
    trajectories_dir = Path(output_path) / "trajectories"
    if not trajectories_dir.exists():
        return set()

    return {
        path.name
        for path in trajectories_dir.iterdir()
        if path.is_dir() and path.name.startswith("stock_")
    }


def filter_existing_source_ids(
    source_ids: list[str],
    existing_stock_ids: set[str],
) -> list[str]:
    return [
        source_id
        for source_id in source_ids
        if f"stock_{source_id}" not in existing_stock_ids
    ]


def main() -> None:
    args = parse_args()
    paths = create_dataset_buffer_layout(args.output_root)

    source_path = Path(args.source_path) if args.source_path else paths.raw_trajectories
    source_ids = list_source_trajectory_ids(source_path)
    selected_ids = select_source_ids(
        source_ids=source_ids,
        n_samples=args.n_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    selected_before_skip = len(selected_ids)

    if args.skip_existing and not args.overwrite:
        existing_stock_ids = list_existing_stock_trajectory_ids(paths.stock_trajectories)
        selected_ids = filter_existing_source_ids(
            source_ids=selected_ids,
            existing_stock_ids=existing_stock_ids,
        )
    elif args.skip_existing and args.overwrite:
        print(
            "[dataset_generation] --skip-existing se ignora porque --overwrite recrea el buffer.",
            flush=True,
        )

    worker = StockAdjustmentTrajectoryWorker(
        source_buffer_path=source_path,
        config=StockAdjustmentConfig(p_stock=args.p_stock),
        trajectory_id_prefix="stock",
    )

    orchestrator = TrajectoryDatasetOrchestrator(
        config=DatasetGenerationConfig(
            output_path=str(paths.stock_trajectories),
            n_workers=args.workers,
            overwrite=args.overwrite,
            progress_interval=args.progress_interval,
        ),
        worker=worker,
    )

    jobs = build_stock_adjustment_jobs(
        source_trajectory_ids=selected_ids,
        seed=args.seed,
    )

    print(f"[dataset_generation] source={source_path}", flush=True)
    print(f"[dataset_generation] output={paths.stock_trajectories}", flush=True)
    print(f"[dataset_generation] workers={args.workers}", flush=True)
    print(f"[dataset_generation] p_stock={args.p_stock}", flush=True)
    print(f"[dataset_generation] source_trajectories={len(source_ids)}", flush=True)
    print(f"[dataset_generation] selected_trajectories={len(selected_ids)}", flush=True)
    if args.skip_existing and not args.overwrite:
        skipped = selected_before_skip - len(selected_ids)
        print(f"[dataset_generation] skipped_existing={skipped}", flush=True)

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

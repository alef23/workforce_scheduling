from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetBufferPaths:
    root: Path
    raw_trajectories: Path
    stock_trajectories: Path
    mcts_trajectories: Path
    samples: Path
    reports: Path


def build_dataset_buffer_paths(root: str | Path = "datasets") -> DatasetBufferPaths:
    root_path = Path(root)
    return DatasetBufferPaths(
        root=root_path,
        raw_trajectories=root_path / "raw" / "trajectories.zarr",
        stock_trajectories=root_path / "derived" / "stock_adjusted" / "trajectories.zarr",
        mcts_trajectories=root_path / "derived" / "mcts" / "trajectories.zarr",
        samples=root_path / "samples" / "samples.zarr",
        reports=root_path / "reports",
    )


def create_dataset_buffer_layout(root: str | Path = "datasets") -> DatasetBufferPaths:
    paths = build_dataset_buffer_paths(root)

    for directory in (
        paths.raw_trajectories.parent,
        paths.stock_trajectories.parent,
        paths.mcts_trajectories.parent,
        paths.samples.parent,
        paths.reports,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return paths

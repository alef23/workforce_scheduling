from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr


SCHEMA_VERSION = "2.0"
NONE_ATTR = "__NONE__"


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    problem_setup: dict[str, Any]
    samples: list[dict[str, Any]]
    final_reward: float
    attrs: dict[str, Any]


class TrajectoryBuffer:
    """
    Buffer Zarr de trayectorias completas.

    Guarda y carga datos crudos. No encodea campos para la red.
    """

    def __init__(self, store_path: str | Path, mode: str = "a") -> None:
        self.store_path = Path(store_path)
        self.root = zarr.open_group(store=str(self.store_path), mode=mode)

        if "trajectories" not in self.root:
            if mode == "r":
                raise KeyError("El store no contiene grupo 'trajectories'.")
            self.trajectories_group = self.root.create_group("trajectories")
        else:
            self.trajectories_group = self.root["trajectories"]

    def list_ids(self) -> list[str]:
        return sorted(str(key) for key in self.trajectories_group.group_keys())

    def __len__(self) -> int:
        return len(self.list_ids())

    def save(
        self,
        trajectory: list[dict[str, Any]],
        problem_setup: Any,
        trajectory_id: str | None = None,
        action_space_size: int = 55,
        temporal_chunk_size: int = 128,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if len(trajectory) == 0:
            raise ValueError("trajectory no puede estar vacía.")
        if trajectory_id is None:
            trajectory_id = self._next_trajectory_id()
        if trajectory_id in self.trajectories_group:
            raise ValueError(f"Ya existe una trayectoria con id={trajectory_id}.")

        for step_index, sample in enumerate(trajectory):
            self._validate_sample(sample, step_index, action_space_size)

        group = self.trajectories_group.create_group(trajectory_id)
        setup = self._problem_setup_to_dict(problem_setup)
        T = len(trajectory)
        chunk_t = min(T, int(temporal_chunk_size))

        arrays = self._trajectory_to_arrays(trajectory, action_space_size)

        for name, data in arrays.items():
            chunks = self._chunks_for_array(name, data, chunk_t)
            self._create_array(group, name, data, chunks=chunks)

        group.attrs["trajectory_id"] = trajectory_id
        group.attrs["length"] = T
        group.attrs["action_space_size"] = int(action_space_size)
        group.attrs["final_reward"] = float(arrays["reward"][-1])
        group.attrs["schema_version"] = SCHEMA_VERSION

        for key, value in setup.items():
            group.attrs[f"problem_setup.{key}"] = self._none_to_attr(value)

        if metadata is not None:
            for key, value in metadata.items():
                group.attrs[f"metadata.{key}"] = self._to_attr_value(value)

        return trajectory_id

    def load(self, trajectory_id: str) -> TrajectoryRecord:
        group = self.trajectories_group[str(trajectory_id)]
        T = int(group.attrs["length"])

        problem_setup = self._read_problem_setup(group)
        samples = []

        for step_index in range(T):
            state = {
                "residual_demand": group["residual_demand"][step_index].astype(np.int32),
                "remaining_stock": group["remaining_stock"][step_index].astype(np.int32),
                "expansion_mode": bool(group["expansion_mode"][step_index]),
                "current_modality": self._minus_one_to_none(
                    group["current_modality"][step_index]
                ),
                "current_entry_hour": self._minus_one_to_none(
                    group["current_entry_hour"][step_index]
                ),
                "assignment_week": int(group["assignment_week"][step_index]),
                "initial_demand_total": int(group["initial_demand_total"][step_index]),
            }

            samples.append(
                {
                    "step_index": step_index,
                    "state": state,
                    "policy": group["policy"][step_index].astype(np.float32),
                    "action_id": int(group["action_id"][step_index]),
                    "reward": float(group["reward"][step_index]),
                }
            )

        return TrajectoryRecord(
            trajectory_id=str(trajectory_id),
            problem_setup=problem_setup,
            samples=samples,
            final_reward=float(group.attrs["final_reward"]),
            attrs=dict(group.attrs),
        )

    def iter_trajectories(
        self,
        trajectory_ids: list[str] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterator[TrajectoryRecord]:
        ids = self._select_ids(trajectory_ids, shuffle, seed)
        for trajectory_id in ids:
            yield self.load(trajectory_id)

    def iter_batches(
        self,
        batch_size: int,
        trajectory_ids: list[str] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> Iterator[list[TrajectoryRecord]]:
        ids = self._select_ids(trajectory_ids, shuffle, seed)
        for start in range(0, len(ids), int(batch_size)):
            batch_ids = ids[start:start + int(batch_size)]
            if drop_last and len(batch_ids) < int(batch_size):
                break
            yield [self.load(trajectory_id) for trajectory_id in batch_ids]

    def _select_ids(
        self,
        trajectory_ids: list[str] | None,
        shuffle: bool,
        seed: int | None,
    ) -> list[str]:
        ids = self.list_ids() if trajectory_ids is None else [str(i) for i in trajectory_ids]
        if shuffle:
            rng = np.random.default_rng(seed)
            ids = list(rng.permutation(ids))
        return ids

    def _next_trajectory_id(self) -> str:
        numeric_ids = []
        for key in self.list_ids():
            try:
                numeric_ids.append(int(key))
            except ValueError:
                continue
        if not numeric_ids:
            return "000000"
        return f"{max(numeric_ids) + 1:06d}"

    @classmethod
    def _trajectory_to_arrays(
        cls,
        trajectory: list[dict[str, Any]],
        action_space_size: int,
    ) -> dict[str, np.ndarray]:
        T = len(trajectory)
        arrays = {
            "residual_demand": np.zeros((T, 24, 28), dtype=np.int32),
            "remaining_stock": np.zeros((T, 3), dtype=np.int32),
            "expansion_mode": np.zeros((T,), dtype=bool),
            "current_modality": np.zeros((T,), dtype=np.int32),
            "current_entry_hour": np.zeros((T,), dtype=np.int32),
            "assignment_week": np.zeros((T,), dtype=np.int32),
            "initial_demand_total": np.zeros((T,), dtype=np.int64),
            "policy": np.zeros((T, action_space_size), dtype=np.float32),
            "action_id": np.zeros((T,), dtype=np.int32),
            "reward": np.zeros((T,), dtype=np.float32),
        }

        for step_index, sample in enumerate(trajectory):
            state = sample["state"]
            arrays["residual_demand"][step_index] = np.asarray(
                cls._get_state_value(state, "residual_demand"),
                dtype=np.int32,
            )
            arrays["remaining_stock"][step_index] = np.asarray(
                cls._get_state_value(state, "remaining_stock"),
                dtype=np.int32,
            )
            arrays["expansion_mode"][step_index] = bool(
                cls._get_state_value(state, "expansion_mode")
            )
            arrays["current_modality"][step_index] = cls._none_to_minus_one(
                cls._get_state_value(state, "current_modality")
            )
            arrays["current_entry_hour"][step_index] = cls._none_to_minus_one(
                cls._get_state_value(state, "current_entry_hour")
            )
            arrays["assignment_week"][step_index] = int(
                cls._get_state_value(state, "assignment_week")
            )
            arrays["initial_demand_total"][step_index] = int(
                cls._get_state_value(state, "initial_demand_total")
            )
            arrays["policy"][step_index] = np.asarray(
                sample["policy"],
                dtype=np.float32,
            )
            arrays["action_id"][step_index] = int(sample["action_id"])
            arrays["reward"][step_index] = float(sample["reward"])

        return arrays

    @classmethod
    def _validate_sample(
        cls,
        sample: dict[str, Any],
        step_index: int,
        action_space_size: int,
    ) -> None:
        required_keys = {"state", "policy", "action_id", "reward"}
        missing_keys = required_keys - set(sample.keys())
        if missing_keys:
            raise ValueError(
                f"Sample {step_index} no contiene claves requeridas: {missing_keys}"
            )

        state = sample["state"]
        required_state_keys = {
            "residual_demand",
            "remaining_stock",
            "expansion_mode",
            "current_modality",
            "current_entry_hour",
            "assignment_week",
            "initial_demand_total",
        }
        missing_state_keys = [
            key for key in required_state_keys if not cls._has_state_value(state, key)
        ]
        if missing_state_keys:
            raise ValueError(
                f"State de sample {step_index} no contiene: {missing_state_keys}"
            )

        if np.asarray(cls._get_state_value(state, "residual_demand")).shape != (24, 28):
            raise ValueError("residual_demand debe tener shape (24, 28).")
        if np.asarray(cls._get_state_value(state, "remaining_stock")).shape != (3,):
            raise ValueError("remaining_stock debe tener shape (3,).")
        if np.asarray(sample["policy"]).shape != (action_space_size,):
            raise ValueError(f"policy debe tener shape ({action_space_size},).")

    @staticmethod
    def _chunks_for_array(name: str, data: np.ndarray, chunk_t: int):
        if data.ndim == 3:
            return (chunk_t, data.shape[1], data.shape[2])
        if data.ndim == 2:
            return (chunk_t, data.shape[1])
        return (chunk_t,)

    @staticmethod
    def _create_array(group, name: str, data: np.ndarray, chunks=None):
        if hasattr(group, "create_array"):
            return group.create_array(
                name=name,
                data=data,
                chunks=chunks,
                overwrite=True,
            )
        return group.create_dataset(
            name=name,
            data=data,
            chunks=chunks,
            overwrite=True,
        )

    @staticmethod
    def _problem_setup_to_dict(problem_setup: Any) -> dict[str, Any]:
        if hasattr(problem_setup, "model_dump"):
            data = problem_setup.model_dump()
        elif hasattr(problem_setup, "dict"):
            data = problem_setup.dict()
        elif isinstance(problem_setup, dict):
            data = problem_setup
        else:
            data = {
                "mobile_days_off_count": problem_setup.mobile_days_off_count,
                "fixed_day_off": problem_setup.fixed_day_off,
                "allowed_entry_hours": problem_setup.allowed_entry_hours,
                "max_overcoverage_tolerance": problem_setup.max_overcoverage_tolerance,
                "closing_hour": problem_setup.closing_hour,
            }

        return {
            "mobile_days_off_count": int(data["mobile_days_off_count"]),
            "fixed_day_off": data["fixed_day_off"],
            "allowed_entry_hours": data["allowed_entry_hours"],
            "max_overcoverage_tolerance": float(data["max_overcoverage_tolerance"]),
            "closing_hour": data["closing_hour"],
        }

    @staticmethod
    def _read_problem_setup(group) -> dict[str, Any]:
        return {
            "mobile_days_off_count": int(
                group.attrs["problem_setup.mobile_days_off_count"]
            ),
            "fixed_day_off": TrajectoryBuffer._restore_attr(
                group.attrs["problem_setup.fixed_day_off"]
            ),
            "allowed_entry_hours": TrajectoryBuffer._restore_attr(
                group.attrs["problem_setup.allowed_entry_hours"]
            ),
            "max_overcoverage_tolerance": float(
                group.attrs["problem_setup.max_overcoverage_tolerance"]
            ),
            "closing_hour": TrajectoryBuffer._restore_attr(
                group.attrs["problem_setup.closing_hour"]
            ),
        }

    @staticmethod
    def _has_state_value(state: Any, key: str) -> bool:
        if isinstance(state, dict):
            return key in state
        return hasattr(state, key)

    @staticmethod
    def _get_state_value(state: Any, key: str) -> Any:
        if isinstance(state, dict):
            return state[key]
        return getattr(state, key)

    @staticmethod
    def _none_to_minus_one(value: Any) -> int:
        return -1 if value is None else int(value)

    @staticmethod
    def _minus_one_to_none(value: Any) -> int | None:
        value = int(value)
        return None if value == -1 else value

    @staticmethod
    def _none_to_attr(value: Any) -> Any:
        return NONE_ATTR if value is None else value

    @staticmethod
    def _restore_attr(value: Any) -> Any:
        return None if value == NONE_ATTR else value

    @staticmethod
    def _to_attr_value(value: Any) -> Any:
        if value is None:
            return NONE_ATTR
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return [TrajectoryBuffer._to_attr_value(item) for item in value]
        if isinstance(value, list):
            return [TrajectoryBuffer._to_attr_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): TrajectoryBuffer._to_attr_value(item)
                for key, item in value.items()
            }
        return value

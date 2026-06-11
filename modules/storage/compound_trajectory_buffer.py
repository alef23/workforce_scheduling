from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr


SCHEMA_VERSION = "compound-1.0"
NONE_ATTR = "__NONE__"
ACTION_SPACE_SIZE = 54


@dataclass(frozen=True)
class CompoundTrajectoryRecord:
    trajectory_id: str
    problem_setup: dict[str, Any]
    samples: list[dict[str, Any]]
    final_reward: float
    attrs: dict[str, Any]


class CompoundTrajectoryBuffer:
    """Buffer Zarr para trayectorias del dominio de acciones compuestas."""

    def __init__(self, store_path: str | Path, mode: str = "a") -> None:
        self.store_path = Path(store_path)
        self.root = zarr.open_group(store=str(self.store_path), mode=mode)

        if "trajectories" not in self.root:
            if mode == "r":
                raise KeyError("El store no contiene grupo 'trajectories'.")
            self.trajectories_group = self.root.create_group("trajectories")
            self.root.attrs["schema_version"] = SCHEMA_VERSION
            self.root.attrs["domain"] = "compound_actions"
            self.root.attrs["action_space_size"] = ACTION_SPACE_SIZE
        else:
            self.trajectories_group = self.root["trajectories"]
            self._validate_root()

    def __len__(self) -> int:
        return len(self.list_ids())

    def list_ids(self) -> list[str]:
        return sorted(str(key) for key in self.trajectories_group.group_keys())

    def save(
        self,
        trajectory: list[dict[str, Any]],
        problem_setup: Any,
        trajectory_id: str,
        metadata: dict[str, Any] | None = None,
        temporal_chunk_size: int = 128,
    ) -> str:
        if not trajectory:
            raise ValueError("trajectory no puede estar vacía.")
        if trajectory_id in self.trajectories_group:
            raise ValueError(f"Ya existe una trayectoria con id={trajectory_id}.")

        for step_index, sample in enumerate(trajectory):
            self._validate_sample(sample, step_index)

        arrays = self._trajectory_to_arrays(trajectory)
        group = self.trajectories_group.create_group(str(trajectory_id))
        chunk_t = min(len(trajectory), int(temporal_chunk_size))
        for name, data in arrays.items():
            self._create_array(
                group,
                name,
                data,
                chunks=self._chunks_for_array(data, chunk_t),
            )

        group.attrs["trajectory_id"] = str(trajectory_id)
        group.attrs["length"] = len(trajectory)
        group.attrs["action_space_size"] = ACTION_SPACE_SIZE
        group.attrs["schema_version"] = SCHEMA_VERSION
        group.attrs["final_reward"] = float(arrays["reward"][-1])

        setup = self._problem_setup_to_dict(problem_setup)
        for key, value in setup.items():
            group.attrs[f"problem_setup.{key}"] = self._to_attr_value(value)
        for key, value in (metadata or {}).items():
            group.attrs[f"metadata.{key}"] = self._to_attr_value(value)

        return str(trajectory_id)

    def load(self, trajectory_id: str) -> CompoundTrajectoryRecord:
        group = self.trajectories_group[str(trajectory_id)]
        length = int(group.attrs["length"])
        samples: list[dict[str, Any]] = []

        for step_index in range(length):
            samples.append(
                {
                    "step_index": step_index,
                    "state": {
                        "residual_demand": group["residual_demand"][
                            step_index
                        ].astype(np.int32),
                        "remaining_stock": group["remaining_stock"][
                            step_index
                        ].astype(np.int32),
                        "expansion_mode": bool(
                            group["expansion_mode"][step_index]
                        ),
                        "current_modality": self._minus_one_to_none(
                            group["current_modality"][step_index]
                        ),
                        "assignment_week": int(
                            group["assignment_week"][step_index]
                        ),
                        "initial_demand_total": int(
                            group["initial_demand_total"][step_index]
                        ),
                    },
                    "policy": group["policy"][step_index].astype(np.float32),
                    "action_id": int(group["action_id"][step_index]),
                    "reward": float(group["reward"][step_index]),
                }
            )

        return CompoundTrajectoryRecord(
            trajectory_id=str(trajectory_id),
            problem_setup=self._read_problem_setup(group),
            samples=samples,
            final_reward=float(group.attrs["final_reward"]),
            attrs=dict(group.attrs),
        )

    def iter_trajectories(
        self,
        trajectory_ids: list[str] | None = None,
    ) -> Iterator[CompoundTrajectoryRecord]:
        ids = self.list_ids() if trajectory_ids is None else trajectory_ids
        for trajectory_id in ids:
            yield self.load(str(trajectory_id))

    @classmethod
    def _trajectory_to_arrays(
        cls,
        trajectory: list[dict[str, Any]],
    ) -> dict[str, np.ndarray]:
        length = len(trajectory)
        arrays = {
            "residual_demand": np.zeros((length, 24, 28), dtype=np.int32),
            "remaining_stock": np.zeros((length, 3), dtype=np.int32),
            "expansion_mode": np.zeros((length,), dtype=bool),
            "current_modality": np.full((length,), -1, dtype=np.int32),
            "assignment_week": np.zeros((length,), dtype=np.int8),
            "initial_demand_total": np.zeros((length,), dtype=np.int64),
            "policy": np.zeros(
                (length, ACTION_SPACE_SIZE),
                dtype=np.float32,
            ),
            "action_id": np.zeros((length,), dtype=np.int16),
            "reward": np.zeros((length,), dtype=np.float32),
        }

        for index, sample in enumerate(trajectory):
            state = sample["state"]
            arrays["residual_demand"][index] = cls._state_value(
                state,
                "residual_demand",
            )
            arrays["remaining_stock"][index] = cls._state_value(
                state,
                "remaining_stock",
            )
            arrays["expansion_mode"][index] = bool(
                cls._state_value(state, "expansion_mode")
            )
            arrays["current_modality"][index] = cls._none_to_minus_one(
                cls._state_value(state, "current_modality")
            )
            arrays["assignment_week"][index] = int(
                cls._state_value(state, "assignment_week")
            )
            arrays["initial_demand_total"][index] = int(
                cls._state_value(state, "initial_demand_total")
            )
            arrays["policy"][index] = np.asarray(
                sample["policy"],
                dtype=np.float32,
            )
            arrays["action_id"][index] = int(sample["action_id"])
            arrays["reward"][index] = float(sample["reward"])

        return arrays

    @classmethod
    def _validate_sample(
        cls,
        sample: dict[str, Any],
        step_index: int,
    ) -> None:
        required = {"state", "policy", "action_id", "reward"}
        missing = required - set(sample)
        if missing:
            raise ValueError(
                f"Sample {step_index} no contiene claves requeridas: {missing}"
            )

        state = sample["state"]
        if np.asarray(cls._state_value(state, "residual_demand")).shape != (
            24,
            28,
        ):
            raise ValueError("residual_demand debe tener shape (24, 28).")
        if np.asarray(cls._state_value(state, "remaining_stock")).shape != (3,):
            raise ValueError("remaining_stock debe tener shape (3,).")
        if np.asarray(sample["policy"]).shape != (ACTION_SPACE_SIZE,):
            raise ValueError("policy debe tener shape (54,).")
        action_id = int(sample["action_id"])
        if action_id < 0 or action_id >= ACTION_SPACE_SIZE:
            raise ValueError("action_id debe estar entre 0 y 53.")

    def _validate_root(self) -> None:
        schema = self.root.attrs.get("schema_version")
        if schema is not None and schema != SCHEMA_VERSION:
            raise ValueError(
                f"Schema incompatible: {schema!r}, esperado {SCHEMA_VERSION!r}."
            )
        action_space = self.root.attrs.get("action_space_size")
        if action_space is not None and int(action_space) != ACTION_SPACE_SIZE:
            raise ValueError("El buffer no utiliza action_space_size=54.")

    @staticmethod
    def _problem_setup_to_dict(problem_setup: Any) -> dict[str, Any]:
        if hasattr(problem_setup, "model_dump"):
            data = problem_setup.model_dump()
        elif isinstance(problem_setup, dict):
            data = problem_setup
        else:
            raise TypeError("problem_setup debe ser un schema o dict.")
        return {
            "mobile_days_off_count": int(data["mobile_days_off_count"]),
            "fixed_day_off": data["fixed_day_off"],
            "allowed_entry_hours": data["allowed_entry_hours"],
            "max_overcoverage_tolerance": float(
                data["max_overcoverage_tolerance"]
            ),
            "closing_hour": data["closing_hour"],
        }

    @staticmethod
    def _read_problem_setup(group) -> dict[str, Any]:
        return {
            "mobile_days_off_count": int(
                group.attrs["problem_setup.mobile_days_off_count"]
            ),
            "fixed_day_off": group.attrs["problem_setup.fixed_day_off"],
            "allowed_entry_hours": list(
                group.attrs["problem_setup.allowed_entry_hours"]
            ),
            "max_overcoverage_tolerance": float(
                group.attrs["problem_setup.max_overcoverage_tolerance"]
            ),
            "closing_hour": group.attrs["problem_setup.closing_hour"],
        }

    @staticmethod
    def _chunks_for_array(data: np.ndarray, chunk_t: int):
        if data.ndim == 3:
            return (chunk_t, data.shape[1], data.shape[2])
        if data.ndim == 2:
            return (chunk_t, data.shape[1])
        return (chunk_t,)

    @staticmethod
    def _create_array(group, name: str, data: np.ndarray, chunks):
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
    def _state_value(state: Any, field: str) -> Any:
        if isinstance(state, dict):
            return state[field]
        return getattr(state, field)

    @staticmethod
    def _none_to_minus_one(value: Any) -> int:
        return -1 if value is None else int(value)

    @staticmethod
    def _minus_one_to_none(value: Any) -> int | None:
        value = int(value)
        return None if value == -1 else value

    @classmethod
    def _to_attr_value(cls, value: Any) -> Any:
        if value is None:
            return NONE_ATTR
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return [cls._to_attr_value(item) for item in value]
        if isinstance(value, list):
            return [cls._to_attr_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): cls._to_attr_value(item)
                for key, item in value.items()
            }
        return value

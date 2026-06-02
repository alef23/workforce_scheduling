from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr

from .trajectory_buffer import TrajectoryBuffer


SCHEMA_VERSION = "1.1"
DEFAULT_POLICY_WEIGHT = 1.0
DEFAULT_SAMPLE_SOURCE = "unknown"
DEFAULT_MODEL_VERSION = -1


@dataclass(frozen=True)
class SampleBatch:
    actions: np.ndarray
    X: dict[str, Any]
    Y: dict[str, np.ndarray]
    metadata: dict[str, np.ndarray]


class SampleBuffer:
    """
    Buffer Zarr de samples aplanados para entrenamiento.

    Guarda X e Y crudos. El encoding queda fuera de este módulo.
    """

    def __init__(self, store_path: str | Path, mode: str = "a") -> None:
        self.store_path = Path(store_path)
        self.root = zarr.open_group(store=str(self.store_path), mode=mode)

        if "samples" not in self.root:
            if mode == "r":
                raise KeyError("El store no contiene grupo 'samples'.")
            self.samples_group = self.root.create_group("samples")
            self.samples_group.attrs["schema_version"] = SCHEMA_VERSION
            self.samples_group.attrs["length"] = 0
        else:
            self.samples_group = self.root["samples"]

    def __len__(self) -> int:
        return int(self.samples_group.attrs.get("length", 0))

    def build_from_trajectory_buffer(
        self,
        trajectory_buffer: TrajectoryBuffer,
        trajectory_ids: list[str] | None = None,
        overwrite: bool = False,
    ) -> int:
        if len(self) > 0 and not overwrite:
            raise ValueError(
                "El sample buffer ya contiene datos. Use overwrite=True para regenerarlo."
            )

        if "samples" in self.root:
            del self.root["samples"]
        self.samples_group = self.root.create_group("samples")
        self.samples_group.attrs["schema_version"] = SCHEMA_VERSION

        records = list(
            trajectory_buffer.iter_trajectories(trajectory_ids=trajectory_ids)
        )
        arrays = self._records_to_arrays(records)

        for name, data in arrays.items():
            chunks = self._chunks_for_array(data)
            self._create_array(self.samples_group, name, data, chunks=chunks)

        self.samples_group.attrs["length"] = int(arrays["action_id"].shape[0])
        return len(self)

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> Iterator[SampleBatch]:
        n_samples = len(self)
        if n_samples == 0:
            return

        indices = np.arange(n_samples)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        for start in range(0, n_samples, int(batch_size)):
            batch_indices = indices[start:start + int(batch_size)]
            if drop_last and len(batch_indices) < int(batch_size):
                break
            yield self.load_batch(batch_indices)

    def load_batch(self, indices: np.ndarray | list[int]) -> SampleBatch:
        idx = np.asarray(indices, dtype=int)
        read_idx, restore_order = self._zarr_read_order(idx)
        g = self.samples_group

        actions = self._restore_order(
            g["action_id"][read_idx].astype(np.int32),
            restore_order,
        )

        X = {
            "residual_demand": self._restore_order(
                g["residual_demand"][read_idx].astype(np.int32),
                restore_order,
            ),
            "remaining_stock": self._restore_order(
                g["remaining_stock"][read_idx].astype(np.int32),
                restore_order,
            ),
            "expansion_mode": self._restore_order(
                g["expansion_mode"][read_idx].astype(bool),
                restore_order,
            ),
            "current_modality": self._restore_order(
                g["current_modality"][read_idx].astype(np.int32),
                restore_order,
            ),
            "current_entry_hour": self._restore_order(
                g["current_entry_hour"][read_idx].astype(np.int32),
                restore_order,
            ),
            "assignment_week": self._restore_order(
                g["assignment_week"][read_idx].astype(np.int32),
                restore_order,
            ),
            "initial_demand_total": self._restore_order(
                g["initial_demand_total"][read_idx].astype(np.int64),
                restore_order,
            ),
            "mobile_days_off_count": self._restore_order(
                g["mobile_days_off_count"][read_idx].astype(np.int32),
                restore_order,
            ),
            "fixed_day_off": self._restore_order(
                g["fixed_day_off"][read_idx].astype(np.int32),
                restore_order,
            ),
            "allowed_entry_hours": [
                self._decode_allowed_entry_hours(mask)
                for mask in self._restore_order(
                    g["allowed_entry_hours_mask"][read_idx].astype(np.int8),
                    restore_order,
                )
            ],
            "max_overcoverage_tolerance": self._restore_order(
                g["max_overcoverage_tolerance"][read_idx].astype(np.float32),
                restore_order,
            ),
            "closing_hour": self._restore_order(
                g["closing_hour"][read_idx].astype(np.int32),
                restore_order,
            ),
        }

        Y = {
            "policy": self._restore_order(
                g["policy"][read_idx].astype(np.float32),
                restore_order,
            ),
            "value": self._restore_order(
                g["value"][read_idx].astype(np.float32),
                restore_order,
            ),
            "policy_weight": self._restore_order(
                self._read_or_default(
                    name="policy_weight",
                    read_idx=read_idx,
                    dtype=np.float32,
                    default_value=DEFAULT_POLICY_WEIGHT,
                ),
                restore_order,
            ),
        }

        metadata = {
            "trajectory_id": self._restore_order(
                g["trajectory_id"][read_idx].astype(str),
                restore_order,
            ),
            "step_index": self._restore_order(
                g["step_index"][read_idx].astype(np.int32),
                restore_order,
            ),
            "sample_source": self._restore_order(
                self._read_or_default(
                    name="sample_source",
                    read_idx=read_idx,
                    dtype=str,
                    default_value=DEFAULT_SAMPLE_SOURCE,
                ),
                restore_order,
            ),
            "source_trajectory_id": self._restore_order(
                self._read_or_default(
                    name="source_trajectory_id",
                    read_idx=read_idx,
                    dtype=str,
                    default_value="",
                ),
                restore_order,
            ),
            "model_version": self._restore_order(
                self._read_or_default(
                    name="model_version",
                    read_idx=read_idx,
                    dtype=np.int32,
                    default_value=DEFAULT_MODEL_VERSION,
                ),
                restore_order,
            ),
            "sample_index": idx.astype(np.int64),
        }

        return SampleBatch(
            actions=actions,
            X=X,
            Y=Y,
            metadata=metadata,
        )

    def append_trajectories(self, trajectories: list[Any]) -> int:
        """
        Aplana trayectorias finalizadas y las agrega al buffer.

        Cada item debe exponer atributos o claves compatibles con:
        - trajectory
        - problem_setup
        - trajectory_id
        - metadata
        """
        samples = []
        for trajectory_record in trajectories:
            trajectory = self._get_value(trajectory_record, "trajectory")
            problem_setup = self._get_value(trajectory_record, "problem_setup")
            trajectory_id = self._get_value(
                trajectory_record,
                "trajectory_id",
                default="",
            )
            metadata = self._get_value(
                trajectory_record,
                "metadata",
                default={},
            )

            for step_index, sample in enumerate(trajectory):
                sample_metadata = dict(metadata)
                sample_metadata.update(sample.get("metadata", {}))
                samples.append(
                    {
                        "trajectory_id": trajectory_id,
                        "step_index": int(sample.get("step_index", step_index)),
                        "state": sample["state"],
                        "problem_setup": problem_setup,
                        "policy": sample["policy"],
                        "action_id": sample["action_id"],
                        "value": sample.get("value", sample.get("reward")),
                        "policy_weight": sample.get(
                            "policy_weight",
                            sample_metadata.get(
                                "policy_weight",
                                DEFAULT_POLICY_WEIGHT,
                            ),
                        ),
                        "metadata": sample_metadata,
                    }
                )

        return self.append_samples(samples)

    def append_samples(self, samples: list[dict[str, Any]]) -> int:
        """
        Agrega samples aplanados al final del buffer.

        Retorna la cantidad de samples agregados.
        """
        if not samples:
            return 0

        arrays = self._samples_to_arrays(samples)
        self._ensure_appendable_arrays(arrays)

        start = len(self)
        end = start + int(arrays["action_id"].shape[0])

        for name, data in arrays.items():
            array = self.samples_group[name]
            array.resize((end,) + array.shape[1:])
            array[start:end] = data

        self.samples_group.attrs["length"] = int(end)
        self.samples_group.attrs["schema_version"] = SCHEMA_VERSION
        return int(arrays["action_id"].shape[0])

    @classmethod
    def _records_to_arrays(cls, records) -> dict[str, np.ndarray]:
        samples = []
        for record in records:
            setup = record.problem_setup
            for sample in record.samples:
                state = sample["state"]
                samples.append(
                    {
                        "trajectory_id": record.trajectory_id,
                        "step_index": int(sample["step_index"]),
                        "state": state,
                        "problem_setup": setup,
                        "policy": sample["policy"],
                        "action_id": sample["action_id"],
                        "value": sample["reward"],
                        "policy_weight": sample.get(
                            "policy_weight",
                            DEFAULT_POLICY_WEIGHT,
                        ),
                        "metadata": {
                            "sample_source": sample.get(
                                "sample_source",
                                "trajectory_buffer",
                            ),
                            "source_trajectory_id": record.trajectory_id,
                            "model_version": sample.get(
                                "model_version",
                                DEFAULT_MODEL_VERSION,
                            ),
                        },
                    }
                )

        return cls._samples_to_arrays(samples)

    @classmethod
    def _samples_to_arrays(cls, samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        n = len(samples)
        arrays = {
            "trajectory_id": np.empty((n,), dtype="U32"),
            "step_index": np.zeros((n,), dtype=np.int32),
            "residual_demand": np.zeros((n, 24, 28), dtype=np.int32),
            "remaining_stock": np.zeros((n, 3), dtype=np.int32),
            "expansion_mode": np.zeros((n,), dtype=bool),
            "current_modality": np.zeros((n,), dtype=np.int32),
            "current_entry_hour": np.zeros((n,), dtype=np.int32),
            "assignment_week": np.zeros((n,), dtype=np.int32),
            "initial_demand_total": np.zeros((n,), dtype=np.int64),
            "mobile_days_off_count": np.zeros((n,), dtype=np.int32),
            "fixed_day_off": np.zeros((n,), dtype=np.int32),
            "allowed_entry_hours_mask": np.zeros((n, 24), dtype=np.int8),
            "max_overcoverage_tolerance": np.zeros((n,), dtype=np.float32),
            "closing_hour": np.zeros((n,), dtype=np.int32),
            "policy": np.zeros((n, 55), dtype=np.float32),
            "action_id": np.zeros((n,), dtype=np.int32),
            "value": np.zeros((n,), dtype=np.float32),
            "policy_weight": np.ones((n,), dtype=np.float32),
            "sample_source": np.empty((n,), dtype="U64"),
            "source_trajectory_id": np.empty((n,), dtype="U64"),
            "model_version": np.full(
                (n,),
                DEFAULT_MODEL_VERSION,
                dtype=np.int32,
            ),
        }

        for i, sample in enumerate(samples):
            state = sample["state"]
            setup = sample["problem_setup"]
            metadata = sample.get("metadata", {})

            arrays["trajectory_id"][i] = sample["trajectory_id"]
            arrays["step_index"][i] = int(sample["step_index"])
            arrays["residual_demand"][i] = np.asarray(
                cls._get_state_value(state, "residual_demand"),
                dtype=np.int32,
            )
            arrays["remaining_stock"][i] = np.asarray(
                cls._get_state_value(state, "remaining_stock"),
                dtype=np.int32,
            )
            arrays["expansion_mode"][i] = bool(
                cls._get_state_value(state, "expansion_mode")
            )
            arrays["current_modality"][i] = cls._none_to_minus_one(
                cls._get_state_value(state, "current_modality")
            )
            arrays["current_entry_hour"][i] = cls._none_to_minus_one(
                cls._get_state_value(state, "current_entry_hour")
            )
            arrays["assignment_week"][i] = int(
                cls._get_state_value(state, "assignment_week")
            )
            arrays["initial_demand_total"][i] = int(
                cls._get_state_value(state, "initial_demand_total")
            )
            arrays["mobile_days_off_count"][i] = int(
                cls._get_setup_value(setup, "mobile_days_off_count")
            )
            arrays["fixed_day_off"][i] = cls._none_to_minus_one(
                cls._get_setup_value(setup, "fixed_day_off")
            )
            arrays["allowed_entry_hours_mask"][i] = cls._encode_allowed_entry_hours(
                cls._get_setup_value(setup, "allowed_entry_hours")
            )
            arrays["max_overcoverage_tolerance"][i] = float(
                cls._get_setup_value(setup, "max_overcoverage_tolerance")
            )
            arrays["closing_hour"][i] = cls._none_to_minus_one(
                cls._get_setup_value(setup, "closing_hour")
            )
            arrays["policy"][i] = np.asarray(sample["policy"], dtype=np.float32)
            arrays["action_id"][i] = int(sample["action_id"])
            arrays["value"][i] = float(sample["value"])
            arrays["policy_weight"][i] = float(
                sample.get("policy_weight", DEFAULT_POLICY_WEIGHT)
            )
            arrays["sample_source"][i] = str(
                metadata.get("sample_source", DEFAULT_SAMPLE_SOURCE)
            )
            arrays["source_trajectory_id"][i] = str(
                metadata.get(
                    "source_trajectory_id",
                    sample.get("trajectory_id", ""),
                )
            )
            arrays["model_version"][i] = int(
                metadata.get("model_version", DEFAULT_MODEL_VERSION)
            )

        return arrays

    def _read_or_default(
        self,
        name: str,
        read_idx: np.ndarray,
        dtype,
        default_value,
    ) -> np.ndarray:
        if name in self.samples_group:
            return self.samples_group[name][read_idx].astype(dtype)
        return np.full((len(read_idx),), default_value, dtype=dtype)

    def _ensure_appendable_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        current_length = len(self)

        for name, data in arrays.items():
            if name not in self.samples_group:
                self._create_empty_array(
                    self.samples_group,
                    name,
                    dtype=data.dtype,
                    shape=(current_length,) + data.shape[1:],
                    chunks=self._chunks_for_array(data),
                )
                if current_length > 0:
                    self.samples_group[name][:] = self._default_array_values(
                        name=name,
                        shape=(current_length,) + data.shape[1:],
                        dtype=data.dtype,
                    )

    @classmethod
    def _default_array_values(
        cls,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> np.ndarray:
        if name == "policy_weight":
            return np.full(shape, DEFAULT_POLICY_WEIGHT, dtype=dtype)
        if name == "model_version":
            return np.full(shape, DEFAULT_MODEL_VERSION, dtype=dtype)
        if name == "sample_source":
            return np.full(shape, DEFAULT_SAMPLE_SOURCE, dtype=dtype)
        if name == "source_trajectory_id":
            return np.full(shape, "", dtype=dtype)
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def _encode_allowed_entry_hours(value: list[int] | None) -> np.ndarray:
        mask = np.zeros(24, dtype=np.int8)
        if value is None:
            mask[:] = 1
            return mask
        for hour in value:
            mask[int(hour)] = 1
        return mask

    @staticmethod
    def _decode_allowed_entry_hours(mask: np.ndarray) -> list[int] | None:
        if np.all(mask == 1):
            return None
        return [int(hour) for hour in np.flatnonzero(mask)]

    @staticmethod
    def _zarr_read_order(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(indices)
        read_idx = indices[order]
        restore_order = np.argsort(order)
        return read_idx, restore_order

    @staticmethod
    def _restore_order(data: np.ndarray, restore_order: np.ndarray) -> np.ndarray:
        return data[restore_order]

    @staticmethod
    def _chunks_for_array(data: np.ndarray):
        if data.ndim == 3:
            return (min(128, data.shape[0]), data.shape[1], data.shape[2])
        if data.ndim == 2:
            return (min(128, data.shape[0]), data.shape[1])
        return (min(128, data.shape[0]),)

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
    def _create_empty_array(
        group,
        name: str,
        dtype,
        shape: tuple[int, ...],
        chunks=None,
    ):
        if hasattr(group, "create_array"):
            return group.create_array(
                name=name,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                overwrite=True,
            )
        return group.create_dataset(
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            overwrite=True,
        )

    @staticmethod
    def _none_to_minus_one(value: Any) -> int:
        return -1 if value is None else int(value)

    @staticmethod
    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _get_state_value(cls, state: Any, key: str) -> Any:
        return cls._get_value(state, key)

    @classmethod
    def _get_setup_value(cls, setup: Any, key: str) -> Any:
        return cls._get_value(setup, key)

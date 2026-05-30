from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr

# ============================================================
# Utilidades generales
# ============================================================

def _create_zarr_array(
    group,
    name: str,
    data: np.ndarray,
    chunks=None,
):
    """
    Crea un array Zarr de forma compatible con distintas versiones de zarr.
    """

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


def _none_to_minus_one(value: Any) -> int:
    """
    Convierte None a -1 para guardar campos opcionales como enteros.
    """

    if value is None:
        return -1

    return int(value)


def _none_to_attr_value(value: Any) -> Any:
    """
    Convierte None en un valor explícito para attrs de Zarr.
    """

    if value is None:
        return "__NONE__"

    return value


def _get_next_trajectory_id(trajectories_group) -> str:
    """
    Genera el próximo ID incremental de trayectoria.

    Ejemplo:
        000000, 000001, 000002, ...
    """

    existing_keys = list(trajectories_group.group_keys())

    if len(existing_keys) == 0:
        return "000000"

    numeric_ids = []

    for key in existing_keys:
        try:
            numeric_ids.append(int(key))
        except ValueError:
            continue

    if len(numeric_ids) == 0:
        return "000000"

    next_id = max(numeric_ids) + 1

    return f"{next_id:06d}"


# ============================================================
# Serialización de ProblemSetup
# ============================================================

def _problem_setup_to_dict(problem_setup: Any) -> dict[str, Any]:
    """
    Convierte ProblemSetup a un diccionario simple.

    Compatible con:
    - Pydantic v2: model_dump()
    - Pydantic v1: dict()
    - Objetos con atributos equivalentes
    """

    if hasattr(problem_setup, "model_dump"):
        data = problem_setup.model_dump()

    elif hasattr(problem_setup, "dict"):
        data = problem_setup.dict()

    else:
        data = {
            "mobile_days_off_count": problem_setup.mobile_days_off_count,
            "fixed_day_off": problem_setup.fixed_day_off,
            "allowed_entry_hours": problem_setup.allowed_entry_hours,
            "max_overcoverage_tolerance": problem_setup.max_overcoverage_tolerance,
            "closing_hour": problem_setup.closing_hour,
        }

    return {
        "mobile_days_off_count": data["mobile_days_off_count"],
        "fixed_day_off": data["fixed_day_off"],
        "allowed_entry_hours": data["allowed_entry_hours"],
        "max_overcoverage_tolerance": data["max_overcoverage_tolerance"],
        "closing_hour": data["closing_hour"],
    }


def _save_problem_setup_attrs(
    trajectory_group,
    problem_setup: Any,
) -> None:
    """
    Guarda ProblemSetup como atributos del grupo Zarr de la trayectoria.
    """

    setup = _problem_setup_to_dict(problem_setup)

    trajectory_group.attrs["problem_setup.mobile_days_off_count"] = int(
        setup["mobile_days_off_count"]
    )

    trajectory_group.attrs["problem_setup.fixed_day_off"] = _none_to_attr_value(
        setup["fixed_day_off"]
    )

    trajectory_group.attrs["problem_setup.allowed_entry_hours"] = _none_to_attr_value(
        setup["allowed_entry_hours"]
    )

    trajectory_group.attrs["problem_setup.max_overcoverage_tolerance"] = float(
        setup["max_overcoverage_tolerance"]
    )

    trajectory_group.attrs["problem_setup.closing_hour"] = _none_to_attr_value(
        setup["closing_hour"]
    )


# ============================================================
# Validaciones
# ============================================================

def _validate_trajectory_sample(
    sample: dict[str, Any],
    t: int,
    action_space_size: int,
) -> None:
    """
    Valida una muestra individual de trayectoria.
    """

    required_keys = {"state", "policy", "action_id", "reward"}

    missing_keys = required_keys - set(sample.keys())

    if missing_keys:
        raise ValueError(
            f"La muestra t={t} no contiene las claves requeridas: {missing_keys}"
        )

    state = sample["state"]

    required_state_attrs = {
        "residual_demand",
        "remaining_stock",
        "expansion_mode",
        "current_modality",
        "current_entry_hour",
        "assignment_week",
        "initial_demand_total",
    }

    missing_state_attrs = [
        attr for attr in required_state_attrs if not hasattr(state, attr)
    ]

    if missing_state_attrs:
        raise ValueError(
            f"El state de la muestra t={t} no contiene atributos: "
            f"{missing_state_attrs}"
        )

    residual_demand = np.asarray(state.residual_demand)

    if residual_demand.shape != (24, 28):
        raise ValueError(
            f"state.residual_demand en t={t} debe tener shape (24, 28), "
            f"pero tiene {residual_demand.shape}."
        )

    remaining_stock = np.asarray(state.remaining_stock)

    if remaining_stock.shape != (3,):
        raise ValueError(
            f"state.remaining_stock en t={t} debe tener shape (3,), "
            f"pero tiene {remaining_stock.shape}."
        )

    policy = np.asarray(sample["policy"])

    if policy.shape != (action_space_size,):
        raise ValueError(
            f"policy en t={t} debe tener shape ({action_space_size},), "
            f"pero tiene {policy.shape}."
        )


# ============================================================
# Función principal
# ============================================================

def save_trajectory_to_zarr(
    trajectory: list[dict[str, Any]],
    store_path: str | Path,
    problem_setup: Any,
    trajectory_id: str | None = None,
    action_space_size: int = 55,
    chunk_size: int = 128,
) -> str:
    """
    Guarda una trayectoria simulada en formato Zarr.

    Cada trayectoria se guarda como un grupo independiente dentro de:

        simulated_trajectories.zarr/
        └── trajectories/
            └── 000000/

    Datasets guardados por trayectoria:
        - residual_demand      shape (T, 24, 28)
        - remaining_stock      shape (T, 3)
        - expansion_mode       shape (T,)
        - current_modality     shape (T,)
        - current_entry_hour   shape (T,)
        - assignment_week      shape (T,)
        - initial_demand_total shape (T,)
        - policy               shape (T, 55)
        - action_id            shape (T,)
        - reward               shape (T,)

    Atributos guardados:
        - trajectory_id
        - length
        - action_space_size
        - final_reward
        - schema_version
        - problem_setup.mobile_days_off_count
        - problem_setup.fixed_day_off
        - problem_setup.allowed_entry_hours
        - problem_setup.max_overcoverage_tolerance
        - problem_setup.closing_hour

    Parameters
    ----------
    trajectory:
        Lista de muestras.

        Cada muestra debe tener:

            {
                "state": WorkforceState,
                "policy": np.ndarray shape (55,),
                "action_id": int,
                "reward": float
            }

    store_path:
        Ruta del store Zarr.

        Ejemplo:
            "simulated_trajectories.zarr"

    problem_setup:
        Objeto ProblemSetup asociado a la trayectoria.

    trajectory_id:
        ID opcional de trayectoria.
        Si no se informa, se genera automáticamente.

    action_space_size:
        Tamaño total del espacio de acciones.
        Por defecto: 55.

    chunk_size:
        Tamaño de chunk sobre la dimensión temporal T.

    Returns
    -------
    trajectory_id:
        ID con el que fue guardada la trayectoria.
    """

    if len(trajectory) == 0:
        raise ValueError("trajectory no puede estar vacía.")

    if not isinstance(action_space_size, int) or action_space_size <= 0:
        raise ValueError("action_space_size debe ser un entero positivo.")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size debe ser un entero positivo.")

    for t, sample in enumerate(trajectory):
        _validate_trajectory_sample(
            sample=sample,
            t=t,
            action_space_size=action_space_size,
        )

    store_path = Path(store_path)

    root = zarr.open_group(
        store=str(store_path),
        mode="a",
    )

    if "trajectories" not in root:
        trajectories_group = root.create_group("trajectories")
    else:
        trajectories_group = root["trajectories"]

    if trajectory_id is None:
        trajectory_id = _get_next_trajectory_id(trajectories_group)

    if trajectory_id in trajectories_group:
        raise ValueError(f"Ya existe una trayectoria con id={trajectory_id}.")

    trajectory_group = trajectories_group.create_group(trajectory_id)

    T = len(trajectory)
    chunk_t = min(T, chunk_size)

    residual_demand = np.zeros((T, 24, 28), dtype=np.int32)
    remaining_stock = np.zeros((T, 3), dtype=np.int32)
    expansion_mode = np.zeros((T,), dtype=bool)
    current_modality = np.zeros((T,), dtype=np.int32)
    current_entry_hour = np.zeros((T,), dtype=np.int32)
    assignment_week = np.zeros((T,), dtype=np.int32)
    initial_demand_total = np.zeros((T,), dtype=np.float32)

    policy = np.zeros((T, action_space_size), dtype=np.float32)
    action_id = np.zeros((T,), dtype=np.int32)
    reward = np.zeros((T,), dtype=np.float32)

    for t, sample in enumerate(trajectory):
        state = sample["state"]

        residual_demand[t] = np.asarray(
            state.residual_demand,
            dtype=np.int32,
        )

        remaining_stock[t] = np.asarray(
            state.remaining_stock,
            dtype=np.int32,
        )

        expansion_mode[t] = bool(state.expansion_mode)

        current_modality[t] = _none_to_minus_one(
            state.current_modality,
        )

        current_entry_hour[t] = _none_to_minus_one(
            state.current_entry_hour,
        )

        assignment_week[t] = int(state.assignment_week)

        initial_demand_total[t] = float(state.initial_demand_total)

        policy[t] = np.asarray(
            sample["policy"],
            dtype=np.float32,
        )

        action_id[t] = int(sample["action_id"])

        reward[t] = float(sample["reward"])

    # ============================================================
    # Datasets del WorkforceState
    # ============================================================

    _create_zarr_array(
        trajectory_group,
        "residual_demand",
        residual_demand,
        chunks=(chunk_t, 24, 28),
    )

    _create_zarr_array(
        trajectory_group,
        "remaining_stock",
        remaining_stock,
        chunks=(chunk_t, 3),
    )

    _create_zarr_array(
        trajectory_group,
        "expansion_mode",
        expansion_mode,
        chunks=(chunk_t,),
    )

    _create_zarr_array(
        trajectory_group,
        "current_modality",
        current_modality,
        chunks=(chunk_t,),
    )

    _create_zarr_array(
        trajectory_group,
        "current_entry_hour",
        current_entry_hour,
        chunks=(chunk_t,),
    )

    _create_zarr_array(
        trajectory_group,
        "assignment_week",
        assignment_week,
        chunks=(chunk_t,),
    )

    _create_zarr_array(
        trajectory_group,
        "initial_demand_total",
        initial_demand_total,
        chunks=(chunk_t,),
    )

    # ============================================================
    # Targets / acciones
    # ============================================================

    _create_zarr_array(
        trajectory_group,
        "policy",
        policy,
        chunks=(chunk_t, action_space_size),
    )

    _create_zarr_array(
        trajectory_group,
        "action_id",
        action_id,
        chunks=(chunk_t,),
    )

    _create_zarr_array(
        trajectory_group,
        "reward",
        reward,
        chunks=(chunk_t,),
    )

    # ============================================================
    # Metadata general
    # ============================================================

    trajectory_group.attrs["trajectory_id"] = trajectory_id
    trajectory_group.attrs["length"] = T
    trajectory_group.attrs["action_space_size"] = action_space_size
    trajectory_group.attrs["final_reward"] = float(reward[-1])
    trajectory_group.attrs["schema_version"] = "1.1"

    _save_problem_setup_attrs(
        trajectory_group=trajectory_group,
        problem_setup=problem_setup,
    )

    return trajectory_id


#######################################################################################################################################################




# ============================================================
# Helpers
# ============================================================

def _restore_none_attr(value: Any) -> Any:
    """
    Restaura valores None guardados como '__NONE__' en attrs.
    """
    if value == "__NONE__":
        return None
    return value


def _attr_none_to_minus_one(value: Any) -> int:
    """
    Convierte atributos opcionales del ProblemSetup a enteros estables.

    None / '__NONE__' -> -1
    entero -> entero
    """
    value = _restore_none_attr(value)

    if value is None:
        return -1

    return int(value)


def _allowed_entry_hours_to_mask(value: Any) -> np.ndarray:
    """
    Convierte allowed_entry_hours a máscara binaria de 24 posiciones.

    Casos:
    - None / '__NONE__' -> todas las horas permitidas.
    - lista de horas -> 1 en horas permitidas, 0 en el resto.
    """
    value = _restore_none_attr(value)

    mask = np.zeros(24, dtype=np.int8)

    if value is None:
        mask[:] = 1
        return mask

    for hour in value:
        hour = int(hour)

        if hour < 0 or hour > 23:
            raise ValueError(f"Hora inválida en allowed_entry_hours: {hour}")

        mask[hour] = 1

    return mask


def _get_trajectory_ids(
    trajectories_group,
    trajectory_ids: list[str] | None,
    shuffle_trajectories: bool,
    seed: int | None,
) -> list[str]:
    """
    Obtiene y opcionalmente mezcla los IDs de trayectoria.
    """
    if trajectory_ids is None:
        ids = sorted(list(trajectories_group.group_keys()))
    else:
        ids = list(trajectory_ids)

    if shuffle_trajectories:
        rng = np.random.default_rng(seed)
        ids = list(rng.permutation(ids))

    return ids


def _split_in_chunks(
    items: list[str],
    chunk_size: int,
    drop_last: bool,
) -> Iterator[list[str]]:
    """
    Parte una lista de trajectory_ids en chunks de tamaño chunk_size.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size debe ser un entero positivo.")

    for start in range(0, len(items), chunk_size):
        end = min(start + chunk_size, len(items))
        chunk = items[start:end]

        if drop_last and len(chunk) < chunk_size:
            break

        yield chunk


def _read_problem_setup_for_samples(g, T: int) -> dict[str, np.ndarray]:
    """
    Lee el ProblemSetup desde attrs de una trayectoria y lo replica T veces.
    """
    mobile_days_off_count = int(
        g.attrs["problem_setup.mobile_days_off_count"]
    )

    fixed_day_off = _attr_none_to_minus_one(
        g.attrs["problem_setup.fixed_day_off"]
    )

    allowed_entry_hours_mask = _allowed_entry_hours_to_mask(
        g.attrs["problem_setup.allowed_entry_hours"]
    )

    max_overcoverage_tolerance = float(
        g.attrs["problem_setup.max_overcoverage_tolerance"]
    )

    closing_hour = _attr_none_to_minus_one(
        g.attrs["problem_setup.closing_hour"]
    )

    return {
        "mobile_days_off_count": np.full(
            shape=(T,),
            fill_value=mobile_days_off_count,
            dtype=np.int32,
        ),
        "fixed_day_off": np.full(
            shape=(T,),
            fill_value=fixed_day_off,
            dtype=np.int32,
        ),
        "allowed_entry_hours": np.repeat(
            allowed_entry_hours_mask[None, :],
            repeats=T,
            axis=0,
        ).astype(np.int8),
        "max_overcoverage_tolerance": np.full(
            shape=(T,),
            fill_value=max_overcoverage_tolerance,
            dtype=np.float32,
        ),
        "closing_hour": np.full(
            shape=(T,),
            fill_value=closing_hour,
            dtype=np.int32,
        ),
    }


def _read_single_trajectory_for_training(g) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Lee una trayectoria completa y devuelve:

    actions:
        action_id en orden.

    X:
        estado crudo extendido = WorkforceState + ProblemSetup.

    Y:
        targets = policy, value.
    """
    T = int(g.attrs["length"])

    actions = g["action_id"][:].astype(np.int32)

    X = {
        # WorkforceState
        "residual_demand": g["residual_demand"][:].astype(np.int32),
        "remaining_stock": g["remaining_stock"][:].astype(np.int32),
        "expansion_mode": g["expansion_mode"][:].astype(bool),
        "current_modality": g["current_modality"][:].astype(np.int32),
        "current_entry_hour": g["current_entry_hour"][:].astype(np.int32),
        "assignment_week": g["assignment_week"][:].astype(np.int32),
        "initial_demand_total": g["initial_demand_total"][:].astype(np.float32),
    }

    problem_setup_X = _read_problem_setup_for_samples(g, T)
    X.update(problem_setup_X)

    Y = {
        "policy": g["policy"][:].astype(np.float32),
        "value": g["reward"][:].astype(np.float32),
    }

    return actions, X, Y


def _concat_chunk(
    chunk_data: list[tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]]
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Concatena varias trayectorias leídas en un único chunk.
    """
    actions_list = [item[0] for item in chunk_data]
    X_list = [item[1] for item in chunk_data]
    Y_list = [item[2] for item in chunk_data]

    actions = np.concatenate(actions_list, axis=0)

    X_keys = X_list[0].keys()
    Y_keys = Y_list[0].keys()

    X = {
        key: np.concatenate([x[key] for x in X_list], axis=0)
        for key in X_keys
    }

    Y = {
        key: np.concatenate([y[key] for y in Y_list], axis=0)
        for key in Y_keys
    }

    return actions, X, Y


# ============================================================
# Loader principal
# ============================================================

def iter_zarr_training_chunks(
    store_path: str | Path,
    chunk_size: int = 1,
    trajectory_ids: list[str] | None = None,
    shuffle_trajectories: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """
    Itera chunks de entrenamiento desde un dataset Zarr.

    Importante:
    - chunk_size representa cantidad de trayectorias por chunk.
    - Cada trayectoria se aplana en samples.
    - Si un chunk contiene N trayectorias y entre todas suman B pasos,
      entonces la salida tendrá B samples.

    Returns
    -------
    actions:
        np.ndarray shape (B,)
        action_id ejecutado en cada sample.

    X:
        dict con estado crudo extendido:
            WorkforceState + ProblemSetup

    Y:
        dict con targets:
            policy, value
    """

    store_path = Path(store_path)

    root = zarr.open_group(
        store=str(store_path),
        mode="r",
    )

    trajectories_group = root["trajectories"]

    ids = _get_trajectory_ids(
        trajectories_group=trajectories_group,
        trajectory_ids=trajectory_ids,
        shuffle_trajectories=shuffle_trajectories,
        seed=seed,
    )

    for chunk_ids in _split_in_chunks(
        items=ids,
        chunk_size=chunk_size,
        drop_last=drop_last,
    ):
        chunk_data = []

        for trajectory_id in chunk_ids:
            g = trajectories_group[trajectory_id]

            trajectory_data = _read_single_trajectory_for_training(g)
            chunk_data.append(trajectory_data)

        actions, X, Y = _concat_chunk(chunk_data)

        yield actions, X, Y
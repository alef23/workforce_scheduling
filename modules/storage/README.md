# Storage Buffers

Buffers Zarr para persistir trayectorias y samples de entrenamiento.

Este módulo separa dos responsabilidades:

- `TrajectoryBuffer`: guarda y carga trayectorias completas e identificables.
- `SampleBuffer`: guarda y carga samples aplanados en batches para entrenamiento.

El encoding para redes neuronales no vive en este módulo. La conversión a tensores y one-hot/multi-hot es responsabilidad de `StateEncoder`.

## TrajectoryBuffer

Guarda trayectorias completas bajo:

```text
store.zarr/
└── trajectories/
    ├── 000000/
    ├── 000001/
    └── ...
```

Cada trayectoria conserva:

- `trajectory_id`
- `ProblemSetup` como metadata cruda
- estados de cada paso
- `policy`
- `action_id`
- `reward`

Uso:

```python
from modules.storage import TrajectoryBuffer

buffer = TrajectoryBuffer("dataset.zarr")

trajectory_id = buffer.save(
    trajectory=trajectory,
    problem_setup=problem_setup,
)

record = buffer.load(trajectory_id)
```

`record.problem_setup["allowed_entry_hours"]` conserva el valor crudo, por ejemplo:

```python
[6, 14, 18]
```

## SampleBuffer

Guarda samples aplanados bajo:

```text
store.zarr/
└── samples/
```

Se construye desde un `TrajectoryBuffer`:

```python
from modules.storage import SampleBuffer, TrajectoryBuffer

trajectory_buffer = TrajectoryBuffer("dataset.zarr")
sample_buffer = SampleBuffer("dataset.zarr")

sample_buffer.build_from_trajectory_buffer(
    trajectory_buffer=trajectory_buffer,
    overwrite=True,
)
```

Tambien puede recibir samples nuevos de forma incremental. Esto se usa cuando un
orquestador recibe trayectorias finalizadas desde workers y las aplana para
entrenamiento:

```python
sample_buffer.append_trajectories(generated_trajectories)
```

Cada trayectoria debe incluir:

- `trajectory`
- `problem_setup`
- `trajectory_id`
- `metadata`

Los samples pueden incluir `policy_weight`; si no esta presente, se usa `1.0`.

Itera batches:

```python
for batch in sample_buffer.iter_batches(batch_size=256, shuffle=True, seed=123):
    actions = batch.actions
    X = batch.X
    Y = batch.Y
    metadata = batch.metadata
```

`metadata["trajectory_id"]` y `metadata["step_index"]` existen solo para trazabilidad.

`TrajectoryBuffer.save()` tambien acepta metadata opcional por trayectoria:

```python
trajectory_buffer.save(
    trajectory=trajectory,
    problem_setup=problem_setup,
    trajectory_id="raw_000001",
    metadata={
        "stage": "raw",
        "seed": 123,
        "initial_stock": [10, 5, 3],
    },
)
```

La metadata se guarda en atributos Zarr bajo el prefijo `metadata.`.

## Contrato de X e Y

`X` contiene estado y setup crudos:

```python
X = {
    "residual_demand": ...,
    "remaining_stock": ...,
    "expansion_mode": ...,
    "current_modality": ...,
    "current_entry_hour": ...,
    "assignment_week": ...,
    "initial_demand_total": ...,
    "mobile_days_off_count": ...,
    "fixed_day_off": ...,
    "allowed_entry_hours": ...,
    "max_overcoverage_tolerance": ...,
    "closing_hour": ...,
}
```

`allowed_entry_hours` se devuelve como dato crudo por sample:

```python
[6, 14, 18]
```

Si todas las horas están permitidas, se devuelve:

```python
None
```

`Y` contiene targets:

```python
Y = {
    "policy": ...,
    "value": ...,
    "policy_weight": ...,
}
```

`value` corresponde al reward final asignado a todos los estados de la trayectoria.
`policy_weight` escala la loss de policy por sample. No modifica `policy`: la
policy sigue siendo una distribucion target normalizada.

## Nota sobre almacenamiento interno

Para poder guardar batches de forma eficiente en Zarr, `SampleBuffer` almacena internamente `allowed_entry_hours` como máscara de 24 posiciones. Esa representación es un detalle de persistencia. Al cargar batches, el loader la convierte nuevamente al dato crudo para que el encoder sea el único responsable del encoding neuronal.

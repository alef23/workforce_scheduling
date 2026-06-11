# Dataset Generation

Orquestador para generar datasets de trayectorias resueltas.

El modulo separa dos responsabilidades:

- El `TrajectoryGenerationWorker` genera trayectorias.
- El `TrajectoryDatasetOrchestrator` ejecuta jobs, coordina workers y guarda en `TrajectoryBuffer`.

La primera implementacion disponible es `RawDemandTrajectoryWorker`.

## Worker Protocol

Todo worker debe cumplir:

```python
class TrajectoryGenerationWorker(Protocol):
    worker_type: str

    def run(self, job: GenerationJob) -> GenerationWorkerResult:
        ...
```

Esto permite reutilizar el mismo orquestador para etapas futuras:

- trayectorias crudas desde `DemandSimulator`
- trayectorias derivadas ajustando stock para inducir `expansion_mode`
- trayectorias iniciadas desde estados intermedios con MCTS

## RawDemandTrajectoryWorker

Secuencia por job:

```text
samplear ProblemSetup
samplear stock inicial
generar cobertura con DemandSimulator
agregar ruido con DemandNoiseGenerator
replay de acciones con WorkforceEngine
devolver trayectoria resuelta
```

El proceso principal guarda las trayectorias en Zarr. Los workers no escriben en
el store compartido.

## StockAdjustmentTrajectoryWorker

Genera una segunda etapa a partir de trayectorias raw.

Por cada trayectoria fuente:

1. carga `ProblemSetup`, demanda inicial, stock inicial y acciones originales;
2. con probabilidad `p_stock`, activa la derivacion con stock reducido;
3. si no reduce stock, copia la trayectoria raw directamente;
4. si reduce stock, corta aleatoriamente el listado ordenado de chunks de
   recursos;
5. define el stock inicial como la cantidad de chunks por modalidad antes del
   corte;
6. conserva el orden original de chunks y replayea con `WorkforceEngine` para
   recalcular estados y activar `expansion_mode`;
7. devuelve siempre una trayectoria derivada.

Ejemplo:

```python
from modules.dataset_generation import (
    StockAdjustmentConfig,
    StockAdjustmentTrajectoryWorker,
    build_stock_adjustment_jobs,
)

worker = StockAdjustmentTrajectoryWorker(
    source_buffer_path="datasets/raw/trajectories.zarr",
    config=StockAdjustmentConfig(p_stock=0.2),
)

jobs = build_stock_adjustment_jobs(
    source_trajectory_ids=["raw_000000", "raw_000001"],
)
```

## Ejemplo

```python
from modules.dataset_generation import (
    DatasetGenerationConfig,
    NoiseGenerationConfig,
    ProblemSetupSamplingConfig,
    RawDemandTrajectoryWorker,
    ResourceSamplingConfig,
    TrajectoryDatasetOrchestrator,
    build_generation_jobs,
)

worker = RawDemandTrajectoryWorker(
    setup_config=ProblemSetupSamplingConfig(
        allowed_entry_hours=[6, 12, 18],
        closing_hour=22,
        max_overcoverage_tolerance=0.1,
    ),
    resource_config=ResourceSamplingConfig(
        mod_4_max=10,
        mod_6_max=10,
        mod_8_max=5,
    ),
    noise_config=NoiseGenerationConfig(k_max=0.8),
)

orchestrator = TrajectoryDatasetOrchestrator(
    config=DatasetGenerationConfig(
        output_path="data/raw_trajectories.zarr",
        n_workers=4,
        overwrite=True,
        progress_interval=100,
    ),
    worker=worker,
)

jobs = build_generation_jobs(n_jobs=1000)
report = orchestrator.run(jobs)
```

`report.saved_trajectories` indica cuantas trayectorias fueron persistidas.

## Script Raw

Para generar un buffer raw desde consola:

```bash
uv run python scripts/generate_raw_demand_dataset.py 1000 --workers 4 --overwrite
```

La documentacion operativa completa esta en:

```text
scripts/README.md
```

Por defecto crea esta estructura:

```text
datasets/
├── raw/
│   └── trajectories.zarr
├── derived/
│   ├── stock_adjusted/
│   │   └── trajectories.zarr
│   └── mcts/
│       └── trajectories.zarr
├── samples/
│   └── samples.zarr
└── reports/
```

Los buffers derivados quedan solo como estructura de carpetas; se llenaran en
las etapas posteriores de stock-adjustment y MCTS.

Si se necesita reproducibilidad, se puede pasar una seed fija:

```python
jobs = build_generation_jobs(n_jobs=1000, seed=123)
```

El reporte incluye estadisticas agregadas de:

- `initial_demand_total`
- `final_reward`
- `final_value`
- recursos totales por modalidad
- `trajectory_id` guardados

Durante la ejecucion, el orquestador imprime progreso en pantalla por defecto:

```text
[dataset_generation] jobs=100/1000 ok=100 failed=0 saved=100 rate=12.50 jobs/s
```

Se puede desactivar con:

```python
DatasetGenerationConfig(
    output_path="data/raw_trajectories.zarr",
    print_progress=False,
)
```

## Pipeline de acciones compuestas

`CompoundFullTrajectoryWorker` ejecuta el circuito completo dentro de un mismo
proceso:

```text
generar cobertura y trayectoria base
-> aplicar ruido a la cobertura
-> reproducir acciones con CompoundWorkforceEngine
-> ajustar stock por chunks de recurso
-> devolver únicamente la trayectoria final
```

`CompoundDatasetOrchestrator` distribuye jobs entre procesos y es el único que
escribe en `CompoundTrajectoryBuffer`. Esto evita escrituras concurrentes sobre
Zarr. Cada worker usa entropía del sistema; no recibe una seed de producción.

```python
from modules.dataset_generation import (
    CompoundDatasetOrchestrator,
    CompoundFullTrajectoryWorker,
    CompoundOrchestratorConfig,
    NoiseGenerationConfig,
    build_compound_generation_jobs,
)
from modules.workforce_engine.schemas import ProblemSetup

setup = ProblemSetup(
    mobile_days_off_count=1,
    fixed_day_off=6,
    allowed_entry_hours=[6, 12, 18],
    max_overcoverage_tolerance=0.1,
    closing_hour=22,
)
worker = CompoundFullTrajectoryWorker(
    problem_setup=setup,
    n_resources=20,
    p_stock=0.2,
    noise_config=NoiseGenerationConfig(k_max=0.8),
)
orchestrator = CompoundDatasetOrchestrator(
    config=CompoundOrchestratorConfig(
        output_path="datasets/compound/trajectories.zarr",
        n_workers=4,
        overwrite=True,
    ),
    worker=worker,
)
report = orchestrator.run(build_compound_generation_jobs(1000))
```

`n_resources` define el máximo por job. Cada worker samplea uniformemente una
cantidad entera entre `1` y ese máximo. La metadata guarda `n_resources` como
cantidad efectiva y `max_n_resources` como límite configurado.

El método de inicio multiproceso predeterminado es `spawn`, para no heredar
recursos internos de Zarr ni contextos de GPU. El reporte incluye rewards,
longitud de trayectoria, demanda inicial y cantidad de recursos de salida.

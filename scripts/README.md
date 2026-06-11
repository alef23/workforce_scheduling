# Scripts

Comandos operativos para ejecutar tareas del proyecto desde consola.

Para una guia completa del flujo de entrenamiento, desde trayectorias raw hasta
MCTS + learner + reload de pesos, ver:

```text
docs/training_pipeline.md
```

## Dashboards

El comando:

```bash
uv run python scripts/build_model_dashboard.py
```

genera dos paginas:

```text
datasets/reports/model_dashboard.html
datasets/reports/partial_evaluation_dashboard.html
```

La primera concentra entrenamiento, ciclos y checkpoints. La segunda contiene
exclusivamente el analisis interactivo del test parcial:

- distribuciones del dataset fijo;
- filtros por checkpoint y `tail`;
- scores MCTS y curva `better or equal rate`;
- tabla buscable, ordenable, paginada y exportable a CSV;
- metadata completa de cada evaluacion y de su trayectoria fuente.

La evaluacion desde el estado inicial se representa como un caso particular del
test parcial usando un `tail` suficientemente grande.

## `generate_compound_dataset.py`

Genera directamente el dataset del dominio de acciones compuestas. Cada worker
ejecuta en memoria:

```text
CompoundDemandSimulator
-> DemandNoiseGenerator
-> CompoundTrajectoryReplayer
-> CompoundStockAdjuster
```

El proceso principal guarda únicamente las trayectorias finales en:

```text
datasets/compound/trajectories.zarr
```

Comando mínimo:

```bash
uv run python scripts/generate_compound_dataset.py 100
```

Comando recomendado para una prueba:

```bash
uv run python scripts/generate_compound_dataset.py 100 \
  --workers 4 \
  --n-resources 20 \
  --p-stock 0.2 \
  --overwrite \
  --progress-interval 10
```

`--n-resources` es un máximo. Con `--n-resources 20`, cada trayectoria samplea
uniformemente una cantidad entera de recursos en `[1, 20]`.

Defaults:

| Parámetro | Default |
|---|---:|
| `n_samples` | requerido |
| `--workers` | `4` |
| `--n-resources` | `20` (máximo; sample uniforme desde `1`) |
| `--p-stock` | `0.2` |
| `--output-path` | `datasets/compound/trajectories.zarr` |
| `--progress-interval` | `100` |
| `--temporal-chunk-size` | `128` |
| `--noise-k-max` | `0.8` |
| `--noise-k-lambda` | `10.0` |
| `--max-overcoverage-tolerance` | `0.1` |
| `--run-prefix` | `compound` |
| `--multiprocessing-start-method` | `spawn` |
| `--overwrite` | desactivado |

El setup estructural permanece fijo:

```text
allowed_entry_hours = [6, 12, 18]
closing_hour = 22
fixed_day_off = 6
mobile_days_off_count = 1
```

No se expone una seed: cada job utiliza aleatoriedad nueva del sistema. Sin
`--overwrite`, una nueva corrida agrega trayectorias con IDs únicos al buffer.

## `generate_raw_demand_dataset.py`

Genera trayectorias raw resueltas usando `RawDemandTrajectoryWorker` y las guarda
en un `TrajectoryBuffer`.

Comando minimo:

```bash
uv run python scripts/generate_raw_demand_dataset.py 40
```

Comando recomendado para pruebas:

```bash
uv run python scripts/generate_raw_demand_dataset.py 40 --workers 4 --overwrite --progress-interval 10
```

## Defaults

| Parametro | Default |
|---|---:|
| `n_samples` | requerido |
| `--workers` | `4` |
| `--output-root` | `datasets` |
| `--progress-interval` | `100` |
| `--seed` | `None` |
| `--allowed-entry-hours` | `6 12 18` |
| `--closing-hour` | `22` |
| `--max-overcoverage-tolerance` | `0.1` |
| `--noise-k-max` | `0.8` |
| `--noise-k-lambda` | `10.0` |
| `--mod-4-max` | `20` |
| `--mod-6-max` | `20` |
| `--mod-8-max` | `20` |
| `--overwrite` | desactivado |

El `k` efectivo del ruido se samplea directamente en `[0, --noise-k-max]`
desde una exponencial truncada. `--noise-k-lambda` controla cuanta masa queda
cerca de cero: valores mas altos generan descuentos efectivos mas chicos con
mayor frecuencia.

## Salida

Por defecto crea:

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

El script solo llena:

```text
datasets/raw/trajectories.zarr
```

Las carpetas derivadas quedan preparadas para etapas posteriores.

## Progreso

Durante la ejecucion imprime:

```text
[dataset_generation] jobs=10/40 ok=10 failed=0 saved=10 rate=12.50 jobs/s
```

Al finalizar imprime:

```text
completed_jobs=40
failed_jobs=0
saved_trajectories=40
resource_totals={...}
stats={...}
```

## Reproducibilidad

Por defecto `--seed` queda en `None`, por lo que cada corrida usa seeds nuevas.

Para reproducir una corrida:

```bash
uv run python scripts/generate_raw_demand_dataset.py 40 --seed 123 --overwrite
```

## Inspeccion Rapida

Contar trayectorias guardadas:

```bash
find datasets/raw/trajectories.zarr/trajectories \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  -name 'raw_*' \
  | wc -l
```

Ver metadata de una trayectoria:

```bash
sed -n '1,220p' datasets/raw/trajectories.zarr/trajectories/raw_000000/zarr.json
```

## `generate_stock_adjusted_dataset.py`

Genera la segunda etapa del dataset desde el buffer raw.

Por cada trayectoria raw:

- con probabilidad `p_stock`, corta la secuencia de chunks de recursos y reduce
  stock para inducir `expansion_mode`;
- si no reduce stock, copia la trayectoria raw directamente al nuevo buffer;
- si reduce stock, conserva el orden original de chunks y replayea con
  `WorkforceEngine` para recalcular estados, `remaining_stock` y
  `expansion_mode`;
- guarda siempre una trayectoria derivada.

Comando minimo:

```bash
uv run python scripts/generate_stock_adjusted_dataset.py --overwrite
```

Comando recomendado para pruebas:

```bash
uv run python scripts/generate_stock_adjusted_dataset.py \
  --workers 4 \
  --p-stock 0.2 \
  --overwrite \
  --progress-interval 10
```

Salida por defecto:

```text
datasets/derived/stock_adjusted/trajectories.zarr
```

Procesar solo algunas trayectorias raw:

```bash
uv run python scripts/generate_stock_adjusted_dataset.py \
  --n-samples 40 \
  --shuffle \
  --overwrite
```

Continuar por tandas sin repetir trayectorias ya generadas:

```bash
uv run python scripts/generate_stock_adjusted_dataset.py \
  --n-samples 1000 \
  --shuffle \
  --skip-existing
```

`--skip-existing` busca IDs `stock_<raw_id>` ya guardados en el output y los
saltea. Si se usa junto con `--overwrite`, se ignora porque `--overwrite` recrea
el buffer.

## `generate_initial_state_test_set.py`

Genera un `SampleBuffer` de validacion con solo el estado inicial de cada
trayectoria. La generacion aplica el pipeline completo
`raw + noise -> stock_adjusted`; luego toma el estado inicial resultante y
guarda `value = final_reward`.

Comando recomendado:

```bash
uv run python scripts/generate_initial_state_test_set.py 100 \
  --output-path datasets/test/initial_states.zarr \
  --workers 4 \
  --p-stock 0.2 \
  --overwrite \
  --seed 12345
```

El formato es compatible con `SampleBuffer`; `sample_source` queda como
`test_initial_stock` y `step_index` queda en `0`. Si el buffer ya existe, el
script exige `--overwrite` para mantener fijo el conjunto de evaluacion.

## `generate_partial_trajectory_test_set.py`

Genera un unico `TrajectoryBuffer` fijo con trayectorias completas para
evaluaciones parciales. Usa el mismo pipeline
`RawDemandTrajectoryWorker + noise -> StockAdjustmentTrajectoryWorker` que el
test de estados iniciales.

```bash
uv run python scripts/generate_partial_trajectory_test_set.py 100 \
  --output-path datasets/test/partial_trajectories.zarr \
  --workers 4 \
  --p-stock 0.2 \
  --seed 12345 \
  --overwrite
```

Sin `--overwrite`, el comando falla si el buffer ya existe. Esto permite
comparar checkpoints diferentes siempre contra las mismas trayectorias.

## `generate_mcts_samples.py`

Genera samples planos desde el buffer `stock_adjusted`.

Por cada trayectoria stock:

- con probabilidad `--p-mcts`, genera trayectorias con MCTS;
- si no usa MCTS, recalcula la policy dando mas peso a la accion elegida;
- el orquestador aplana las trayectorias resultantes en `SampleBuffer`.

Comando minimo de prueba:

```bash
uv run python scripts/generate_mcts_samples.py \
  --workers 1 \
  --n-trajectories 1 \
  --p-mcts 0 \
  --overwrite-samples
```

## `evaluate_test_set_mcts.py`

Evalua el test set fijo de estados iniciales con MCTS usando la ultima ResNet
por defecto. Guarda las trayectorias en un `TrajectoryBuffer` separado y deja
metricas agregadas en JSON.

Comando recomendado:

```bash
uv run python scripts/evaluate_test_set_mcts.py \
  --sample-path datasets/test/initial_states.zarr \
  --output-root datasets/evaluation/mcts_test \
  --workers 4 \
  --mcts-simulations 500 \
  --device cuda \
  --overwrite
```

Salida:

```text
datasets/evaluation/mcts_test/
+-- trajectories.zarr
+-- reports/
    +-- runs.jsonl
    +-- trajectories.jsonl
    +-- run_summary.json
```

`runs.jsonl` mantiene el historial de corridas para que
`build_model_dashboard.py` grafique la evolucion de positivos sobre total y
el resto de metricas del test set.

El script guarda incrementalmente cada trayectoria y su fila JSONL apenas
termina el job correspondiente.

Para evaluar parcialmente las trayectorias completas, usar `partial` y pasar
obligatoriamente la cantidad de estados contados desde el final:

```bash
uv run python scripts/evaluate_test_set_mcts.py \
  --input-mode partial \
  --partial-trajectory-path datasets/test/partial_trajectories.zarr \
  --tail-states 30 \
  --workers 4 \
  --mcts-simulations 500 \
  --device cuda \
  --overwrite
```

Para una trayectoria de longitud `T`, el estado de inicio se selecciona con
`max(0, T - tail_states)`. Los resultados se guardan por defecto en
`datasets/evaluation/mcts_partial` y registran la longitud fuente, el indice
inicial y la distancia efectiva desde el final.

Comando con evaluador centralizado y MCTS:

```bash
uv run python scripts/generate_mcts_samples.py \
  --workers 2 \
  --n-trajectories 10 \
  --p-mcts 0.2 \
  --start-mode initial_only \
  --mcts-simulations 16 \
  --device cuda \
  --sample-limit-per-cycle 10000 \
  --overwrite-samples
```

Comando con entrenamiento sincrono al cerrar cada ciclo:

```bash
uv run python scripts/generate_mcts_samples.py \
  --workers 2 \
  --n-trajectories 100 \
  --p-mcts 0.2 \
  --sample-limit-per-cycle 10000 \
  --train-on-cycle \
  --learner-batch-size 64 \
  --checkpoint-dir modules/evaluators/resnet/checkpoints \
  --device cuda
```

Si no se pasa `--checkpoint-path`, cada nueva ejecucion selecciona el `.pt`
con mayor step numerico dentro de `--checkpoint-dir`. Un path explicito sigue
teniendo prioridad. El checkpoint resuelto se imprime como
`[mcts_generation] checkpoint=...` y queda registrado en los logs de la corrida.

Con `--train-on-cycle`, el orquestador espera a que los workers terminen sus
trayectorias activas. El learner mezcla una vez el rango nuevo del ciclo y lo
consume en batches sin reposicion. Los steps se calculan como
`ceil(samples_del_ciclo / learner_batch_size)`. Tras guardar el checkpoint y
recargar el evaluator, ese rango queda marcado como entrenado y no se reutiliza.

Cada ciclo registra tiempos de generacion y entrenamiento en
`mcts_generation_cycles.jsonl` y los muestra en el dashboard:

- `generation_wall_seconds`: tiempo real transcurrido desde que comienza el
  ciclo hasta que sus samples quedan guardados.
- `mcts_generation_total_seconds` y `zarr_read_total_seconds`: suma de los
  tiempos medidos dentro de los workers. Con varios workers pueden superar el
  tiempo real del ciclo porque las tareas corren en paralelo.
- `zarr_write_total_seconds`: tiempo empleado por el proceso orquestador para
  agregar samples al buffer.
- El bloque `learner` separa tiempo total, lectura Zarr, encoding, optimizacion
  y guardado de checkpoints, ademas de samples por segundo.

Para concentrar las semillas MCTS en una ventana anterior al terminal:

```bash
uv run python scripts/generate_mcts_samples.py \
  --start-mode tail_forward_sampled \
  --tail-window-size 30 \
  --max-seed-states 15 \
  --seed-state-probability 0.15
```

Si el terminal esta en `T`, se evaluan en orden los candidatos desde `T-30`
hasta `T-1`. El terminal queda excluido y el estado inicial se agrega siempre
como semilla independiente.

Logs persistentes:

Por defecto `generate_mcts_samples.py` escribe JSONL en `datasets/reports/`:

```text
mcts_generation_runs.jsonl
mcts_generation_cycles.jsonl
mcts_generation_learner_steps.jsonl
```

Se puede fijar un ID de corrida o cambiar el directorio:

```bash
uv run python scripts/generate_mcts_samples.py \
  --run-id smoke_001 \
  --reports-dir datasets/reports \
  ...
```

Para desactivar estos logs:

```bash
uv run python scripts/generate_mcts_samples.py --disable-report-logging ...
```

Dashboards HTML:

`build_model_dashboard.py` construye el visor liviano del modelo desde logs
JSONL, reportes de evaluación y checkpoints. No abre buffers Zarr.

```bash
uv run python scripts/build_model_dashboard.py
```

Durante el entrenamiento puede refrescarse cada 30 segundos:

```bash
watch -n 30 'uv run python scripts/build_model_dashboard.py'
```

`build_zarr_dashboard.py` construye el resumen de buffers y el explorador estado
por estado:

```bash
uv run python scripts/build_zarr_dashboard.py
```

Generan `datasets/reports/model_dashboard.html` y
`datasets/reports/zarr_dashboard.html`, respectivamente. El segundo comando
realiza lecturas intensivas de Zarr y puede competir por I/O con el
entrenamiento.

`build_training_dashboard.py` se conserva como comando legado para generar
ambos en una sola ejecución.
Salida por defecto:

```text
datasets/samples/samples.zarr
```

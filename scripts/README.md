# Scripts

Comandos operativos para ejecutar tareas del proyecto desde consola.

Para una guia completa del flujo de entrenamiento, desde trayectorias raw hasta
MCTS + learner + reload de pesos, ver:

```text
docs/training_pipeline.md
```

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
trayectoria raw simulada. La trayectoria completa se resuelve internamente para
guardar `value = final_reward`, pero no se guarda en un `TrajectoryBuffer`.

Comando recomendado:

```bash
uv run python scripts/generate_initial_state_test_set.py 100 \
  --output-path datasets/test/initial_states.zarr \
  --workers 4 \
  --overwrite \
  --seed 12345
```

El formato es compatible con `SampleBuffer`; `sample_source` queda como
`test_initial_raw` y `step_index` queda en `0`.

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
`build_training_dashboard.py` grafique la evolucion de positivos sobre total y
el resto de metricas del test set.

El script guarda incrementalmente cada trayectoria y su fila JSONL apenas
termina el job correspondiente.

Comando con evaluador centralizado y MCTS:

```bash
uv run python scripts/generate_mcts_samples.py \
  --workers 2 \
  --n-trajectories 10 \
  --p-mcts 0.2 \
  --start-mode initial_only \
  --mcts-simulations 16 \
  --checkpoint-path modules/evaluators/resnet/checkpoints/workforce_resnet_000.pt \
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
  --learner-steps 100 \
  --learner-batch-size 64 \
  --checkpoint-path modules/evaluators/resnet/checkpoints/workforce_resnet_000.pt \
  --checkpoint-dir modules/evaluators/resnet/checkpoints \
  --device cuda
```

Con `--train-on-cycle`, el orquestador espera a que los workers terminen sus
trayectorias activas, el learner entrena desde el `SampleBuffer`, guarda un
checkpoint y devuelve ese path para que el evaluator recargue pesos.

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

Dashboard HTML:

`build_training_dashboard.py` construye un visor estatico desde los logs JSONL,
checkpoints y buffers Zarr existentes. No modifica los modulos ni escribe en los
buffers.

```bash
uv run python scripts/build_training_dashboard.py
```

Por defecto genera:

```text
datasets/reports/training_dashboard.html
datasets/reports/trajectory_explorer.html
```

`training_dashboard.html` contiene monitoreo y analisis agregado de
distribuciones. `trajectory_explorer.html` contiene el explorador estado por
estado de las ultimas trayectorias seleccionadas de cada buffer.

Mientras corre el entrenamiento se puede refrescar cada 30 segundos:

```bash
watch -n 30 'uv run python scripts/build_training_dashboard.py'
```

Salida por defecto:

```text
datasets/samples/samples.zarr
```

# Training Pipeline

Guia operativa para generar datos, crear samples y entrenar la ResNet con el
ciclo MCTS + learner.

## Contexto

El objetivo es entrenar una `WorkforceResNet` que pueda evaluar estados del
problema de workforce scheduling. La red devuelve:

- `policy_logits`: preferencia por cada accion del espacio discreto de 55
  acciones.
- `value`: estimacion escalar del estado en `[-1, 1]`.

El dataset no nace directamente de MCTS puro. Primero se generan trayectorias
resueltas con demanda ruidosa, luego se derivan trayectorias con stock reducido
para forzar casos de `expansion_mode`, y finalmente los workers MCTS producen
samples de entrenamiento. Cuando no se usa MCTS para una trayectoria, se
recalcula una policy artificial con menor `policy_weight`.

El entrenamiento se organiza por ciclos:

1. Los workers procesan trayectorias del buffer `stock_adjusted`.
2. El orquestador guarda samples planos en `SampleBuffer`.
3. Cuando se alcanza un limite de samples por ciclo, el orquestador deja de
   asignar nuevos jobs y espera a que terminen las trayectorias activas.
4. El learner toma batches aleatorios del `SampleBuffer`, entrena la ResNet y
   guarda un checkpoint.
5. El evaluador centralizado recarga el checkpoint.
6. Los workers continuan con el siguiente ciclo.

## Arquitectura

```text
scripts/
+-- generate_raw_demand_dataset.py
+-- generate_stock_adjusted_dataset.py
+-- generate_mcts_samples.py

modules/
+-- dataset_generation/
|   +-- RawDemandTrajectoryWorker
|   +-- StockAdjustmentTrajectoryWorker
|   +-- TrajectoryDatasetOrchestrator
+-- storage/
|   +-- TrajectoryBuffer
|   +-- SampleBuffer
+-- mcts_generation/
|   +-- MCTSGenerationWorker
|   +-- MCTSGenerationOrchestrator
+-- evaluators/
|   +-- centralized/
|   |   +-- CentralizedEvaluatorServer
|   |   +-- CentralizedEvaluatorClient
|   +-- resnet/
|       +-- StateEncoder
|       +-- WorkforceResNet
|       +-- ResNetStateEvaluator
+-- learning/
    +-- ResNetSampleLearner
```

### Componentes

`RawDemandTrajectoryWorker`

Genera una trayectoria inicial:

- samplea `ProblemSetup`;
- samplea stock inicial;
- genera cobertura base con `DemandSimulator`;
- agrega ruido con `DemandNoiseGenerator`;
- replayea acciones con `WorkforceEngine`;
- devuelve una trayectoria terminal.

`StockAdjustmentTrajectoryWorker`

Parte de una trayectoria raw:

- con probabilidad `p_stock` activa una derivacion con stock reducido;
- si no reduce stock, copia la trayectoria raw directamente;
- si reduce stock, corta aleatoriamente el listado de chunks en orden original;
- define el stock inicial contando chunks por modalidad antes del corte;
- replayea con `WorkforceEngine` para recalcular estados y activar
  `expansion_mode`;
- guarda una trayectoria `stock_<raw_id>`.

`MCTSGenerationWorker`

Parte de una trayectoria `stock_adjusted`:

- con probabilidad `p_mcts` genera trayectorias con MCTS;
- con probabilidad `1 - p_mcts` no corre MCTS y recalcula la policy;
- devuelve trayectorias finalizadas, no escribe buffers.

`MCTSGenerationOrchestrator`

Coordina workers, evaluator y `SampleBuffer`:

- asigna jobs a workers;
- recibe trayectorias terminadas;
- aplana estados en samples;
- cierra ciclos cuando se alcanza `sample_limit_per_cycle`;
- llama al hook de learner;
- pide reload de pesos al evaluator.

`CentralizedEvaluatorServer`

Proceso unico que carga la ResNet. En modo multiproceso evita que cada worker
inicialice su propio modelo en GPU. Los workers hacen requests sincronicos y
esperan la respuesta.

`ResNetSampleLearner`

Entrena desde el `SampleBuffer`:

- toma batches aleatorios;
- encodea `X` con `StateEncoder`;
- calcula policy loss soft cross entropy;
- escala solo policy loss con `policy_weight`;
- calcula value loss MSE;
- guarda checkpoints `.pt`.

## Estructura de datasets

Por defecto los scripts usan:

```text
datasets/
+-- raw/
|   +-- trajectories.zarr
+-- derived/
|   +-- stock_adjusted/
|   |   +-- trajectories.zarr
|   +-- mcts/
|       +-- trajectories.zarr
+-- samples/
|   +-- samples.zarr
+-- reports/
```

## Formato de TrajectoryBuffer

Cada buffer de trayectorias guarda grupos bajo:

```text
trajectories.zarr/
+-- trajectories/
    +-- raw_000000/
    +-- raw_000001/
    +-- ...
```

En la etapa stock, los IDs quedan:

```text
stock_raw_000000
stock_raw_000001
```

Cada trayectoria contiene arrays:

```text
residual_demand        shape (T, 24, 28) int32
remaining_stock        shape (T, 3)      int32
expansion_mode         shape (T,)        bool
current_modality       shape (T,)        int32, None como -1
current_entry_hour     shape (T,)        int32, None como -1
assignment_week        shape (T,)        int32
initial_demand_total   shape (T,)        int64
policy                 shape (T, 55)     float32
action_id              shape (T,)        int32
reward                 shape (T,)        float32
```

Y atributos:

```text
trajectory_id
length
action_space_size
final_reward
schema_version
problem_setup.*
metadata.*
```

`problem_setup.*` incluye, entre otros:

```text
mobile_days_off_count
fixed_day_off
allowed_entry_hours
max_overcoverage_tolerance
closing_hour
```

`metadata.*` depende de la etapa. Por ejemplo, raw guarda seeds, stock inicial,
total de demanda y reward final. Stock-adjusted agrega `source_trajectory_id`,
`stock_was_reduced`, `output_stock`, `has_expansion_mode` y
`first_expansion_step`.

## Formato de SampleBuffer

`SampleBuffer` aplana trayectorias en samples independientes:

```text
samples.zarr/
+-- samples/
```

Cada sample contiene `X`, `Y` y metadata.

`X`:

```text
residual_demand
remaining_stock
expansion_mode
current_modality
current_entry_hour
assignment_week
initial_demand_total
mobile_days_off_count
fixed_day_off
allowed_entry_hours
max_overcoverage_tolerance
closing_hour
```

`Y`:

```text
policy
value
policy_weight
```

Metadata:

```text
trajectory_id
step_index
sample_source
source_trajectory_id
model_version
sample_index
```

`policy_weight` no cambia la policy target. Solo escala la loss de policy:

```text
policy_loss_i = -sum(policy_target * log_softmax(policy_logits))
policy_loss   = mean(policy_loss_i * policy_weight_i)
```

## Checklist operativo

### Paso 1. Generar trayectorias raw con ruido

Comando recomendado para una prueba chica:

```bash
uv run python scripts/generate_raw_demand_dataset.py 100 \
  --workers 4 \
  --overwrite \
  --progress-interval 10 \
  --seed 123 \
  --allowed-entry-hours 6 12 18 \
  --closing-hour 22 \
  --max-overcoverage-tolerance 0.1 \
  --noise-k-max 0.8 \
  --mod-4-max 20 \
  --mod-6-max 20 \
  --mod-8-max 20
```

Salida:

```text
datasets/raw/trajectories.zarr
```

Que genera:

- una trayectoria terminal por job;
- IDs `raw_000000`, `raw_000001`, ...;
- estados con demanda residual, stock, modalidad actual, hora actual y semana;
- `policy` por estado;
- `action_id` elegido;
- `reward` final propagado en la trayectoria;
- metadata de seeds, stock inicial, ruido y reward final.

Validacion rapida:

```bash
find datasets/raw/trajectories.zarr/trajectories \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  -name 'raw_*' \
  | wc -l
```

### Paso 2. Generar trayectorias stock_adjusted y muestras con expansion_mode

Este paso toma las trayectorias raw y genera una version derivada. Con
probabilidad `p_stock` reduce stock a partir de un corte aleatorio del listado de
chunks de recursos. No mezcla chunks: conserva el orden original y replayea con
`WorkforceEngine` para recalcular `remaining_stock` y `expansion_mode`.

Comando recomendado:

```bash
uv run python scripts/generate_stock_adjusted_dataset.py \
  --workers 4 \
  --p-stock 0.2 \
  --overwrite \
  --progress-interval 10 \
  --seed 123
```

Salida:

```text
datasets/derived/stock_adjusted/trajectories.zarr
```

Que genera:

- una trayectoria `stock_<raw_id>` por raw procesada;
- copia directa de la raw cuando no reduce stock;
- replay con `WorkforceEngine` cuando reduce stock;
- estados con el mismo formato del `TrajectoryBuffer`;
- `expansion_mode=True` en estados donde ya no queda stock y el engine opera en
  modo expansion;
- metadata `stock_was_reduced`, `stock_cut_index`, `output_stock`,
  `has_expansion_mode`, `first_expansion_step` y `source_trajectory_id`.

Procesar por tandas sin repetir:

```bash
uv run python scripts/generate_stock_adjusted_dataset.py \
  --n-samples 1000 \
  --shuffle \
  --skip-existing \
  --progress-interval 100
```

Validacion rapida:

```bash
find datasets/derived/stock_adjusted/trajectories.zarr/trajectories \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  -name 'stock_*' \
  | wc -l
```

### Paso 3. Generar samples MCTS/reweighted sin entrenar

Este paso usa el buffer `stock_adjusted` como fuente y escribe samples planos en
`SampleBuffer`.

Comando de prueba:

```bash
uv run python scripts/generate_mcts_samples.py \
  --workers 2 \
  --n-trajectories 20 \
  --p-mcts 0.2 \
  --start-mode initial_only \
  --mcts-simulations 16 \
  --sample-path datasets/samples/samples.zarr \
  --overwrite-samples \
  --device cpu
```

Salida:

```text
datasets/samples/samples.zarr
```

Que genera:

- samples `X` e `Y` aplanados;
- `sample_source="mcts"` para trayectorias generadas con MCTS;
- `sample_source="stock_reweighted"` para trayectorias sin MCTS;
- `policy_weight=1.0` por defecto para MCTS;
- `policy_weight=0.5` por defecto para reweighted.

Para los casos sin MCTS, la policy se recalcula usando acciones legales
originalmente no cero:

```text
selected_action = 1 / (Nl - 1)
other_legal     = (Nl - 2) / (Nl - 1)^2
illegal         = 0
```

Casos borde:

- `Nl == 1`: la accion seleccionada recibe `1.0`.
- `Nl == 2`: la accion seleccionada recibe `1.0` y la otra accion legal `0.0`.

Modos MCTS:

```text
initial_only       toma solo el estado inicial
forward_sampled   toma estado inicial + hasta N estados hacia adelante
backward_sampled  toma estado inicial + hasta N estados hacia atras, sin terminal
```

Ejemplo con semillas adicionales hacia el final:

```bash
uv run python scripts/generate_mcts_samples.py \
  --workers 2 \
  --n-trajectories 20 \
  --p-mcts 0.3 \
  --start-mode backward_sampled \
  --max-seed-states 2 \
  --seed-state-probability 0.5 \
  --mcts-simulations 16 \
  --overwrite-samples
```

### Paso 4. Correr el ciclo completo MCTS + learner + reload

Este es el flujo de entrenamiento sincronico.

Comando de prueba controlada:

```bash
uv run python scripts/generate_mcts_samples.py \
  --workers 2 \
  --n-trajectories 20 \
  --p-mcts 0.2 \
  --start-mode initial_only \
  --mcts-simulations 16 \
  --sample-limit-per-cycle 2000 \
  --train-on-cycle \
  --learner-steps 50 \
  --learner-batch-size 64 \
  --learner-learning-rate 0.0001 \
  --learner-weight-decay 0.0001 \
  --checkpoint-path modules/evaluators/resnet/checkpoints/workforce_resnet_000.pt \
  --checkpoint-dir modules/evaluators/resnet/checkpoints \
  --device cuda \
  --overwrite-samples
```

Que sucede:

- se inicia un `CentralizedEvaluatorServer` si `--workers > 1`;
- los workers comparten el evaluator;
- el orquestador guarda samples;
- al llegar a `sample_limit_per_cycle`, se deja de asignar trabajo nuevo;
- workers activos terminan sus trayectorias;
- `ResNetSampleLearner` entrena desde el `SampleBuffer`;
- se guarda un checkpoint `workforce_resnet_<global_step>.pt`;
- el evaluator recarga pesos;
- continua el siguiente ciclo.

Con GPU local de 8GB, el modo recomendado es:

- `--workers > 1` con evaluador centralizado;
- no crear evaluadores por worker;
- mantener entrenamiento y reload sincronicos;
- empezar con pocos `mcts-simulations` y pocos `learner-steps`;
- subir gradualmente si la memoria y el tiempo son estables.

### Paso 5. Inspeccionar checkpoints y samples

Checkpoints:

```bash
ls -lh modules/evaluators/resnet/checkpoints
```

Samples:

```bash
find datasets/samples/samples.zarr/samples -maxdepth 1 -type f -o -type d
```

Ver el resumen de una corrida desde consola:

```text
completed_jobs=...
failed_jobs=...
generated_trajectories=...
saved_samples=...
used_mcts_jobs=...
reweighted_jobs=...
cycles=...
```

Durante entrenamiento se imprime:

```text
[mcts_generation] learner_done checkpoint=... global_step=... loss=...
```

Ademas se escriben logs JSONL en `datasets/reports/`:

```text
mcts_generation_runs.jsonl
mcts_generation_cycles.jsonl
mcts_generation_learner_steps.jsonl
```

`mcts_generation_runs.jsonl` guarda una fila por corrida con argumentos,
cantidad de jobs, samples y errores. `mcts_generation_cycles.jsonl` guarda una
fila por ciclo con MCTS vs reweighted y el checkpoint del learner si entreno.
`mcts_generation_learner_steps.jsonl` guarda una fila por step de learner con
loss, policy loss, value loss y `mean_policy_weight`.

Flags utiles:

```text
--run-id                  fija el ID de corrida
--reports-dir             cambia el directorio de logs
--disable-report-logging  desactiva logs persistentes
```

### Visor de entrenamiento

El entrenamiento deja suficientes datos para generar un visor HTML sin tocar el
pipeline: logs JSONL, checkpoints, `TrajectoryBuffer` raw/stock y
`SampleBuffer`.

Para construir o refrescar el visor:

```bash
uv run python scripts/build_training_dashboard.py
```

Salida por defecto:

```text
datasets/reports/training_dashboard.html
```

Durante una corrida larga se puede regenerar periodicamente y refrescar el
navegador:

```bash
watch -n 30 'uv run python scripts/build_training_dashboard.py'
```

El visor muestra corridas, ciclos, losses del learner, checkpoints, conteos de
MCTS vs reweighted, resumen de buffers Zarr y un preview de trayectorias y
samples. Es una foto del estado actual; no escribe ni modifica buffers.

### Paso 6. Escalar la corrida

Cuando la prueba chica sea estable:

```bash
uv run python scripts/generate_mcts_samples.py \
  --workers 4 \
  --p-mcts 0.2 \
  --start-mode backward_sampled \
  --max-seed-states 1 \
  --seed-state-probability 0.3 \
  --mcts-simulations 32 \
  --sample-limit-per-cycle 10000 \
  --train-on-cycle \
  --learner-steps 100 \
  --learner-batch-size 64 \
  --device cuda
```

Ajustes manuales esperados a medida que aprende el modelo:

- subir `p_mcts`;
- pasar de `initial_only` a `backward_sampled` y luego `forward_sampled`;
- aumentar `max_seed_states`;
- aumentar `mcts-simulations`;
- aumentar `sample_limit_per_cycle`;
- ajustar `learner-steps`, `batch-size` y learning rate.

## Comandos utiles

Contar raw trajectories:

```bash
find datasets/raw/trajectories.zarr/trajectories \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  -name 'raw_*' \
  | wc -l
```

Contar stock trajectories:

```bash
find datasets/derived/stock_adjusted/trajectories.zarr/trajectories \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  -name 'stock_*' \
  | wc -l
```

Ver ayuda de scripts:

```bash
uv run python scripts/generate_raw_demand_dataset.py --help
uv run python scripts/generate_stock_adjusted_dataset.py --help
uv run python scripts/generate_mcts_samples.py --help
```

## Pruebas recomendadas despues de cambios

```bash
.venv/bin/python -m compileall scripts modules
.venv/bin/python -m pytest tests/test_mcts_generation_helpers.py tests/test_mcts_generation_worker.py tests/test_resnet_sample_learner.py -q
```

El test de orquestador usa Zarr real y, si el sandbox cuelga al abrir buffers,
conviene correrlo con timeout fuera del sandbox:

```bash
timeout 30 .venv/bin/python -m pytest tests/test_mcts_generation_orchestrator.py -q
```

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
+-- generate_initial_state_test_set.py
+-- generate_stock_adjusted_dataset.py
+-- generate_mcts_samples.py
+-- evaluate_test_set_mcts.py

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
+-- test/
|   +-- initial_states.zarr
+-- evaluation/
|   +-- mcts_test/
|       +-- trajectories.zarr
|       +-- reports/
|           +-- runs.jsonl
|           +-- trajectories.jsonl
|           +-- run_summary.json
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

El test set fijo de estados iniciales tambien usa formato `SampleBuffer`, pero
guarda solo un sample por problema. Ese buffer vive separado del entrenamiento,
por defecto en:

```text
datasets/test/initial_states.zarr
```

No se mezcla con `datasets/samples/samples.zarr`.

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
  --noise-k-lambda 10.0 \
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

El ruido descuenta una parte de la cobertura factible. Para cada trayectoria se
samplea `noise_k_effective` directamente en `[0, noise_k_max]` desde una
exponencial truncada con `noise_k_lambda` por defecto `10.0`, lo que concentra
mas probabilidad cerca de `0`.

Validacion rapida:

```bash
find datasets/raw/trajectories.zarr/trajectories \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  -name 'raw_*' \
  | wc -l
```

### Paso 1b. Generar test set fijo de estados iniciales

Este paso es opcional, pero recomendado para evaluar checkpoints contra siempre
el mismo conjunto de problemas. Genera trayectorias raw internamente para conocer
el reward final, pero guarda solo el estado inicial como `SampleBuffer`.

```bash
uv run python scripts/generate_initial_state_test_set.py 100 \
  --output-path datasets/test/initial_states.zarr \
  --workers 4 \
  --overwrite \
  --seed 12345 \
  --allowed-entry-hours 6 12 18 \
  --closing-hour 22 \
  --max-overcoverage-tolerance 0.1 \
  --noise-k-max 0.8 \
  --noise-k-lambda 10.0 \
  --mod-4-max 20 \
  --mod-6-max 20 \
  --mod-8-max 20
```

Formato:

- un sample por problema;
- `step_index = 0`;
- `sample_source = test_initial_raw`;
- `value = final_reward` de la trayectoria completa;
- `policy` y `action_id` del estado inicial;
- campos `X` y setup compatibles con `SampleBuffer`.

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

### Paso 4b. Evaluar el test set fijo con MCTS

Este paso toma `datasets/test/initial_states.zarr`, corre MCTS desde cada estado
inicial usando un checkpoint ResNet, y guarda las trayectorias generadas en un
buffer separado de evaluacion. No entrena, no escribe `SampleBuffer` y no toca
los buffers raw/stock/MCTS de entrenamiento.

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

Si no se pasa `--checkpoint-path`, el script usa el checkpoint `.pt` con mayor
step numerico en:

```text
modules/evaluators/resnet/checkpoints
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

`trajectories.zarr` guarda una trayectoria por sample evaluado, con el estado
terminal incluido como ultimo estado. El estado terminal usa `action_id = -1` y
policy cero porque ya no hay accion a elegir.

Las trayectorias y sus metricas se escriben incrementalmente a medida que cada
job termina. Si una corrida larga se corta, los jobs ya finalizados quedan
guardados en `trajectories.zarr` y `trajectories.jsonl`.

`trajectories.jsonl` guarda una fila por trayectoria con:

```text
trajectory_id
source_sample_index
source_trajectory_id
checkpoint_path
checkpoint_step
mcts_simulations
elapsed_seconds
states_count
final_reward
original_value
value_error
```

`run_summary.json` agrega la ultima corrida completa: cantidad de rewards
positivos, cuantas trayectorias mejoraron al reward original del sample,
promedios y errores. `runs.jsonl` mantiene una fila por corrida para graficar la
evolucion historica de checkpoints.

Comando chico para prueba:

```bash
uv run python scripts/evaluate_test_set_mcts.py \
  --n-samples 5 \
  --workers 1 \
  --mcts-simulations 16 \
  --device cpu \
  --output-root /tmp/mcts_test_eval \
  --overwrite
```

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
--run-id                  fija manualmente el ID de corrida
--run-prefix              prefijo del ID correlativo automatico
--reports-dir             cambia el directorio de logs
--disable-report-logging  desactiva logs persistentes
```

Si no se pasa `--run-id`, el logger reserva un correlativo por prefijo:

```bash
--run-prefix train_gpu_mid
```

Genera:

```text
train_gpu_mid_001
train_gpu_mid_002
train_gpu_mid_003
```

Los ciclos internos quedan identificados como:

```text
train_gpu_mid_001_cycle_000
train_gpu_mid_001_cycle_001
```

La secuencia se guarda en `datasets/reports/run_sequences.json`. Este archivo
debe conservarse al limpiar logs si se quiere continuar la numeracion.

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
datasets/reports/trajectory_explorer.html
```

Durante una corrida larga se puede regenerar periodicamente y refrescar el
navegador:

```bash
watch -n 30 'uv run python scripts/build_training_dashboard.py'
```

El dashboard principal muestra corridas, ciclos, losses del learner,
checkpoints, conteos de MCTS vs reweighted, resumen de buffers Zarr, analisis
agregado de distribuciones y la seccion `Evaluacion MCTS del test set`. Esa
seccion lee `datasets/evaluation/mcts_test/reports/runs.jsonl` para graficar la
evolucion de positivos sobre total, `positive_rate`, mejora contra el reward
original, rewards medios y tiempos. El explorador de trayectorias vive en
`trajectory_explorer.html` y permite navegar estado por estado las ultimas
trayectorias cargadas, incluyendo el buffer `Test MCTS` si existe. Ambos son
una foto del estado actual; no escriben ni modifican buffers.

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

## Mapa operativo de entrenamiento

### Flujo principal

| # | Operacion | Descripcion | Script | Buffer In | Buffer Out | Parametros clave | Logs / reportes |
|---:|---|---|---|---|---|---|---|
| 0 | Generacion del test set fijo | Genera problemas de validacion que se mantienen constantes para comparar checkpoints. Internamente resuelve trayectorias completas, pero guarda solo el estado inicial con su `value=final_reward`. | `scripts/generate_initial_state_test_set.py` | n/a | `datasets/test/initial_states.zarr` | `n_samples=100`, `--workers 10`, `--noise-k-max 0.8`, `--noise-k-lambda 10.0`, `--overwrite` | Progreso en consola. El buffer queda en formato `SampleBuffer`. |
| 1 | Simulacion de demanda raw | Genera trayectorias resueltas a partir de cobertura base, ruido de demanda y replay con `WorkforceEngine`. Estas trayectorias sirven como base factible. | `scripts/generate_raw_demand_dataset.py` | n/a | `datasets/raw/trajectories.zarr` | `n_samples`, `--workers`, `--allowed-entry-hours`, `--closing-hour`, `--noise-k-max`, `--noise-k-lambda`, `--mod-*-max` | Progreso en consola `[dataset_generation]`. Metadata por trayectoria en Zarr. |
| 2 | Simulacion stock adjusted | Toma trayectorias raw y reduce stock con probabilidad `p_stock` para inducir casos con `expansion_mode`. Si no reduce, copia la raw directa. | `scripts/generate_stock_adjusted_dataset.py` | `datasets/raw/trajectories.zarr` | `datasets/derived/stock_adjusted/trajectories.zarr` | `--workers 10`, `--p-stock 0.25`, `--skip-existing`, `--overwrite` | Progreso en consola. Metadata: `stock_was_reduced`, `has_expansion_mode`, `first_expansion_step`. |
| 3 | Generacion MCTS + reweighted | Procesa trayectorias stock. Con probabilidad `p_mcts` genera trayectorias MCTS; con `1-p_mcts` recalcula policy artificial con menor `policy_weight`. El orquestador aplana todo en `SampleBuffer`. | `scripts/generate_mcts_samples.py` | `datasets/derived/stock_adjusted/trajectories.zarr` | `datasets/samples/samples.zarr` | `--workers 10`, `--p-mcts`, `--start-mode`, `--max-seed-states`, `--seed-state-probability`, `--mcts-simulations`, `--evaluator-batch-wait 0` | `datasets/reports/mcts_generation_runs.jsonl`, `mcts_generation_cycles.jsonl`; progreso `[mcts_generation]`. |
| 4 | Ciclo MCTS + learner + reload | Igual que el paso 3, pero al llegar a `sample_limit_per_cycle` pausa asignacion, espera workers activos, entrena la ResNet, guarda checkpoint y recarga evaluator. | `scripts/generate_mcts_samples.py --train-on-cycle` | `stock_adjusted` + `samples.zarr` acumulado | `samples.zarr` + checkpoints `.pt` | `--sample-limit-per-cycle`, `--learner-steps`, `--learner-batch-size`, `--learner-learning-rate`, `--checkpoint-dir`, `--device cuda` | `mcts_generation_learner_steps.jsonl`, `cycle_ready`, `learner_done`, checkpoints `workforce_resnet_<step>.pt`. |
| 5 | Evaluacion del test set | Corre MCTS desde cada estado inicial del test set usando el ultimo checkpoint o uno indicado. Guarda trayectorias completas y metricas por corrida. | `scripts/evaluate_test_set_mcts.py` | `datasets/test/initial_states.zarr` | `datasets/evaluation/mcts_test/trajectories.zarr` | `--workers 10`, `--mcts-simulations 300`, `--evaluator-batch-size 10`, `--evaluator-batch-wait 0`, `--device cuda` | `datasets/evaluation/mcts_test/reports/trajectories.jsonl`, `run_summary.json`, `runs.jsonl`. |
| 6 | Dashboard y explorador | Construye HTML estatico con logs, buffers, distribuciones, curvas de entrenamiento y evaluacion MCTS del test set. | `scripts/build_training_dashboard.py` | logs + buffers Zarr | `datasets/reports/training_dashboard.html`, `trajectory_explorer.html` | `--reports-dir`, `--test-eval-reports-dir`, `--max-sample-scan`, `--explorer-trajectory-count` | No genera entrenamiento; solo snapshot HTML. |

### Plan sugerido de ciclos MCTS

| Fase | Objetivo | `p_mcts` | `start_mode` | `max_seed_states` | `seed_state_probability` | `mcts-simulations` | `sample_limit_per_cycle` | `learner-steps` | Evaluacion recomendada |
|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Aprender finales y estabilizar value | `0.15` | `backward_sampled` | `1` | `0.40` | `32` | `8000` | `80` | 100 samples, 100-300 sims |
| 2 | Aumentar MCTS cerca del final | `0.25` | `backward_sampled` | `2` | `0.50` | `64` | `12000` | `100` | 100 samples, 300 sims |
| 3 | Balancear final e inicio | `0.35` | `forward_sampled` | `2` | `0.35` | `96` | `16000` | `120` | 100 samples, 300 sims |
| 4 | Dar mas libertad a MCTS | `0.50` | `forward_sampled` | `3` | `0.40` | `128` | `20000` | `150` | 100 samples, 300-500 sims |

### Parametros por etapa

| Parametro | Etapa | Valor inicial sugerido | Uso | Efecto esperado | Riesgo / nota |
|---|---|---:|---|---|---|
| `n_samples` | test set | `100` | Cantidad de problemas fijos de validacion. | Comparacion estable entre checkpoints. | No cambiarlo frecuentemente si se quiere comparabilidad historica. |
| `n_samples` | raw | `5000` | Cantidad de trayectorias raw iniciales. | Mayor diversidad base. | Mas costo de generacion y almacenamiento. |
| `--workers` | raw / stock / MCTS / eval | `10` | Paralelismo de generacion. | Mejor throughput en Ryzen 9 5900X. | Subir demasiado puede saturar CPU o colas. |
| `--device` | MCTS / eval | `cuda` | Device del evaluador ResNet. | Usa GPU local. | Valores validos: `auto`, `cpu`, `cuda`; no usar `gpu`. |
| `--noise-k-max` | raw / test | `0.8` | Maximo descuento de ruido. | Genera demanda residual variada. | Valores muy altos pueden generar casos demasiado dificiles. |
| `--noise-k-lambda` | raw / test | `10.0` | Concentracion de ruido cerca de cero. | Muchos casos cercanos a factibles y algunos mas duros. | Menor lambda aumenta ruido efectivo promedio. |
| `--p-stock` | stock | `0.25` | Probabilidad de reducir stock. | Aumenta casos con `expansion_mode`. | Si es muy alto, sesga demasiado hacia falta de recursos. |
| `--p-mcts` | MCTS training | `0.15 -> 0.50` | Probabilidad de generar trayectorias con MCTS. | Aumenta calidad de policy targets a medida que aprende. | Muy alto al inicio puede gastar mucho sin suficientes rewards buenos. |
| `--start-mode` | MCTS training | `backward_sampled`, luego `forward_sampled` | Seleccion de estados semilla. | Backward concentra aprendizaje cerca del final; forward da libertad progresiva. | Cambiar muy pronto puede diluir muestras ganadoras. |
| `--max-seed-states` | MCTS training | `1 -> 3` | Estados adicionales para arrancar MCTS. | Mas trayectorias MCTS por job. | Multiplica costo y samples correlacionados. |
| `--seed-state-probability` | MCTS training | `0.35 -> 0.50` | Probabilidad de elegir cada estado candidato. | Controla densidad de semillas intermedias. | Si es `1`, toma los primeros candidatos hasta el maximo. |
| `--mcts-simulations` | MCTS training | `32 -> 128` | Simulaciones por decision. | Policies MCTS mas informadas. | Es uno de los mayores multiplicadores de tiempo. |
| `--mcts-simulations` | evaluacion | `300` | Profundidad de evaluacion contra test set. | Medicion mas robusta de reward final. | `500` puede ser caro; usar para evaluaciones puntuales. |
| `--sample-limit-per-cycle` | MCTS + learner | `8000 -> 20000` | Samples antes de cerrar ciclo. | Controla frecuencia de entrenamiento/reload. | Al alcanzarlo, deja de asignar jobs y espera activos; puede parecer pausa. |
| `--learner-steps` | learner | `80 -> 150` | Updates por ciclo. | Mas aprendizaje por ciclo. | Demasiados steps pueden sobreajustar al buffer reciente. |
| `--learner-batch-size` | learner | `64` | Batch de entrenamiento. | Estable con GPU 8GB. | Subirlo aumenta VRAM. |
| `--evaluator-batch-size` | MCTS / eval | igual a `--workers` | Maximo batch del evaluator centralizado. | Mejor uso de GPU sin esperar batches enormes. | Si es muy alto con pocos workers no aporta. |
| `--evaluator-batch-wait` | MCTS / eval | `0` | Espera para juntar requests. | Reduce latencia de MCTS sincrono. | `0.01` puede acumular mucha espera. |
| `--request-timeout` | MCTS / eval | `120` | Timeout de requests al evaluator. | Evita bloqueos silenciosos si evaluator muere. | Debe ser mayor que una inferencia razonable bajo carga. |
| `--run-prefix` | MCTS training | `train_gpu_mid` | Prefijo para IDs correlativos de corrida. | Genera IDs ordenables como `train_gpu_mid_001`. | Cada prefijo mantiene su propia secuencia. |
| `--run-id` | MCTS training | `None` | ID manual de corrida. | Permite nombrar una corrida excepcional. | Tiene prioridad y no consume el correlativo automatico. |
| `--overwrite-samples` | MCTS training | solo fase 1 | Recrea `samples.zarr`. | Arranque limpio. | No usar si se quiere acumular muestras entre fases. |
| `--overwrite` | evaluacion | activado para corrida unica | Recrea output de evaluacion. | Dashboard limpio de ultima evaluacion. | Si se quiere historico por trayectoria, no usar overwrite o separar `output-root`. |

### Logs y salidas de monitoreo

| Archivo / salida | Generado por | Frecuencia | Contenido principal | Uso recomendado |
|---|---|---|---|---|
| Consola `[dataset_generation] jobs=...` | raw / stock | Cada `progress_interval` | Jobs procesados, OK, failed, saved, rate. | Detectar fallas o caidas de throughput. |
| Consola `[mcts_generation] jobs=...` | MCTS generation | Cada `progress_interval` | Jobs OK/failed, samples acumulados, rate. | Ver avance de workers y cantidad de samples. |
| Consola `[mcts_generation] cycle_ready=...` | MCTS + learner | Al cerrar ciclo | Resumen de ciclo: jobs, samples, MCTS vs reweighted. | Confirmar que `sample_limit_per_cycle` disparo entrenamiento. |
| Consola `[mcts_generation] learner_done ...` | learner | Al terminar entrenamiento de ciclo | Checkpoint, global_step, loss, policy_loss, value_loss. | Ver si el modelo entrena y que checkpoint recargo. |
| `datasets/reports/mcts_generation_runs.jsonl` | `generate_mcts_samples.py` | Una fila por corrida | Args, report final, jobs, samples, errores. | Historial general de corridas de entrenamiento. |
| `datasets/reports/mcts_generation_cycles.jsonl` | `generate_mcts_samples.py` | Una fila por ciclo | Ciclo, samples, MCTS jobs, reweighted jobs, checkpoint. | Analizar curriculum y frecuencia de entrenamiento. |
| `datasets/reports/mcts_generation_learner_steps.jsonl` | learner | Una fila por step | `loss`, `policy_loss`, `value_loss`, `mean_policy_weight`. | Graficar curva de aprendizaje. |
| `datasets/reports/run_sequences.json` | logger MCTS | Al crear una corrida automatica | Ultimo correlativo reservado por `run-prefix`. | Mantenerlo al limpiar logs para no repetir IDs. |
| `datasets/evaluation/mcts_test/reports/trajectories.jsonl` | `evaluate_test_set_mcts.py` | Una fila por trayectoria evaluada | Reward final, value original, value_error, states_count, elapsed. | Analizar casos buenos/malos y tiempos. |
| `datasets/evaluation/mcts_test/reports/run_summary.json` | evaluacion | Ultima corrida | Positivos, mejores que original, medias, errores. | Resumen rapido de la evaluacion actual. |
| `datasets/evaluation/mcts_test/reports/runs.jsonl` | evaluacion | Una fila por corrida | Historial de summaries de evaluacion. | Curvas de `positive_rate`, reward medio y mejora por checkpoint. |
| `datasets/reports/training_dashboard.html` | dashboard | Al ejecutar builder | Snapshot visual de entrenamiento y evaluacion. | Monitoreo principal. |
| `datasets/reports/trajectory_explorer.html` | dashboard | Al ejecutar builder | Ultimas trayectorias raw, stock, samples y test MCTS. | Inspeccion estado por estado. |

## Referencia de parametros

### Parametros comunes

| Parametro | Scripts | Default | Que controla | Notas |
|---|---|---:|---|---|
| `--workers` | todos los generadores | varia por script | Cantidad de procesos worker. | En MCTS con `workers > 1` se usa un evaluador centralizado. |
| `--seed` | raw, stock, MCTS, test set | `None` | Semilla para ordenar jobs y decisiones aleatorias. | Si queda en `None`, cada corrida genera variacion nueva. |
| `--overwrite` | raw, stock, test set, evaluacion | desactivado | Recrea el buffer destino. | Borra/recrea el output del script correspondiente. |
| `--progress-interval` | raw, stock, MCTS, evaluacion | varia | Frecuencia de logs de progreso. | No cambia la ejecucion, solo la impresion. |
| `--device` | MCTS, evaluacion | `auto` o indicado | Device del evaluador ResNet. | Valores validos: `auto`, `cpu`, `cuda`. No usar `gpu`. |
| `--checkpoint-path` | MCTS, evaluacion | varia | Checkpoint ResNet a cargar. | En evaluacion, si no se pasa, usa el `.pt` con mayor step numerico. |
| `--checkpoint-dir` | MCTS, evaluacion | `modules/evaluators/resnet/checkpoints` | Directorio de checkpoints. | El learner escribe nuevos `.pt` ahi. |

### Ruido y datos raw

| Parametro | Default | Que controla | Notas |
|---|---:|---|---|
| `n_samples` | requerido | Cantidad de trayectorias raw a generar. | Argumento posicional de `generate_raw_demand_dataset.py` y `generate_initial_state_test_set.py`. |
| `--allowed-entry-hours` | `6 12 18` | Horarios de entrada permitidos. | Si se restringen demasiado, cambia el espacio legal. |
| `--closing-hour` | `22` | Hora de cierre. | Ningun horario de entrada permitido puede ser mayor o igual al cierre. |
| `--max-overcoverage-tolerance` | `0.1` | Tolerancia maxima de sobrecobertura. | Es el `k` del `ProblemSetup`, no el ruido efectivo. |
| `--noise-k-max` | `0.8` | Maximo del descuento de ruido. | El `k` efectivo se samplea en `[0, noise_k_max]`. |
| `--noise-k-lambda` | `10.0` | Forma de la exponencial truncada del ruido. | Mas alto concentra mas masa cerca de `0`. |
| `--mod-4-max`, `--mod-6-max`, `--mod-8-max` | `20` | Stock maximo por modalidad. | Define el rango de recursos iniciales por modalidad. |

### Stock adjusted

| Parametro | Default | Que controla | Notas |
|---|---:|---|---|
| `--source-path` | `datasets/raw/trajectories.zarr` | Buffer raw fuente. | Si no se pasa, usa el layout default. |
| `--output-path` | `datasets/derived/stock_adjusted/trajectories.zarr` | Buffer stock destino. | Guarda `stock_<raw_id>`. |
| `--p-stock` | `0.2` | Probabilidad de reducir stock. | Si no reduce, copia la raw directamente. |
| `--n-samples` | `None` | Cantidad de raw a procesar. | Si queda en `None`, procesa todas. |
| `--shuffle` | desactivado | Mezcla IDs fuente antes de aplicar `--n-samples`. | Util para procesar tandas representativas. |
| `--skip-existing` | desactivado | Saltea trayectorias stock ya generadas. | Se ignora si se usa `--overwrite`. |

### MCTS generation y learner

| Parametro | Default | Que controla | Notas |
|---|---:|---|---|
| `--source-path` | `datasets/derived/stock_adjusted/trajectories.zarr` | Buffer stock fuente. | Cada job toma una trayectoria stock. |
| `--sample-path` | `datasets/samples/samples.zarr` | SampleBuffer destino. | El orquestador aplana estados `X` e `Y`. |
| `--n-trajectories` | `None` | Cantidad de trayectorias stock a procesar. | Si queda en `None`, procesa todas. |
| `--p-mcts` | `0.2` | Probabilidad de usar MCTS por trayectoria stock. | Con `1 - p_mcts`, solo recalcula policy reweighted. |
| `--start-mode` | `initial_only` | Estados desde los que arranca MCTS. | Opciones: `initial_only`, `forward_sampled`, `backward_sampled`. |
| `--max-seed-states` | `0` | Cantidad maxima de estados adicionales. | Ademas del estado inicial en modos sampled. |
| `--seed-state-probability` | `0.0` | Probabilidad de seleccionar cada estado candidato. | Solo aplica en `forward_sampled` y `backward_sampled`. |
| `--mcts-simulations` | `16` | Simulaciones por decision MCTS. | Aumentarlo mejora busqueda pero multiplica tiempo. |
| `--c-puct` | `1.0` | Exploracion PUCT. | Mas alto explora mas acciones con prior. |
| `--mcts-policy-weight` | `1.0` | Peso de policy loss para samples MCTS. | Solo afecta policy loss durante entrenamiento. |
| `--reweighted-policy-weight` | `0.5` | Peso de policy loss para samples reweighted. | Baja importancia de policies artificiales. |
| `--sample-limit-per-cycle` | `None` | Samples maximos antes de cerrar ciclo. | Al alcanzarlo, deja de asignar jobs nuevos y espera jobs activos. |
| `--train-on-cycle` | desactivado | Entrena learner al cerrar cada ciclo. | Luego recarga pesos del evaluador. |
| `--learner-steps` | `100` | Steps de entrenamiento por ciclo. | Solo aplica con `--train-on-cycle`. |
| `--learner-batch-size` | `64` | Batch size del learner. | Batches aleatorios desde `SampleBuffer`. |
| `--learner-learning-rate` | `1e-4` | Learning rate. | Solo aplica con `--train-on-cycle`. |
| `--learner-weight-decay` | `1e-4` | Weight decay. | Regularizacion del optimizer. |
| `--learner-value-loss-weight` | `1.0` | Peso global de value loss. | No usa `policy_weight`. |
| `--learner-policy-loss-weight` | `1.0` | Peso global de policy loss. | Se combina con `policy_weight` por sample. |
| `--evaluator-batch-size` | `32` | Batch maximo del evaluador centralizado. | Con pocos workers, usar un valor cercano a `--workers` puede reducir latencia. |
| `--evaluator-batch-wait` | `0.01` | Espera maxima para armar batch de inferencia. | En MCTS pesado conviene probar `0`. |
| `--request-timeout` | `None` | Timeout de requests al evaluador. | Conviene usarlo para evitar bloqueos silenciosos. |
| `--run-prefix` | `train_gpu_mid` | Prefijo para ID correlativo automatico. | Ejemplo: `train_gpu_mid_001`, luego `_002`. |
| `--run-id` | `None` | ID manual de corrida. | Si se pasa, tiene prioridad sobre `--run-prefix`. |

### Evaluacion del test set con MCTS

| Parametro | Default | Que controla | Notas |
|---|---:|---|---|
| `--sample-path` | `datasets/test/initial_states.zarr` | Test set fijo fuente. | Formato `SampleBuffer`, un estado inicial por problema. |
| `--output-root` | `datasets/evaluation/mcts_test` | Directorio de salida. | Contiene `trajectories.zarr` y `reports/`. |
| `--trajectory-path` | `<output-root>/trajectories.zarr` | Buffer destino de trayectorias evaluadas. | Separado de entrenamiento. |
| `--reports-dir` | `<output-root>/reports` | Reportes JSON. | Escribe `trajectories.jsonl`, `run_summary.json`, `runs.jsonl`. |
| `--n-samples` | `None` | Cantidad de samples del test a evaluar. | Si queda en `None`, evalua todos. |
| `--mcts-simulations` | `500` | Simulaciones por decision MCTS. | Es mucho mas caro que el default de entrenamiento. |
| `--evaluator-batch-wait` | `0.01` | Espera para batching del evaluador. | Para evaluar con pocos workers, probar `0`. |
| `--request-timeout` | `None` | Timeout de requests al evaluador. | Recomendado: `60` o mas, para detectar evaluator caido. |

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
uv run python scripts/generate_initial_state_test_set.py --help
uv run python scripts/generate_stock_adjusted_dataset.py --help
uv run python scripts/generate_mcts_samples.py --help
uv run python scripts/evaluate_test_set_mcts.py --help
```

## Pruebas recomendadas despues de cambios

```bash
.venv/bin/python -m compileall scripts modules
.venv/bin/python -m pytest tests/test_demand_noise.py tests/test_initial_state_test_set.py tests/test_mcts_generation_helpers.py tests/test_mcts_generation_worker.py tests/test_resnet_sample_learner.py -q
```

El test de orquestador usa Zarr real y, si el sandbox cuelga al abrir buffers,
conviene correrlo con timeout fuera del sandbox:

```bash
timeout 30 .venv/bin/python -m pytest tests/test_mcts_generation_orchestrator.py -q
```

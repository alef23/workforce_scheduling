# Diseño experimental: acciones compuestas

Este documento registra los criterios acordados para la rama
`experiment/compound-actions`. Los cambios se implementarán por etapas para
poder validar cada decisión de diseño de forma aislada.

## Dominio fijo

- Modalidades disponibles: `4`, `6` y `8` horas.
- Horarios de ingreso permitidos: `6`, `12` y `18`.
- Horario de cierre: `22`.
- Franco fijo: domingo, representado por el día `6`.
- Un franco móvil, seleccionable entre los días `0` a `5`.
- Máximo de recursos totales iniciales: `20`, sumando las tres modalidades.

## Espacio de acciones

Cada acción representa una asignación semanal completa:

```text
(modalidad, horario de ingreso, franco móvil)
```

Los índices válidos son:

```text
modality_index  = 0, 1, 2  -> 4, 6, 8 horas
entry_hour_index = 0, 1, 2 -> 6, 12, 18 horas
mobile_day_off   = 0..5
```

El espacio global contiene `3 x 3 x 6 = 54` acciones:

```text
action_id = modality_index * 18
          + entry_hour_index * 6
          + mobile_day_off
```

Decodificación:

```text
modality_index  = action_id // 18
remainder       = action_id % 18
entry_hour_index = remainder // 6
mobile_day_off   = remainder % 6
```

El espacio global siempre contiene 54 IDs. La máscara legal también aplica el
cierre operativo `entry_hour + modality <= 22`. Por eso, con stock disponible
en las tres modalidades, al iniciar un recurso son legales como máximo:

```text
4 horas: 18 acciones
6 horas: 12 acciones
8 horas: 12 acciones
total:    42 acciones
```

Durante las semanas restantes, la modalidad activa queda fijada: pueden ser
legales 18 acciones para modalidad 4h o 12 para modalidades 6h y 8h.

## Estado e inputs de la red

### Demanda residual

- Shape crudo: `(24, 28)`.
- Se mantiene como entrada principal de la red.
- Cada celda se normaliza dividiéndola por `20`.

La constante `20` representa el máximo esperado por celda. Esta hipótesis debe
ser validada al generar el nuevo dataset para evitar valores normalizados muy
superiores a `1`.

### Demanda inicial total

- Se mantiene en el estado y como input de la red.
- Se mantiene para el cálculo del score final.

Como es la suma de las `24 x 28` celdas, no debe normalizarse únicamente por
`20`. La referencia consistente con la demanda residual es:

```text
initial_demand_total_ref = 20 * 24 * 28 = 13.440
```

Por lo tanto:

```text
normalized_initial_demand_total =
    initial_demand_total / 13.440
```

### Stock disponible

- Se conservan tres valores, uno por modalidad.
- Cada valor se normaliza dividiéndolo por `20`.
- La generación de problemas debe garantizar:

```text
stock_4 + stock_6 + stock_8 <= 20
```

- El stock inicial total debe ser mayor que cero.

### Semana activa

- Valores crudos: `0`, `1`, `2`, `3`.
- Identifica cuál de las cuatro semanas del recurso se está asignando.
- Se mantendrá inicialmente el encoding one-hot reducido actual:
  - semana `0`: implícita;
  - semana `1`: canal propio;
  - semana `2`: canal propio;
  - semana `3`: canal propio.

### Modalidad activa

Debe mantenerse como input aunque el setup sea fijo:

- `None` indica que comienza un recurso y habilita hasta 54 acciones.
- `4`, `6` u `8` fija la modalidad durante las semanas restantes y restringe
  la máscara legal a 18 acciones.

Se mantendrá inicialmente su representación one-hot.

## Score y value

- Se conserva `initial_demand_total`.
- Se conserva el score actual.
- El reward terminal continúa retropropagándose sin transformación a todos los
  estados de la trayectoria.

El cambio del espacio de acciones se evaluará antes de experimentar con una
definición distinta del value.

## Variables que dejan de ser inputs dinámicos

Como estos valores quedan fijados para todo el experimento, no aportan
información para distinguir estados:

- `allowed_entry_hours = [6, 12, 18]`
- `closing_hour = 22`
- `fixed_day_off = 6`
- `mobile_days_off_count = 1`

En una primera implementación pueden conservarse como canales constantes para
reducir el alcance del cambio. Luego podrán eliminarse del encoder una vez
validado el nuevo engine y el nuevo MCTS.

`current_entry_hour` deja de ser necesario porque cada acción compuesta aplica
directamente modalidad, horario y franco móvil para una semana completa.

## Decisiones pendientes

- Confirmar que `20` es también el máximo esperado de demanda por celda.
- Definir si los canales constantes del setup se conservan durante el primer
  prototipo o se eliminan inmediatamente.
- Definir la nueva versión del schema para buffers y checkpoints.

## Prototipo de red

La primera variante experimental utiliza:

```text
input_channels = 11
action_space_size = 54
hidden_channels = 128
num_res_blocks = 8
```

Los 11 canales son:

```text
1 demanda residual
1 demanda inicial total
3 stock por modalidad
3 modalidad activa
3 semana activa, con semana 0 implícita
```

El checkpoint inicial se guarda separado de los modelos anteriores:

```text
modules/evaluators/resnet/checkpoints_compound_actions/
```

El wrapper de inferencia experimental es:

```text
modules/evaluators/resnet/compound_evaluator.py
```

`CompoundResNetEvaluator` cumple el contrato actual de MCTS mediante
`predict(state)`, soporta inferencia agrupada con `predict_batch(states)` y
permite actualizar el modelo con `reload_weights(checkpoint_path)`. La policy
mantiene las 54 acciones; la máscara legal y su renormalización continúan
siendo responsabilidad de `CompoundWorkforceEngine`.

El MCTS existente no requiere una variante propia. La integración se valida
sin modificarlo mediante una trayectoria completa de cuatro semanas y el
benchmark:

```bash
uv run python scripts/benchmark_compound_mcts.py
```

La generación de cobertura utiliza un simulador propio del dominio:

```text
modules/demand_simulator/compound_demand_simulator.py
```

`CompoundDemandSimulator` recibe únicamente `n_resources`, genera directamente
cuatro acciones semanales por recurso y deriva el stock por modalidad desde la
primera acción de cada bloque. Luego reconstruye la trayectoria positiva con
`CompoundWorkforceState` y policies de 54 posiciones. No consulta el engine
durante esta etapa; el `DemandSimulator` anterior permanece intacto.

Después de aplicar `DemandNoiseGenerator`, `CompoundTrajectoryReplayer`
reproduce los `action_id` mediante `CompoundWorkforceEngine`. El replay se
detiene en la primera terminalidad, recalcula el reward real y lo retropropaga
a todos los samples resultantes.

`CompoundStockAdjuster` divide esa trayectoria en chunks de cuatro acciones,
samplea chunks completos para formar el stock reducido, ubica primero los
seleccionados y vuelve a ejecutar `CompoundTrajectoryReplayer`. Los chunks no
seleccionados continúan después del agotamiento del stock en `expansion_mode`.

El script reproducible de inicialización y benchmark es:

```bash
uv run python scripts/benchmark_compound_resnet.py
```

Los resultados completos de inferencia, entrenamiento, convivencia
learner/actor, CPU y memoria del replay buffer se encuentran en:

```text
docs/compound_actions_benchmarks.md
```

Conclusiones actuales:

- replay buffer de 500.000 samples crudos: viable;
- generar secuencialmente los 42 sucesores iniciales tarda aproximadamente
  1,43 ms;
- inferencia GPU aislada con batch 3.072: viable;
- entrenamiento FP32 con batch físico 512: OOM;
- batch efectivo 3.072: viable mediante acumulación de gradientes;
- learner y actor pueden residir simultáneamente en GPU;
- el MCTS actual funciona sin cambios con el engine y evaluador compuestos;
- en CPU, la inferencia ResNet concentra más del 96% del tiempo del MCTS.

## Generación paralela del dataset compuesto

La generación definitiva utiliza un único buffer. Cada
`CompoundFullTrajectoryWorker` mantiene en memoria la secuencia:

```text
CompoundDemandSimulator
-> DemandNoiseGenerator
-> CompoundTrajectoryReplayer
-> CompoundStockAdjuster
```

Solo la trayectoria resultante se envía al proceso principal. El
`CompoundDatasetOrchestrator` reparte jobs independientes con
`ProcessPoolExecutor`, recibe los resultados a medida que terminan y centraliza
la escritura en `CompoundTrajectoryBuffer`.

Decisiones:

- no existen buffers intermedios para raw, noise o stock;
- los workers no escriben directamente en Zarr;
- cada job obtiene aleatoriedad desde el sistema y no expone seed;
- cada job samplea uniformemente sus recursos totales en
  `[1, max_n_resources]`;
- el multiproceso usa `spawn` por defecto;
- el buffer conserva estados crudos, policies de 54 acciones, rewards,
  `ProblemSetup` y metadata de trazabilidad.

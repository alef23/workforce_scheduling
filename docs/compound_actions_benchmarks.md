# Benchmarks del prototipo de acciones compuestas

Este documento consolida las mediciones realizadas sobre la variante
experimental de ResNet con acciones compuestas.

Las pruebas cubren únicamente:

- transición individual del engine compuesto;
- encoding de estados;
- inferencia ResNet;
- entrenamiento ResNet;
- convivencia de un modelo learner y un modelo actor;
- inferencia CPU;
- memoria estimada del replay buffer.
- integración y costo de trayectorias MCTS completas.

## Hardware y software

```text
GPU: NVIDIA GeForce RTX 3060
VRAM física: 8.192 MB
CPU: AMD Ryzen 9 5900X
CPU física: 12 cores / 24 threads
PyTorch: 2.12.0+cu130
CUDA de PyTorch: 13.0
```

PyTorch utilizó 12 threads para las mediciones CPU.

## Modelo medido

```text
input_channels: 11
action_space_size: 54
hidden_channels: 128
num_res_blocks: 8
policy_channels: 8
value_channels: 4
value_hidden_dim: 256
```

Checkpoint inicial:

```text
modules/evaluators/resnet/checkpoints_compound_actions/
workforce_resnet_compound_000000.pt
```

El checkpoint contiene pesos inicializados y no entrenados.

## Engine compuesto

Se midió el engine NumPy actual llamando individualmente a `step()`:

- una transición seleccionada;
- todas las transiciones legales desde el mismo estado.

No se utilizó ni implementó una API batch. Cada sucesor se construyó mediante
una llamada independiente y validada a `step()`.

| Contexto | Acciones legales | Un `step` | Todos los sucesores |
|---|---:|---:|---:|
| Inicio de recurso | 42 | 0,035 ms | 1,426 ms |
| Modalidad 4h activa | 18 | 0,027 ms | 0,465 ms |
| Modalidad 6h activa | 12 | 0,027 ms | 0,310 ms |
| Modalidad 8h activa | 12 | 0,027 ms | 0,308 ms |

Los valores son medianas de 500 repeticiones.

También se compararon las transiciones contra el engine anterior:

```text
engine anterior:
modalidad -> horario -> franco móvil

engine compuesto:
acción semanal única
```

Las pruebas verifican igualdad de demanda residual, stock, modalidad activa,
semana, expansion mode, terminalidad y reward. También se validó un recurso
completo de cuatro semanas.

Conclusiones:

- El engine compuesto reproduce la semántica del engine anterior para el
  dominio fijo.
- Generar secuencialmente todos los sucesores cuesta menos de 1,5 ms.
- Con la implementación actual, la transición no aparece como cuello de
  botella frente a la inferencia ResNet.
- No hay evidencia actual que justifique introducir una transición batch.

## Inferencia GPU

La medición incluye:

```text
inputs NumPy crudos -> encoder GPU -> ResNet
```

Se realizaron warm-ups y múltiples repeticiones sincronizando CUDA antes de
leer los tiempos.

| Batch | Mediana | P95 | Estados/s | VRAM asignada | VRAM reservada |
|---:|---:|---:|---:|---:|---:|
| 256 | 144,61 ms | 147,30 ms | 1.770 | 534 MB | 640 MB |
| 512 | 291,67 ms | 292,75 ms | 1.755 | 1.033 MB | 1.128 MB |
| 1.024 | 592,66 ms | 594,90 ms | 1.728 | 2.029 MB | 2.108 MB |
| 2.048 | 1.183,65 ms | 1.186,33 ms | 1.730 | 4.023 MB | 4.852 MB |
| 3.072 | 1.939,06 ms | 1.964,18 ms | 1.584 | 4.843 MB | 6.028 MB |
| 3.328 | 2.109,91 ms | 2.114,91 ms | 1.577 | 5.243 MB | 6.548 MB |
| 3.584 | 2.288,74 ms | 2.289,77 ms | 1.566 | 5.644 MB | 7.046 MB |
| 3.840 | OOM | - | - | - | - |
| 4.056 | OOM | - | - | - | - |

Conclusiones:

- La eficiencia se mantiene cercana a 1.700 estados/s hasta batch 2.048.
- El throughput comienza a degradarse en 3.072.
- Batch 3.584 entra, pero deja poco margen operativo.
- Batch 3.840 y 4.056 producen OOM.
- Batch 3.072 es un máximo prudente para inferencia aislada.

## Entrenamiento GPU

La prueba ejecutó un paso FP32 completo:

```text
encoder
-> forward
-> policy loss + value loss
-> backward
-> AdamW.step
```

No se guardó ni modificó ningún checkpoint.

| Batch físico | Tiempo por step | VRAM asignada | VRAM reservada | Resultado |
|---:|---:|---:|---:|---|
| 256 | 817,89 ms | 3.708 MB | 4.162 MB | OK |
| 384 | 1.045,09 ms | 5.560 MB | 6.190 MB | OK |
| 416 | 1.119,80 ms | 6.006 MB | 6.682 MB | OK |
| 448 | 1.194,50 ms | 6.444 MB | 6.800 MB | OK |
| 480 | 1.295,70 ms | 6.870 MB | 7.088 MB | OK |
| 512 | - | - | - | OOM |

Conclusiones:

- Batch físico 512 no entra en FP32 con AdamW.
- Batch 480 entra, pero no deja un margen operativo cómodo.
- Batch 384 es una opción prudente si el learner usa la GPU en exclusividad.
- Un batch efectivo de 3.072 puede lograrse con:

```text
microbatch 256 x 12 acumulaciones
microbatch 384 x 8 acumulaciones
```

## Learner y actor residentes en GPU

Se cargaron simultáneamente:

- un modelo learner en modo entrenamiento;
- un modelo actor en modo inferencia;
- un optimizer AdamW para el learner.

Después del paso de entrenamiento se ejecutó una inferencia de 10 o 20 estados.
Las operaciones se ejecutaron secuencialmente sobre la GPU, no en paralelo.

| Batch learner | Batch actor | Training | Inferencia | VRAM reservada |
|---:|---:|---:|---:|---:|
| 384 | 10 | 1.072,17 ms | 12,16 ms | 6.202 MB |
| 384 | 20 | 1.046,61 ms | 17,42 ms | 6.202 MB |
| 416 | 10 | 1.107,77 ms | 12,34 ms | 6.694 MB |
| 416 | 20 | 1.106,74 ms | 16,89 ms | 6.694 MB |
| 448 | 10 | 1.187,07 ms | 13,38 ms | 6.810 MB |
| 448 | 20 | 1.205,21 ms | 18,47 ms | 6.810 MB |

Los dos modelos residentes, sin activaciones del entrenamiento, ocuparon
aproximadamente 40 MB asignados. Las activaciones, gradientes y estados del
optimizer dominan el uso de VRAM.

Conclusiones:

- Dos modelos caben simultáneamente en la RTX 3060.
- Una inferencia de 10 o 20 estados agrega poco al pico de memoria.
- Batch learner 384 sigue siendo viable con el actor residente.
- Estas pruebas no demuestran que entrenamiento e inferencia puedan ejecutarse
  concurrentemente sin aumentar latencias. Ambos competirían por la misma GPU.

## Inferencia CPU

| Batch | Mediana | P95 | Estados/s |
|---:|---:|---:|---:|
| 1 | 9,85 ms | 10,48 ms | 102 |
| 10 | 54,73 ms | 58,48 ms | 183 |
| 20 | 112,29 ms | 125,72 ms | 178 |

Comparación para batch 20:

```text
CPU: aproximadamente 112 ms
GPU con actor y learner residentes: aproximadamente 17 ms
```

La inferencia CPU es unas 6,5 veces más lenta para batch 20. Todavía no puede
decidirse si la CPU será suficiente para el actor en producción; la medición
integral posterior confirma que la inferencia domina el costo del MCTS.

## Memoria del replay buffer

Se midieron 1.000 samples con:

```text
X:
  residual_demand int32, shape (24, 28)
  initial_demand_total int64
  remaining_stock int32, shape (3,)
  current_modality int32
  assignment_week int32

Y:
  policy float32, shape (54,)
  value float32
  policy_weight float32

action_id int32
```

| Representación | 1.000 samples | Por sample | 500.000 estimados |
|---|---:|---:|---:|
| Arrays NumPy compactos | 2,944 MB | 2.944 bytes | 1,472 GB |
| `deque` de samples individuales | 3,785 MB | 3.785 bytes | 1,893 GB |
| X encodeado float32 + Y | 34,724 MB | 34.724 bytes | 17,362 GB |

Conclusiones:

- Un replay de 500.000 samples crudos es viable en RAM.
- Una estructura compacta o buffer circular reduce el overhead frente a una
  `deque` de objetos individuales.
- Los estados deben almacenarse crudos.
- El tensor encodeado debe materializarse únicamente al preparar cada batch.

La estimación no incluye metadata adicional, fragmentación del allocator,
índices auxiliares ni copias temporales durante el muestreo.

## MCTS compuesto

Se utilizó el MCTS existente, sin modificaciones, junto con:

```text
CompoundWorkforceEngine
CompoundResNetEvaluator
workforce_resnet_compound_000000.pt
```

La prueba funcional determinista genera una trayectoria de cuatro semanas,
produce una policy `(54,)` en cada estado, cierra el recurso, alcanza un estado
terminal y verifica la retropropagación del reward en `Q`.

También se midieron trayectorias completas sobre CPU con demanda inicial
unitaria, stock `[7, 7, 6]` y tres repeticiones por configuración:

| Simulaciones por búsqueda | Profundidad máxima | Llamadas ResNet | Tiempo ResNet | Tiempo engine | Tiempo total |
|---:|---:|---:|---:|---:|---:|
| 10 | 3 | 72 | 0,563 s | 0,005 s | 0,581 s |
| 25 | 5 | 136 | 1,063 s | 0,010 s | 1,099 s |
| 50 | 6 | 202 | 1,569 s | 0,014 s | 1,627 s |

Todos los runs terminaron tras ocho acciones semanales y dos recursos
completos. El reward fue negativo porque el checkpoint contiene pesos
inicializados, sin entrenamiento; no representa la calidad esperada del
modelo.

Conclusiones:

- La integración completa funciona con policies de 54 acciones.
- Más del 96% del tiempo total corresponde a inferencia ResNet.
- El costo acumulado de `step()` permanece por debajo del 1% del wall time.
- Aumentar simulaciones incrementa la profundidad observada.
- Estas cifras CPU no sustituyen una medición posterior del actor sobre GPU.

## Conclusiones provisionales

- Replay buffer de 500.000 samples crudos: viable, alrededor de 1,5 a 1,9 GB.
- Inferencia GPU aislada: batch 3.072 viable.
- Entrenamiento físico FP32: batch prudente entre 256 y 384.
- Batch efectivo 3.072: viable mediante acumulación de gradientes.
- Learner y actor pueden residir simultáneamente en GPU.
- Inferencia CPU es posible, pero considerablemente más lenta.
- El MCTS actual funciona sin cambios con el dominio compuesto.
- La decisión final de arquitectura queda pendiente de medir el actor MCTS
  sobre GPU y de coordinar learner y actor.

## Reproducción

Inicialización, inferencia GPU y memoria RAM:

```bash
uv run python scripts/benchmark_compound_resnet.py
```

El script no ejecuta entrenamiento. Las pruebas de entrenamiento realizadas
fueron experimentos puntuales sin escritura de checkpoints.

Benchmark del engine:

```bash
uv run python scripts/benchmark_compound_engine.py
```

Benchmark integral del MCTS compuesto:

```bash
uv run python scripts/benchmark_compound_mcts.py
```

El benchmark utiliza el MCTS existente sin modificarlo. Para cada cantidad de
simulaciones registra trayectoria, reward, recursos completados, nodos,
profundidad máxima observada, llamadas y tiempo del evaluador, llamadas y
tiempo de `step`, y duración total. El checkpoint inicial no está entrenado,
por lo que estas mediciones describen costo e integración, no calidad.

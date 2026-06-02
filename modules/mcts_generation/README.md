# MCTS Generation

Generacion de muestras de entrenamiento desde trayectorias `stock_adjusted`.

Este modulo define las piezas base para el futuro worker MCTS:

- seleccion de estados semilla;
- policy artificial reweighted para trayectorias sin MCTS;
- worker que devuelve trayectorias finalizadas;
- schemas de resultados que el orquestador podra aplanar en `SampleBuffer`.

## Modos de semillas MCTS

`initial_only`:

- selecciona solo el estado inicial.

`forward_sampled`:

- incluye siempre el estado inicial;
- recorre la trayectoria fuente hacia adelante;
- selecciona hasta `max_seed_states` estados adicionales con probabilidad
  `seed_state_probability`.

`backward_sampled`:

- incluye siempre el estado inicial;
- recorre la trayectoria fuente hacia atras;
- no considera el estado terminal;
- selecciona hasta `max_seed_states` estados adicionales con probabilidad
  `seed_state_probability`.

Los indices seleccionados son puntos de partida. Cada trayectoria MCTS generada
desde esos estados debe continuar hasta terminalidad.

## Policy reweighted

Para la rama sin MCTS se parte de la policy original de la trayectoria stock. Las
acciones legales son las entradas no cero.

Para `Nl` acciones legales:

```text
selected_action = 1 / (Nl - 1)
other_legal     = (Nl - 2) / (Nl - 1)^2
illegal         = 0
```

Casos borde:

- `Nl == 1`: la accion seleccionada recibe `1.0`.
- `Nl == 2`: la accion seleccionada recibe `1.0` y la otra accion legal `0.0`.

La policy sigue sumando `1.0`. El peso de confianza se guarda aparte como
`policy_weight`.

## Worker

`MCTSGenerationWorker` toma una trayectoria `stock_adjusted` y sortea:

```text
usar_mcts ~ Bernoulli(p_mcts)
```

Si usa MCTS:

- selecciona semillas segun el modo configurado;
- genera una trayectoria completa desde cada semilla;
- usa `policy_weight = mcts_policy_weight`.

Si no usa MCTS:

- no genera trayectorias nuevas;
- devuelve la trayectoria fuente con policy reweighted;
- usa `policy_weight = reweighted_policy_config.policy_weight`.

El worker no escribe `SampleBuffer`. Devuelve trayectorias finalizadas para que
el orquestador las aplaste y persista.

## Orquestador minimo

`MCTSGenerationOrchestrator` conecta:

- `MCTSGenerationWorker`;
- evaluator;
- `SampleBuffer`.

La primera version es secuencial (`n_workers=1`) y no maneja learner ni ciclos.
Sirve para validar end-to-end que el worker devuelve trayectorias finalizadas y
que el orquestador las guarda como samples planos.

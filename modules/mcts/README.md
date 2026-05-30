# MCTS

Implementación agnóstica de dominio de Monte Carlo Tree Search.

El módulo no importa `WorkforceEngine`, `ProblemSetup` ni `WorkforceState`. Solo conoce interfaces mínimas para interactuar con un entorno y un evaluador.

## Archivos

```text
modules/mcts/
├── mcts.py
├── mcts_schemas.py
└── README.md
```

## Responsabilidad

`MCTS` explora un árbol de decisiones a partir de un estado raíz.

Sus responsabilidades son:

- mantener nodos y estadísticas del árbol;
- expandir nodos usando un evaluador policy/value;
- seleccionar acciones durante simulaciones con PUCT;
- avanzar estados delegando transiciones al engine;
- construir una policy final desde visit counts;
- seleccionar una acción final para entrenamiento o inferencia.

El MCTS no contiene reglas de workforce scheduling. Toda legalidad y transición pertenece al engine.

## Interfaces esperadas

### Engine

El objeto `engine` debe implementar:

```python
action_space_size: int

step(state, action_id) -> StepResult
legal_mask(state, policy) -> np.ndarray
check_terminality(state) -> bool
compute_reward(state) -> float
```

`StepResult` debe exponer:

```python
next_state
is_terminal: bool
reward: float
```

En este proyecto, `WorkforceEngine` cumple esta interfaz.

### Evaluator

El objeto `evaluator` debe cumplir `EvaluatorProtocol`:

```python
predict(state) -> tuple[np.ndarray, float]
```

Debe devolver:

- `policy`: vector de probabilidades o scores no negativos para todo el espacio de acciones.
- `value`: estimación escalar del valor del estado.

La policy se pasa por `engine.legal_mask`, que filtra acciones ilegales y renormaliza. El value no se modifica.

## Schemas

### `MCTSConfig`

Configuración de búsqueda:

| Campo | Descripción |
|---|---|
| `num_simulations` | Cantidad de simulaciones por llamada a `search`. |
| `c_puct` | Peso de exploración en la fórmula PUCT. |
| `temperature` | Reservado para control de policy en entrenamiento. Actualmente no se aplica en `select_final_action`. |
| `mode` | `training` o `inference`. |
| `random_seed` | Semilla opcional para selección estocástica en entrenamiento. |
| `debug` | Si es `True`, agrega diagnósticos simples al resultado. |

### `MCTSMode`

```python
MCTSMode.INFERENCE
MCTSMode.TRAINING
```

En `INFERENCE`, la acción final es:

```python
argmax(policy)
```

En `TRAINING`, la acción final se samplea usando la policy final.

### `MCTSResult`

Resultado de `search`:

| Campo | Descripción |
|---|---|
| `root_node_id` | ID del nodo raíz usado. |
| `selected_action_id` | Acción final elegida. |
| `policy` | Policy final sobre todo el espacio de acciones. |
| `root_stats` | Estadísticas por acción legal de la raíz. |
| `num_simulations` | Simulaciones ejecutadas. |
| `diagnostics` | Diagnósticos opcionales si `debug=True`. |

## Flujo de `search`

```text
root_state
    -> crear root si no existe
    -> expandir root si no estaba expandido
    -> repetir num_simulations:
        -> seleccionar acciones con PUCT hasta hoja
        -> expandir hoja o leer reward terminal
        -> backpropagar value/reward
    -> construir policy desde visit counts
    -> seleccionar acción final
    -> devolver MCTSResult
```

## PUCT

Para cada acción legal de un nodo expandido:

```text
score(s, a) = Q(s, a) + U(s, a)
```

donde:

```text
U(s, a) = c_puct * prior(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

`prior` es la probabilidad legal enmascarada que viene del evaluador.

## Uso básico

```python
from modules.evaluators.dummy.dummy_model import DummyEvaluator
from modules.mcts.mcts import MCTS
from modules.mcts.mcts_schemas import MCTSConfig, MCTSMode


config = MCTSConfig(
    num_simulations=50,
    c_puct=1.5,
    mode=MCTSMode.INFERENCE,
    random_seed=123,
)

evaluator = DummyEvaluator(action_space_size=engine.action_space_size)
mcts = MCTS(engine=engine, evaluator=evaluator, config=config)

result = mcts.search(root_state)

selected_action_id = result.selected_action_id
policy = result.policy
root_stats = result.root_stats
```

## Uso con ResNet

```python
from modules.evaluators.resnet.resnet_state_evaluator import ResNetStateEvaluator


evaluator = ResNetStateEvaluator(
    setup=problem_setup,
    checkpoint_path="modules/evaluators/resnet/checkpoints/workforce_resnet_000.pt",
    device="auto",
)

mcts = MCTS(engine=engine, evaluator=evaluator, config=config)
result = mcts.search(root_state)
```

## Reutilización del árbol

`search(root_state)` mantiene el árbol entre llamadas.

Después de ejecutar una acción real, se puede avanzar la raíz:

```python
mcts.advance_root(action_id)
```

Esto conserva el subárbol ya explorado bajo esa acción.

Si se quiere empezar desde cero:

```python
mcts.reset_tree()
```

## Consideraciones para este proyecto

- El estado raíz debe ser coherente con el engine.
- La terminalidad efectiva del `WorkforceEngine` ocurre al cerrar un recurso mensual.
- La policy del evaluador puede asignar masa a acciones ilegales; `legal_mask` la corrige.
- El value del evaluador se backpropaga sin transformación.
- Si un hijo es terminal, se backpropaga el reward terminal devuelto por el engine.

## Puntos pendientes

- Usar `temperature` para suavizar o concentrar la policy final en modo entrenamiento.
- Agregar tests unitarios sobre selección PUCT, expansión y reutilización de raíz.
- Documentar ejemplos específicos de integración con trayectorias de entrenamiento.

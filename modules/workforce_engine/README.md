# Workforce Engine

Motor determinístico para validar acciones, aplicar transiciones de estado y evaluar terminalidad en el problema de planificación de dotación.

El engine no decide cuál es la mejor acción. Recibe un `WorkforceState`, recibe una acción y devuelve el próximo estado. Está pensado para ser usado por MCTS, heurísticas, simuladores de trayectorias o evaluadores neuronales.

## Archivos

```text
modules/workforce_engine/
├── compound_actions.py
├── compound_engine.py
├── compound_schemas.py
├── engine.py
├── schemas.py
├── workforce_example.ipynb
└── README.md
```

Los archivos `compound_*` contienen una variante experimental independiente.
El engine original permanece sin cambios.

## Engine de acciones compuestas

`CompoundWorkforceEngine` restringe el dominio a:

```text
modalidades: [4, 6, 8]
horarios: [6, 12, 18]
franco fijo: 6
un franco móvil: 0..5
cierre: 22
```

Cada acción aplica una cobertura semanal completa y el espacio global contiene
54 IDs. La restricción de cierre reduce la cantidad máxima de acciones legales
a 42 al iniciar un recurso. Durante las semanas restantes, la modalidad queda
fijada.

El engine conserva el contrato requerido por el MCTS actual:

```python
action_space_size
step(state, action_id)
legal_mask(state, priors)
check_terminality(state)
compute_reward(state)
```

Ejemplo:

```python
from modules.workforce_engine.compound_actions import encode_action
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine
from modules.workforce_engine.compound_schemas import CompoundWorkforceState

engine = CompoundWorkforceEngine(setup)
state = CompoundWorkforceState(
    residual_demand=residual_demand,
    remaining_stock=remaining_stock,
    expansion_mode=False,
    current_modality=None,
    assignment_week=0,
    initial_demand_total=initial_demand_total,
)

action_id = encode_action(
    modality_index=0,
    entry_hour_index=1,
    mobile_day_off=3,
)
result = engine.step(state, action_id)
```

## Conceptos principales

El horizonte de planificación es fijo:

- 4 semanas.
- 28 días.
- 24 horas por día.

La demanda se representa como una matriz entera:

```python
residual_demand.shape == (24, 28)
```

La interpretación de cada celda es:

- `> 0`: demanda pendiente.
- `= 0`: demanda exactamente cubierta.
- `< 0`: sobrecobertura.

Las modalidades soportadas son:

```python
4, 6, 8
```

El stock se codifica como:

```python
remaining_stock = np.array([stock_4h, stock_6h, stock_8h])
```

## Schemas

Los dos objetos centrales son `ProblemSetup` y `WorkforceState`.

### `ProblemSetup`

Define las reglas fijas del problema. No contiene demanda ni estado dinámico.

```python
from modules.workforce_engine.schemas import ProblemSetup

setup = ProblemSetup(
    mobile_days_off_count=1,
    fixed_day_off=None,
    allowed_entry_hours=[8, 10, 12, 14],
    max_overcoverage_tolerance=0.2,
    closing_hour=22,
)
```

Campos:

| Campo | Descripción |
|---|---|
| `mobile_days_off_count` | Cantidad de francos móviles semanales. Puede ser `0`, `1` o `2`. |
| `fixed_day_off` | Franco fijo semanal. Puede ser `None` o un entero entre `0` y `6`. |
| `allowed_entry_hours` | Horarios de entrada permitidos. Si es `None`, se permiten todas las horas `0-23`. |
| `max_overcoverage_tolerance` | Tolerancia usada para terminalidad y recompensa por sobrecobertura. |
| `closing_hour` | Hora de cierre operativo. Si es `None`, los turnos pueden cruzar medianoche. |

Validaciones relevantes:

- La suma de franco fijo y francos móviles no puede superar `2`.
- `allowed_entry_hours` no puede tener duplicados.
- Si existe `closing_hour`, ningún horario permitido puede ser mayor o igual al cierre.
- La compatibilidad fina `entry_hour + modality <= closing_hour` se valida en el engine.

### `WorkforceState`

Representa el estado dinámico de una trayectoria.

```python
from modules.workforce_engine.schemas import WorkforceState

state = WorkforceState(
    residual_demand=np.ones((24, 28), dtype=int),
    remaining_stock=np.array([10, 5, 3], dtype=int),
    expansion_mode=False,
    current_modality=None,
    current_entry_hour=None,
    assignment_week=0,
    initial_demand_total=24 * 28,
)
```

Campos:

| Campo | Descripción |
|---|---|
| `residual_demand` | Matriz `(24, 28)` con demanda residual. |
| `remaining_stock` | Vector `(3,)` con stock de recursos de 4, 6 y 8 horas. |
| `expansion_mode` | Si es `False`, solo se pueden elegir modalidades con stock. Si es `True`, el stock ya no restringe. |
| `current_modality` | Modalidad del recurso actual: `None`, `4`, `6` u `8`. |
| `current_entry_hour` | Hora de entrada de la semana actual, o `None` si todavía no fue elegida. |
| `assignment_week` | Semana actual del recurso, de `0` a `3`. |
| `initial_demand_total` | Suma de la demanda inicial. Se usa para normalizar sobrecobertura. |

`WorkforceState.copy_state(**updates)` crea una copia validada y copia los arrays NumPy para evitar contaminación entre ramas de búsqueda.

## Espacio de acciones

El espacio de acciones tiene tamaño fijo:

```python
ACTION_SPACE_SIZE = 55
```

Se divide en tres bloques:

| Rango | Tipo | Significado |
|---:|---|---|
| `0-2` | `MODALITY` | Selección de modalidad: 4, 6 u 8 horas. |
| `3-26` | `ENTRY_HOUR` | Selección de hora de entrada: 0 a 23. |
| `27-54` | `DAY_OFFS` | Selección de francos semanales. |

El engine provee helpers para no manipular IDs a mano:

```python
from modules.workforce_engine.schemas import ActionType

action_id = engine.encode_action(ActionType.MODALITY, 8)
decoded = engine.decode_action(action_id)
```

## API de uso directo

La clase principal es:

```python
from modules.workforce_engine.engine import WorkforceEngine

engine = WorkforceEngine(setup)
```

Los métodos más importantes para el problema son los de legalidad y transición.

### `validate_action(state, action_id)`

Valida una acción concreta para el estado recibido.

```python
is_valid, reason = engine.validate_action(state, action_id)
```

Devuelve:

- `(True, None)` si la acción es legal.
- `(False, "motivo")` si la acción no es legal.

Este método es útil para depuración, logs o tests puntuales.

### `get_legal_actions(state)`

Devuelve un vector booleano de tamaño `55`.

```python
legal = engine.get_legal_actions(state)
legal_action_ids = np.flatnonzero(legal)
```

Cada posición indica si el `action_id` correspondiente es legal en el estado actual.

### `legal_mask(state, priors)`

Recibe probabilidades o scores no negativos de tamaño `55`, enmascara acciones ilegales y devuelve una distribución normalizada solo sobre acciones legales.

```python
priors = np.ones(engine.action_space_size) / engine.action_space_size
policy = engine.legal_mask(state, priors)
```

Es el método pensado para conectar el engine con un evaluador o una red neuronal que produzca una policy sobre todo el espacio de acciones. Si toda la masa de probabilidad cae en acciones ilegales, el engine usa una distribución uniforme sobre las acciones legales.

### `step(state, action_id)`

Aplica una acción legal y devuelve un `StepResult`.

```python
result = engine.step(state, action_id)

next_state = result.next_state
is_terminal = result.is_terminal
reward = result.reward
```

Si la acción es ilegal, lanza `ValueError`.

`step` es el método principal para avanzar la simulación. Internamente:

1. Valida la acción.
2. Determina el tipo de acción esperada.
3. Aplica selección de modalidad, selección de hora o selección de francos.
4. Cuando corresponde, aplica cobertura semanal.
5. Al cerrar la semana 4 del recurso, descuenta stock, evalúa terminalidad y calcula recompensa.

## Flujo de decisión

El flujo depende de `mobile_days_off_count`.

### Sin francos móviles

Si `mobile_days_off_count == 0`:

```text
MODALITY -> ENTRY_HOUR -> ENTRY_HOUR -> ENTRY_HOUR -> ENTRY_HOUR
```

La selección de `ENTRY_HOUR` aplica inmediatamente la cobertura de esa semana. Al completar la cuarta semana, se cierra el recurso mensual.

### Con francos móviles

Si `mobile_days_off_count > 0`:

```text
MODALITY -> ENTRY_HOUR -> DAY_OFFS -> ENTRY_HOUR -> DAY_OFFS -> ...
```

Cada semana requiere una hora de entrada y una acción de francos. La acción `DAY_OFFS` aplica la cobertura semanal.

## Ejemplo mínimo

```python
import numpy as np

from modules.workforce_engine.engine import WorkforceEngine
from modules.workforce_engine.schemas import ActionType, ProblemSetup, WorkforceState


setup = ProblemSetup(
    mobile_days_off_count=0,
    fixed_day_off=0,
    allowed_entry_hours=[8, 10, 12],
    max_overcoverage_tolerance=0.2,
    closing_hour=22,
)

initial_demand = np.ones((24, 28), dtype=int)

state = WorkforceState(
    residual_demand=initial_demand,
    remaining_stock=np.array([2, 1, 1], dtype=int),
    initial_demand_total=int(initial_demand.sum()),
)

engine = WorkforceEngine(setup)

modality_action = engine.encode_action(ActionType.MODALITY, 8)
is_valid, reason = engine.validate_action(state, modality_action)

if not is_valid:
    raise ValueError(reason)

result = engine.step(state, modality_action)
state = result.next_state

entry_action = engine.encode_action(ActionType.ENTRY_HOUR, 10)
result = engine.step(state, entry_action)
state = result.next_state

print(state.assignment_week)
print(result.is_terminal)
print(result.reward)
```

## Ejemplo con máscara legal

```python
import numpy as np

legal = engine.get_legal_actions(state)
print(np.flatnonzero(legal))

priors = np.ones(engine.action_space_size) / engine.action_space_size
policy = engine.legal_mask(state, priors)

next_action = int(np.argmax(policy))
result = engine.step(state, next_action)
```

## Terminalidad y recompensa

La terminalidad se evalúa con:

```python
engine.check_terminality(state)
```

Un estado puede ser terminal si:

- Toda la demanda residual queda cubierta (`residual_demand <= 0` en todas las celdas).
- La sobrecobertura excede el límite definido por `max_overcoverage_tolerance`.

La recompensa terminal se calcula con:

```python
engine.compute_reward(state)
```

El índice de sobrecobertura se calcula con:

```python
engine.compute_overcoverage_index(state)
```

En el flujo normal, `step` solo devuelve `is_terminal=True` después de cerrar el ciclo mensual de un recurso, es decir, al terminar la semana `3`.

## Notas de diseño

- El engine no mantiene memoria mutable de la trayectoria.
- Cada transición devuelve un nuevo `WorkforceState`.
- La legalidad de acciones depende del estado actual y del `ProblemSetup`.
- La cobertura resta `1` a cada celda cubierta de `residual_demand`.
- Si `closing_hour` es `None`, un turno puede cruzar medianoche siempre que no exceda el horizonte de 28 días.
- `expansion_mode` se activa cuando el stock real llega a cero en todas las modalidades.

## Próximos puntos a documentar o testear

- Casos mínimos para `validate_action`.
- Casos mínimos para `step`.
- Ejemplos con `mobile_days_off_count=1` y `mobile_days_off_count=2`.
- Criterios esperados de reward y terminalidad.

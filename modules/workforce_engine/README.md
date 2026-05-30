# Workforce Engine

Módulo Python para representar la dinámica operativa de un entorno de planificación de recursos laborales sobre un horizonte fijo de **4 semanas** y resolución **horaria**.

El objetivo del módulo es actuar como **engine determinístico de transición**, legalidad, cobertura, terminalidad y scoring. Está pensado para ser consumido por otros componentes, como un **MCTS**, una heurística, una red neuronal o un simulador de trayectorias.

---

## 1. Propósito del módulo

El `WorkforceEngine` recibe un estado actual de planificación y una acción seleccionada por un agente externo. A partir de eso, devuelve el nuevo estado resultante.

El engine no decide cuál es la mejor acción. Esa decisión queda fuera del módulo.

El engine responde preguntas como:

```text
Dado este estado y esta acción, ¿cuál es el nuevo estado?
Dado este estado, ¿qué acciones son legales?
Dado este estado y un vector de logits, ¿cuál es la distribución legal normalizada?
¿El nuevo estado es terminal?
Si es terminal, ¿cuál es la recompensa?
```

---

## 2. Estructura del paquete

La estructura generada es:

```text
workforce_engine_module/
│
├── workforce/
│   ├── __init__.py
│   ├── schemas.py
│   └── engine.py
│
└── example_usage.py
```

### 2.1. `schemas.py`

Contiene las estructuras de datos compartidas por el engine y por otros módulos.

Incluye:

```python
ActionType
ProblemSetup
WorkforceState
StepResult
```

Se utiliza el nombre `schemas.py` para evitar confusiones futuras con modelos de machine learning.

### 2.2. `engine.py`

Contiene la clase principal:

```python
WorkforceEngine
```

y las constantes operativas del entorno:

```python
ACTION_SPACE_SIZE
DAY_OFF_ACTION_MATRIX
MODALITIES
```

También contiene la lógica de:

```text
validación de acciones
máscara legal
transición de estado
aplicación de cobertura semanal
avance de semana
cierre de recurso mensual
descuento de stock
activación de expansion_mode
terminalidad
scoring
```

### 2.3. `example_usage.py`

Archivo mínimo de ejemplo para instanciar el setup, crear un estado inicial y ejecutar algunas acciones.

---

## 3. Dependencias

El módulo utiliza:

```text
numpy
pydantic
```

Instalación sugerida:

```bash
pip install numpy pydantic
```

La implementación fue pensada para `pydantic` versión 2.

---

## 4. Conceptos centrales

## 4.1. Horizonte de planificación

El horizonte es fijo:

```text
4 semanas
28 días
24 horas por día
```

La demanda se representa con una matriz:

```python
residual_demand.shape == (24, 28)
```

donde:

```text
filas    -> horas del día, de 0 a 23
columnas -> días del horizonte, de 0 a 27
```

La interpretación de cada celda es:

| Valor | Interpretación |
|---:|---|
| `> 0` | demanda pendiente |
| `= 0` | demanda exactamente cubierta |
| `< 0` | sobrecobertura |

---

## 4.2. Recurso laboral

Un recurso tiene una modalidad fija durante sus cuatro semanas de asignación.

Las modalidades soportadas son:

```python
4
6
8
```

Cada recurso aporta:

```text
1 FTE por cada hora trabajada
```

Por lo tanto, si un recurso trabaja una celda determinada, se resta `1` a esa celda de `residual_demand`.

---

## 4.3. Flujo de decisión

El flujo de decisión depende del estado.

Primero se elige modalidad. Luego, para cada semana del recurso, se elige horario de entrada y, si corresponde, francos móviles.

### Con francos móviles

Si:

```python
mobile_days_off_count > 0
```

el flujo es:

```text
Seleccionar modalidad

Semana 1:
    seleccionar hora de entrada
    seleccionar francos
    aplicar cobertura

Semana 2:
    seleccionar hora de entrada
    seleccionar francos
    aplicar cobertura

Semana 3:
    seleccionar hora de entrada
    seleccionar francos
    aplicar cobertura

Semana 4:
    seleccionar hora de entrada
    seleccionar francos
    aplicar cobertura
    descontar stock
    resetear recurso
    evaluar terminalidad
```

### Sin francos móviles

Si:

```python
mobile_days_off_count == 0
```

el flujo es:

```text
Seleccionar modalidad

Semana 1:
    seleccionar hora de entrada
    aplicar cobertura

Semana 2:
    seleccionar hora de entrada
    aplicar cobertura

Semana 3:
    seleccionar hora de entrada
    aplicar cobertura

Semana 4:
    seleccionar hora de entrada
    aplicar cobertura
    descontar stock
    resetear recurso
    evaluar terminalidad
```

En este caso no se usa el bloque de acciones de francos.

---

## 5. `ProblemSetup`

`ProblemSetup` representa las reglas fijas del problema. No contiene estado dinámico ni demanda inicial.

Se define en:

```python
workforce/schemas.py
```

### 5.1. Atributos

```python
ProblemSetup(
    mobile_days_off_count: int,
    fixed_day_off: int | None,
    allowed_entry_hours: list[int] | None,
    max_overcoverage_tolerance: float,
    closing_hour: int | None,
)
```

| Campo | Descripción |
|---|---|
| `mobile_days_off_count` | Cantidad de francos móviles por semana. Puede ser `0`, `1` o `2`. |
| `fixed_day_off` | Día fijo de franco semanal. Puede ser `None` o un entero de `0` a `6`. |
| `allowed_entry_hours` | Lista de horarios de entrada permitidos. Si es `None`, se permiten todas las horas de `0` a `23`. |
| `max_overcoverage_tolerance` | Tolerancia máxima de sobrecobertura usada en el scoring. |
| `closing_hour` | Hora de cierre operativo. Si es `None`, no hay cierre operativo. |

---

## 5.2. Codificación de días

Los días de la semana se codifican así:

| Valor | Día |
|---:|---|
| `0` | lunes |
| `1` | martes |
| `2` | miércoles |
| `3` | jueves |
| `4` | viernes |
| `5` | sábado |
| `6` | domingo |

---

## 5.3. Validaciones de `ProblemSetup`

### `mobile_days_off_count`

Debe ser:

```python
0, 1 o 2
```

### `fixed_day_off`

Debe ser:

```python
None
```

o un entero entre:

```python
0 y 6
```

### Francos totales

La suma de franco fijo y francos móviles no puede superar `2`.

Ejemplos válidos:

```python
mobile_days_off_count=0, fixed_day_off=None
mobile_days_off_count=1, fixed_day_off=None
mobile_days_off_count=2, fixed_day_off=None
mobile_days_off_count=0, fixed_day_off=3
mobile_days_off_count=1, fixed_day_off=3
```

Ejemplos inválidos:

```python
mobile_days_off_count=2, fixed_day_off=3
mobile_days_off_count=3, fixed_day_off=None
```

### `allowed_entry_hours`

Puede ser `None` o una lista no vacía de enteros entre `0` y `23`.

No puede tener duplicados.

Si existe `closing_hour`, ningún horario permitido puede ser mayor o igual al cierre.

Por ejemplo:

```python
ProblemSetup(
    mobile_days_off_count=1,
    fixed_day_off=None,
    allowed_entry_hours=[8, 9, 10],
    max_overcoverage_tolerance=0.2,
    closing_hour=20,
)
```

es válido.

Pero:

```python
ProblemSetup(
    mobile_days_off_count=1,
    fixed_day_off=None,
    allowed_entry_hours=[18, 20],
    max_overcoverage_tolerance=0.2,
    closing_hour=20,
)
```

es inválido porque `20` es igual al horario de cierre.

### `max_overcoverage_tolerance`

Debe cumplir:

```python
0 < max_overcoverage_tolerance <= 1
```

### `closing_hour`

Puede ser `None` o un entero entre `0` y `23`.

Si existe, un turno puede terminar exactamente en el cierre, pero no puede cubrir la hora de cierre.

La condición de legalidad es:

```python
entry_hour + modality <= closing_hour
```

Ejemplo:

```python
entry_hour = 16
modality = 4
closing_hour = 20
```

El recurso trabaja las horas:

```text
16, 17, 18, 19
```

y se retira a las `20`.

Por lo tanto, es legal.

---

## 6. `WorkforceState`

`WorkforceState` representa el estado dinámico de una trayectoria.

Se define en:

```python
workforce/schemas.py
```

### 6.1. Atributos

```python
WorkforceState(
    residual_demand: np.ndarray,
    remaining_stock: np.ndarray,
    expansion_mode: bool,
    current_modality: int | None,
    current_entry_hour: int | None,
    assignment_week: int,
    initial_demand_total: int,
)
```

| Campo | Descripción |
|---|---|
| `residual_demand` | Matriz entera de demanda residual, shape `(24, 28)`. |
| `remaining_stock` | Array entero de stock remanente por modalidad, shape `(3,)`. |
| `expansion_mode` | Indica si el sistema ya agotó el stock real. |
| `current_modality` | Modalidad actual del recurso en curso: `None`, `4`, `6` u `8`. |
| `current_entry_hour` | Hora de entrada actual de la semana en curso. |
| `assignment_week` | Semana actual del recurso, de `0` a `3`. |
| `initial_demand_total` | Suma total de la demanda inicial. |

---

## 6.2. `residual_demand`

Debe ser una matriz entera:

```python
residual_demand.shape == (24, 28)
```

No se aceptan valores decimales porque no se asignan fracciones de recurso.

La demanda puede tener valores negativos, ya que estos representan sobrecobertura.

---

## 6.3. `remaining_stock`

Debe ser un array entero de shape `(3,)`.

La codificación es:

| Índice | Modalidad |
|---:|---:|
| `0` | 4 horas |
| `1` | 6 horas |
| `2` | 8 horas |

Ejemplo:

```python
remaining_stock = np.array([10, 5, 3])
```

significa:

```text
10 recursos de 4 horas
5 recursos de 6 horas
3 recursos de 8 horas
```

---

## 6.4. `expansion_mode`

Mientras:

```python
expansion_mode == False
```

solo son legales las modalidades con stock disponible.

Ejemplo:

```python
remaining_stock = np.array([0, 1, 0])
expansion_mode = False
```

Legalidad:

| Modalidad | Legalidad |
|---:|---|
| 4 horas | ilegal |
| 6 horas | legal |
| 8 horas | ilegal |

El `expansion_mode` se activa únicamente cuando todos los contadores quedan en cero:

```python
remaining_stock = np.array([0, 0, 0])
```

A partir de ese momento, la falta de stock deja de invalidar modalidades.

---

## 6.5. `current_modality`

Puede ser:

```python
None, 4, 6, 8
```

La modalidad se elige una vez por recurso y se mantiene durante las cuatro semanas.

---

## 6.6. `current_entry_hour`

Puede ser:

```python
None, 0, 1, ..., 23
```

Se elige por semana.

Después de aplicar cobertura semanal, vuelve a `None`.

---

## 6.7. `assignment_week`

Entero entre `0` y `3`.

| Valor | Semana | Días absolutos |
|---:|---|---|
| `0` | semana 1 | `0` a `6` |
| `1` | semana 2 | `7` a `13` |
| `2` | semana 3 | `14` a `20` |
| `3` | semana 4 | `21` a `27` |

Al finalizar la semana `3`, se completa el ciclo mensual del recurso.

---

## 6.8. `initial_demand_total`

Es la suma total de la demanda inicial.

Se utiliza para calcular el índice de sobrecobertura normalizado.

No es necesario almacenar la matriz de demanda inicial completa.

---

## 7. Espacio de acciones

El espacio de acciones es fijo y tiene tamaño:

```python
ACTION_SPACE_SIZE = 55
```

Se divide en tres bloques:

| Rango | Tipo | Cantidad |
|---:|---|---:|
| `0-2` | modalidad | `3` |
| `3-26` | hora de entrada | `24` |
| `27-54` | francos | `28` |

---

## 7.1. Acciones de modalidad

| `action_id` | Modalidad |
|---:|---:|
| `0` | 4 horas |
| `1` | 6 horas |
| `2` | 8 horas |

Constante:

```python
MODALITIES = [4, 6, 8]
```

---

## 7.2. Acciones de hora de entrada

Las horas se codifican como:

```python
action_id = 3 + entry_hour
```

Por lo tanto:

| `action_id` | Hora |
|---:|---:|
| `3` | `0` |
| `4` | `1` |
| `5` | `2` |
| `...` | `...` |
| `26` | `23` |

---

## 7.3. Acciones de francos

Las acciones de francos ocupan:

```python
27 a 54
```

El action id real se obtiene como:

```python
action_id = 27 + internal_day_off_action_id
```

La matriz de acciones internas es:

```python
DAY_OFF_ACTION_MATRIX = np.array([
    [0, 1, 2, 3, 4, 5, 6],
    [1, 7, 8, 9, 10, 11, 12],
    [2, 8, 13, 14, 15, 16, 17],
    [3, 9, 14, 18, 19, 20, 21],
    [4, 10, 15, 19, 22, 23, 24],
    [5, 11, 16, 20, 23, 25, 26],
    [6, 12, 17, 21, 24, 26, 27],
])
```

La matriz es simétrica.

Eso significa que:

```python
DAY_OFF_ACTION_MATRIX[d1, d2] == DAY_OFF_ACTION_MATRIX[d2, d1]
```

---

## 8. Legalidad de francos

## 8.1. Sin francos móviles

Si:

```python
mobile_days_off_count == 0
```

no se usa el bloque de acciones de francos.

La acción de hora de entrada aplica directamente la cobertura semanal.

---

## 8.2. Un franco móvil y sin franco fijo

Si:

```python
fixed_day_off is None
mobile_days_off_count == 1
```

se usa la diagonal de la matriz.

Valores internos legales:

```python
[0, 7, 13, 18, 22, 25, 27]
```

Acciones reales legales:

```python
[27, 34, 40, 45, 49, 52, 54]
```

---

## 8.3. Un franco fijo y un franco móvil

Si:

```python
fixed_day_off is not None
mobile_days_off_count == 1
```

se usa la fila del franco fijo, excluyendo la columna del mismo día.

Ejemplo:

```python
fixed_day_off = 2
```

Fila 2:

```python
[2, 8, 13, 14, 15, 16, 17]
```

Se excluye el valor diagonal:

```python
13
```

Valores internos legales:

```python
[2, 8, 14, 15, 16, 17]
```

Acciones reales legales:

```python
[29, 35, 41, 42, 43, 44]
```

---

## 8.4. Dos francos móviles y sin franco fijo

Si:

```python
fixed_day_off is None
mobile_days_off_count == 2
```

se permiten todos los pares de días distintos.

Por lo tanto, son legales todos los valores de la matriz excepto la diagonal.

Diagonal interna ilegal:

```python
[0, 7, 13, 18, 22, 25, 27]
```

Valores internos legales:

```python
[
    1, 2, 3, 4, 5, 6,
    8, 9, 10, 11, 12,
    14, 15, 16, 17,
    19, 20, 21,
    23, 24,
    26
]
```

Acciones reales legales:

```python
[
    28, 29, 30, 31, 32, 33,
    35, 36, 37, 38, 39,
    41, 42, 43, 44,
    46, 47, 48,
    50, 51,
    53
]
```

---

## 9. Clase `WorkforceEngine`

Se importa desde:

```python
from workforce import WorkforceEngine
```

Se inicializa con:

```python
engine = WorkforceEngine(setup)
```

---

## 9.1. Métodos principales

| Método | Descripción |
|---|---|
| `step(state, action_id)` | Aplica una acción legal y devuelve un `StepResult`. |
| `legal_mask(state, priors)` | Enmascara acciones ilegales y normaliza logits legales. |
| `get_legal_actions(state)` | Devuelve un vector booleano de acciones legales. |
| `encode_action(action_type, value)` | Codifica una acción conceptual como `action_id`. |
| `decode_action(action_id)` | Decodifica un `action_id` a una representación interpretable. |
| `get_action_type(action_id)` | Devuelve el tipo de acción del `action_id`. |
| `check_terminality(state)` | Evalúa si un estado cumple condiciones terminales. |
| `compute_overcoverage_index(state)` | Calcula el índice de sobrecobertura normalizado. |
| `compute_reward(state)` | Calcula la recompensa terminal. |

---

## 10. `step`

`step` es el método central del engine.

Firma:

```python
result = engine.step(state, action_id)
```

Devuelve:

```python
StepResult(
    next_state=...,
    is_terminal=...,
    reward=...
)
```

Si la acción es ilegal, lanza:

```python
ValueError
```

---

## 10.1. Comportamiento general

El método:

```text
1. Valida que la acción sea legal.
2. Determina el tipo de acción.
3. Aplica la transición correspondiente.
4. Si corresponde, aplica cobertura semanal.
5. Si se completó la semana 4, descuenta stock.
6. Si se completó la semana 4, evalúa terminalidad.
7. Si el estado es terminal, calcula reward.
8. Si no es terminal, devuelve reward = 0.
```

---

## 10.2. Acción de modalidad

Ejemplo:

```python
action_id = engine.encode_action(ActionType.MODALITY, 4)
result = engine.step(state, action_id)
```

Efecto:

```text
current_modality = 4
```

No modifica:

```text
residual_demand
remaining_stock
assignment_week
current_entry_hour
```

El stock no se descuenta al seleccionar la modalidad. Se descuenta al finalizar la semana 4 del recurso.

---

## 10.3. Acción de hora de entrada

Ejemplo:

```python
action_id = engine.encode_action(ActionType.ENTRY_HOUR, 8)
result = engine.step(state, action_id)
```

Efecto:

```text
current_entry_hour = 8
```

Si `mobile_days_off_count > 0`, todavía no se aplica cobertura.

Si `mobile_days_off_count == 0`, la acción de hora aplica directamente cobertura semanal.

---

## 10.4. Acción de francos

Ejemplo:

```python
action_id = engine.encode_action(ActionType.DAY_OFFS, (0, 0))
result = engine.step(state, action_id)
```

Si el setup tiene un solo franco móvil y no hay franco fijo, `(0, 0)` representa franco lunes.

La acción de francos:

```text
decodifica los días de franco
determina los días trabajados
aplica cobertura semanal
resetea current_entry_hour
avanza assignment_week
```

Si la semana actual es `3`, además:

```text
resetea current_modality
descuenta stock
puede activar expansion_mode
evalúa terminalidad
calcula reward si corresponde
```

---

## 11. `legal_mask`

Firma:

```python
masked_probs = engine.legal_mask(state, priors)
```

`priors` debe ser un vector de logits de tamaño:

```python
(55,)
```

El método:

```text
1. Identifica las acciones legales.
2. Conserva los logits de las acciones legales.
3. Aplica softmax solo sobre las legales.
4. Devuelve probabilidad 0 para las ilegales.
```

Ejemplo:

```python
priors = np.zeros(55)
masked_probs = engine.legal_mask(state, priors)
```

Si en el estado inicial solo son legales las acciones de modalidad con stock, la probabilidad se distribuirá solamente entre esas modalidades.

---

## 12. `get_legal_actions`

Firma:

```python
legal = engine.get_legal_actions(state)
```

Devuelve un vector booleano de tamaño `55`.

Ejemplo:

```python
legal[action_id] == True
```

significa que esa acción es legal para el estado actual.

---

## 13. Cobertura semanal

La cobertura semanal se aplica sobre la matriz `residual_demand`.

Para cada día trabajado, se cubren las horas:

```python
entry_hour
entry_hour + 1
...
entry_hour + modality - 1
```

Cada celda cubierta se reduce en `1`.

Ejemplo:

```python
entry_hour = 8
modality = 4
```

Cubre:

```text
8, 9, 10, 11
```

---

## 13.1. Días trabajados

Primero se determina el conjunto de francos:

```python
days_off = fixed_day_off + mobile_days_off
```

Luego:

```python
working_days = {0, 1, 2, 3, 4, 5, 6} - days_off
```

---

## 13.2. Cruce de medianoche

Si no existe `closing_hour`, un turno puede cruzar al día siguiente.

Ejemplo:

```python
entry_hour = 22
modality = 6
```

Cubre:

```text
día d:     22, 23
día d + 1: 0, 1, 2, 3
```

Si el cruce excede el día `27`, la acción se considera ilegal.

---

## 13.3. Cierre operativo

Si existe `closing_hour`, debe cumplirse:

```python
entry_hour + modality <= closing_hour
```

Ejemplo legal:

```python
entry_hour = 16
modality = 4
closing_hour = 20
```

Trabaja:

```text
16, 17, 18, 19
```

y se retira a las `20`.

Ejemplo ilegal:

```python
entry_hour = 17
modality = 4
closing_hour = 20
```

Trabajaría:

```text
17, 18, 19, 20
```

lo cual cubre la hora de cierre.

---

## 14. Stock y `expansion_mode`

El stock se descuenta solamente al finalizar la semana 4 del recurso.

Ejemplo:

```python
remaining_stock = np.array([1, 0, 0])
current_modality = 4
assignment_week = 0
```

Durante semanas `0`, `1` y `2`, el stock sigue siendo:

```python
[1, 0, 0]
```

Al finalizar semana `3`, se descuenta:

```python
[0, 0, 0]
```

y se activa:

```python
expansion_mode = True
```

Mientras `expansion_mode == False`, una modalidad sin stock es ilegal.

Cuando `expansion_mode == True`, todas las modalidades quedan permitidas desde el punto de vista del stock.

---

## 15. Terminalidad

La terminalidad efectiva se evalúa únicamente al finalizar la semana 4 del recurso.

No se evalúa después de:

```text
seleccionar modalidad
seleccionar hora, salvo que complete cobertura semanal en caso sin francos móviles
aplicar semana 0
aplicar semana 1
aplicar semana 2
```

---

## 15.1. Condición de demanda cubierta

El estado es terminal si:

```python
np.all(residual_demand <= 0)
```

Esto significa que no queda ninguna celda con demanda positiva.

---

## 15.2. Condición de sobrecobertura excesiva

El estado también es terminal si:

```python
rho <= -2 * max_overcoverage_tolerance
```

donde `rho` es el índice de sobrecobertura normalizado.

---

## 16. Índice de sobrecobertura normalizado

Se calcula con:

```python
negative_residual_sum = np.minimum(residual_demand, 0).sum()
rho = negative_residual_sum / initial_demand_total
```

Importante:

```text
los valores positivos de demanda pendiente no compensan la sobrecobertura
solo se consideran los valores negativos
```

Interpretación:

| Valor | Interpretación |
|---:|---|
| `rho = 0` | no hay sobrecobertura |
| `rho < 0` | hay sobrecobertura |
| `rho` más negativo | mayor sobrecobertura relativa |

---

## 17. Recompensa terminal

La recompensa solo se calcula si el estado es terminal.

Para estados no terminales:

```python
reward = 0.0
```

Para estados terminales:

```python
reward = np.tanh(2 * (1 - abs(rho) / k))
```

donde:

```python
rho = índice de sobrecobertura normalizado
k = max_overcoverage_tolerance
```

Interpretación:

| Caso | Resultado |
|---|---|
| `rho = 0` | recompensa cercana a `1`, exactamente `tanh(2)` |
| `abs(rho) = k` | recompensa igual a `0` |
| `abs(rho) > k` | recompensa negativa |
| sobrecobertura muy alta | recompensa cercana a `-1` |

---

## 18. Ejemplo de uso básico

```python
import numpy as np

from workforce import ActionType, ProblemSetup, WorkforceEngine, WorkforceState


setup = ProblemSetup(
    mobile_days_off_count=1,
    fixed_day_off=None,
    allowed_entry_hours=[8, 9, 10],
    max_overcoverage_tolerance=0.2,
    closing_hour=20,
)

initial_demand = np.ones((24, 28), dtype=int)

state = WorkforceState(
    residual_demand=initial_demand,
    remaining_stock=np.array([1, 1, 1], dtype=int),
    expansion_mode=False,
    current_modality=None,
    current_entry_hour=None,
    assignment_week=0,
    initial_demand_total=int(initial_demand.sum()),
)

engine = WorkforceEngine(setup)

# Seleccionar modalidad de 4 horas.
result = engine.step(
    state,
    engine.encode_action(ActionType.MODALITY, 4)
)
state = result.next_state

# Seleccionar hora de entrada 8.
result = engine.step(
    state,
    engine.encode_action(ActionType.ENTRY_HOUR, 8)
)
state = result.next_state

# Seleccionar franco lunes.
# Con un franco móvil y sin franco fijo, se usa la diagonal.
result = engine.step(
    state,
    engine.encode_action(ActionType.DAY_OFFS, (0, 0))
)
state = result.next_state

print(state.assignment_week)
print(state.current_modality)
print(state.current_entry_hour)
print(result.is_terminal)
print(result.reward)
```

---

## 19. Ejemplo de máscara legal

```python
priors = np.zeros(55)
masked_probs = engine.legal_mask(state, priors)

legal_actions = np.where(masked_probs > 0)[0]
print(legal_actions)
print(masked_probs[legal_actions])
```

Si todos los logits legales son iguales, la probabilidad se reparte uniformemente entre las acciones legales.

---

## 20. Ejemplo de acción ilegal

```python
try:
    engine.step(state, 999)
except ValueError as error:
    print(error)
```

También se lanza `ValueError` si se intenta seleccionar una modalidad sin stock cuando `expansion_mode == False`.

---

## 21. Ejecución del ejemplo

Desde la carpeta raíz del paquete:

```bash
python example_usage.py
```

---

## 22. Reglas finales implementadas

```text
El espacio de acciones tiene tamaño fijo 55.
Las acciones 0-2 corresponden a modalidad.
Las acciones 3-26 corresponden a hora de entrada.
Las acciones 27-54 corresponden a francos.
ProblemSetup y WorkforceState están en schemas.py.
El engine no mantiene estado interno de trayectoria.
El stock se descuenta al finalizar semana 4.
expansion_mode se activa solo cuando todos los stocks quedan en cero.
Mientras expansion_mode=False, una modalidad sin stock es ilegal.
Si mobile_days_off_count=0, no se usa acción de francos.
Si existe fixed_day_off, el franco móvil no puede repetir ese día.
Si hay dos francos móviles, no se permite la diagonal.
residual_demand debe ser una matriz entera.
Cada recurso resta 1 FTE por hora trabajada.
Los turnos pueden cruzar medianoche si no hay closing_hour.
Si existe closing_hour, debe cumplirse entry_hour + modality <= closing_hour.
La terminalidad efectiva se evalúa al finalizar semana 4.
La recompensa solo se calcula en estados terminales.
step lanza ValueError ante acciones ilegales.
legal_mask aplica softmax solo sobre acciones legales.
```

---

## 23. Consideraciones para próximos módulos

Este paquete está preparado para ser usado por:

```text
MCTS
codificador de estado para red neuronal
simulador de demanda
generador de trayectorias
evaluador heurístico
scripts de entrenamiento
```

Los objetos `ProblemSetup`, `WorkforceState` y `StepResult` deben funcionar como contrato común entre esos módulos.

Más adelante, si el proyecto crece, se puede separar `engine.py` en módulos más específicos:

```text
actions.py
coverage.py
scoring.py
legal_mask.py
```

Por ahora se mantiene en un único archivo para facilitar la lectura, la prueba y la iteración.

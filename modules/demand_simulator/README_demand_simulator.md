# Demand Simulator

Este documento describe la primera versión de la clase `DemandSimulator`, cuyo objetivo es generar una **matriz de cobertura acumulada** para un problema de planificación de demanda horaria sobre un horizonte de 28 días.

En esta etapa, el simulador **no genera todavía demanda inicial con ruido**, ni matriz de descuento, ni scoring. El foco está puesto únicamente en construir la cobertura que resultaría de asignar recursos de 4, 6 y 8 horas a lo largo de 4 semanas.

---

## 1. Objetivo de la clase

La clase `DemandSimulator` permite simular la cobertura generada por una dotación de recursos disponibles.

Dado un conjunto de recursos de distintas modalidades contractuales:

| Modalidad | Significado |
|---:|---|
| 4 | Recurso con jornada diaria de 4 horas |
| 6 | Recurso con jornada diaria de 6 horas |
| 8 | Recurso con jornada diaria de 8 horas |

la clase asigna aleatoriamente, para cada recurso y para cada una de las 4 semanas del horizonte:

1. la modalidad del recurso;
2. el horario de ingreso semanal;
3. los días francos de la semana.

A partir de esas decisiones, construye una matriz de cobertura acumulada:

```text
R ∈ Z≥0^(24 x 28)
```

Donde:

| Dimensión | Significado |
|---|---|
| Filas | Horas del día, de 0 a 23 |
| Columnas | Días del horizonte, de 0 a 27 |
| Valor de la celda | Cantidad de recursos cubriendo esa hora y ese día |

---

## 2. Convenciones utilizadas

### 2.1 Grilla temporal

La matriz de cobertura tiene forma:

```python
(24, 28)
```

La interpretación es:

```text
R[hora, dia]
```

Por ejemplo:

| Celda | Interpretación |
|---|---|
| `R[8, 0]` | Cobertura a las 8 hs del día 0 |
| `R[15, 10]` | Cobertura a las 15 hs del día 10 |
| `R[23, 27]` | Cobertura a las 23 hs del día 27 |

---

### 2.2 Días del horizonte

El horizonte tiene 28 días:

```text
0, 1, 2, ..., 27
```

Se agrupan en 4 semanas de 7 días:

| Semana | Días globales |
|---:|---|
| 0 | 0 a 6 |
| 1 | 7 a 13 |
| 2 | 14 a 20 |
| 3 | 21 a 27 |

---

### 2.3 Días de la semana

La codificación de días de la semana es:

| Código | Día |
|---:|---|
| 0 | Domingo |
| 1 | Lunes |
| 2 | Martes |
| 3 | Miércoles |
| 4 | Jueves |
| 5 | Viernes |
| 6 | Sábado |

Para obtener el día de semana a partir del día global:

```python
day_of_week = global_day % 7
```

---

## 3. Inicialización de la clase

La clase se inicializa con los parámetros operativos que condicionan las asignaciones.

```python
sim = DemandSimulator(
    entry_hours=[8, 10, 12, 14, 16],
    close_hour=22,
    fixed_holidays=None,
    var_holidays=2,
    seed=42,
)
```

### 3.1 Parámetros del constructor

| Parámetro | Tipo | Descripción |
|---|---:|---|
| `entry_hours` | `list[int]` | Lista de horarios de ingreso posibles. Cada valor debe estar entre 0 y 23. |
| `close_hour` | `int | None` | Hora de cierre. Si es `None`, no se aplica restricción de cierre. |
| `fixed_holidays` | `int | None` | Día franco fijo semanal. Debe estar entre 0 y 6, o ser `None`. |
| `var_holidays` | `int` | Cantidad de francos variables. Puede ser 0, 1 o 2. |
| `seed` | `int | None` | Semilla aleatoria opcional para reproducibilidad. |

---

## 4. Validaciones principales

La clase valida que la configuración sea consistente antes de ejecutar la simulación.

### 4.1 `entry_hours`

Debe ser una lista no vacía de enteros entre 0 y 23.

Ejemplos válidos:

```python
[8, 10, 12, 14, 16]
list(range(24))
[0, 6, 12, 18]
```

Ejemplos inválidos:

```python
[]
[-1, 8, 12]
[8, 8, 10]
[24]
```

---

### 4.2 `close_hour`

Puede ser:

| Valor | Significado |
|---|---|
| `None` | No existe horario de cierre. Se permite que una jornada cruce al día siguiente. |
| `0` a `23` | Hora máxima de operación. La jornada no puede exceder esa hora. |

Ejemplo:

Si `close_hour=22` y el recurso es de 8 horas:

| Ingreso | Horas cubiertas | ¿Legal? |
|---:|---|---|
| 14 | 14 a 21 | Sí |
| 15 | 15 a 22 | Sí |
| 16 | 16 a 23 | No |
| 18 | 18 a 25 | No |

---

### 4.3 `fixed_holidays` y `var_holidays`

`fixed_holidays` puede ser un día fijo entre 0 y 6, o `None`.

`var_holidays` puede ser:

```python
0, 1, 2
```

La cantidad total de francos no puede superar 2:

```text
total_francos = fixed_holidays_existente + var_holidays ≤ 2
```

Ejemplos válidos:

| `fixed_holidays` | `var_holidays` | Total francos | Válido |
|---:|---:|---:|---|
| `None` | 0 | 0 | Sí |
| `None` | 1 | 1 | Sí |
| `None` | 2 | 2 | Sí |
| 0 | 0 | 1 | Sí |
| 0 | 1 | 2 | Sí |

Ejemplos inválidos:

| `fixed_holidays` | `var_holidays` | Total francos | Motivo |
|---:|---:|---:|---|
| 0 | 2 | 3 | Supera 2 francos |
| 7 | 0 | - | Día inválido |
| `None` | 3 | - | `var_holidays` inválido |

---

## 5. Método principal: `compute_coverage`

El método principal es:

```python
R, T = sim.compute_coverage(
    mod_4=2,
    mod_6=1,
    mod_8=1,
)
```

### 5.1 Inputs del método

| Parámetro | Tipo | Descripción |
|---|---:|---|
| `mod_4` | `int` | Cantidad de recursos disponibles de 4 horas |
| `mod_6` | `int` | Cantidad de recursos disponibles de 6 horas |
| `mod_8` | `int` | Cantidad de recursos disponibles de 8 horas |

Todos deben ser enteros mayores o iguales a cero.

---

### 5.2 Outputs del método

| Output | Tipo | Descripción |
|---|---|---|
| `R` | `np.ndarray` | Matriz de cobertura acumulada de dimensión `(24, 28)` |
| `T` | `list[dict]` | Trayectoria de acciones y matrices acumuladas |

---

## 6. Flujo lógico del método `compute_coverage`

El método sigue un pipeline simple basado en listas paralelas.

### Paso 1: generar lista de recursos

Primero se genera una lista con todas las modalidades de recursos disponibles.

```python
resources_list = [4 for _ in range(mod_4)] + \
                 [6 for _ in range(mod_6)] + \
                 [8 for _ in range(mod_8)]

random.shuffle(resources_list)
```

Ejemplo:

```python
mod_4 = 2
mod_6 = 1
mod_8 = 1
```

Antes del shuffle:

```python
[4, 4, 6, 8]
```

Después del shuffle:

```python
[6, 4, 8, 4]
```

---

### Paso 2: expandir cada recurso a sus 4 semanas

Cada recurso debe asignarse durante las 4 semanas del horizonte.

Si:

```python
resources_list = [6, 4, 8, 4]
```

Entonces se generan tres listas paralelas:

```python
resource_id_list = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
modality_list    = [6,6,6,6, 4,4,4,4, 8,8,8,8, 4,4,4,4]
week_list        = [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3]
```

Cada posición representa una asignación semanal base:

```python
(resource_id, modality, week)
```

---

### Paso 3: seleccionar horarios de ingreso legales

Para cada elemento de `modality_list`, se selecciona aleatoriamente un horario legal de ingreso.

```python
entry_hour_list = []

for modality in modality_list:
    legal_hours = get_legal_entry_hours(modality)
    selected_hour = random.choice(legal_hours)
    entry_hour_list.append(selected_hour)
```

La legalidad depende de:

- modalidad del recurso;
- horarios de ingreso permitidos;
- horario de cierre, si existe.

---

### Paso 4: seleccionar francos

Se genera el dominio válido de francos según:

- `fixed_holidays`;
- `var_holidays`.

Luego se selecciona aleatoriamente una combinación de francos por cada asignación semanal.

```python
holidays_list = [
    random.choice(holiday_options)
    for _ in range(len(modality_list))
]
```

---

### Paso 5: combinar listas en asignaciones semanales

Las listas paralelas se combinan en tuplas:

```python
assignments = list(zip(
    resource_id_list,
    modality_list,
    week_list,
    entry_hour_list,
    holidays_list,
))
```

Cada asignación tiene la forma:

```python
(resource_id, modality, week, entry_hour, holidays)
```

Ejemplo:

```python
(0, 6, 0, 10, (0, 6))
```

Interpretación:

| Elemento | Valor | Significado |
|---|---:|---|
| `resource_id` | 0 | Primer recurso de la trayectoria |
| `modality` | 6 | Recurso de 6 horas |
| `week` | 0 | Semana 0 |
| `entry_hour` | 10 | Ingresa a las 10 hs |
| `holidays` | `(0, 6)` | Franco domingo y sábado |

---

### Paso 6: construir cobertura y trayectoria

Para cada asignación semanal:

1. se registra la acción de modalidad;
2. se registra la acción de horario de ingreso;
3. se calcula la cobertura semanal;
4. se acumula en `R`;
5. se registra la acción de francos con la matriz `Rt` actualizada.

La cobertura semanal se aplica recién después de la acción de francos, porque recién en ese momento queda definido el bloque semanal completo.

---

## 7. Dominio de acciones

La trayectoria `T` guarda acciones codificadas mediante `action_id`.

### 7.1 Modalidad

| `action_id` | Acción |
|---:|---|
| 0 | Elegir modalidad 4h |
| 1 | Elegir modalidad 6h |
| 2 | Elegir modalidad 8h |

---

### 7.2 Horario de ingreso

Los horarios de ingreso comienzan en `action_id = 3`.

| `action_id` | Acción |
|---:|---|
| 3 | Elegir hora de ingreso 0 |
| 4 | Elegir hora de ingreso 1 |
| 5 | Elegir hora de ingreso 2 |
| ... | ... |
| 26 | Elegir hora de ingreso 23 |

La fórmula es:

```python
action_id = 3 + entry_hour
```

---

### 7.3 Francos

Las acciones de francos comienzan en `action_id = 27`.

| Rango | Significado |
|---:|---|
| 27 | Sin francos `()` |
| 28 a 34 | Un franco |
| 35 en adelante | Dos francos |

La tabla completa se puede obtener con:

```python
sim.get_holiday_action_table()
```

Ejemplo de codificación:

| `action_id` | Francos |
|---:|---|
| 27 | `()` |
| 28 | `(0,)` |
| 29 | `(1,)` |
| 30 | `(2,)` |
| 31 | `(3,)` |
| 32 | `(4,)` |
| 33 | `(5,)` |
| 34 | `(6,)` |
| 35 | `(0, 1)` |
| 36 | `(0, 2)` |
| ... | ... |

---

## 8. Estructura de la trayectoria `T`

La trayectoria es una lista de diccionarios.

Cada paso tiene la forma:

```python
{
    "step": int,
    "resource_id": int,
    "week": int,
    "action_type": str,
    "action_id": int,
    "action_value": Any,
    "Rt": np.ndarray,
}
```

### 8.1 Campos

| Campo | Descripción |
|---|---|
| `step` | Número secuencial de acción |
| `resource_id` | Identificador del recurso dentro de la simulación |
| `week` | Semana de asignación, entre 0 y 3 |
| `action_type` | Tipo de acción: `modality`, `entry_hour` o `holidays` |
| `action_id` | Código numérico de la acción |
| `action_value` | Valor real de la acción tomada |
| `Rt` | Matriz acumulada de cobertura en ese paso |

---

### 8.2 Ejemplo de trayectoria para una asignación semanal

Supongamos la asignación:

```python
(resource_id=0, modality=6, week=0, entry_hour=10, holidays=(0, 6))
```

La trayectoria registra tres pasos:

| Paso | `action_type` | `action_value` | Comentario |
|---:|---|---|---|
| 0 | `modality` | 6 | Se elige modalidad 6h |
| 1 | `entry_hour` | 10 | Se elige ingreso a las 10 hs |
| 2 | `holidays` | `(0, 6)` | Se eligen francos y se actualiza `Rt` |

---

## 9. Construcción de la cobertura semanal

Una asignación semanal genera una matriz de cobertura parcial de dimensión `(24, 28)`.

La cobertura se construye recorriendo los días de la semana correspondiente.

Para una semana `week`:

```python
start_day = week * 7
end_day = start_day + 6
```

Por ejemplo:

| Semana | `start_day` | `end_day` |
|---:|---:|---:|
| 0 | 0 | 6 |
| 1 | 7 | 13 |
| 2 | 14 | 20 |
| 3 | 21 | 27 |

Para cada día de esa semana:

1. se calcula el día de semana:

```python
day_of_week = global_day % 7
```

2. si el día está en `holidays`, no se asigna cobertura;
3. si no es franco, se cubren `modality` horas desde `entry_hour`.

---

## 10. Manejo del horario de cierre

Si `close_hour` está definido, los horarios ilegales se filtran antes de generar la asignación.

La regla actual es:

```python
last_covered_hour = entry_hour + modality - 1
legal = last_covered_hour <= close_hour
```

Ejemplo para `close_hour=22`:

| Modalidad | Ingreso | Horas cubiertas | Legal |
|---:|---:|---|---|
| 8 | 14 | 14 a 21 | Sí |
| 8 | 15 | 15 a 22 | Sí |
| 8 | 16 | 16 a 23 | No |
| 6 | 17 | 17 a 22 | Sí |
| 6 | 18 | 18 a 23 | No |

---

## 11. Overflow al día siguiente

Si `close_hour=None`, se permite que una jornada cruce al día siguiente.

Ejemplo:

| Modalidad | Ingreso | Cobertura |
|---:|---:|---|
| 8 | 20 | Día actual: 20, 21, 22, 23. Día siguiente: 0, 1, 2, 3 |

Si el overflow cae fuera del horizonte de 28 días, la cobertura fuera del horizonte se descarta.

---

## 12. Ejemplo de uso

```python
sim = DemandSimulator(
    entry_hours=[8, 10, 12, 14, 16],
    close_hour=22,
    fixed_holidays=None,
    var_holidays=2,
    seed=42,
)

R, T = sim.compute_coverage(
    mod_4=2,
    mod_6=1,
    mod_8=1,
)

print("Shape R:", R.shape)
print("Cobertura total:", R.sum())
print("Cantidad de acciones:", len(T))

print("Primer paso:")
print(T[0])

print("Último paso:")
print(T[-1])

print("Tabla de francos:")
print(sim.get_holiday_action_table())
```

---

## 13. Resultado esperado del ejemplo

Dado:

```python
mod_4=2
mod_6=1
mod_8=1
```

La cantidad de recursos es:

```text
2 + 1 + 1 = 4 recursos
```

Cada recurso se asigna durante 4 semanas:

```text
4 recursos x 4 semanas = 16 asignaciones semanales
```

Cada asignación semanal genera 3 acciones:

```text
modalidad + horario + francos = 3 acciones
```

Por lo tanto, la trayectoria tendrá:

```text
16 x 3 = 48 acciones
```

---

## 14. Métodos principales de la clase

| Método | Descripción |
|---|---|
| `compute_coverage()` | Método principal. Genera matriz de cobertura y trayectoria. |
| `_build_resources_list()` | Genera y mezcla aleatoriamente la lista de modalidades de recursos. |
| `_expand_resources_by_week()` | Repite cada recurso para las 4 semanas. |
| `_sample_entry_hours()` | Elige un horario legal por asignación semanal. |
| `_get_legal_entry_hours()` | Calcula horarios legales según modalidad y cierre. |
| `_build_holiday_options()` | Genera el dominio válido de francos. |
| `_sample_holidays_list()` | Selecciona francos aleatorios para cada asignación semanal. |
| `_build_coverage_and_trajectory()` | Construye la matriz acumulada y la trayectoria. |
| `_build_weekly_coverage()` | Construye la cobertura de una asignación semanal. |
| `_apply_daily_shift()` | Aplica la cobertura diaria de un recurso. |
| `_make_trajectory_step()` | Construye un registro de la trayectoria. |
| `get_holiday_action_table()` | Devuelve la tabla de acciones asociadas a francos. |

---

## 15. Alcance actual

Esta versión implementa solamente la primera parte del simulador:

| Componente | Estado |
|---|---|
| Generación de recursos | Implementado |
| Asignación semanal por recurso | Implementado |
| Horarios legales | Implementado |
| Francos fijos y variables | Implementado |
| Matriz de cobertura acumulada | Implementado |
| Trayectoria de acciones | Implementado |
| Generación de demanda inicial | Pendiente |
| Matriz de descuento aleatorio | Pendiente |
| Ruido estructurado | Pendiente |
| Scoring | Pendiente |
| Terminación anticipada | Pendiente |

---

## 16. Próximos pasos

Los próximos componentes a incorporar serán:

1. generación de la matriz de propensión deseada de demanda `Q`;
2. transformación de `Q` en pesos de descuento;
3. generación de matriz de descuento `D` o `R_discount`;
4. construcción de demanda inicial a partir de cobertura;
5. revisión de terminación anticipada;
6. cálculo del score final.

---

## 17. Notas de diseño

El diseño actual prioriza la claridad y trazabilidad.

La lógica central se basa en listas paralelas:

```python
resource_id_list
modality_list
week_list
entry_hour_list
holidays_list
```

Estas listas se combinan en:

```python
assignments
```

y luego cada asignación se transforma en cobertura.

Este enfoque permite inspeccionar fácilmente cada etapa del simulador, imprimir resultados intermedios y validar que la trayectoria generada tenga sentido antes de avanzar hacia la generación de demanda inicial con ruido.

# README — Codificación de acciones del `DemandSimulator`

Este documento describe la codificación discreta de acciones utilizada por la clase `DemandSimulator` para representar la trayectoria de decisiones asociada a la construcción de la matriz de cobertura.

La trayectoria `T` se guarda como una lista de pasos. Cada paso contiene una acción discreta identificada por `action_id`, junto con el tipo de decisión tomada y el valor real asociado.

---

## 1. Estructura general de una acción en la trayectoria

Cada elemento de la trayectoria tiene la siguiente estructura conceptual:

| Campo | Tipo | Descripción |
|---|---:|---|
| `step` | `int` | Número secuencial del paso dentro de la trayectoria. |
| `resource_id` | `int` | Identificador interno del recurso asignado. |
| `week` | `int` | Semana del horizonte en la que se aplica la acción. Valores posibles: `0`, `1`, `2`, `3`. |
| `action_type` | `str` | Tipo de decisión: `modality`, `entry_hour` o `holidays`. |
| `action_id` | `int` | Identificador discreto de la acción. |
| `action_value` | `Any` | Valor real representado por el `action_id`. |
| `Rt` | `np.ndarray` | Matriz de cobertura acumulada luego de ese paso. |

---

## 2. Tipos de acciones

Para cada asignación semanal se registran tres decisiones consecutivas:

| Orden | `action_type` | Qué representa | Cuándo cambia `Rt` |
|---:|---|---|---|
| 1 | `modality` | Elección de modalidad del recurso: 4h, 6h u 8h. | No cambia. |
| 2 | `entry_hour` | Elección de la hora de ingreso. | No cambia. |
| 3 | `holidays` | Elección del patrón de francos. | Sí cambia, porque la asignación semanal queda completa. |

La matriz de cobertura acumulada `Rt` solo se actualiza después de la acción `holidays`, porque recién en ese momento se conoce la asignación semanal completa:

```text
modalidad + hora de ingreso + francos
```

---

## 3. Convenciones temporales

| Concepto | Dominio | Convención |
|---|---:|---|
| Hora del día | `0..23` | `0` representa la hora 00:00. |
| Semana del horizonte | `0..3` | El horizonte mensual tiene 4 semanas. |
| Día global del horizonte | `0..27` | 28 días totales. |
| Día de semana | `0..6` | Se usa para francos. |

### Codificación de día de semana

| Día | Valor |
|---|---:|
| Domingo | `0` |
| Lunes | `1` |
| Martes | `2` |
| Miércoles | `3` |
| Jueves | `4` |
| Viernes | `5` |
| Sábado | `6` |

---

## 4. Dominio completo de acciones

La codificación se divide en tres bloques:

| Bloque | Rango de `action_id` | Tipo de acción | Descripción |
|---|---:|---|---|
| Modalidad | `0..2` | `modality` | Modalidad del recurso. |
| Hora de ingreso | `3..26` | `entry_hour` | Hora de ingreso entre 0 y 23. |
| Francos | `27..55` | `holidays` | Combinaciones posibles de 0, 1 o 2 días francos. |

---

## 5. Acciones de modalidad

| `action_id` | `action_type` | `value_type` | `action_value` | Descripción |
|---:|---|---|---:|---|
| `0` | `modality` | `int` | `4` | Selección de recurso de 4 horas. |
| `1` | `modality` | `int` | `6` | Selección de recurso de 6 horas. |
| `2` | `modality` | `int` | `8` | Selección de recurso de 8 horas. |

---

## 6. Acciones de hora de ingreso

Las acciones de hora de ingreso usan la fórmula:

```text
action_id = 3 + entry_hour
```

Por lo tanto:

| `action_id` | `action_type` | `value_type` | `action_value` | Descripción |
|---:|---|---|---:|---|
| `3` | `entry_hour` | `int` | `0` | Ingreso a la hora 0. |
| `4` | `entry_hour` | `int` | `1` | Ingreso a la hora 1. |
| `5` | `entry_hour` | `int` | `2` | Ingreso a la hora 2. |
| `6` | `entry_hour` | `int` | `3` | Ingreso a la hora 3. |
| `7` | `entry_hour` | `int` | `4` | Ingreso a la hora 4. |
| `8` | `entry_hour` | `int` | `5` | Ingreso a la hora 5. |
| `9` | `entry_hour` | `int` | `6` | Ingreso a la hora 6. |
| `10` | `entry_hour` | `int` | `7` | Ingreso a la hora 7. |
| `11` | `entry_hour` | `int` | `8` | Ingreso a la hora 8. |
| `12` | `entry_hour` | `int` | `9` | Ingreso a la hora 9. |
| `13` | `entry_hour` | `int` | `10` | Ingreso a la hora 10. |
| `14` | `entry_hour` | `int` | `11` | Ingreso a la hora 11. |
| `15` | `entry_hour` | `int` | `12` | Ingreso a la hora 12. |
| `16` | `entry_hour` | `int` | `13` | Ingreso a la hora 13. |
| `17` | `entry_hour` | `int` | `14` | Ingreso a la hora 14. |
| `18` | `entry_hour` | `int` | `15` | Ingreso a la hora 15. |
| `19` | `entry_hour` | `int` | `16` | Ingreso a la hora 16. |
| `20` | `entry_hour` | `int` | `17` | Ingreso a la hora 17. |
| `21` | `entry_hour` | `int` | `18` | Ingreso a la hora 18. |
| `22` | `entry_hour` | `int` | `19` | Ingreso a la hora 19. |
| `23` | `entry_hour` | `int` | `20` | Ingreso a la hora 20. |
| `24` | `entry_hour` | `int` | `21` | Ingreso a la hora 21. |
| `25` | `entry_hour` | `int` | `22` | Ingreso a la hora 22. |
| `26` | `entry_hour` | `int` | `23` | Ingreso a la hora 23. |

> Nota: no todos los `action_id` de hora son necesariamente legales en todos los estados. La legalidad depende de `entry_hours`, `close_hour` y la modalidad del recurso.

---

## 7. Acciones de francos

Las acciones de francos comienzan en:

```text
HOLIDAY_ACTION_OFFSET = 27
```

La codificación incluye:

1. sin francos;
2. un franco;
3. dos francos.

Aunque el dominio completo incluye todas las combinaciones posibles, durante la simulación solo se seleccionan las combinaciones compatibles con `fixed_holidays` y `var_holidays`.

---

## 8. Acción sin francos

| `action_id` | `action_type` | `value_type` | `action_value` | Descripción |
|---:|---|---|---|---|
| `27` | `holidays` | `tuple[int, ...]` | `()` | Sin días francos. |

---

## 9. Acciones con un franco

| `action_id` | `action_type` | `value_type` | `action_value` | Descripción |
|---:|---|---|---|---|
| `28` | `holidays` | `tuple[int, ...]` | `(0,)` | Franco domingo. |
| `29` | `holidays` | `tuple[int, ...]` | `(1,)` | Franco lunes. |
| `30` | `holidays` | `tuple[int, ...]` | `(2,)` | Franco martes. |
| `31` | `holidays` | `tuple[int, ...]` | `(3,)` | Franco miércoles. |
| `32` | `holidays` | `tuple[int, ...]` | `(4,)` | Franco jueves. |
| `33` | `holidays` | `tuple[int, ...]` | `(5,)` | Franco viernes. |
| `34` | `holidays` | `tuple[int, ...]` | `(6,)` | Franco sábado. |

---

## 10. Acciones con dos francos

| `action_id` | `action_type` | `value_type` | `action_value` | Descripción |
|---:|---|---|---|---|
| `35` | `holidays` | `tuple[int, ...]` | `(0, 1)` | Franco domingo y lunes. |
| `36` | `holidays` | `tuple[int, ...]` | `(0, 2)` | Franco domingo y martes. |
| `37` | `holidays` | `tuple[int, ...]` | `(0, 3)` | Franco domingo y miércoles. |
| `38` | `holidays` | `tuple[int, ...]` | `(0, 4)` | Franco domingo y jueves. |
| `39` | `holidays` | `tuple[int, ...]` | `(0, 5)` | Franco domingo y viernes. |
| `40` | `holidays` | `tuple[int, ...]` | `(0, 6)` | Franco domingo y sábado. |
| `41` | `holidays` | `tuple[int, ...]` | `(1, 2)` | Franco lunes y martes. |
| `42` | `holidays` | `tuple[int, ...]` | `(1, 3)` | Franco lunes y miércoles. |
| `43` | `holidays` | `tuple[int, ...]` | `(1, 4)` | Franco lunes y jueves. |
| `44` | `holidays` | `tuple[int, ...]` | `(1, 5)` | Franco lunes y viernes. |
| `45` | `holidays` | `tuple[int, ...]` | `(1, 6)` | Franco lunes y sábado. |
| `46` | `holidays` | `tuple[int, ...]` | `(2, 3)` | Franco martes y miércoles. |
| `47` | `holidays` | `tuple[int, ...]` | `(2, 4)` | Franco martes y jueves. |
| `48` | `holidays` | `tuple[int, ...]` | `(2, 5)` | Franco martes y viernes. |
| `49` | `holidays` | `tuple[int, ...]` | `(2, 6)` | Franco martes y sábado. |
| `50` | `holidays` | `tuple[int, ...]` | `(3, 4)` | Franco miércoles y jueves. |
| `51` | `holidays` | `tuple[int, ...]` | `(3, 5)` | Franco miércoles y viernes. |
| `52` | `holidays` | `tuple[int, ...]` | `(3, 6)` | Franco miércoles y sábado. |
| `53` | `holidays` | `tuple[int, ...]` | `(4, 5)` | Franco jueves y viernes. |
| `54` | `holidays` | `tuple[int, ...]` | `(4, 6)` | Franco jueves y sábado. |
| `55` | `holidays` | `tuple[int, ...]` | `(5, 6)` | Franco viernes y sábado. |

---

## 11. Resumen completo del dominio

| Rango | Cantidad de acciones | `action_type` | Descripción |
|---:|---:|---|---|
| `0..2` | 3 | `modality` | Modalidades 4h, 6h y 8h. |
| `3..26` | 24 | `entry_hour` | Horas de ingreso 0 a 23. |
| `27` | 1 | `holidays` | Sin franco. |
| `28..34` | 7 | `holidays` | Un franco. |
| `35..55` | 21 | `holidays` | Dos francos. |
| **Total** | **56** |  | Dominio completo de acciones. |

---

## 12. Acciones legales vs. dominio completo

El dominio completo tiene 56 acciones posibles, pero no todas son necesariamente seleccionables en una instancia concreta.

### Modalidad

Las modalidades seleccionables dependen de los recursos disponibles:

| Condición | Acción legal |
|---|---|
| `mod_4 > 0` | `action_id = 0` |
| `mod_6 > 0` | `action_id = 1` |
| `mod_8 > 0` | `action_id = 2` |

### Hora de ingreso

Las horas seleccionables dependen de:

| Parámetro | Efecto |
|---|---|
| `entry_hours` | Limita las horas candidatas. |
| `close_hour` | Elimina horas que exceden el cierre. |
| modalidad | Define la duración de la jornada. |

Ejemplo:

| Modalidad | `entry_hour` | `close_hour` | Horas cubiertas | Legal |
|---:|---:|---:|---|---|
| 8 | 15 | 22 | 15..22 | Sí |
| 8 | 16 | 22 | 16..23 | No |
| 4 | 18 | 22 | 18..21 | Sí |

### Francos

Las combinaciones legales dependen de:

| Parámetro | Efecto |
|---|---|
| `fixed_holidays` | Obliga a incluir un día fijo de franco. |
| `var_holidays` | Define cuántos francos adicionales se seleccionan. |

Ejemplos:

| `fixed_holidays` | `var_holidays` | Combinaciones legales |
|---:|---:|---|
| `None` | `0` | `()` |
| `None` | `1` | `(0,)`, `(1,)`, ..., `(6,)` |
| `None` | `2` | todas las combinaciones de dos días |
| `0` | `0` | `(0,)` |
| `0` | `1` | `(0,1)`, `(0,2)`, ..., `(0,6)` |

---

## 13. Ejemplo de trayectoria

Una asignación semanal completa podría generar tres pasos:

| `step` | `resource_id` | `week` | `action_type` | `action_id` | `action_value` | Comentario |
|---:|---:|---:|---|---:|---|---|
| `0` | `0` | `0` | `modality` | `2` | `8` | Recurso de 8 horas. |
| `1` | `0` | `0` | `entry_hour` | `11` | `8` | Ingreso a la hora 8. |
| `2` | `0` | `0` | `holidays` | `55` | `(5, 6)` | Franco viernes y sábado. |

La cobertura semanal se aplica luego del tercer paso, porque recién ahí se conoce la asignación completa.

---

## 14. Fórmulas útiles

### Codificación de modalidad

```text
4h -> 0
6h -> 1
8h -> 2
```

### Codificación de hora de ingreso

```text
action_id = 3 + entry_hour
```

### Decodificación de hora de ingreso

```text
entry_hour = action_id - 3
```

válido para:

```text
3 <= action_id <= 26
```

### Offset de francos

```text
HOLIDAY_ACTION_OFFSET = 27
```

---

## 15. Consideraciones de diseño

La codificación está separada en bloques para que sea fácil identificar el tipo de acción a partir del `action_id`:

| Condición sobre `action_id` | Tipo inferido |
|---|---|
| `0 <= action_id <= 2` | Modalidad |
| `3 <= action_id <= 26` | Hora de ingreso |
| `action_id >= 27` | Francos |

Esta estructura permite:

- mantener un dominio discreto único de acciones;
- registrar trayectorias compatibles con entrenamiento posterior;
- aplicar máscaras legales por etapa;
- separar decisiones parciales sin perder la trazabilidad de la asignación semanal completa.

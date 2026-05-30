# ResNet Evaluator

Evaluador neuronal para Workforce Scheduling basado en una red residual.

Este módulo separa tres responsabilidades:

- `StateEncoder`: transforma `ProblemSetup + WorkforceState` en un tensor.
- `WorkforceResNet`: red neuronal pura que recibe tensores y devuelve policy/value.
- `ResNetStateEvaluator`: wrapper compatible con la interfaz común de evaluadores.

## Archivos

```text
modules/evaluators/resnet/
├── encoder.py
├── resnet_evaluator.py
├── resnet_state_evaluator.py
├── checkpoints/
└── README.md
```

## Interfaz esperada

El wrapper `ResNetStateEvaluator` cumple el contrato definido en `modules/evaluators/base.py`:

```python
predict(state: WorkforceState) -> tuple[np.ndarray, float]
```

Devuelve:

- `policy`: `np.ndarray` de shape `(55,)`, con probabilidades sobre todo el espacio de acciones.
- `value`: `float`, estimación escalar del estado en `[-1, 1]`.

La policy se pasa luego a `WorkforceEngine.legal_mask`, que filtra acciones ilegales y renormaliza las probabilidades legales. El `value` no se modifica.

## Uso básico

```python
from modules.evaluators.resnet.resnet_state_evaluator import ResNetStateEvaluator

evaluator = ResNetStateEvaluator(
    setup=problem_setup,
    checkpoint_path="modules/evaluators/resnet/checkpoints/workforce_resnet_000.pt",
    device="auto",
)

policy, value = evaluator.predict(state)
```

Si no se pasa `checkpoint_path`, el wrapper busca el checkpoint `.pt` más reciente dentro de `checkpoints/`.

## StateEncoder

`StateEncoder` convierte un diccionario con datos de `ProblemSetup` y `WorkforceState` en un tensor:

```python
torch.Tensor shape (B, 93, 28, 28)
```

El wrapper arma ese diccionario automáticamente desde:

- `ProblemSetup`
- `WorkforceState`

### Datos usados desde `WorkforceState`

| Campo | Codificación |
|---|---|
| `residual_demand` | Canal 0, normalizado por `demand_ref`, con padding vertical para pasar de `24 x 28` a `28 x 28`. |
| `initial_demand_total` | Canal 1, normalizado por `demand_ref * 24 * 28`. |
| `remaining_stock` | Canales 2-4, uno por modalidad 4h, 6h y 8h, normalizado por `stock_ref`. |
| `current_modality` | Canales 5-7, one-hot para 4h, 6h y 8h. `None` queda todo en cero. |
| `assignment_week` | Canales 8-10. La semana 0 queda implícita en cero; semanas 1, 2 y 3 usan one-hot. |
| `current_entry_hour` | Canales 69-92, one-hot de hora de entrada. `None` queda todo en cero. |

### Datos usados desde `ProblemSetup`

| Campo | Codificación |
|---|---|
| `mobile_days_off_count` | Canales 11-13, one-hot para 0, 1 y 2 francos móviles. |
| `fixed_day_off` | Canales 14-20, one-hot del día fijo. `None` queda todo en cero. |
| `allowed_entry_hours` | Canales 21-44, multi-hot de horas permitidas. `None` equivale a todas las horas permitidas. |
| `closing_hour` | Canales 45-68, one-hot de hora de cierre. `None` queda todo en cero. |

`allowed_entry_hours` puede llegar crudo como lista simple para un sample:

```python
[6, 14, 18]
```

o como lista por sample para batches:

```python
[[6, 14, 18], None, [8, 16]]
```

La conversión a multi-hot ocurre dentro del encoder.

## Datos excluidos del encoder

Hay dos campos del dominio que no se codifican explícitamente:

### `expansion_mode`

No se agrega como canal porque, en el diseño actual del engine, es equivalente a:

```python
remaining_stock == np.array([0, 0, 0])
```

La red puede inferir esa condición desde los canales de stock.

### `max_overcoverage_tolerance`

No se agrega como canal porque se tratará como una variable fija del sistema en esta etapa.

La decisión reduce el espacio de variación de los samples de entrenamiento y evita sumar un canal adicional. Si en el futuro se entrena una red que deba generalizar entre distintas tolerancias de sobrecobertura, este campo debería incorporarse al encoder.

## Mapa de canales

| Canales | Contenido |
|---:|---|
| `0` | `residual_demand` |
| `1` | `initial_demand_total` |
| `2-4` | `remaining_stock` por modalidad |
| `5-7` | `current_modality` |
| `8-10` | `assignment_week` |
| `11-13` | `mobile_days_off_count` |
| `14-20` | `fixed_day_off` |
| `21-44` | `allowed_entry_hours` |
| `45-68` | `closing_hour` |
| `69-92` | `current_entry_hour` |

Total:

```python
93 canales
```

## WorkforceResNet

`WorkforceResNet` es la red neuronal pura.

Input:

```python
x.shape == (B, 93, 28, 28)
```

Output de `forward`:

```python
policy_logits.shape == (B, 55)
value.shape == (B,)
```

`forward` devuelve logits crudos para entrenamiento. El wrapper `ResNetStateEvaluator` aplica `softmax` para cumplir la interfaz de evaluador, que trabaja con probabilidades.

## Checkpoints

El checkpoint actual sigue esta estructura:

```python
{
    "model_state_dict": ...,
    "model_config": ...,
    "training_state": ...,
}
```

`ResNetStateEvaluator` usa `model_config` para reconstruir la arquitectura y luego carga `model_state_dict`.

# Demand Simulator

Este modulo genera escenarios de demanda para entrenar y validar el planificador.

La responsabilidad esta separada en dos piezas:

- `DemandSimulator`: genera una cobertura factible y una trayectoria positiva.
- `CompoundDemandSimulator`: variante experimental que genera directamente
  acciones semanales compuestas.
- `DemandNoiseGenerator`: toma esa cobertura y genera demanda inicial con descuento/ruido.

El modulo no guarda datos. La persistencia corresponde a `modules.storage`.

## Flujo

1. Se define un `ProblemSetup`.
2. `DemandSimulator.compute_coverage()` genera una matriz de cobertura `C`.
3. Esa cobertura se interpreta como demanda perfecta y se reconstruye una trayectoria.
4. `DemandNoiseGenerator.generate(C)` genera una demanda inicial `D0 = C - R`.
5. Las acciones de la trayectoria base pueden replayearse con `WorkforceEngine` sobre `D0`.

## DemandSimulator

Uso basico:

```python
from modules.demand_simulator import DemandSimulator
from modules.workforce_engine.schemas import ProblemSetup

setup = ProblemSetup(
    mobile_days_off_count=1,
    fixed_day_off=None,
    allowed_entry_hours=[8, 10, 12],
    max_overcoverage_tolerance=0.2,
    closing_hour=20,
)

simulator = DemandSimulator(problem_setup=setup, seed=42)

coverage_matrix, trajectory = simulator.compute_coverage(
    mod_4=10,
    mod_6=5,
    mod_8=3,
)
```

`coverage_matrix` tiene shape `(24, 28)`.

`trajectory` es una lista de samples compatibles con entrenamiento:

```python
{
    "state": WorkforceState,
    "policy": np.ndarray,  # shape (55,)
    "action_id": int,
    "reward": 1.0,
}
```

La `policy` es uniforme sobre las acciones legales del estado. Las acciones ilegales tienen probabilidad `0`.

## CompoundDemandSimulator

La variante compuesta utiliza el dominio fijo:

```text
modalidades: 4, 6 y 8
horarios: 6, 12 y 18
cierre: 22
franco fijo: día 6
un franco móvil: días 0 a 5
stock total máximo: 20
```

Cada acción aplica una semana completa y la policy tiene shape `(54,)`.
Cada recurso genera exactamente cuatro acciones:

```python
from modules.demand_simulator import CompoundDemandSimulator

simulator = CompoundDemandSimulator(problem_setup=setup, seed=42)
coverage_matrix, trajectory = simulator.compute_coverage(
    n_resources=20,
)
```

El simulador samplea directamente `action_id` legales por cierre:

- la primera acción de cada recurso se elige entre las 42 acciones posibles;
- esa acción determina la modalidad;
- las siguientes tres acciones se eligen dentro del bloque de esa modalidad.

La distribución de stock por modalidad surge de esas primeras acciones. Luego
se reconstruyen los snapshots con `CompoundWorkforceState`; cada acción
semanal modifica la demanda residual. El simulador no consulta ni ejecuta el
engine durante la generación.

## DemandNoiseGenerator

Uso basico:

```python
from modules.demand_simulator import DemandNoiseGenerator

noise_generator = DemandNoiseGenerator(
    k=0.30,
    k_exponential_lambda=10.0,
    max_daily_peaks=4,
    max_hourly_peaks=2,
    seed=42,
)

noise_result = noise_generator.generate(coverage_matrix)
```

Campos principales de `noise_result`:

- `initial_demand`: demanda inicial simulada, shape `(24, 28)`.
- `discount_matrix`: demanda removida desde la cobertura.
- `demand_propensity`: propension relativa de demanda.
- `k_effective`: descuento efectivo aplicado.
- `discount_total`: total descontado.

`k_effective` se samplea directamente en `[0, k]` desde una exponencial
truncada. `k_exponential_lambda` controla la concentracion cerca de cero; el
default `10.0` favorece descuentos efectivos bajos.

Garantias:

- `initial_demand >= 0`
- `discount_matrix >= 0`
- `discount_matrix <= coverage_matrix`
- `initial_demand = coverage_matrix - discount_matrix`

La trayectoria compuesta se corrige después del ruido mediante:

```python
from modules.trajectory_generation import CompoundTrajectoryReplayer
from modules.workforce_engine.compound_engine import CompoundWorkforceEngine

replayer = CompoundTrajectoryReplayer(
    CompoundWorkforceEngine(setup)
)
result = replayer.replay_trajectory(
    initial_demand=noise_result.initial_demand,
    source_trajectory=trajectory,
)
```

El replayer se detiene cuando el engine declara terminalidad, aunque todavía
queden acciones base. Luego retropropaga el reward terminal a todos los
samples generados.

La reducción posterior de stock trabaja con recursos completos:

```python
from modules.trajectory_generation import CompoundStockAdjuster

result = CompoundStockAdjuster(
    engine=CompoundWorkforceEngine(setup),
    p_stock=0.2,
).adjust(result["trajectory"])
```

Cada recurso es un chunk de cuatro acciones consecutivas. Cuando se activa la
reducción, se samplea una lista de chunks que estarán incluidos en el stock;
esos chunks se ubican primero y los restantes se reproducen en
`expansion_mode`. El ajustador llama internamente a
`CompoundTrajectoryReplayer`.

## Relacion con otros modulos

El simulador usa los schemas canonicos de `modules.workforce_engine.schemas`.

Para generar escenarios completos en memoria, incluyendo ruido, replay, stock scenario, augmentation y MCTS opcional, usar:

```python
from modules.trajectory_generation import generate_one_scenario
```

Para guardar trayectorias o samples, usar:

```python
from modules.storage import TrajectoryBuffer, SampleBuffer
```

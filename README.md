# Workforce Scheduling AI

Proyecto orientado al diseño e implementación de un sistema inteligente para la planificación de recursos humanos bajo restricciones operativas.

El objetivo general es construir una arquitectura capaz de asistir la generación de cronogramas laborales a partir de una demanda horaria previamente estimada, considerando modalidades de trabajo, disponibilidad de recursos, horarios de inicio, descansos y restricciones de operación.

## Contexto del problema

La planificación de recursos humanos, también conocida como *workforce scheduling*, es un problema operativo relevante en organizaciones intensivas en mano de obra, como retail, supermercados, centros logísticos, e-commerce, call centers, salud, manufactura y servicios.

En términos generales, el problema consiste en decidir cómo asignar recursos disponibles a distintos períodos de tiempo, de manera tal de cubrir una demanda variable respetando restricciones laborales y operativas.

Una planificación deficiente puede generar dos tipos de problemas:

- Subasignación de personal, con impacto en nivel de servicio, tiempos de espera, ventas, productividad o sobrecarga operativa.
- Sobreasignación de personal, con incremento de costos laborales y baja utilización de recursos.

El proyecto aborda este problema desde una perspectiva secuencial: cada decisión de asignación modifica el estado del sistema y condiciona las decisiones futuras.

## Alcance inicial

El modelo trabaja sobre un horizonte discreto de planificación compuesto por cuatro semanas consecutivas, equivalentes a 28 días calendario.

Cada día se divide en franjas horarias. En el caso base, se consideran 24 franjas horarias por día, por lo que la demanda puede representarse como una matriz de 24 x 28.

La demanda requerida se asume conocida al inicio del proceso de planificación. Esta demanda puede provenir de modelos de predicción, modelos de dimensionamiento, teoría de colas, escenarios simulados o definiciones expertas.

El proyecto se concentra en la capa de scheduling: transformar una demanda requerida por hora y día en asignaciones concretas de recursos.

## Stack tecnológico

La implementación inicial del proyecto se desarrollará en Python.

Para la gestión del entorno, dependencias y ejecución de comandos del proyecto se utilizará `uv`, con el objetivo de simplificar la instalación, acelerar la resolución de paquetes y mantener un entorno reproducible.

El entrenamiento y la implementación de modelos neuronales se realizarán con `PyTorch`. En particular, PyTorch será utilizado posteriormente para implementar el evaluador basado en una arquitectura residual tipo ResNet.

El stack inicial considerado es:

- Python
- uv
- PyTorch
- NumPy
- Pandas
- Pydantic
- Pytest
- Jupyter / notebooks exploratorios

## Hardware disponible

El desarrollo y las primeras pruebas experimentales se realizarán sobre una estación de trabajo local con GPU dedicada.

Hardware relevante disponible:

- GPU NVIDIA RTX 3060
- VRAM: 8 GB

Esta capacidad permite ejecutar pruebas iniciales con PyTorch utilizando aceleración por GPU. No obstante, el diseño del proyecto debe contemplar que algunos procesos puedan ejecutarse también en CPU, especialmente durante etapas de desarrollo, validación de reglas, pruebas unitarias y depuración del engine.

El uso intensivo de GPU se reservará principalmente para etapas posteriores vinculadas al entrenamiento del evaluador neuronal.

## Componentes principales del problema

El sistema considera los siguientes elementos principales.

### Demanda

La demanda representa la cantidad mínima de recursos requeridos para cada hora y día del horizonte de planificación.

Conceptualmente:

```text
D[h, t]
```

donde:

- `h` representa la franja horaria.
- `t` representa el día dentro del horizonte.
- `D[h, t]` representa la cantidad requerida de recursos.

### Modalidades laborales

La dotación disponible se organiza en modalidades horarias. En la formulación inicial se consideran tres modalidades:

- 4 horas
- 6 horas
- 8 horas

Cada recurso pertenece previamente a una modalidad. El sistema no decide la duración diaria libremente para cada recurso, sino que asigna recursos existentes respetando su modalidad.

### Recursos disponibles

El modelo recibe como entrada la cantidad de recursos disponibles por modalidad:

```text
N4, N6, N8
```

donde:

- `N4` es la cantidad de recursos disponibles de 4 horas.
- `N6` es la cantidad de recursos disponibles de 6 horas.
- `N8` es la cantidad de recursos disponibles de 8 horas.

Estos valores funcionan como stock inicial por modalidad.

### Horarios de inicio

Para cada asignación se debe seleccionar un horario de ingreso válido. El horario de inicio, combinado con la modalidad del recurso, determina la ventana horaria cubierta.

Por ejemplo, un recurso de 8 horas que inicia a las 10 cubre el intervalo desde las 10 hasta las 18.

### Horario de cierre

El modelo puede contemplar un horario de cierre operativo.

Cuando existe horario de cierre, una asignación es válida únicamente si cumple:

```text
start_hour + modality <= closing_hour
```

Cuando no existe horario de cierre definido, se permite que la jornada continúe más allá del día calendario, interpretándose como un desborde horario hacia el día siguiente.

### Descansos

El modelo contempla descansos semanales mediante dos conceptos:

- Franco fijo general.
- Francos móviles.

El franco fijo general representa un día de descanso común para toda la dotación. Los francos móviles se definen por recurso y por semana, permitiendo mayor flexibilidad para adaptar la cobertura a la demanda.

La cantidad máxima de días no laborables por semana es igual a dos.

## Decisiones del modelo

El cronograma se construye de manera secuencial.

Para cada recurso, el sistema debe definir una asignación compatible con su modalidad, las restricciones de descanso y las reglas operativas vigentes.

Las principales decisiones son:

1. Seleccionar la modalidad del recurso.
2. Seleccionar el horario de ingreso.
3. Seleccionar los francos correspondientes.
4. Aplicar la cobertura generada sobre la demanda residual.
5. Avanzar a la siguiente semana o al siguiente recurso.

Cada decisión altera la demanda residual y modifica las opciones futuras disponibles.

## Estado del sistema

En cada instante de decisión, el sistema se representa mediante un estado que contiene la información necesaria para decidir la próxima acción.

El estado incluye, de forma conceptual:

- Demanda residual pendiente de cobertura.
- Recursos disponibles por modalidad.
- Semana actual.
- Modalidad seleccionada, si corresponde.
- Horario de ingreso seleccionado, si corresponde.
- Configuración de descansos.
- Horario de cierre operativo, si corresponde.
- Información de progreso dentro de la asignación actual.

Esta representación permite modelar el problema como una secuencia de transiciones entre estados.

## Acciones

El espacio de acciones se descompone en subacciones simples para reducir la amplitud del árbol de decisión.

En lugar de seleccionar simultáneamente modalidad, horario y francos, el sistema decide en etapas:

```text
Modalidad -> Horario de ingreso -> Francos -> Aplicación de cobertura
```

Esta descomposición aumenta la profundidad del proceso, pero reduce significativamente la cantidad de alternativas evaluadas en cada paso.

## Arquitectura general

La arquitectura propuesta se organiza en módulos desacoplados.

### Workforce Engine

El `Workforce Engine` representa el entorno operativo del sistema.

Sus responsabilidades principales son:

- Validar acciones legales.
- Aplicar transiciones de estado.
- Actualizar la demanda residual.
- Gestionar el stock de recursos.
- Detectar estados terminales.
- Calcular el score final de una trayectoria.

El engine debe ser funcional y no depender de memoria interna mutable. Cada invocación recibe un estado y una acción, y devuelve el nuevo estado resultante.

### Legal Action Mask

La máscara de legalidad filtra las acciones inválidas en función del estado actual.

Debe considerar, entre otras restricciones:

- Etapa actual del proceso de decisión.
- Stock disponible por modalidad.
- Horarios de inicio válidos.
- Horario de cierre.
- Modalidad seleccionada.
- Franco fijo.
- Francos móviles.
- Cantidad máxima de descansos.
- Condición terminal.

### Search / Decision Module

El sistema puede utilizar un módulo de búsqueda secuencial para explorar consecuencias futuras antes de seleccionar una acción definitiva.

Este módulo consulta al `Workforce Engine` para simular transiciones y evaluar alternativas posibles.

### Evaluators

Los evaluadores de estado estiman la calidad relativa de un estado y la conveniencia inicial de las acciones posibles.

La arquitectura permite que existan distintos tipos de evaluadores, manteniendo la interfaz desacoplada del mecanismo concreto de evaluación.

En esta etapa del proyecto, el foco está puesto en definir correctamente el entorno, las acciones, las transiciones y las restricciones antes de profundizar en estrategias avanzadas de evaluación o entrenamiento.

## Objetivo funcional del proyecto

El objetivo inicial del proyecto es construir una base sólida para representar y resolver el problema de planificación.

La primera versión debería permitir:

1. Cargar una matriz de demanda.
2. Definir recursos disponibles por modalidad.
3. Configurar horarios de inicio válidos.
4. Configurar horario de cierre, si aplica.
5. Configurar reglas de descanso.
6. Generar estados iniciales.
7. Consultar acciones legales.
8. Aplicar acciones y obtener nuevos estados.
9. Actualizar demanda residual.
10. Detectar condiciones terminales.
11. Evaluar la calidad final de un cronograma.

## Estructura sugerida del proyecto

```text
project/
│
├── README.md
├── AGENTS.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
│
├── src/
│   └── scheduling_ai/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── state.py
│       │   ├── action.py
│       │   └── config.py
│       │
│       ├── rules/
│       │   ├── __init__.py
│       │   └── legal_action_mask.py
│       │
│       ├── search/
│       │   ├── __init__.py
│       │   └── search.py
│       │
│       ├── evaluators/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── resnet.py
│       │
│       └── utils/
│           ├── __init__.py
│           └── validation.py
│
├── tests/
│   ├── test_engine.py
│   ├── test_state.py
│   ├── test_action.py
│   └── test_legal_action_mask.py
│
├── notebooks/
│   └── exploration.ipynb
│
└── docs/
    ├── problem_context.md
    ├── architecture.md
    └── decisions.md
```

## Principios de diseño

El proyecto debe priorizar:

- Modularidad.
- Separación clara de responsabilidades.
- Estados explícitos e inmutables cuando sea posible.
- Transiciones determinísticas.
- Reglas centralizadas.
- Tests unitarios para la lógica crítica.
- Facilidad para experimentar con distintos evaluadores o estrategias de búsqueda.
- Compatibilidad con ejecución en CPU y GPU.

## Supuestos iniciales

La primera versión del sistema adopta los siguientes supuestos:

- La demanda es conocida al inicio del horizonte.
- El horizonte base es de 28 días.
- Cada día se divide en 24 franjas horarias.
- Los recursos son funcionalmente equivalentes dentro de cada modalidad.
- Las modalidades consideradas son 4, 6 y 8 horas.
- El ausentismo no se modela explícitamente.
- Las horas extra quedan fuera del alcance inicial.
- Los costos laborales quedan fuera del alcance inicial.
- La productividad individual queda fuera del alcance inicial.

## Fuera del alcance inicial

No forman parte de la primera versión funcional:

- Optimización económica por costos.
- Productividad individual por empleado.
- Horas extra.
- Preferencias individuales de empleados.
- Integración con sistemas externos.
- Interfaz gráfica.
- Entrenamiento avanzado de modelos neuronales.
- Comparación exhaustiva de estrategias de evaluación.

Aunque PyTorch será parte del stack tecnológico del proyecto, la prioridad inicial será consolidar el entorno base: representación del estado, acciones, reglas de legalidad, transiciones y evaluación terminal. La implementación y entrenamiento completo del evaluador neuronal tipo ResNet se abordará en una etapa posterior.

## Instalación

El proyecto utilizará `uv` para la gestión del entorno y las dependencias.

### Crear el proyecto

Si el proyecto todavía no fue inicializado:

```bash
uv init
```

### Crear el entorno virtual

```bash
uv venv
```

### Activar el entorno

En Linux/macOS:

```bash
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Definir versión de Python

Se recomienda usar una versión moderna y estable de Python, por ejemplo:

```bash
uv python pin 3.11
```

Esto genera el archivo:

```text
.python-version
```

### Instalar dependencias base

```bash
uv add numpy pandas pydantic pytest jupyter ipykernel
```

### Instalar PyTorch

Para instalación estándar:

```bash
uv add torch torchvision torchaudio
```

Para usar aceleración GPU con CUDA, verificar previamente la versión compatible con el sistema y la GPU.

### Sincronizar dependencias

Cuando el proyecto ya tenga `pyproject.toml` y `uv.lock`:

```bash
uv sync
```

### Ejecutar comandos dentro del entorno

```bash
uv run python
```

```bash
uv run pytest
```

```bash
uv run jupyter notebook
```

## Tests

El proyecto debería utilizar `pytest` para validar la lógica principal.

Ejecutar tests:

```bash
uv run pytest
```

Las pruebas iniciales deberían cubrir:

- Creación de estados.
- Validación de acciones.
- Máscara de legalidad.
- Transiciones del engine.
- Aplicación de cobertura.
- Detección de estados terminales.
- Cálculo de score final.

## Uso esperado con Codex

Codex debe utilizar este README junto con `AGENTS.md` para comprender el contexto del proyecto y las reglas de trabajo.

El flujo recomendado es:

1. Analizar primero la estructura del proyecto.
2. Proponer un plan antes de modificar archivos.
3. Realizar cambios pequeños y testeables.
4. Explicar los archivos modificados.
5. Indicar cómo validar los cambios.

Prompt inicial recomendado para Codex:

```text
Leé README.md y AGENTS.md. Confirmame cómo entendés el objetivo del proyecto, la arquitectura esperada y las reglas de trabajo. No modifiques archivos todavía.
```

## Estado actual

Proyecto en etapa inicial de configuración y diseño.

La prioridad actual es construir correctamente la representación del problema, el engine de transición y las reglas de legalidad antes de avanzar hacia modelos de evaluación más complejos.
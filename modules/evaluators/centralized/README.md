# Centralized Evaluator

Evaluador ResNet centralizado para workers MCTS.

El objetivo es mantener una sola instancia de `WorkforceResNet` en GPU y exponer
un cliente compatible con la interfaz actual de MCTS:

```python
predict(state) -> tuple[policy, value]
```

De esta forma los workers no cargan su propio modelo ni consumen VRAM adicional.

## Arquitectura

```text
worker MCTS
  └── CentralizedEvaluatorClient.predict(state)
        └── request_queue
              └── CentralizedEvaluatorServer
                    ├── StateEncoder
                    ├── WorkforceResNet
                    └── response_queue por client_id
```

Cada worker debe tener un `client_id` y una `response_queue` propia. Esto evita
que un worker consuma respuestas de otro.

## Sincronia

El cliente es sincrono: `predict` bloquea hasta recibir respuesta.

El server puede juntar requests pendientes hasta `max_batch_size` o esperar
`batch_wait_s` para armar un batch. Esto permite aprovechar GPU sin cambiar el
contrato del MCTS.

## Reload de pesos

`reload_weights(checkpoint_path)` envia una request de control al server.

El server termina el batch actual, carga el checkpoint nuevo, incrementa
`model_version` y luego responde. Mientras eso ocurre, los workers que llamen
`predict` quedan esperando.

El entrenamiento lo debe manejar un learner separado. Este modulo solo provee el
mecanismo de inferencia centralizada y recarga de checkpoint.

## Uso minimo

```python
from modules.evaluators.centralized import (
    CentralizedEvaluatorClient,
    CentralizedEvaluatorConfig,
)

config = CentralizedEvaluatorConfig(
    checkpoint_path="modules/evaluators/resnet/checkpoints/workforce_resnet_000.pt",
    device="auto",
)

client, process = CentralizedEvaluatorClient.start_server(
    config=config,
    setup=problem_setup,
)

policy, value = client.predict(state)
client.shutdown_server()
process.join()
```

## Uso con multiples workers

Un orquestador debe crear:

- una `request_queue` compartida;
- una `response_queue` por worker;
- un `CentralizedEvaluatorServer` con el diccionario `{client_id: response_queue}`;
- un `CentralizedEvaluatorClient` por worker.

El `setup` viaja en cada request porque distintas trayectorias pueden tener
distinto `ProblemSetup`.

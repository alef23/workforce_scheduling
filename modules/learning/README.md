# Learning

Entrenamiento de modelos desde `SampleBuffer`.

## ResNetSampleLearner

`ResNetSampleLearner` recibe un rango `[sample_start_index, sample_end_index)`,
mezcla sus indices una sola vez y los consume en batches sin reposicion.
Encodea `X` con `StateEncoder` y entrena `WorkforceResNet`.

La loss de policy es cross entropy soft:

```text
policy_loss_i = -sum(policy_target * log_softmax(policy_logits))
policy_loss   = mean(policy_loss_i * policy_weight_i)
```

`policy_weight` solo escala la loss de policy por sample. No modifica ni
renormaliza la policy target.

La loss total es:

```text
loss = policy_loss_weight * policy_loss + value_loss_weight * mse(value, target)
```

Uso desde un hook del orquestador:

```python
from modules.learning import ResNetLearnerConfig, ResNetSampleLearner

def on_cycle_ready(cycle_report):
    learner = ResNetSampleLearner(
        ResNetLearnerConfig(
            sample_buffer_path="datasets/samples/samples.zarr",
            checkpoint_path="modules/evaluators/resnet/checkpoints/workforce_resnet_000.pt",
            checkpoint_dir="modules/evaluators/resnet/checkpoints",
            sample_start_index=cycle_report.sample_start_index,
            sample_end_index=cycle_report.sample_end_index,
            batch_size=64,
            device="cuda",
        )
    )
    report = learner.train()
    return report.checkpoint_path
```

El path devuelto puede pasarse al evaluador centralizado para `reload_weights`.

import numpy as np
from typing import Any

class DummyEvaluator:
    """
    Evaluador dummy para pruebas.

    Devuelve:
    - priors/logits aleatorios para las 55 acciones del problema.
    - value aleatorio en [-1, 1].
    """

    def __init__(
        self,
        action_space_size: int = 55,
        random_seed: int | None = None,
    ):
        self.action_space_size = action_space_size
        self.rng = np.random.default_rng(random_seed)

    def predict(self, state: Any) -> tuple[np.ndarray, float]:
        priors = self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=self.action_space_size,
        )

        value = float(self.rng.uniform(-1.0, 1.0))

        return priors, value
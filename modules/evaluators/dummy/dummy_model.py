import numpy as np
from typing import Any

class DummyEvaluator:
    """
    Evaluador dummy para pruebas.

    Devuelve:
    - probabilidades aleatorias para las 55 acciones del problema.
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
        policy = self.rng.random(
            size=self.action_space_size,
        )
        policy = policy / policy.sum()

        value = float(self.rng.uniform(-1.0, 1.0))

        return policy, value

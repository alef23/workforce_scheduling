from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class EvaluatorProtocol(Protocol):
    """
    Interfaz mínima para evaluadores compatibles con MCTS.

    Un evaluador recibe un estado del entorno y devuelve:
    - probabilidades para todo el espacio de acciones.
    - una estimación escalar del valor del estado.
    """

    action_space_size: int

    def predict(self, state: Any) -> tuple[np.ndarray, float]:
        """
        Devuelve policy y value.

        Convenciones esperadas:
        - policy.shape == (action_space_size,)
        - policy contiene valores no negativos
        - value es un escalar, idealmente normalizado en [-1, 1]
        """

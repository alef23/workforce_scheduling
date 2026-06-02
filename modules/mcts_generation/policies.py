from __future__ import annotations

import numpy as np


def build_reweighted_policy(
    original_policy: np.ndarray,
    selected_action_id: int,
) -> np.ndarray:
    """
    Recalcula una policy artificial dando mas peso a la accion elegida.

    Las acciones legales se infieren desde entradas no cero de original_policy.

    Para Nl acciones legales:
    - accion seleccionada: 1 / (Nl - 1)
    - resto legal: (Nl - 2) / (Nl - 1)^2

    Casos borde:
    - Nl == 1: la accion seleccionada recibe 1.0
    - Nl == 2: la accion seleccionada recibe 1.0 y la otra accion legal 0.0
    """
    policy = np.asarray(original_policy, dtype=np.float32)
    if policy.ndim != 1:
        raise ValueError("original_policy debe ser un vector 1D.")

    selected_action_id = int(selected_action_id)
    if selected_action_id < 0 or selected_action_id >= policy.shape[0]:
        raise ValueError("selected_action_id esta fuera del rango de la policy.")

    legal_ids = np.flatnonzero(policy > 0)
    if len(legal_ids) == 0:
        raise ValueError("original_policy no contiene acciones legales.")
    if selected_action_id not in set(int(action_id) for action_id in legal_ids):
        raise ValueError("selected_action_id no esta entre las acciones legales.")

    output = np.zeros_like(policy, dtype=np.float32)
    n_legal = int(len(legal_ids))

    if n_legal == 1:
        output[selected_action_id] = 1.0
        return output

    selected_prob = 1.0 / float(n_legal - 1)
    other_prob = float(n_legal - 2) / float((n_legal - 1) ** 2)

    output[legal_ids] = other_prob
    output[selected_action_id] = selected_prob
    return output

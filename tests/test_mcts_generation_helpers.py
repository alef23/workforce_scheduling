import random

import numpy as np

from modules.mcts_generation import MCTSStartMode
from modules.mcts_generation.policies import build_reweighted_policy
from modules.mcts_generation.seed_selection import select_seed_step_indices


def test_reweighted_policy_three_legal_actions() -> None:
    policy = np.zeros(55, dtype=np.float32)
    policy[[0, 1, 2]] = 1 / 3

    output = build_reweighted_policy(policy, selected_action_id=1)

    assert np.isclose(output.sum(), 1.0)
    assert np.isclose(output[1], 0.5)
    assert np.isclose(output[0], 0.25)
    assert np.isclose(output[2], 0.25)
    assert np.count_nonzero(output) == 3


def test_reweighted_policy_two_legal_actions() -> None:
    policy = np.zeros(55, dtype=np.float32)
    policy[[9, 15]] = 0.5

    output = build_reweighted_policy(policy, selected_action_id=15)

    assert np.isclose(output.sum(), 1.0)
    assert np.isclose(output[15], 1.0)
    assert np.isclose(output[9], 0.0)


def test_seed_selection_initial_only() -> None:
    indices = select_seed_step_indices(
        trajectory_length=10,
        start_mode=MCTSStartMode.INITIAL_ONLY,
        max_seed_states=5,
        seed_state_probability=1.0,
        rng=random.Random(123),
    )

    assert indices == [0]


def test_seed_selection_forward_includes_initial_and_samples_after_it() -> None:
    indices = select_seed_step_indices(
        trajectory_length=10,
        start_mode=MCTSStartMode.FORWARD_SAMPLED,
        max_seed_states=3,
        seed_state_probability=1.0,
        rng=random.Random(123),
    )

    assert indices == [0, 1, 2, 3]


def test_seed_selection_backward_skips_terminal_and_includes_initial() -> None:
    indices = select_seed_step_indices(
        trajectory_length=10,
        start_mode=MCTSStartMode.BACKWARD_SAMPLED,
        max_seed_states=3,
        seed_state_probability=1.0,
        rng=random.Random(123),
    )

    assert indices == [0, 8, 7, 6]

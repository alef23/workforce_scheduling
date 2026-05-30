# MCTS Python scripts

This folder contains only the scripts directly related to the MCTS module.

## Files

- `mcts_schemas.py`: Pydantic schemas for `MCTSConfig`, `MCTSNode`, `MCTSAction`, and `MCTSResult`.
- `mcts.py`: domain-agnostic MCTS implementation.

## External dependencies expected by MCTS

The MCTS does not import the Workforce Engine. Instead, it expects an `engine` object with this interface:

- `action_space_size: int`
- `step(state, action_id) -> (next_state, is_terminal, reward)`
- `legal_mask(state, priors) -> masked_probs`
- `check_terminality(state) -> bool`
- `compute_reward(state) -> float`

It also expects an `evaluator` object with:

- `predict(state) -> (priors, value)`

## Dependencies

- `numpy`
- `pydantic`

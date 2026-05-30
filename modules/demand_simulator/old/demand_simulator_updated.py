from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np

from schemas import ProblemSetup


ACTION_SPACE_SIZE = 55
MODALITY_ACTION_START = 0
MODALITY_ACTION_END = 2
ENTRY_HOUR_ACTION_START = 3
ENTRY_HOUR_ACTION_END = 26
DAY_OFF_ACTION_START = 27
DAY_OFF_ACTION_END = 54

MODALITIES = [4, 6, 8]
MODALITY_TO_ACTION_ID = {4: 0, 6: 1, 8: 2}

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
WEEKS = 4
DAYS_IN_HORIZON = 28

DAY_OFF_ACTION_MATRIX = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6],
        [1, 7, 8, 9, 10, 11, 12],
        [2, 8, 13, 14, 15, 16, 17],
        [3, 9, 14, 18, 19, 20, 21],
        [4, 10, 15, 19, 22, 23, 24],
        [5, 11, 16, 20, 23, 25, 26],
        [6, 12, 17, 21, 24, 26, 27],
    ],
    dtype=int,
)


def _build_day_off_action_to_pair() -> dict[int, tuple[int, int]]:
    mapping: dict[int, tuple[int, int]] = {}
    for row in range(DAYS_PER_WEEK):
        for col in range(row, DAYS_PER_WEEK):
            internal_id = int(DAY_OFF_ACTION_MATRIX[row, col])
            mapping[internal_id] = (row, col)
    return mapping


DAY_OFF_ACTION_TO_PAIR = _build_day_off_action_to_pair()


class DemandSimulator:
    """
    Coverage simulator aligned with WorkforceEngine and MCTS action encoding.

    Returns:
    - coverage_matrix: accumulated coverage matrix, shape (24, 28).
    - trajectory: list of MCTS-like samples with exactly:
        state, policy, action_id, reward

    Notes:
    - state is None because this simulator does not build WorkforceState objects.
    - policy is uniform over legal actions for the simulated decision.
    - reward is always 1.0 because this constructive trajectory represents
      a perfectly covered demand scenario.
    """

    def __init__(self, problem_setup: ProblemSetup, seed: Optional[int] = None):
        self.problem_setup = problem_setup
        self.seed = seed
        self.rng = random.Random(seed)

        self.entry_hours = self._resolve_entry_hours(problem_setup.allowed_entry_hours)
        self.closing_hour = problem_setup.closing_hour
        self.fixed_day_off = problem_setup.fixed_day_off
        self.mobile_days_off_count = problem_setup.mobile_days_off_count

        self._validate_setup()

    def compute_coverage(
        self,
        mod_4: int,
        mod_6: int,
        mod_8: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Generates accumulated coverage and an MCTS-compatible trajectory.
        """
        self._validate_resource_inputs(mod_4, mod_6, mod_8)
        resources = self._build_resources_list(mod_4, mod_6, mod_8)
        return self._build_coverage_and_trajectory(resources)

    @staticmethod
    def _resolve_entry_hours(allowed_entry_hours: Optional[list[int]]) -> list[int]:
        if allowed_entry_hours is None:
            return list(range(HOURS_PER_DAY))
        return list(allowed_entry_hours)

    def _validate_setup(self) -> None:
        if not isinstance(self.entry_hours, list) or len(self.entry_hours) == 0:
            raise ValueError("allowed_entry_hours must resolve to a non-empty list.")
        if any(not isinstance(h, int) or h < 0 or h > 23 for h in self.entry_hours):
            raise ValueError("allowed_entry_hours must contain integers between 0 and 23.")
        if len(set(self.entry_hours)) != len(self.entry_hours):
            raise ValueError("allowed_entry_hours must not contain duplicated values.")
        if self.closing_hour is not None and (
            not isinstance(self.closing_hour, int)
            or self.closing_hour < 0
            or self.closing_hour > 23
        ):
            raise ValueError("closing_hour must be an integer between 0 and 23 or None.")
        if self.fixed_day_off is not None and (
            not isinstance(self.fixed_day_off, int)
            or self.fixed_day_off < 0
            or self.fixed_day_off > 6
        ):
            raise ValueError("fixed_day_off must be an integer between 0 and 6 or None.")
        if self.mobile_days_off_count not in (0, 1, 2):
            raise ValueError("mobile_days_off_count must be 0, 1 or 2.")
        fixed_count = 1 if self.fixed_day_off is not None else 0
        if fixed_count + self.mobile_days_off_count > 2:
            raise ValueError("Total days off cannot exceed 2.")

    @staticmethod
    def _validate_resource_inputs(mod_4: int, mod_6: int, mod_8: int) -> None:
        for name, value in {"mod_4": mod_4, "mod_6": mod_6, "mod_8": mod_8}.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0.")

    def _build_resources_list(self, mod_4: int, mod_6: int, mod_8: int) -> list[int]:
        resources = [4] * mod_4 + [6] * mod_6 + [8] * mod_8
        self.rng.shuffle(resources)
        return resources

    def _build_coverage_and_trajectory(
        self, resources: list[int]
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        coverage_matrix = np.zeros((HOURS_PER_DAY, DAYS_IN_HORIZON), dtype=int)
        trajectory: list[dict[str, Any]] = []

        for modality in resources:
            trajectory.append(
                self._make_trajectory_sample(
                    action_id=self._modality_action_id(modality),
                    legal_action_ids=self._legal_modality_action_ids(),
                )
            )

            for week in range(WEEKS):
                entry_hour = self._sample_entry_hour(modality)
                trajectory.append(
                    self._make_trajectory_sample(
                        action_id=self._entry_hour_action_id(entry_hour),
                        legal_action_ids=self._legal_entry_hour_action_ids(modality),
                    )
                )

                if self.mobile_days_off_count > 0:
                    day_off_action_id = self._sample_day_off_action_id()
                    days_off = self.decode_day_off_action(day_off_action_id)
                    trajectory.append(
                        self._make_trajectory_sample(
                            action_id=day_off_action_id,
                            legal_action_ids=self.get_legal_day_off_action_ids(),
                        )
                    )
                else:
                    days_off = self._get_days_off_without_mobile_action()

                coverage_matrix += self._build_weekly_coverage(
                    modality=modality,
                    week=week,
                    entry_hour=entry_hour,
                    days_off=days_off,
                )

        return coverage_matrix, trajectory

    @staticmethod
    def _make_trajectory_sample(
        action_id: int, legal_action_ids: list[int]
    ) -> dict[str, Any]:
        return {
            "state": None,
            "policy": DemandSimulator._uniform_policy_over_legal_actions(legal_action_ids),
            "action_id": action_id,
            "reward": 1.0,
        }

    @staticmethod
    def _uniform_policy_over_legal_actions(legal_action_ids: list[int]) -> np.ndarray:
        if len(legal_action_ids) == 0:
            raise ValueError("At least one legal action is required.")
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=float)
        policy[legal_action_ids] = 1.0 / len(legal_action_ids)
        return policy

    def _sample_entry_hour(self, modality: int) -> int:
        legal_hours = self._get_legal_entry_hours(modality)
        if not legal_hours:
            raise ValueError(
                f"No legal entry hours for modality={modality} and closing_hour={self.closing_hour}."
            )
        return self.rng.choice(legal_hours)

    def _get_legal_entry_hours(self, modality: int) -> list[int]:
        if modality not in MODALITIES:
            raise ValueError(f"Invalid modality: {modality}")
        if self.closing_hour is None:
            return list(self.entry_hours)
        return [hour for hour in self.entry_hours if hour + modality <= self.closing_hour]

    def _legal_entry_hour_action_ids(self, modality: int) -> list[int]:
        return [self._entry_hour_action_id(hour) for hour in self._get_legal_entry_hours(modality)]

    def get_legal_day_off_action_ids(self) -> list[int]:
        if self.mobile_days_off_count == 0:
            return []

        legal_internal_ids: set[int] = set()

        if self.fixed_day_off is None and self.mobile_days_off_count == 1:
            for day in range(DAYS_PER_WEEK):
                legal_internal_ids.add(int(DAY_OFF_ACTION_MATRIX[day, day]))
        elif self.fixed_day_off is not None and self.mobile_days_off_count == 1:
            fixed_day = self.fixed_day_off
            for mobile_day in range(DAYS_PER_WEEK):
                if mobile_day != fixed_day:
                    legal_internal_ids.add(int(DAY_OFF_ACTION_MATRIX[fixed_day, mobile_day]))
        elif self.fixed_day_off is None and self.mobile_days_off_count == 2:
            for d1 in range(DAYS_PER_WEEK):
                for d2 in range(d1 + 1, DAYS_PER_WEEK):
                    legal_internal_ids.add(int(DAY_OFF_ACTION_MATRIX[d1, d2]))
        else:
            raise ValueError("Unsupported day-off configuration in ProblemSetup.")

        return sorted(DAY_OFF_ACTION_START + internal_id for internal_id in legal_internal_ids)

    def _sample_day_off_action_id(self) -> int:
        legal_actions = self.get_legal_day_off_action_ids()
        if not legal_actions:
            raise ValueError("No legal day-off actions for current setup.")
        return self.rng.choice(legal_actions)

    def decode_day_off_action(self, action_id: int) -> set[int]:
        if action_id < DAY_OFF_ACTION_START or action_id > DAY_OFF_ACTION_END:
            raise ValueError("action_id does not belong to day-off block.")
        internal_id = action_id - DAY_OFF_ACTION_START
        if internal_id not in DAY_OFF_ACTION_TO_PAIR:
            raise ValueError(f"Invalid internal day-off id: {internal_id}")

        d1, d2 = DAY_OFF_ACTION_TO_PAIR[internal_id]

        if self.fixed_day_off is None:
            if self.mobile_days_off_count == 1:
                return {d1}
            if self.mobile_days_off_count == 2:
                return {d1, d2}
        if self.fixed_day_off is not None and self.mobile_days_off_count == 1:
            return {d1, d2}
        raise ValueError("Day-off action is incompatible with current setup.")

    def _get_days_off_without_mobile_action(self) -> set[int]:
        if self.fixed_day_off is None:
            return set()
        return {self.fixed_day_off}

    def _build_weekly_coverage(
        self,
        modality: int,
        week: int,
        entry_hour: int,
        days_off: set[int],
    ) -> np.ndarray:
        coverage = np.zeros((HOURS_PER_DAY, DAYS_IN_HORIZON), dtype=int)
        week_start_day = week * DAYS_PER_WEEK
        working_days = set(range(DAYS_PER_WEEK)) - set(days_off)

        for relative_day in sorted(working_days):
            absolute_day = week_start_day + relative_day
            self._apply_daily_shift(
                coverage=coverage,
                absolute_day=absolute_day,
                entry_hour=entry_hour,
                modality=modality,
            )
        return coverage

    def _apply_daily_shift(
        self,
        coverage: np.ndarray,
        absolute_day: int,
        entry_hour: int,
        modality: int,
    ) -> None:
        for offset in range(modality):
            raw_hour = entry_hour + offset
            target_day = absolute_day + raw_hour // HOURS_PER_DAY
            target_hour = raw_hour % HOURS_PER_DAY
            if target_day >= DAYS_IN_HORIZON:
                raise ValueError("Coverage exceeds the 28-day horizon.")
            coverage[target_hour, target_day] += 1

    @staticmethod
    def _legal_modality_action_ids() -> list[int]:
        return [0, 1, 2]

    @staticmethod
    def _modality_action_id(modality: int) -> int:
        if modality not in MODALITY_TO_ACTION_ID:
            raise ValueError("Modality must be one of 4, 6 or 8.")
        return MODALITY_TO_ACTION_ID[modality]

    @staticmethod
    def _entry_hour_action_id(entry_hour: int) -> int:
        if not isinstance(entry_hour, int) or entry_hour < 0 or entry_hour > 23:
            raise ValueError("entry_hour must be an integer between 0 and 23.")
        return ENTRY_HOUR_ACTION_START + entry_hour

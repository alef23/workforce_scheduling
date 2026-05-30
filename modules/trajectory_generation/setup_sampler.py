from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from modules.workforce_engine.schemas import ProblemSetup


@dataclass
class ProblemSetupSampler:
    """
    Genera ProblemSetup con parámetros fijos o aleatorios.

    Regla:
    - Si un parámetro viene informado, se usa tal cual.
    - Si viene en None, se samplea aleatoriamente.

    Importante:
    - scoring_k representa el k del problema/scoring.
    - Este valor se guarda en ProblemSetup.max_overcoverage_tolerance.
    - El kmax del ruido NO vive acá; se define en ScenarioGenerationConfig.noise_k_max.
    """

    allowed_entry_hours: Optional[list[int]] = None
    closing_hour: Optional[int] = 22
    mobile_days_off_count: Optional[int] = None
    fixed_day_off: Optional[int] = None
    scoring_k: float = 0.15

    random_entry_hours_count: int = 3
    random_entry_hours_pool: tuple[int, ...] = tuple(range(0, 24))
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def sample(self) -> ProblemSetup:
        mobile_days_off_count = self._sample_mobile_days_off_count()
        fixed_day_off = self._sample_fixed_day_off(mobile_days_off_count)
        allowed_entry_hours = self._sample_allowed_entry_hours()
        closing_hour = self._sample_closing_hour()

        return ProblemSetup(
            mobile_days_off_count=mobile_days_off_count,
            fixed_day_off=fixed_day_off,
            allowed_entry_hours=allowed_entry_hours,
            closing_hour=closing_hour,
            max_overcoverage_tolerance=float(self.scoring_k),
        )

    def _sample_mobile_days_off_count(self) -> int:
        if self.mobile_days_off_count is not None:
            return int(self.mobile_days_off_count)
        return self.rng.randint(0, 2)

    def _sample_fixed_day_off(self, mobile_days_off_count: int) -> int | None:
        if self.fixed_day_off is not None:
            return int(self.fixed_day_off)

        if mobile_days_off_count >= 2:
            return None

        sampled = self.rng.randint(-1, 6)
        return None if sampled == -1 else int(sampled)

    def _sample_allowed_entry_hours(self) -> list[int]:
        if self.allowed_entry_hours is not None:
            return [int(h) for h in self.allowed_entry_hours]

        hours = self.rng.sample(
            list(self.random_entry_hours_pool),
            k=int(self.random_entry_hours_count),
        )
        return sorted(int(h) for h in hours)

    def _sample_closing_hour(self) -> int | None:
        if self.closing_hour is None:
            return None
        return int(self.closing_hour)

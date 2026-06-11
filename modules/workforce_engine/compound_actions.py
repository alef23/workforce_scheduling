from __future__ import annotations

from dataclasses import dataclass


ACTION_SPACE_SIZE = 54
MODALITIES = (4, 6, 8)
ENTRY_HOURS = (6, 12, 18)
MOBILE_DAYS_OFF = (0, 1, 2, 3, 4, 5)
FIXED_DAY_OFF = 6

MODALITY_BLOCK_SIZE = len(ENTRY_HOURS) * len(MOBILE_DAYS_OFF)
ENTRY_HOUR_BLOCK_SIZE = len(MOBILE_DAYS_OFF)


@dataclass(frozen=True)
class CompoundAction:
    action_id: int
    modality_index: int
    entry_hour_index: int
    mobile_day_off: int
    modality: int
    entry_hour: int

    @property
    def days_off(self) -> frozenset[int]:
        return frozenset((FIXED_DAY_OFF, self.mobile_day_off))


def encode_action(
    modality_index: int,
    entry_hour_index: int,
    mobile_day_off: int,
) -> int:
    if modality_index not in range(len(MODALITIES)):
        raise ValueError("modality_index debe ser 0, 1 o 2.")
    if entry_hour_index not in range(len(ENTRY_HOURS)):
        raise ValueError("entry_hour_index debe ser 0, 1 o 2.")
    if mobile_day_off not in MOBILE_DAYS_OFF:
        raise ValueError("mobile_day_off debe estar entre 0 y 5.")

    return (
        modality_index * MODALITY_BLOCK_SIZE
        + entry_hour_index * ENTRY_HOUR_BLOCK_SIZE
        + mobile_day_off
    )


def decode_action(action_id: int) -> CompoundAction:
    if not isinstance(action_id, int):
        raise TypeError("action_id debe ser entero.")
    if action_id < 0 or action_id >= ACTION_SPACE_SIZE:
        raise ValueError(f"action_id fuera de rango: {action_id}")

    modality_index = action_id // MODALITY_BLOCK_SIZE
    remainder = action_id % MODALITY_BLOCK_SIZE
    entry_hour_index = remainder // ENTRY_HOUR_BLOCK_SIZE
    mobile_day_off = remainder % ENTRY_HOUR_BLOCK_SIZE

    return CompoundAction(
        action_id=action_id,
        modality_index=modality_index,
        entry_hour_index=entry_hour_index,
        mobile_day_off=mobile_day_off,
        modality=MODALITIES[modality_index],
        entry_hour=ENTRY_HOURS[entry_hour_index],
    )

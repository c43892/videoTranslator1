"""DurationMatcher v1 policy math (§7.1).

Pure functions so the worker's fitting decisions are unit-testable without
any media files. ``atempo`` is always generated/available — the legacy
AudioStitcher passed target/generated and slowed long speech down further.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class DurationPolicy:
    duration_tolerance_ms: int = 50
    soft_max_tempo: float = 1.20
    hard_max_tempo: float = 1.35
    max_borrow_silence_ms: int = 500
    max_borrow_ratio: float = 0.20
    max_translation_compaction_attempts: int = 2
    policy_version: str = "duration-v1"


class FitAction(StrEnum):
    PAD = "pad"  # shorter than base: keep natural pace, pad trailing silence
    NATURAL = "natural"  # fits base within tolerance: no adjustment
    BORROW = "borrow"  # overflows base but fits with borrowed following silence
    SPEED_UP = "speed_up"  # atempo within soft limit
    COMPACT = "compact"  # over soft limit: re-translate shorter and re-TTS
    FAIL = "fail"  # still over hard limit after compaction retries


@dataclass(frozen=True)
class FitDecision:
    action: FitAction
    available_duration_ms: int
    borrowed_gap_ms: int
    required_tempo_ratio: float
    atempo: float  # generated/available; 1.0 when no speed change


def borrowable_gap_ms(base_duration_ms: int, gap_to_next_ms: int, policy: DurationPolicy) -> int:
    return int(
        min(
            max(0, gap_to_next_ms),
            policy.max_borrow_silence_ms,
            int(base_duration_ms * policy.max_borrow_ratio),
        )
    )


def atempo_factor(generated_duration_ms: int, target_duration_ms: int) -> float:
    """FFmpeg atempo multiplier; 8s→5s must yield 1.6, never 0.625."""
    if target_duration_ms <= 0:
        raise ValueError("target duration must be positive")
    return generated_duration_ms / target_duration_ms


def decide_fit(
    generated_duration_ms: int,
    base_duration_ms: int,
    gap_to_next_ms: int,
    compaction_attempts_used: int,
    policy: DurationPolicy = DurationPolicy(),
) -> FitDecision:
    borrowed = borrowable_gap_ms(base_duration_ms, gap_to_next_ms, policy)
    available = base_duration_ms + borrowed

    if generated_duration_ms <= base_duration_ms - policy.duration_tolerance_ms:
        return FitDecision(FitAction.PAD, available, 0, 1.0, 1.0)
    if generated_duration_ms <= base_duration_ms:
        return FitDecision(FitAction.NATURAL, available, 0, 1.0, 1.0)

    ratio = generated_duration_ms / available
    if ratio <= 1.0:
        return FitDecision(FitAction.BORROW, available, borrowed, ratio, 1.0)
    if ratio <= policy.soft_max_tempo:
        return FitDecision(FitAction.SPEED_UP, available, borrowed, ratio, atempo_factor(generated_duration_ms, available))
    if compaction_attempts_used < policy.max_translation_compaction_attempts:
        return FitDecision(FitAction.COMPACT, available, borrowed, ratio, 1.0)
    if ratio <= policy.hard_max_tempo:
        return FitDecision(FitAction.SPEED_UP, available, borrowed, ratio, atempo_factor(generated_duration_ms, available))
    return FitDecision(FitAction.FAIL, available, borrowed, ratio, 1.0)

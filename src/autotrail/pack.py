"""Hound profiles and pack presets.

A ``HoundProfile`` holds the few numbers the tier 1 simulator needs. Paces are
stored in seconds per metre internally but are set in minutes per mile, the
way hashers talk about them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

METRES_PER_MILE = 1609.344


def pace_from_min_per_mile(min_per_mile: float) -> float:
    """Minutes per mile -> seconds per metre."""
    return min_per_mile * 60.0 / METRES_PER_MILE


def min_per_mile(pace_s_per_m: float) -> float:
    return pace_s_per_m * METRES_PER_MILE / 60.0


def fmt_pace(pace_s_per_m: float) -> str:
    total = round(min_per_mile(pace_s_per_m) * 60)
    return f"{total // 60}:{total % 60:02d}/mi"


@dataclass(frozen=True)
class HoundProfile:
    name: str
    pace_s_per_m: float          # base pace
    pace_jitter: float = 0.03    # each run draws pace * (1 +/- jitter)
    detect_prob: float = 0.9     # chance of spotting any one mark while running
    give_up_m: float = 200.0     # run this far past the last mark seen, then turn back
    check_search_m: float = 150.0  # how far out a hound runs a direction from a check
    confirm_marks: int = 2       # marks found past a check before calling On-On
    sight_m: float = 40.0        # a mark this close past a corner can be seen from it
    check_pause_s: float = 4.0   # time to look around before picking a direction
    follow_s: float = 45.0       # takes a corner the right way if someone did within this
    split_preference: str = "eagle"  # phase 2 (Turkey/Eagle splits)
    role: str = "runner"         # runner, walker, frb

    @property
    def pace_label(self) -> str:
        return fmt_pace(self.pace_s_per_m)


def runners(n: int = 12, fastest: float = 7.5, slowest: float = 9.5, **kw) -> list[HoundProfile]:
    """``n`` runners with paces spread evenly from ``fastest`` to ``slowest`` min/mile."""
    out = []
    for i in range(n):
        mpm = fastest if n == 1 else fastest + (slowest - fastest) * i / (n - 1)
        out.append(HoundProfile(name=f"Runner {i + 1}", pace_s_per_m=pace_from_min_per_mile(mpm),
                                role="runner", **kw))
    return out


def default_pack() -> list[HoundProfile]:
    """MVP default: 12 runners at 7:30 to 9:30 min/mile. Walkers join in phase 2."""
    return runners(12, 7.5, 9.5)


def perfect(profile: HoundProfile) -> HoundProfile:
    """A hound that never misses a mark (for tests and the true trail baseline)."""
    return replace(profile, detect_prob=1.0, pace_jitter=0.0)

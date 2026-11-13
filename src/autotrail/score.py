"""Turn sim results into a ScoreCard: one number per term plus a weighted total.

MVP terms (design doc, "Trail scoring"):
  * distance_fit: true trail vs the target length.
  * cohesion: finish spread, front-to-back gap along the trail, and how much of
    the time the back of the pack runs with nobody in sight.
  * check_effectiveness: how much of the front-to-back gap the checks close.

Each term is a ``ScoreTerm`` plug-in returning a value from 0 (bad) to 1
(ideal) plus the raw numbers behind it. New terms join with weight 0 until
someone tunes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Protocol

import numpy as np

from .pack import fmt_pace
from .sim import SimResult, sample_tracks

METRES_PER_MILE = 1609.344


def _half_at(x: float, half: float) -> float:
    """1 at 0, 0.5 at ``half``, 0.25 at twice that: a gradient, not a cliff."""
    return float(0.5 ** (max(0.0, x) / half))


@dataclass
class TermScore:
    name: str
    value: float
    weight: float
    details: dict = field(default_factory=dict)


@dataclass
class ScoreCard:
    terms: list[TermScore]
    summary: dict

    @property
    def total(self) -> float:
        w = sum(t.weight for t in self.terms)
        return 100.0 * sum(t.value * t.weight for t in self.terms) / w if w else 0.0

    def term(self, name: str) -> TermScore:
        return next(t for t in self.terms if t.name == name)


class ScoreTerm(Protocol):
    name: str
    weight: float

    def score(self, true_trail_m: float, results: list[SimResult]) -> TermScore: ...


@dataclass
class DistanceFit:
    """Full marks within +/- ``ok`` of the target, zero at +/- ``zero``."""
    target_m: float | None
    weight: float = 1.0
    ok: float = 0.05
    zero: float = 0.25
    name: str = "distance_fit"

    def score(self, true_trail_m, results):
        if not self.target_m:
            return TermScore(self.name, 1.0, 0.0, {"note": "no target distance"})
        err = (true_trail_m - self.target_m) / self.target_m
        v = 1.0 - max(0.0, abs(err) - self.ok) / (self.zero - self.ok)
        return TermScore(self.name, float(min(1.0, max(0.0, v))), self.weight,
                         {"true_trail_mi": true_trail_m / METRES_PER_MILE,
                          "target_mi": self.target_m / METRES_PER_MILE, "error_pct": 100 * err})


def alone_fraction_of_last(result: SimResult, sight_m: float = 150.0, dt: float = 20.0) -> float:
    """Share of the last finisher's running time with no other hound within ``sight_m``."""
    fin = [(h.finish_s, i) for i, h in enumerate(result.hounds) if h.finish_s is not None]
    if len(result.hounds) < 2 or not fin:
        return 0.0
    _, last = max(fin)
    times, _, xy = sample_tracks(result, dt)
    active = times <= result.hounds[last].finish_s
    me = xy[last, active]
    others = np.delete(xy[:, active], last, axis=0)
    d = np.hypot(others[..., 0] - me[:, 0], others[..., 1] - me[:, 1])
    alone = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=0) > sight_m
    return float(alone.mean()) if len(alone) else 0.0


@dataclass
class Cohesion:
    """Ideal: everyone arrives at once. Each part scores 0.5 at ``half_s`` (3 min)."""
    weight: float = 2.0
    half_s: float = 180.0
    sight_m: float = 150.0
    name: str = "cohesion"

    def score(self, true_trail_m, results):
        spreads, mean_gaps, worst_gaps, alone = [], [], [], []
        for r in results:
            fin = [h.finish_s for h in r.finishers]
            spreads.append(max(fin) - min(fin) if len(fin) >= 2 else 0.0)
            gaps = [g for _, g in r.gap_profile]
            mean_gaps.append(mean(gaps) if gaps else 0.0)
            worst_gaps.append(max(gaps) if gaps else 0.0)
            alone.append(alone_fraction_of_last(r, self.sight_m))
        d = {"finish_spread_min": median(spreads) / 60, "mean_gap_min": mean(mean_gaps) / 60,
             "worst_gap_min": median(worst_gaps) / 60, "back_alone_pct": 100 * mean(alone)}
        v = (0.5 * _half_at(median(spreads), self.half_s)
             + 0.3 * _half_at(mean(mean_gaps), self.half_s)
             + 0.2 * _half_at(median(worst_gaps), self.half_s))
        v *= 1.0 - 0.3 * mean(alone)
        # Hounds that never reach the Beer Near are the worst kind of spread.
        n = sum(len(r.hounds) for r in results)
        done = sum(len(r.finishers) for r in results) / max(1, n)
        if done < 1.0:
            d["dnf_pct"] = 100 * (1 - done)
            v *= done ** 2
        return TermScore(self.name, v, self.weight, d)


@dataclass
class CheckEffectiveness:
    """Share of the front-to-back gap that checks close (gap in vs gap just after)."""
    weight: float = 1.0
    name: str = "check_effectiveness"

    def score(self, true_trail_m, results):
        gin = gout = 0.0
        losses = []
        n = 0
        for r in results:
            for c in r.checks:
                if c.gap_in_s is None or c.gap_out_s is None:
                    continue
                gin += c.gap_in_s
                gout += c.gap_out_s
                n += 1
                if c.front_loss_s is not None:
                    losses.append(c.front_loss_s)
        if not n:
            return TermScore(self.name, 0.0, self.weight, {"note": "no checks"})
        closed = (gin - gout) / max(gin, 1.0)
        v = min(1.0, max(0.0, 0.5 + 0.5 * closed))
        return TermScore(self.name, v, self.weight,
                         {"gap_closed_pct": 100 * closed, "checks": n // max(1, len(results)),
                          "front_loss_min": (mean(losses) / 60) if losses else 0.0})


def default_terms(target_m: float | None) -> list:
    return [DistanceFit(target_m), Cohesion(), CheckEffectiveness()]


def summarize(results: list[SimResult]) -> dict:
    """Per-hound averages over runs: finish, distance, extra over true trail."""
    L = results[0].true_trail_m
    rows = []
    for i, h0 in enumerate(results[0].hounds):
        hs = [r.hounds[i] for r in results]
        fin = [h.finish_s for h in hs if h.finish_s is not None]
        rows.append({
            "name": h0.name, "role": h0.role, "pace": fmt_pace(h0.pace_s_per_m),
            "pace_s_per_m": h0.pace_s_per_m,
            "finish_min": (mean(fin) / 60) if fin else None,
            "dnf": len(hs) - len(fin),
            "distance_mi": mean(h.distance_m for h in hs) / METRES_PER_MILE,
            "extra_mi": mean(h.extra_m for h in hs) / METRES_PER_MILE,
        })
    k = max(1, len(rows) // 4)
    by_pace = sorted(rows, key=lambda r: r["pace_s_per_m"])
    return {
        "runs": len(results),
        "true_trail_mi": L / METRES_PER_MILE,
        "hounds": rows,
        "front_extra_mi": mean(r["extra_mi"] for r in by_pace[:k]),
        "back_extra_mi": mean(r["extra_mi"] for r in by_pace[-k:]),
        "pack_extra_mi": mean(r["extra_mi"] for r in rows),
    }


def score_results(results: list[SimResult], target_m: float | None = None,
                  terms: list | None = None) -> ScoreCard:
    terms = terms if terms is not None else default_terms(target_m)
    L = results[0].true_trail_m
    return ScoreCard([t.score(L, results) for t in terms], summarize(results))


def format_report(card: ScoreCard, results: list[SimResult] | None = None) -> str:
    s = card.summary
    lines = [f"Score: {card.total:.0f} / 100   ({s['runs']} simulated runs)", ""]
    for t in card.terms:
        det = ", ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in t.details.items())
        lines.append(f"  {t.name:<20} {t.value:5.2f}  x{t.weight:g}   {det}")
    lines += ["", f"True trail {s['true_trail_mi']:.2f} mi.  Extra distance run: front quarter "
              f"{s['front_extra_mi']:.2f} mi, back quarter {s['back_extra_mi']:.2f} mi, "
              f"pack average {s['pack_extra_mi']:.2f} mi.", ""]
    lines.append(f"  {'hound':<10} {'pace':>8} {'finish':>8} {'ran':>8} {'extra':>8}")
    for r in s["hounds"]:
        fin = f"{r['finish_min']:.1f}m" if r["finish_min"] is not None else "DNF"
        if r["dnf"] and r["finish_min"] is not None:
            fin += "*"
        lines.append(f"  {r['name']:<10} {r['pace']:>8} {fin:>8} {r['distance_mi']:>6.2f}mi "
                     f"{r['extra_mi']:>+6.2f}mi")
    if results and results[0].checks:
        lines += ["", "  check   at mi   front lost   gap in -> out  (minutes, mean over runs)"]
        for i, c0 in enumerate(results[0].checks):
            cs = [r.checks[i] for r in results]

            def avg(attr):
                v = [getattr(c, attr) for c in cs if getattr(c, attr) is not None]
                return mean(v) / 60 if v else float("nan")
            lines.append(f"  {i + 1:>5}   {c0.s / METRES_PER_MILE:5.2f}   {avg('front_loss_s'):9.1f}"
                         f"   {avg('gap_in_s'):5.1f} -> {avg('gap_out_s'):.1f}")
    return "\n".join(lines)

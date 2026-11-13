import random
import time

import pytest

from autotrail.generate import GenerateOptions, add_marks, blob_positions, needs_marks, place_marks
from autotrail.pack import HareProfile, default_pack, pace_from_min_per_mile, perfect, runners
from autotrail.score import score_results
from autotrail.sim import TrailModel, simulate, simulate_many
from autotrail.trail import EdgePos, EdgeRef, Mark, MarkKind, Trail
from autotrail.viz import replay_map

N = 16  # the city fixture is N x N


def street(g, a, b):
    return [EdgeRef(*e) for e in g.shortest_edges(a, b)]


def straight_trail(g, cols=8, row=3):
    """East along one street of the city grid: corners only at crossings."""
    a, b = row * N + 1, row * N + cols
    return Trail(start=a, end=b, true_path=street(g, a, b))


def l_trail(g):
    """East along row 3, then north from column 8 (the city grid has gaps, so maybe a jog or two)."""
    corner = 3 * N + 8
    edges = street(g, 3 * N + 1, corner) + street(g, corner, 10 * N + 8)
    return Trail(start=edges[0].u, end=edges[-1].v, true_path=edges), corner


def long_trail(g, seed=0, **kw):
    route = street(g, 1, N * N)  # corner to corner, ~3.5 km with plenty of turns
    t = Trail(start=route[0].u, end=route[-1].v, true_path=route)
    return place_marks(t, g, random.Random(seed), GenerateOptions(**kw))


def test_perfect_hound_runs_exactly_true_trail(city):
    t = add_marks(straight_trail(city), city)
    hound = perfect(runners(1)[0])
    r = simulate(t, city, [hound], seed=1)
    L = t.true_trail_m(city)
    h = r.hounds[0]
    assert h.finish_s == pytest.approx(L * hound.pace_s_per_m, rel=1e-6)
    assert h.distance_m == pytest.approx(L)
    assert h.extra_m == pytest.approx(0.0, abs=1e-6)


def blobbed(g, t):
    """The trail with blobs every ~40 m and no other marks."""
    from autotrail.generate import _cum, _pos
    edges = [(e.u, e.v, e.key) for e in t.true_path]
    cum = _cum(g, edges)
    blobs = [Mark(MarkKind.BLOB, _pos(edges, cum, s)) for s in blob_positions(float(cum[-1]), 40.0, random.Random(0))]
    return Trail(start=t.start, end=t.end, true_path=t.true_path, marks=blobs)


def test_unmarked_turn_needs_a_mark(city):
    t, corner = l_trail(city)
    t = blobbed(city, t)
    model = TrailModel(t, city)
    assert model.unmarked_turns and model.warnings
    assert needs_marks(t, city)
    t2 = add_marks(t, city)
    arrows = [m for m in t2.marks if m.kind == MarkKind.ARROW]
    assert corner in [m.at.node for m in arrows]
    assert not TrailModel(t2, city).unmarked_turns
    r = simulate(t2, city, [perfect(runners(1)[0])])
    assert r.hounds[0].extra_m == pytest.approx(0.0, abs=1e-6)


def test_hounds_go_straight_at_an_unmarked_turn(city):
    """No mark at the turn: hounds run straight on, lose trail and come back. Nobody intuits it."""
    t, corner = l_trail(city)
    r = simulate(blobbed(city, t), city, [perfect(runners(1)[0])])
    assert r.hounds[0].extra_m > 50.0


def test_long_trail_pack_all_finish_and_fast_hounds_pay_more(city):
    t = long_trail(city)
    model = TrailModel(t, city)
    assert len(model.checks) >= 2
    assert not model.unmarked_turns
    assert all(c.options[0].kind == "true" and len(c.options) >= 2 for c in model.checks)
    results = simulate_many(t, city, default_pack(), runs=10, seed=3, model=model)
    L = t.true_trail_m(city)
    for r in results:
        assert len(r.finishers) == 12
        assert all(h.distance_m >= L - 1e-6 for h in r.hounds)
        assert all(c.solved_s is not None for c in r.checks)
    card = score_results(results, L, hare=HareProfile())
    s = card.summary
    assert s["front_extra_mi"] > s["back_extra_mi"]
    for term in card.terms:
        assert 0.0 <= term.value <= 1.0
    assert 0 < card.total <= 100


def test_checks_take_three_marks_and_whichy_ways_one(city):
    t = long_trail(city, seed=2, arrow_share=0.0)
    model = TrailModel(t, city)
    kinds = {c.kind for c in model.checks}
    assert kinds == {"check", "whichy"}
    for c in model.checks:
        assert (c.confirm or 3) == (1 if c.kind == "whichy" else 3)
    r = simulate(t, city, seed=0, model=model)
    calls = [w for h in r.raw for _, w in h.events]
    assert "On one" in calls and "On two" in calls
    assert any(w.startswith("On-On! (check") for w in calls)
    assert any(w.startswith("On-On! (whichy") for w in calls)


def test_falses_are_few_short_and_pull_hounds(city):
    t = long_trail(city, falses=(1, 1))
    falses = [b for b in t.branches if b.purpose.value == "FALSE"]
    assert falses
    assert sum(m.kind == MarkKind.FALSE for m in t.marks) == len(t.branches)
    model = TrailModel(t, city)
    results = simulate_many(t, city, runs=10, seed=0, model=model)
    went = sum(1 for r in results for h in r.raw for _, w in h.events if w.startswith("On-On! (down a false"))
    assert went > 0
    assert all(len(r.finishers) == 12 for r in results)


def test_a_check_just_before_the_beer_near_still_gets_solved(city):
    t = add_marks(straight_trail(city), city)
    last = t.true_path[-1]
    t.marks = [m for m in t.marks if not (m.kind == MarkKind.BLOB and m.at.edge == last)]
    t.marks.append(Mark(MarkKind.CHECK, EdgePos(node=last.u)))
    for r in simulate_many(t, city, runs=5):
        assert len(r.finishers) == 12


def test_hare_caught_without_a_head_start(city):
    t = long_trail(city)
    results = simulate_many(t, city, runs=3)
    L = t.true_trail_m(city)
    slow = HareProfile(pace_s_per_m=pace_from_min_per_mile(12.0), head_start_s=0.0)
    assert score_results(results, L, hare=slow).term("hare_safety").value == 0.0
    early = HareProfile(head_start_s=3600.0)
    assert score_results(results, L, hare=early).term("hare_safety").value == pytest.approx(1.0)


def test_same_seed_same_run(city):
    t = long_trail(city)
    a = simulate(t, city, seed=7)
    b = simulate(t, city, seed=7)
    assert [h.distance_m for h in a.hounds] == [h.distance_m for h in b.hounds]
    assert [h.finish_s for h in a.hounds] == [h.finish_s for h in b.hounds]


def test_one_run_is_fast(city):
    t = long_trail(city)
    model = TrailModel(t, city)
    t0 = time.perf_counter()
    simulate_many(t, city, runs=5, seed=0, model=model)
    assert (time.perf_counter() - t0) / 5 < 1.0


def test_blob_spacing_and_gaps():
    s = blob_positions(1000.0, 50.0, random.Random(0), gaps=[(300.0, 350.0)])
    gaps = [b - a for a, b in zip(s, s[1:])]
    assert max(gaps) < 75.0 + 50.0
    assert min(gaps) > 0.0
    assert not any(300.0 < x < 350.0 for x in s)
    assert 350.0 in s  # first mark after a check's gap


def test_replay_map(city):
    t = add_marks(straight_trail(city), city)
    results = simulate_many(t, city, runs=2)
    card = score_results(results, t.true_trail_m(city), hare=HareProfile())
    html = replay_map(t, city, results[0], card, hare=HareProfile()).get_root().render()
    assert "rp-play" in html and "Runner 12" in html and "Score" in html


def test_whichy_way_wrong_leg_is_unmarked_and_free_for_the_hare(city):
    t = long_trail(city, seed=2, arrow_share=0.0, falses=(0, 0))
    whichies = [m for m in t.marks if m.kind == MarkKind.WHICHY_WAY]
    assert whichies
    assert not t.branches  # nothing laid down any wrong way
    assert not any(m.kind == MarkKind.FALSE for m in t.marks)
    model = TrailModel(t, city)
    assert model.side_detours == []  # the hare runs no extra distance
    w = [c for c in model.checks if c.kind == "whichy"]
    assert all(any(o.kind == "probe" for o in c.options) for c in w)
    assert all(len(c.options) == 2 for c in w)  # the two ways drawn, nothing else

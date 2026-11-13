import random
import time

import pytest

from autotrail.generate import GenerateOptions, add_blobs, blob_positions, has_blobs, place_marks
from autotrail.pack import default_pack, perfect, runners
from autotrail.score import score_results
from autotrail.sim import TrailModel, simulate, simulate_many
from autotrail.trail import EdgeRef, MarkKind, Trail
from autotrail.viz import replay_map


def straight_trail(g, cols=8, row=3, n=16):
    """East along one street of the city grid: a simple trail with corners only at crossings."""
    a, b = row * n + 1, row * n + cols
    edges = [EdgeRef(*e) for e in g.shortest_edges(a, b)]
    return Trail(start=a, end=b, true_path=edges)


def marked(g, trail, seed=0, **kw):
    return place_marks(trail, g, random.Random(seed), GenerateOptions(**kw))


def test_perfect_hound_runs_exactly_true_trail(city):
    t = straight_trail(city)
    t = add_blobs(t, city)
    hound = perfect(runners(1)[0])
    r = simulate(t, city, [hound], seed=1)
    L = t.true_trail_m(city)
    h = r.hounds[0]
    assert h.finish_s == pytest.approx(L * hound.pace_s_per_m, rel=1e-6)
    assert h.distance_m == pytest.approx(L)
    assert h.extra_m == pytest.approx(0.0, abs=1e-6)


def test_long_trail_pack_all_finish_and_fast_hounds_pay_more(city):
    route = [EdgeRef(*e) for e in city.shortest_edges(1, 16 * 16)]  # corner to corner, ~3.5 km
    t = marked(city, Trail(start=route[0].u, end=route[-1].v, true_path=route), check_every_m=500)
    model = TrailModel(t, city)
    assert len(model.checks) >= 3
    assert all(c.options[0].kind == "true" and len(c.options) >= 2 for c in model.checks)
    results = simulate_many(t, city, default_pack(), runs=10, seed=3, model=model)
    L = t.true_trail_m(city)
    for r in results:
        assert len(r.finishers) == 12
        assert all(h.distance_m >= L - 1e-6 for h in r.hounds)
        assert all(c.solved_s is not None for c in r.checks)
    card = score_results(results, L)
    s = card.summary
    assert s["front_extra_mi"] > s["back_extra_mi"]
    for term in card.terms:
        assert 0.0 <= term.value <= 1.0
    assert card.term("distance_fit").value == pytest.approx(1.0)
    assert 0 < card.total <= 100


def test_same_seed_same_run(city):
    route = [EdgeRef(*e) for e in city.shortest_edges(1, 16 * 16)]
    t = marked(city, Trail(start=route[0].u, end=route[-1].v, true_path=route), check_every_m=500)
    a = simulate(t, city, seed=7)
    b = simulate(t, city, seed=7)
    assert [h.distance_m for h in a.hounds] == [h.distance_m for h in b.hounds]
    assert [h.finish_s for h in a.hounds] == [h.finish_s for h in b.hounds]


def test_falses_send_hounds_back_to_the_check(city):
    route = [EdgeRef(*e) for e in city.shortest_edges(1, 16 * 16)]
    t = marked(city, Trail(start=route[0].u, end=route[-1].v, true_path=route), check_every_m=500,
               falses_per_check=(2, 3), false_blobs=2)
    assert any(m.kind == MarkKind.FALSE for m in t.marks)
    model = TrailModel(t, city)
    assert any(o.kind == "false" for c in model.checks for o in c.options)
    results = simulate_many(t, city, runs=10, seed=0, model=model)
    falses = sum(1 for r in results for h in r.raw for _, what in h.events if what == "false")
    assert falses > 0
    assert all(len(r.finishers) == 12 for r in results)


def test_one_run_is_fast(city):
    route = [EdgeRef(*e) for e in city.shortest_edges(1, 16 * 16)]
    t = marked(city, Trail(start=route[0].u, end=route[-1].v, true_path=route), check_every_m=500)
    model = TrailModel(t, city)
    t0 = time.perf_counter()
    simulate_many(t, city, runs=5, seed=0, model=model)
    assert (time.perf_counter() - t0) / 5 < 1.0


def test_add_blobs_leaves_a_gap_after_checks(city):
    t = straight_trail(city)
    node = t.true_path[3].u
    from autotrail.trail import EdgePos, Mark
    t.marks = [Mark(MarkKind.CHECK, EdgePos(node=node))]
    assert not has_blobs(t)
    t2 = add_blobs(t, city)
    assert has_blobs(t2)
    m = TrailModel(t2, city)
    c = m.checks[0]
    assert c.gap_end - c.s >= 5.0
    assert not any(c.s < s < c.gap_end for s in m.blobs)


def test_blob_spacing(city):
    t = straight_trail(city)
    edges = [(e.u, e.v, e.key) for e in t.true_path]
    s = blob_positions(city, edges, 50.0, 15.0, random.Random(0))
    gaps = [b - a for a, b in zip(s, s[1:])]
    assert max(gaps) < 75.0
    assert min(gaps) > 0.0


def test_replay_map(city):
    t = add_blobs(straight_trail(city), city)
    results = simulate_many(t, city, runs=2)
    card = score_results(results, t.true_trail_m(city))
    html = replay_map(t, city, results[0], card).get_root().render()
    assert "rp-play" in html and "Runner 12" in html and "Score" in html

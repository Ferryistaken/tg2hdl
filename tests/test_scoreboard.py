from benchmarks.scoreboard import build_cases, run_scoreboard


def test_scoreboard_case_names_are_unique():
    cases = build_cases()
    names = [case.name for case in cases]
    assert len(names) == len(set(names))


def test_scoreboard_smoke_tier0():
    results = run_scoreboard(max_tier=0)
    assert results
    assert all(r.status == "pass" for r in results)

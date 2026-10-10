"""Tests for the loop. The server run and the LLM are fakes, so this is instant."""

import json
from pathlib import Path

from optexity.memory import ActionCache, compile_cache
from optexity.memory.llm_builder import BuildResult
from optexity.memory.loop import is_converged, read_report, run_loop
from optexity.memory.writeback import step_report_path

FIXTURE = Path(__file__).parent / "fixtures" / "roboform_actions.json"


def _cache():
    return ActionCache.model_validate_json(FIXTURE.read_text())


def _report(*outcomes):
    return [{"node_index": i, "outcome": o} for i, o in enumerate(outcomes)]


class _FakeServer:
    """Writes the per-step report the real server would write, one per round."""

    def __init__(self, workflow, reports):
        self.workflow = workflow
        self.reports = list(reports)
        self.runs = 0

    def __call__(self) -> float:
        self.runs += 1
        entries = self.reports.pop(0)
        step_report_path(self.workflow).write_text("".join(json.dumps(e) + "\n" for e in entries))
        return 10.0


def _builder(changes: bool = True):
    calls = []

    def build(cache, current, report):
        calls.append(report)
        if not changes:
            return BuildResult(current, False, ["kept baseline"])
        better = json.loads(json.dumps(current))
        better["automation_description"] = f"improved {len(calls)}"
        return BuildResult(better, True, ["accepted"])

    build.calls = calls
    return build


def test_is_converged():
    assert is_converged(_report("command_ok", "no_command"))
    assert not is_converged(_report("command_ok", "fallback_ok"))
    assert not is_converged([])


def test_stops_as_soon_as_every_step_uses_its_command(tmp_path):
    wf = tmp_path / "cached.json"
    server = _FakeServer(wf, [_report("command_ok", "command_ok")])
    result = run_loop(_cache(), wf, server, _builder(), rounds=3)
    assert result.converged and len(result.rounds) == 1 and server.runs == 1
    assert wf.is_file()  # created from the actions because it did not exist


def test_rebuilds_with_the_report_then_converges_and_keeps_a_backup(tmp_path):
    wf = tmp_path / "cached.json"
    server = _FakeServer(wf, [_report("command_ok", "fallback_ok"), _report("command_ok", "command_ok")])
    build = _builder()
    result = run_loop(_cache(), wf, server, build, rounds=3)

    assert [r.number for r in result.rounds] == [1, 2] and result.converged
    assert build.calls[0] == _report("command_ok", "fallback_ok")  # the report reached the builder
    assert json.loads(wf.read_text())["automation_description"] == "improved 1"
    original = json.loads((tmp_path / "cached.round1.json").read_text())
    assert original["automation_description"] == compile_cache(_cache()).automation["automation_description"]
    assert "table" and "converged" in result.table()


def test_stops_at_the_round_limit_without_a_final_rebuild(tmp_path):
    wf = tmp_path / "cached.json"
    bad = _report("failed")
    server = _FakeServer(wf, [bad, bad])
    build = _builder()
    result = run_loop(_cache(), wf, server, build, rounds=2)
    assert not result.converged and server.runs == 2
    assert len(build.calls) == 1  # rebuilt after round 1 only
    assert result.rounds[-1].note == "round limit reached"


def test_stops_early_when_the_builder_changes_nothing(tmp_path):
    wf = tmp_path / "cached.json"
    server = _FakeServer(wf, [_report("agentic"), _report("agentic")])
    result = run_loop(_cache(), wf, server, _builder(changes=False), rounds=5)
    assert server.runs == 1
    assert "nothing to change" in result.rounds[-1].note


def test_a_missing_report_is_explained_not_looped_on(tmp_path):
    wf = tmp_path / "cached.json"
    result = run_loop(_cache(), wf, lambda: 1.0, _builder(), rounds=3)
    assert len(result.rounds) == 1 and "STEP_REPORT" in result.rounds[0].note
    assert read_report(wf) == []

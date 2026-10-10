"""The iterative loop: run, read how each step went, rebuild, run again.

What this file is for
---------------------
`llm_builder.py` can improve a workflow once. This file repeats that until the
workflow is good, or we run out of rounds:

    round 1: run the workflow  -> read the per-step report
             every step passed on its exact command?  -> done
             otherwise: ask the LLM builder to improve the workflow using
             that report, validate it, save it (old file kept as a backup)
    round 2: run the improved workflow ... and so on, up to `--rounds`.

How it talks to Optexity
------------------------
It does not start a browser itself. It sends the task to a local Optexity
server (the same one you start for any run) and reads the per-step report that
the server writes next to the workflow file. Start the server with:

    OPTEXITY_TEST_AUTOMATION=<workflow file> OPTEXITY_STEP_REPORT=true \\
    ENV_PATH=~/optexity/.env optexity inference --port 9000 --child_process_id 0

Then run (the endpoint name is any workflow name from your dashboard):

    python -m optexity.memory.loop --actions actions.json --workflow cached.json \\
        --endpoint <name> --input stock_ticker=AAPL --rounds 3

The pieces that touch the outside world (running a task, calling the LLM) are
passed in as functions, so the tests run the whole loop with fakes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from optexity.memory.compiler import compile_cache
from optexity.memory.llm_builder import BuildResult, build_with_llm, default_ask
from optexity.memory.schema import ActionCache
from optexity.memory.writeback import step_report_path

logger = logging.getLogger(__name__)

# Outcomes that mean "this step is already as good as it gets".
_GOOD_OUTCOMES = {"command_ok", "no_command"}

RunTask = Callable[[], float]  # runs the task once, returns seconds taken
Builder = Callable[[ActionCache, dict[str, Any], list[dict[str, Any]]], BuildResult]


@dataclass
class RoundInfo:
    """What happened in one round."""

    number: int
    seconds: float
    outcomes: dict[str, int]  # e.g. {"command_ok": 3, "fallback_ok": 1}
    converged: bool
    note: str = ""  # what the loop did after the round
    llm_used: bool | None = None  # None if no rebuild happened


@dataclass
class LoopResult:
    """The whole loop's outcome."""

    rounds: list[RoundInfo] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        """True if the last round had every step on its exact command."""
        return bool(self.rounds) and self.rounds[-1].converged

    def table(self) -> str:
        """A small plain-text table of the rounds, for the terminal and RESULTS.md."""
        lines = ["round | seconds | outcomes | after the round", "--- | --- | --- | ---"]
        for r in self.rounds:
            outcomes = ", ".join(f"{k}={v}" for k, v in sorted(r.outcomes.items()))
            lines.append(f"{r.number} | {r.seconds:.0f} | {outcomes} | {r.note}")
        return "\n".join(lines)


def read_report(workflow_path: str | Path) -> list[dict[str, Any]]:
    """Read the per-step report the server wrote for this workflow ([] if none)."""
    path = step_report_path(workflow_path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_converged(report: list[dict[str, Any]]) -> bool:
    """True if there is a report and every step passed on its exact command.

    Example: [command_ok, command_ok] -> True; [command_ok, fallback_ok] -> False;
    an empty report (nothing ran) -> False.
    """
    return bool(report) and all(e.get("outcome") in _GOOD_OUTCOMES for e in report)


def _write_json_atomically(path: Path, data: dict[str, Any]) -> None:
    """Write JSON so a crash can never leave a half-written workflow."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def run_loop(
    cache: ActionCache,
    workflow_path: str | Path,
    run_task: RunTask,
    build: Builder,
    rounds: int = 3,
) -> LoopResult:
    """Run the improve-and-retry loop.

    Steps per round: delete the old report, run the task, read the report.
    If every step passed on its command we stop. If it is not the last round
    we ask `build` for a better workflow using the report, back up the old file
    as `<name>.round<N>.json`, and write the new one.
    We also stop early if the builder could not change anything (no point
    running the same file again).
    """
    path = Path(workflow_path)
    if not path.is_file():
        _write_json_atomically(path, compile_cache(cache).automation)

    result = LoopResult()
    for number in range(1, rounds + 1):
        step_report_path(path).unlink(missing_ok=True)
        seconds = run_task()
        report = read_report(path)
        info = RoundInfo(
            number=number,
            seconds=seconds,
            outcomes=dict(Counter(e.get("outcome", "?") for e in report)),
            converged=is_converged(report),
        )
        result.rounds.append(info)

        if info.converged:
            info.note = "converged: every step used its exact command"
            break
        if not report:
            info.note = "no step report found; is OPTEXITY_STEP_REPORT=true on the server?"
            break
        if number == rounds:
            info.note = "round limit reached"
            break

        current = json.loads(path.read_text())
        built = build(cache, current, report)
        info.llm_used = built.used_llm
        if built.automation == current:
            info.note = "builder had nothing to change; stopped"
            break
        shutil.copyfile(path, path.with_name(f"{path.stem}.round{number}.json"))
        _write_json_atomically(path, built.automation)
        info.note = "rebuilt (LLM answer accepted)" if built.used_llm else "rebuilt (baseline kept)"
    return result


def make_server_runner(
    endpoint: str, port: int, inputs: dict[str, list[str]], timeout_seconds: int = 600
) -> RunTask:
    """Make a `run_task` function that uses a local Optexity server.

    It posts one task to `/inference`, waits until the server reports no task
    running and an empty queue, and returns the seconds it took.
    Raises RuntimeError if the server rejects the task or never goes idle.
    """
    import httpx

    base = f"http://localhost:{port}"

    def run_task() -> float:
        started = time.time()
        reply = httpx.post(
            f"{base}/inference",
            json={"endpoint_name": endpoint, "input_parameters": inputs},
            timeout=60,
        )
        if reply.status_code >= 300:
            raise RuntimeError(f"Server rejected the task: {reply.text}")
        time.sleep(5)  # give the server time to pick the task up
        while time.time() - started < timeout_seconds:
            health = httpx.get(f"{base}/health", timeout=10).json()
            if not health.get("task_running") and not health.get("queued_tasks"):
                return time.time() - started
            time.sleep(3)
        raise RuntimeError("The task did not finish in time.")

    return run_task


def main() -> None:
    """Command line entry point: `python -m optexity.memory.loop ...` (see the top of this file)."""
    parser = argparse.ArgumentParser(description="Improve a cached workflow by running it in a loop.")
    parser.add_argument("--actions", required=True, help="actions.json written by the recorder")
    parser.add_argument("--workflow", required=True, help="the file the server runs (OPTEXITY_TEST_AUTOMATION)")
    parser.add_argument("--endpoint", required=True, help="any workflow name from your dashboard")
    parser.add_argument("--input", action="append", default=[], help="name=value, repeatable")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model", default=None, help="LLM model name (default: Optexity's LLM_MODEL)")
    args = parser.parse_args()

    inputs = {k: [v] for k, v in (item.split("=", 1) for item in args.input)}
    cache = ActionCache.model_validate_json(Path(args.actions).read_text())
    ask = default_ask(args.model)

    def build(cache: ActionCache, current: dict[str, Any], report: list[dict[str, Any]]) -> BuildResult:
        return build_with_llm(cache, current, ask, feedback=report)

    result = run_loop(
        cache, args.workflow, make_server_runner(args.endpoint, args.port, inputs), build, args.rounds
    )
    print(result.table())


if __name__ == "__main__":
    main()

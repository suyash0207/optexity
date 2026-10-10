"""Let an LLM improve the cached workflow, with a strict safety gate.

What this file is for
---------------------
The rule-based compiler (`compiler.py`) already turns `actions.json` into a
valid workflow. This file adds the bonus: ask an LLM to make that workflow
better, and only accept its answer if it passes our checks.

What "better" means: clearer `prompt_instructions` (the text the fallback uses
if a command breaks), exact steps instead of leftover `agentic_task` steps when
the notes allow it, and no pointless repeated clicks. When a previous run
report says a step needed the LLM fallback, the LLM is also given the locators
the fallback found AND that were checked on the live page.

The gate (an LLM answer is accepted only if ALL of these hold)
-------------------------------------------------------------
  1. It is valid JSON.
  2. It passes Optexity's own `Automation` Pydantic model.
  3. It starts on the same URL as the baseline.
  4. Every `command` in it is one we already trust: a locator that was checked
     on a live page to match exactly one element (from the notes, from the run
     report, or already in the baseline). The LLM may NOT invent a locator.
  5. Every typed text is either a value from the notes or a `{name[0]}`
     placeholder that is declared in `input_parameters`.
  6. Secret placeholders (`secret_N`) and their parameters are untouched.
  7. The number of steps is sensible.
If a check fails, the exact error is sent back to the LLM and it tries again.
After `max_attempts` failures we keep the baseline, so this layer can never make
the workflow worse than the rule-based one.

Nothing in this file talks to a browser. The LLM is passed in as a plain
function, so tests can use a fake one.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from optexity.memory.schema import ActionCache
from optexity.schema.automation import Automation

logger = logging.getLogger(__name__)

# An "ask" function takes (system_instruction, prompt) and returns the LLM's text.
AskLLM = Callable[[str, str], str]

_PLACEHOLDER_RE = re.compile(r"^\{(\w+)\[0\]\}$")

# A short, hand-written reference of the workflow syntax. It is only used as a
# backup when the real Optexity docs folder cannot be found (for example when
# Optexity is installed as a package without the repo's `docs/` folder).
SYNTAX_REFERENCE = """\
A workflow is JSON: {"url", "automation_description", "parameters", "nodes"}.
"parameters" is {"input_parameters": {name: [default]}, "generated_parameters": {}}.
Each node: {"type": "action_node", "interaction_action": {...}, "end_sleep_time": seconds}.
Exactly one action key inside "interaction_action":
  "click_element":  {"command", "prompt_instructions"}
  "input_text":     {"command", "prompt_instructions", "input_text"}
  "select_option":  {"command", "prompt_instructions", "select_values": [..]}
  "check" / "uncheck" / "hover": {"command", "prompt_instructions"}
  "go_to_url":      {"url"}
  "go_back":        {}
  "key_press":      {"key"}  (lowercase, for example "enter")
  "agentic_task":   {"task", "max_steps", "backend": "browser_use"}  (slow: an LLM agent)
"command" is a Playwright locator written WITHOUT the "page." prefix, for example
get_by_role("button", name="Save") or locator("#email").
"prompt_instructions" is plain English used only if the command stops working.
Typed values may be a placeholder like {name[0]} if "name" is in input_parameters.
"""

# The real Optexity docs pages (in the repo's `docs/docs/` folder) that the LLM
# reads, in reading order.
_DOC_PAGES = (
    "building-automations/automation-structure.mdx",
    "building-automations/action-node.mdx",
    "action-types/interaction-action.mdx",
    "action-types/agentic-tasks.mdx",
    "building-automations/parameters.mdx",
    "advanced/locators.mdx",
)
_DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs" / "docs"


@lru_cache(maxsize=1)
def load_optexity_docs() -> str:
    """Read the real Optexity docs pages and return them as one text.

    Reads the pages listed in `_DOC_PAGES` from the repo's `docs/docs/` folder
    and removes the `---` header block at the top of each page. If the folder
    or any page is missing, it returns the short hand-written `SYNTAX_REFERENCE`
    instead, so the builder still works. The result is cached, so the files are
    read only once.
    """
    try:
        pages = []
        for name in _DOC_PAGES:
            text = (_DOCS_ROOT / name).read_text()
            text = re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)
            pages.append(f"### {name}\n{text.strip()}")
        return "\n\n".join(pages)
    except OSError:
        logger.info("Optexity docs folder not found; using the short built-in reference.")
        return SYNTAX_REFERENCE


SYSTEM_INSTRUCTION = (
    "You improve browser automation workflows. You reply with ONE JSON object "
    "(the full improved workflow) and nothing else. Follow the Optexity docs. "
    "Never invent locators: for every `command` you may only reuse a locator that "
    "is listed as verified in the input. Keep secret placeholders such as "
    "{secret_1[0]} exactly as they are."
)


@dataclass
class BuildResult:
    """What `build_with_llm` gives back."""

    automation: dict[str, Any]  # the workflow to use (LLM's if accepted, else baseline)
    used_llm: bool  # True if the LLM's answer passed the gate
    attempts: list[str] = field(default_factory=list)  # one plain-English line per attempt


def _commands(automation: dict[str, Any]) -> list[str]:
    """Every `command` string in a workflow, in order."""
    out = []
    for node in automation.get("nodes", []):
        body = (node.get("interaction_action") or {}) if isinstance(node, dict) else {}
        for action in body.values():
            if isinstance(action, dict) and action.get("command"):
                out.append(action["command"])
    return out


def _typed_texts(automation: dict[str, Any]) -> list[str]:
    """Every `input_text` value in a workflow, in order."""
    out = []
    for node in automation.get("nodes", []):
        body = (node.get("interaction_action") or {}) if isinstance(node, dict) else {}
        action = body.get("input_text")
        if isinstance(action, dict) and action.get("input_text") is not None:
            out.append(str(action["input_text"]))
    return out


def trusted_commands(
    cache: ActionCache, baseline: dict[str, Any], feedback: list[dict[str, Any]] | None
) -> set[str]:
    """Collect every locator the LLM is allowed to use as a `command`.

    These are: locators that matched exactly one element when recorded, locators
    the fallback found and that were checked in a previous run report, and the
    commands already in the baseline. Anything else is an invented locator.
    """
    trusted = set(_commands(baseline))
    for note in cache.notes:
        trusted.update(c.locator for c in note.locators if c.unique)
    for entry in feedback or []:
        trusted.update(v["locator"] for v in entry.get("verified_locators", []))
    return trusted


def check_answer(
    answer: dict[str, Any],
    cache: ActionCache,
    baseline: dict[str, Any],
    feedback: list[dict[str, Any]] | None,
) -> str | None:
    """Run the safety gate on one parsed LLM answer. Returns an error text, or None if OK.

    The text is written so it can be sent straight back to the LLM as feedback.
    """
    try:
        Automation.model_validate(answer)
    except Exception as e:
        return f"The workflow does not pass Optexity's validation: {e}"

    if answer.get("url") != baseline.get("url"):
        return f"`url` must stay {baseline.get('url')!r}."

    allowed = trusted_commands(cache, baseline, feedback)
    for command in _commands(answer):
        if command not in allowed:
            return (
                f"The command {command!r} is not a verified locator. "
                "Only reuse a locator that is listed as verified in the input."
            )

    params = (answer.get("parameters") or {}).get("input_parameters", {}) or {}
    recorded_texts = {str(n.params.get("text")) for n in cache.notes if n.params.get("text")}
    recorded_texts |= set(_typed_texts(baseline))
    for text in _typed_texts(answer):
        match = _PLACEHOLDER_RE.match(text)
        if match:
            if match.group(1) not in params:
                return f"The placeholder {text!r} is not declared in input_parameters."
        elif text not in recorded_texts:
            return f"The typed text {text!r} was not in the recording. Do not invent values."

    base_params = (baseline.get("parameters") or {}).get("input_parameters", {}) or {}
    for name, value in base_params.items():
        if name.startswith("secret_") and params.get(name) != value:
            return f"The secret parameter {name!r} must be kept exactly as it is."

    n_base, n_new = len(baseline.get("nodes", [])), len(answer.get("nodes", []))
    if n_new < 1 or n_new > 2 * n_base + 2:
        return f"The workflow has {n_new} steps; expected between 1 and {2 * n_base + 2}."
    return None


def parse_json_answer(text: str) -> dict[str, Any]:
    """Pull the JSON object out of an LLM reply.

    Handles ```json fences and chatter around the object. Raises ValueError
    if no JSON object can be read.
    """
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("The reply contains no JSON object.")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"The reply is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("The reply JSON must be an object.")
    return data


def _compact_notes(cache: ActionCache) -> list[dict[str, Any]]:
    """A short version of the notes for the prompt (the LLM does not need everything)."""
    out = []
    for note in cache.notes:
        out.append(
            {
                "step": note.step,
                "action": note.action,
                "params": {k: v for k, v in note.params.items() if k != "index"},
                "goal": note.next_goal,
                "verdict_of_previous_step": note.evaluation,
                "success": note.success,
                "url_changed": note.url_before != note.url_after,
                "element": (note.element.model_dump(exclude_none=True) if note.element else None),
                "verified_locators": [c.locator for c in note.locators if c.unique][:4],
                "sensitive": note.sensitive,
            }
        )
    return out


def build_prompt(
    cache: ActionCache,
    baseline: dict[str, Any],
    feedback: list[dict[str, Any]] | None,
    error: str | None,
) -> str:
    """Write the prompt for one attempt.

    Includes the goal, the compact notes, the baseline, the syntax reference,
    the last run report (if any) and the previous error (if the last answer
    was rejected).
    """
    parts = [
        "GOAL the human wrote:\n" + cache.goal,
        "OPTEXITY DOCS (workflow syntax):\n" + load_optexity_docs(),
        "RECORDED ACTIONS (compact):\n" + json.dumps(_compact_notes(cache), indent=1),
        "CURRENT WORKFLOW (valid, built by rules):\n" + json.dumps(baseline, indent=1),
        "TASK: Return an improved workflow. Improve prompt_instructions wording, turn leftover "
        "agentic_task steps into exact steps only when a verified locator in the notes allows it, "
        "and drop clicks that did nothing and were repeated. Keep what already works.",
    ]
    if feedback:
        parts.append(
            "LAST RUN REPORT (one entry per step; node_index counts steps from 0). "
            "Steps with outcome fallback_ok had a broken command; their verified_locators "
            "are safe replacements. Steps with outcome failed or agentic need attention:\n"
            + json.dumps(feedback, indent=1)
        )
    if error:
        parts.append("YOUR PREVIOUS ANSWER WAS REJECTED. Fix this and answer again:\n" + error)
    return "\n\n".join(parts)


def build_with_llm(
    cache: ActionCache,
    baseline: dict[str, Any],
    ask: AskLLM,
    feedback: list[dict[str, Any]] | None = None,
    max_attempts: int = 3,
) -> BuildResult:
    """Ask the LLM for an improved workflow; accept it only if it passes the gate.

    Example: `build_with_llm(cache, compile_cache(cache).automation, ask)`.
    Each rejected answer is explained back to the LLM. After `max_attempts`
    rejections (or if the LLM call itself keeps failing) the baseline is
    returned with `used_llm=False`, so the result is never worse than the rules.
    """
    attempts: list[str] = []
    error: str | None = None

    for number in range(1, max_attempts + 1):
        prompt = build_prompt(cache, baseline, feedback, error)
        try:
            answer = parse_json_answer(ask(SYSTEM_INSTRUCTION, prompt))
            error = check_answer(answer, cache, baseline, feedback)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            answer = None

        if error is None and answer is not None:
            attempts.append(f"attempt {number}: accepted")
            return BuildResult(answer, True, attempts)
        attempts.append(f"attempt {number}: rejected ({error})")
        logger.info(attempts[-1])

    attempts.append("kept the rule-based baseline")
    return BuildResult(baseline, False, attempts)


def default_ask(model_name: str | None = None) -> AskLLM:
    """The real LLM, wrapped as an `AskLLM` function.

    Uses Optexity's existing LiteLLM wrapper, so the model and API key come
    from the same settings as the rest of Optexity (`LLM_MODEL`, `LLM_MODEL_API_KEY`).
    """
    from optexity.inference.models import get_llm_model, normalize_model

    model = get_llm_model(normalize_model(None, model_name), use_structured_output=False)

    def ask(system: str, prompt: str) -> str:
        text, _usage = model.get_model_response(prompt, system)
        return text

    return ask

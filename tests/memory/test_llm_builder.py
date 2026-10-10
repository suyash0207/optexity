"""Tests for the LLM builder. The "LLM" is a fake function, so no key is needed."""

import copy
import json
from pathlib import Path

from optexity.memory import ActionCache, compile_cache
from optexity.memory.llm_builder import (
    build_with_llm,
    check_answer,
    parse_json_answer,
)

FIXTURE = Path(__file__).parent / "fixtures" / "roboform_actions.json"


def _setup():
    cache = ActionCache.model_validate_json(FIXTURE.read_text())
    baseline = compile_cache(cache).automation
    return cache, baseline


def _scripted(*replies):
    """A fake LLM that returns the given replies one by one (and records prompts)."""
    seen = []
    queue = list(replies)

    def ask(system, prompt):
        seen.append(prompt)
        return queue.pop(0)

    ask.prompts = seen
    return ask


def _improved(baseline):
    better = copy.deepcopy(baseline)
    for node in better["nodes"]:
        action = next(iter(node["interaction_action"].values()))
        action["prompt_instructions"] = "A clearer sentence."
    return better


def test_a_valid_improvement_is_accepted_on_the_first_try():
    cache, baseline = _setup()
    better = _improved(baseline)
    result = build_with_llm(cache, baseline, _scripted("```json\n" + json.dumps(better) + "\n```"))
    assert result.used_llm and result.automation == better
    assert result.attempts == ["attempt 1: accepted"]


def test_bad_json_then_invalid_schema_then_valid_uses_retries_and_sends_the_error_back():
    cache, baseline = _setup()
    broken_schema = copy.deepcopy(baseline)
    broken_schema["nodes"][0]["interaction_action"] = {"not_an_action": {}}
    ask = _scripted("sorry, no json here", json.dumps(broken_schema), json.dumps(_improved(baseline)))

    result = build_with_llm(cache, baseline, ask)

    assert result.used_llm
    assert len(result.attempts) == 3
    assert "REJECTED" in ask.prompts[1] and "no JSON object" in ask.prompts[1]
    assert "validation" in ask.prompts[2]


def test_an_invented_locator_is_rejected():
    cache, baseline = _setup()
    sneaky = copy.deepcopy(baseline)
    action = next(iter(sneaky["nodes"][0]["interaction_action"].values()))
    action["command"] = 'locator("#totally-made-up")'
    assert "not a verified locator" in check_answer(sneaky, cache, baseline, None)


def test_a_locator_the_run_report_verified_is_allowed():
    cache, baseline = _setup()
    changed = copy.deepcopy(baseline)
    action = next(iter(changed["nodes"][0]["interaction_action"].values()))
    action["command"] = 'locator("#found-by-fallback")'
    report = [{"node_index": 0, "verified_locators": [{"locator": 'locator("#found-by-fallback")', "kind": "id"}]}]
    assert check_answer(changed, cache, baseline, report) is None


def test_an_invented_typed_value_is_rejected_but_a_declared_placeholder_is_fine():
    cache, baseline = _setup()
    invented = copy.deepcopy(baseline)
    invented["nodes"][0]["interaction_action"]["input_text"]["input_text"] = "hacked"
    assert "not in the recording" in check_answer(invented, cache, baseline, None)

    parametrised = copy.deepcopy(baseline)
    parametrised["nodes"][0]["interaction_action"]["input_text"]["input_text"] = "{full_name[0]}"
    assert "not declared" in check_answer(parametrised, cache, baseline, None)
    parametrised["parameters"]["input_parameters"]["full_name"] = ["myname"]
    assert check_answer(parametrised, cache, baseline, None) is None


def test_changing_the_start_url_or_secrets_is_rejected():
    cache, baseline = _setup()
    moved = copy.deepcopy(baseline)
    moved["url"] = "https://evil.example"
    assert "url" in check_answer(moved, cache, baseline, None)

    with_secret = copy.deepcopy(baseline)
    with_secret["parameters"]["input_parameters"]["secret_1"] = ["REPLACE_ME"]
    stripped = copy.deepcopy(with_secret)
    stripped["parameters"]["input_parameters"]["secret_1"] = ["leaked-value"]
    assert "secret" in check_answer(stripped, cache, with_secret, None)


def test_after_all_attempts_fail_the_baseline_is_kept():
    cache, baseline = _setup()
    result = build_with_llm(cache, baseline, _scripted("no", "still no", "nope"), max_attempts=3)
    assert result.used_llm is False
    assert result.automation == baseline
    assert result.attempts[-1] == "kept the rule-based baseline"


def test_an_llm_that_crashes_is_handled():
    cache, baseline = _setup()

    def ask(system, prompt):
        raise RuntimeError("no network")

    result = build_with_llm(cache, baseline, ask, max_attempts=2)
    assert result.used_llm is False and result.automation == baseline


def test_parse_json_answer_handles_fences_and_chatter():
    assert parse_json_answer('Sure!\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}


def test_the_prompt_contains_the_real_optexity_docs():
    from optexity.memory.llm_builder import SYNTAX_REFERENCE, build_prompt, load_optexity_docs

    cache, baseline = _setup()
    docs = load_optexity_docs()
    assert docs != SYNTAX_REFERENCE  # the repo's docs folder was found
    assert "interaction-action.mdx" in docs and "click_element" in docs
    assert not docs.lstrip().startswith("---")  # page headers were stripped
    assert "OPTEXITY DOCS" in build_prompt(cache, baseline, None, None)

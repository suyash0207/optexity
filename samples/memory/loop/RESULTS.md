# Bonus: LLM builder and iterative loop, real runs

Command used (server started with `OPTEXITY_STEP_REPORT=true` and `OPTEXITY_TEST_AUTOMATION=<workflow>`):

```bash
python -m optexity.memory.loop --actions <actions.json> --workflow <workflow.json> \
  --endpoint <name> --input stock_ticker=AAPL --rounds 3
```

The LLM is Optexity's default (`gemini/gemini-3.5-flash-lite`). In both runs the LLM's answer was accepted by the safety gate (the number of attempts was not printed).

## roboform, one command deliberately broken (`roboform/`)

| Round | Seconds | Steps | What the loop did next |
|---|---|---|---|
| 1 | 43 | 3 command_ok, 1 fallback_ok | The report said step 2 needed the LLM fallback. The builder was given the locators the fallback found and checked on the live page. It replaced the broken `#field-that-no-longer-exists` with a verified one and rewrote all four `prompt_instructions` into clear sentences (for example "Type 'xyz' into the address line 1 input field."). |
| 2 | 27 | 4 command_ok | Converged. |

`round1_before.json` is the workflow before the rebuild, `final.json` after it.

## stockanalysis, baseline with one leftover agentic step (`stockanalysis/`)

| Round | Seconds | Steps | What the loop did next |
|---|---|---|---|
| 1 | 111 | 3 command_ok, 1 agentic | The leftover `agentic_task` (a full browser-use agent, the slow part) was a repeat of a click that had already been done. The builder dropped it. |
| 2 | 30 | 3 command_ok | Converged. |

The same file took about 75 s in the earlier replay (one command timed out and the LLM fallback rescued it). In the loop, round 2 ran with no LLM and no fallback.

## Notes

* The gate never let the LLM invent a locator or typed value. Unit tests cover that (`tests/memory/test_llm_builder.py`).
* If the LLM keeps failing the gate, the loop keeps the rule-based workflow, so it cannot make things worse.
* One run each. LLM output varies from run to run, so treat the exact timings as examples.

"""Experiment 3: tests of the agent's safety checks. No API calls are made.

A scripted stand-in plays the part of the LLM, so every failure can be produced on purpose:
stopping early, wrong applicant ID, a made-up tool, a wrongly copied value, broken JSON, a timeout.
Each test says what the harness must do about it.

Run:   python Exp_3/test_agent.py        (about 10 seconds)
"""
import json
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

import exp3_agent as agent
import exp3_eval as grading
import exp3_tools as tools

APPLICANT = 100047
ALL_TOOLS = tools.TOOL_NAMES
tools.load_data()
THRESHOLDS = grading.load_thresholds()
results = []


def check(name, condition):
    """Record one test result."""
    results.append((name, bool(condition)))
    print("{0}  {1}".format("PASS" if condition else "FAIL", name))


class ScriptedModel:
    """Plays back a fixed list of replies. An entry that is a function is called with the messages."""
    model_name = "scripted"
    temperature = 0.0

    def __init__(self, script):
        self.script = list(script)
        self.position = 0

    def bind_tools(self, lc_tools):
        return self

    def invoke(self, messages):
        step = self.script[self.position]
        self.position += 1
        if callable(step):
            return step(messages)
        return step


def tool_calls(names, applicant_id=APPLICANT):
    """A model reply that asks for these tools."""
    calls = []
    for number, name in enumerate(names):
        calls.append({"name": name, "args": {"applicant_id": applicant_id}, "id": "call-{0}-{1}".format(number, name), "type": "tool_call"})
    return AIMessage(content="", tool_calls=calls)


def evidence_in(messages):
    """The tool results that are in the conversation so far, by tool name."""
    tool_of_call = {}
    evidence = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                tool_of_call[call["id"]] = call["name"]
        if isinstance(message, ToolMessage):
            result = json.loads(message.content)
            if "error" not in result:
                evidence[tool_of_call[message.tool_call_id]] = result
    return evidence


def findings(change=None):
    """A model reply with correct findings. `change` can edit the findings before they are sent."""
    def make(messages):
        evidence = evidence_in(messages)
        outputs = {name: evidence.get(name, {}) for name in ALL_TOOLS}
        truth = grading.policy_truth(outputs, THRESHOLDS)
        checks = []
        for policy_line in agent.POLICY:
            value = evidence.get(policy_line["tool"], {}).get(policy_line["field"])
            checks.append({"rule": policy_line["id"], "value": value, "condition": policy_line["rule"],
                           "applies": truth[policy_line["id"]]})
        reply = {"checks": checks, "notes": [], "own_estimate": 0.1}
        if change is not None:
            change(reply)
        return AIMessage(content="```json\n" + json.dumps(reply) + "\n```")      # wrapped in a code fence on purpose
    return make


REPORT = AIMessage(content="Rating: medium. Short case note.")


def statuses(run):
    return [entry["status"] for entry in run["audit_log"] if entry["type"] == "tool_call"]


# 1. Normal case
run = agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS), findings(), REPORT]))
truth = grading.policy_truth(run["tool_outputs"], THRESHOLDS)
lines = [name for name, applies in truth.items() if applies]
expected_score, basis = agent.score_from_lines(run["tool_outputs"]["get_applicant_profile"]["external_score_percentile"], lines)
check("normal case finishes as complete with 3 model calls", run["status"] == "complete" and run["n_model_calls"] == 3)
check("the score is calculated by code from the findings", run["risk_score"] == expected_score)
check("every citation was checked against the tool output", all(c["check"] == "match" for c in run["citations"]))
check("the external score is always cited", "external_score_mean" in [c["field"] for c in run["citations"]])
check("no action is executed and a human decision is required", run["action_executed"] == False and run["human_decision_required"] == True)

check("the whole case is one conversation with one system prompt", run["n_messages"] == 13)

# 1a. The fields the team's evaluation file reads are logged with every run
check("every tool call is logged with the applicant's SK_ID_CURR",
      len(run["tool_calls"]) == 6 and all(call["args"] == {"SK_ID_CURR": APPLICANT} for call in run["tool_calls"]))
check("tool call statuses use the team's words", set(call["status"] for call in run["tool_calls"]) <= {"ok", "no_history"})
check("every data source has a verdict", sorted(run["source_findings"]) == sorted(agent.SOURCE_OF_TOOL.values()))
no_history_sources = [agent.SOURCE_OF_TOOL[name] for name in run["no_record_sources"]]
check("a source with no record is logged as no_history, never clean",
      all(run["source_findings"][source] == "no_history" for source in no_history_sources))
check("a case is escalated exactly when it is rated high", run["escalated"] == (run["rating"] == "high"))
check("token use and time are logged", run["usage"] == {"input_tokens": run["tokens_in"], "output_tokens": run["tokens_out"], "latency_s": run["seconds"]})

# 1b. The agent asks for a tool again in step 2: it is run (from the cache) and the agent is asked again
run = agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS), tool_calls(ALL_TOOLS[:1]), findings(), REPORT]))
check("a tool request in step 2 is served and the case still finishes", run["status"] == "complete" and statuses(run).count("cached") == 1)

# 2. The model wants to stop after two tools: the gate sends it back
run = agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS[:2]), AIMessage(content="DONE"), tool_calls(ALL_TOOLS[2:]), findings(), REPORT]))
check("stopping early is caught by the evidence gate", run["gate_nudges"] == 1 and run["status"] == "complete" and run["not_checked"] == [])

# 3. Wrong applicant ID, a tool that does not exist, an extra argument
bad_calls = AIMessage(content="", tool_calls=[
    {"name": "get_applicant_profile", "args": {"applicant_id": 100048}, "id": "a", "type": "tool_call"},
    {"name": "approve_loan", "args": {"applicant_id": APPLICANT}, "id": "b", "type": "tool_call"},
    {"name": "get_installment_history", "args": {"applicant_id": APPLICANT, "days": 30}, "id": "c", "type": "tool_call"}])
run = agent.investigate(APPLICANT, ScriptedModel([bad_calls, tool_calls(ALL_TOOLS), findings(), REPORT]))
check("another applicant's ID is refused", statuses(run)[0] == "error:not_permitted")
check("a tool that does not exist is refused", statuses(run)[1] == "error:unknown_tool")
check("an extra argument is refused", statuses(run)[2] == "error:bad_argument")
check("the investigation still finishes after bad calls", run["status"] == "complete" and len(run["tool_outputs"]) == 6)
check("failed tool calls are logged as errors", [call["status"] for call in run["tool_calls"][:3]] == ["error", "error", "error"])
check("the wrong ID is kept in the log", run["tool_calls"][0]["args"] == {"SK_ID_CURR": 100048})
check("a case with a failed tool call is escalated to a human",
      run["escalated"] == True and "human review required" in run["recommended_action"])

# 4. A repeated call is answered from the cache
run = agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS[:3]), tool_calls(ALL_TOOLS[:3]), tool_calls(ALL_TOOLS[3:]), findings(), REPORT]))
check("repeated tool calls come from the cache", statuses(run).count("cached") == 3)

# 5. The budget runs out with three sources unchecked
run = agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS[:3])] + [AIMessage(content="DONE")] * 3 + [findings(), REPORT]))
check("an unfinished investigation is marked incomplete", run["status"] == "incomplete" and len(run["not_checked"]) == 3)
check("an incomplete case asks for human review", "human review required" in run["recommended_action"] and run["escalated"] == True)
check("a source that was not checked is logged as unknown", list(run["source_findings"].values()).count("unknown") == 3)


# 6. A wrongly copied value is rejected and repaired
def wrong_value(reply):
    reply["checks"][0]["value"] = 0.99

run = agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS), findings(wrong_value), findings(), REPORT]))
check("a wrongly copied value is rejected, then repaired", len(run["problems_first_draft"]) == 1 and run["findings_attempts"] == 2 and run["status"] == "complete")


# 7. A wrong value that is never repaired is not counted in the score
def wrong_flag(reply):
    for one_check in reply["checks"]:
        if one_check["rule"] == "INST_SEVERE_LATE":
            one_check["value"] = 400
            one_check["applies"] = True

run = agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS), findings(wrong_flag), findings(wrong_flag), REPORT]))
check("an unrepaired wrong value sends the case to review", run["status"] == "needs_review" and run["escalated"] == True)
check("an unverified citation is not counted in the score", "INST_SEVERE_LATE" not in run["score_basis"]["red_flags_counted"])
check("an unverified value is not offered to the case note", "400" not in run["report_input"].split("Score calculation")[0])


# 8. A policy line is skipped
def drop_line(reply):
    reply["checks"] = reply["checks"][1:]

run = agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS), findings(drop_line), findings(), REPORT]))
check("a skipped policy line is rejected", "have no check" in run["problems_first_draft"][0])


# 9. A rule name that is not in the policy
def invent_rule(reply):
    reply["checks"].append({"rule": "LOOKS_SUSPICIOUS", "value": 1, "condition": "", "applies": True})

run = agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS), findings(invent_rule), findings(), REPORT]))
check("an invented rule is rejected", "no policy line" in run["problems_first_draft"][0])


# 10. A rule is said to apply to a value that is missing
def flag_on_null(reply):
    for one_check in reply["checks"]:
        if one_check["value"] is None:
            one_check["applies"] = True
            break

run = agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS), findings(flag_on_null), findings(), REPORT]))
check("a rule cannot apply to a missing value", "is null" in run["problems_first_draft"][0])

# 11. No valid JSON at all: the run fails loudly
try:
    agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS), AIMessage(content="I cannot."), AIMessage(content="{bad json")]))
    check("broken JSON raises an error", False)
except RuntimeError:
    check("broken JSON raises an error", True)

# 12. Bad IDs given straight to a tool
check("a tool returns an error for a bad ID instead of crashing",
      tools.get_applicant_profile("abc")["error"] == "bad_argument" and tools.get_applicant_profile(1)["error"] == "not_found")


# 13. Batch: a failed run is recorded, a finished run is skipped on the next start
class HalfBroken(ScriptedModel):
    def invoke(self, messages):
        if "100048" in str(messages[1].content):
            raise TimeoutError("vendor timeout")
        last_message = str(messages[-1].content)
        if len(messages) == 2:
            return tool_calls(ALL_TOOLS)
        if "Step 2" in last_message:
            return findings()(messages)
        return REPORT

folder = Path(tempfile.mkdtemp())
first = agent.run_batch([100047, 100048], HalfBroken([]), 1, folder / "runs.jsonl", workers=1)
second = agent.run_batch([100047], HalfBroken([]), 1, folder / "runs.jsonl", workers=1)
check("a failed run goes to the failed file", first == {"saved": 1, "failed": 1} and (folder / "runs_failed.jsonl").exists())
check("a finished run is not repeated", second == {"saved": 0, "failed": 0})

# 14. Runs from another version are never mixed into the same file
record = json.loads((folder / "runs.jsonl").read_text().splitlines()[0])
record["code_version"] = "older"
(folder / "old.jsonl").write_text(json.dumps(record) + "\n")
try:
    agent.run_batch([100047], HalfBroken([]), 1, folder / "old.jsonl", workers=1)
    check("runs from another version are not mixed", False)
except RuntimeError:
    check("runs from another version are not mixed", True)

# 15. The scoring rule
table = agent.SCORE_TABLE["default_rates"]
check("the lowest percentile uses the first row of the table", agent.base_rate(0, 0) == table[0][0] and agent.base_rate(5, 4) == table[0][2])
check("the highest percentile uses the last row", agent.base_rate(100, 2) == table[6][1])
check("a missing external score uses the average rate", agent.base_rate(None, 0) == agent.SCORE_TABLE["no_external_score_rate"])
score, basis = agent.score_from_lines(50, ["RAISE_MOSTLY_REFUSED", "RAISE_HIGH_INTEREST_LOANS", "RAISE_CARD_CASH_WITHDRAWALS"])
check("the adjustment is limited to 2 steps", basis["adjustment_steps"] == 2 and score == round(table[4][0] * 1.25 ** 2, 4))
score, basis = agent.score_from_lines(50, ["LOW_EXT_SCORE", "SHORT_EMPLOYMENT"])
check("LOW_EXT_SCORE is not counted twice", basis["red_flags_counted"] == ["SHORT_EMPLOYMENT"])
check("rating bands", agent.rating_for(0.079) == "low" and agent.rating_for(0.08) == "medium" and agent.rating_for(0.16) == "high")

# 16. The policy, the evaluation's field map and the rule code agree with each other
flag_map = grading.flag_map()
rule_names = [rule["id"] for rule in json.loads((tools.find_shared_folder() / "exp3" / "red_flag_rules.json").read_text())]
red_flag_lines = [line for line in agent.POLICY if line["kind"] == "red_flag"]
check("the policy has the team's 21 red flags", sorted(line["id"] for line in red_flag_lines) == sorted(rule_names))
check("every red-flag line cites a field the evaluation accepts", all(line["field"] in flag_map[line["id"]]["fields"] for line in red_flag_lines))
check("the rule code covers every policy line", set(truth) == set(agent.POLICY_BY_ID))

# 17. The judge's helpers (no model call)
changed = grading.plant_wrong_number("The score is 0.1355 and 3 flags.")
check("a wrong number can be planted in a note", changed[0] == "The score is 0.5355 and 3 flags.")
check("a number that is not in the facts is found by code",
      grading.unsupported_numbers("Score 0.7816 and 3 flags, ratio 1.2.", "score 0.3816; 3 red flags; ratio 1.20") == [0.7816])
grade = grading.parse_judgement('{"grounding": 0, "hedging": 2, "completeness": 1, "evidence": {"grounding": "x"}}')
check("a judge reply in the team's format is parsed", grade.grounding == 0 and grade.hedging == 2 and grade.completeness == 1)
grade = grading.parse_judgement('{"grounding": 1, "hedging": 2, "completeness": 0, "evidence": {"grounding": "a "broken" quote"}}')
check("the scores are still read when a quote breaks the JSON", grade.grounding == 1 and grade.completeness == 0)
try:
    grading.parse_judgement('{"grounding": 3, "hedging": 2, "completeness": 1}')
    check("a judge score outside 0-2 is rejected", False)
except ValueError:
    check("a judge score outside 0-2 is rejected", True)

# 18. The team's evaluation file accepts a run as it is saved
sys.path.insert(0, str(tools.find_shared_folder() / "exp3"))
import eval_exp3 as ev

run = agent.investigate(APPLICANT, ScriptedModel([tool_calls(ALL_TOOLS), findings(), REPORT]))
table, summary = ev.evaluate_agent_runs([run], flag_map=flag_map, tool_map=grading.tool_map(), n_boot=10)
check("the team's file measures tool recall and argument accuracy on a saved run",
      summary["process"]["tool_recall"] == 1.0 and summary["process"]["argument_accuracy"] == 1.0)
check("the team's file measures faithfulness and cost on a saved run",
      summary["process"]["faithfulness"] == 1.0 and "latency_s" in summary["outcome"]["cost per run"])
view = grading.judge_view(run)
check("the judge is shown the six tool outputs and the score calculated by code",
      len(view["tool_outputs"]) == 7 and view["tool_outputs"]["given_to_the_writer_by_the_system"]["risk_score"] == run["risk_score"])

failed = [name for name, passed in results if not passed]
print()
print("{0} of {1} tests passed".format(len(results) - len(failed), len(results)))
if len(failed) > 0:
    sys.exit(1)

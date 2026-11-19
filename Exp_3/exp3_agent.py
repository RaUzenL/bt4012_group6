"""Experiment 3: the investigation agent (one LLM agent with tool calls, plus the harness around it).

The agent gets ONE applicant ID. It never sees TARGET or the Exp1 model score.

It is a single agent: one model, one system prompt and one conversation from start to finish.
The conversation has three steps, and Python (the harness) checks the work between them:

  Step 1  Evidence   The agent chooses which of the six read-only tools to call. Python runs them.
                     Gate: the step ends only when all six sources have an answer
                     ("no record" is an answer) or the budget of model calls is used up.
  Step 2  Findings   In the same conversation, the agent gives a verdict on every line of the written
                     policy, as JSON. Python checks that no line is skipped and that every value
                     equals what the tools returned. One repair attempt.
          Score      Code, not the model, does the arithmetic: it counts the red flags, looks the
                     score up in a table built from TRAIN applicants, and sets the rating.
  Step 3  Case note  Still in the same conversation, the agent writes a short note for the underwriter
                     from the verified findings and the score.
          Decision   Code turns the rating into a recommendation. A human decides.

Everything that happened is saved with the result (audit log, tool outputs, model, prompt version),
so a run can be inspected later without calling the model again. Each run also carries the fields
the team's evaluation file reads (escalated, tool_calls, source_findings, usage, report).
"""
import hashlib
import json
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import openai
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

import exp3_tools as tools

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------
BASE_URL = "https://soclaas-api.comp.nus.edu.sg/v1"   # NUS SoC LLM service (OpenAI-compatible)
MODEL = "qwen3.6:35b"          # the model "default" pointed to on 13 Nov 2026; pinned so every run uses the same model
                               # (the output is still not fully repeatable, even at temperature 0)
TEMPERATURE = 0
MAX_EVIDENCE_TURNS = 4         # budget of model calls in step 1 (evidence)
MAX_FINDINGS_ATTEMPTS = 2      # first draft + one repair
MAX_EXTRA_TOOL_TURNS = 2       # if the agent asks for a tool again in step 2 or 3, it gets this many extra turns
CODE_VERSION = "exp3-v3.1"

# The risk score is an estimated default rate. The average applicant is at about 0.08.
MEDIUM_FROM = 0.08             # at or above the average applicant
HIGH_FROM = 0.16               # at least twice the average applicant
STEP_FACTOR = 1.25             # one adjustment step multiplies the score by 1.25 (or divides it)
MAX_STEPS = 2                  # the adjustment is limited to 2 steps up or down

RECOMMENDED_ACTION = {
    "low": "Routine processing. No extra checks suggested.",
    "medium": "Standard manual review by an underwriter.",
    "high": "Refer to a senior underwriter before any decision.",
}

# The team's evaluation file (eval_exp3.py) has its own name for each of the six data sources.
SOURCE_OF_TOOL = {
    "get_applicant_profile": "application",
    "get_credit_bureau_history": "bureau",
    "get_previous_applications": "previous_application",
    "get_installment_history": "installments",
    "get_credit_card_history": "credit_card",
    "get_pos_cash_history": "pos_cash",
}

# The case note gives every data source a verdict, also the sources where nothing is wrong (the team's
# rubric asks for that). Code hands the note one line per source: its name, how much history it holds
# (these fields, read from the tool output) and the verdict worked out from the verified findings.
SOURCE_SUMMARY = {
    "get_applicant_profile": ("Applicant profile", "external score percentile: {0}, loan to yearly income: {1}",
                              ["external_score_percentile", "credit_to_income"]),
    "get_credit_bureau_history": ("Credit bureau", "loans on file: {0}, active: {1}, overdue amount: {2}",
                                  ["bureau_n_loans", "bureau_n_active_loans", "bureau_overdue_amount"]),
    "get_previous_applications": ("Previous Home Credit applications", "total: {0}, approved: {1}, refused: {2}",
                                  ["prev_n_applications", "prev_n_approved", "prev_n_refused"]),
    "get_installment_history": ("Instalment payments", "instalments: {0}, late rate: {1}",
                                ["inst_n_installments", "inst_late_rate"]),
    "get_credit_card_history": ("Credit card", "months of history: {0}, latest utilisation: {1}",
                                ["card_months_of_history", "card_latest_utilisation"]),
    "get_pos_cash_history": ("POS and cash loans", "contracts: {0}, worst days past due: {1}",
                             ["pos_n_contracts", "pos_max_days_past_due"]),
}

# The written policy: 21 red-flag rules, 6 facts that raise the risk and 6 that lower it.
# Each line names one tool field and the condition it must meet (prompts/policy.json).
POLICY = json.loads((HERE / "prompts" / "policy.json").read_text())
POLICY_BY_ID = {}
for policy_line in POLICY:
    POLICY_BY_ID[policy_line["id"]] = policy_line

# The scoring table: default rates of TRAIN applicants. It is built once by exp3_checks.build_score_table().
PERCENTILE_BANDS = [5, 10, 20, 30, 50, 70, 100]     # upper edge of each external-score percentile band (the rows)
FLAG_BAND_LABELS = ["0-1", "2-3", "4+"]             # number of other red flags (the columns)
SCORE_TABLE_FILE = HERE / "data" / "score_table.json"
SCORE_TABLE = None
if SCORE_TABLE_FILE.exists():
    SCORE_TABLE = json.loads(SCORE_TABLE_FILE.read_text())


# ----------------------------------------------------------------------------------------------
# Prompts (the text is kept in the prompts/ folder, one file per prompt)
# ----------------------------------------------------------------------------------------------
PROMPT_FOLDER = HERE / "prompts"


def read_prompt(file_name):
    """Read one prompt file from the prompts/ folder."""
    return (PROMPT_FOLDER / file_name).read_text(encoding="utf-8")


def team_thresholds():
    """The cut-off of every team rule, by rule name (red_flag_rules.json). None where the rule has a fixed cut-off."""
    rules = json.loads((HERE / "data" / "red_flag_rules.json").read_text())
    return {rule["id"]: rule["threshold"] for rule in rules}


def policy_thresholds():
    """The four cut-offs that the team's rules take from TRAIN percentiles, written as the policy shows them."""
    by_name = team_thresholds()
    return {
        "HIGH_ANNUITY_TO_INCOME": "{0:.4f}".format(by_name["HIGH_ANNUITY_TO_INCOME"]),
        "LOW_EXT_SCORE": "{0:.4f}".format(by_name["LOW_EXT_SCORE"]),
        "MANY_ENQUIRIES": "{0:.0f}".format(by_name["MANY_ENQUIRIES"]),
        "PREV_HIGH_FREQUENCY": "{0:.0f}".format(by_name["PREV_HIGH_FREQUENCY"]),
    }


THRESHOLDS = policy_thresholds()


def policy_text():
    """The policy as the model reads it: one line per rule, with the four cut-offs filled in."""
    headings = {
        "red_flag": "Red flags",
        "raises_risk": "Other facts that raise the risk (past applicants with them defaulted more often)",
        "lowers_risk": "Facts that lower the risk (past applicants with them defaulted less often)",
    }
    lines = []
    for kind, heading in headings.items():
        lines.append(heading + ":")
        for policy_line in POLICY:
            if policy_line["kind"] == kind:
                lines.append("- {0}: {1} {2}   [{3}]".format(policy_line["id"], policy_line["field"],
                                                           policy_line["rule"].format(**THRESHOLDS), policy_line["tool"]))
        lines.append("")
    return "\n".join(lines).strip()


# The agent's one system prompt. This file is a template: the policy is filled in by code,
# and the JSON example uses double braces {{ }} so that .format() leaves them alone.
AGENT_PROMPT = read_prompt("agent_prompt.txt").format(policy=policy_text(), n_lines=len(POLICY))

# What the harness says to the agent between the steps (same conversation).
CASE_MESSAGE = "Case: applicant_id = {applicant_id}. Start step 1: gather the evidence."
GATE_MESSAGE = ("Evidence gate: these sources are not checked yet: {missing}. "
                "Call them now for applicant {applicant_id}.")
FINDINGS_MESSAGE = ("Step 1 is finished.\nSources with no record: {no_record}\nSources not checked: {not_checked}\n\n"
                    "Step 2: reply with your findings as one JSON object, with one check for every policy line.")
REPAIR_MESSAGE = "The automatic check rejected the findings:\n- {problems}\nReturn the corrected JSON object only."
REPORT_MESSAGE = ("Your findings are accepted. This is what the case note may use:\n\n{facts}\n\n"
                  "Step 3: write the case note for the underwriter now. Give each of the six sources in "
                  "\"Result per source\" its verdict, also the sources where nothing is wrong. "
                  "Use only the facts in this message, not the earlier tool results. Copy every number exactly "
                  "as it is written here (write 0.7073, not 70.73 percent).")

# Changing the prompt (or a policy line or cut-off in it) changes this version, so runs from different prompts are never mixed.
PROMPT_VERSION = hashlib.sha1((AGENT_PROMPT + CASE_MESSAGE + GATE_MESSAGE + FINDINGS_MESSAGE
                               + REPAIR_MESSAGE + REPORT_MESSAGE).encode()).hexdigest()[:8]


# ----------------------------------------------------------------------------------------------
# The findings the model must return
# ----------------------------------------------------------------------------------------------
class Check(BaseModel):
    """The model's verdict on one policy line."""
    rule: str
    value: Union[bool, int, float, str, None]
    condition: str = ""
    applies: bool


class Note(BaseModel):
    """An extra fact for the underwriter. It does not change the score."""
    tool: str
    field: str
    value: Union[bool, int, float, str, None]
    finding: str


class Findings(BaseModel):
    checks: list[Check]
    notes: list[Note] = Field(default_factory=list, max_length=3)
    own_estimate: Optional[float] = Field(default=None, ge=0, le=1)     # recorded for comparison, never used


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def load_env_file():
    """Read KEY=value lines from a .env file next to this folder (never committed to git)."""
    for folder in [HERE, HERE.parent]:
        env_file = folder / ".env"
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line == "" or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def make_llm(model=MODEL, temperature=TEMPERATURE):
    """The chat model. The key comes from the environment or the .env file."""
    load_env_file()
    api_key = os.environ.get("SOCLAAS_API_KEY", "")
    if api_key == "":
        raise RuntimeError("SOCLAAS_API_KEY is not set. Put it in project/.env (git-ignored).")
    base_url = os.environ.get("SOCLAAS_BASE_URL", BASE_URL)
    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key,
                      temperature=temperature, timeout=120, max_retries=2)


# The LLM sees each tool's name, docstring and argument types. It never sees the function body.
LC_TOOLS = [tool(function) for function in tools.TOOLS]
TOOLS_BY_NAME = {lc_tool.name: lc_tool for lc_tool in LC_TOOLS}


def call_model(state, stage, llm, messages):
    """One model call. Time and token use are written to the audit log."""
    start = time.perf_counter()
    reply = llm.invoke(messages)
    usage = reply.usage_metadata or {}
    state["audit_log"].append({
        "type": "model_call",
        "stage": stage,
        "seconds": round(time.perf_counter() - start, 2),
        "tokens_in": usage.get("input_tokens"),
        "tokens_out": usage.get("output_tokens"),
        "tool_calls_requested": len(getattr(reply, "tool_calls", []) or []),
    })
    state["model_served"] = reply.response_metadata.get("model_name", state["model_served"])
    return reply


# ----------------------------------------------------------------------------------------------
# Step 1: evidence
# ----------------------------------------------------------------------------------------------
def run_tool(state, call):
    """Check one tool request and run it. Always returns a dict (errors are results, not exceptions)."""
    name = call["name"]
    args = call["args"]
    entry = {"type": "tool_call", "tool": name, "args": args, "call_id": call["id"], "status": "started"}
    state["audit_log"].append(entry)           # logged BEFORE running, so the attempt is always on record

    if name not in TOOLS_BY_NAME:              # only registered tools can run
        entry["status"] = "error:unknown_tool"
        return {"error": "unknown_tool", "message": "There is no tool called {0}.".format(name)}
    if set(args) != {"applicant_id"}:
        entry["status"] = "error:bad_argument"
        return {"error": "bad_argument", "message": "Pass exactly one input: applicant_id."}
    if str(args["applicant_id"]) != str(state["SK_ID_CURR"]):     # the ID must be the one in the case
        entry["status"] = "error:not_permitted"
        return {"error": "not_permitted",
                "message": "This case covers applicant {0} only.".format(state["SK_ID_CURR"])}

    cache_key = name + json.dumps(args, sort_keys=True)
    if cache_key in state["cache"]:            # a repeated call is answered from the cache
        entry["status"] = "cached"
        return state["cache"][cache_key]

    try:
        result = TOOLS_BY_NAME[name].invoke(args)
    except Exception as problem:               # a broken tool must not stop the investigation
        entry["status"] = "error:tool_failure"
        return {"error": "tool_failure", "message": "The check did not complete: {0}".format(problem)}

    if "error" in result:
        entry["status"] = "error:" + result["error"]
        return result
    entry["status"] = "ok" if result.get("has_record", True) else "no_record"
    state["cache"][cache_key] = result
    state["evidence"][name] = result           # only completed checks count as evidence
    return result


def missing_sources(state):
    return [name for name in tools.TOOL_NAMES if name not in state["evidence"]]


def run_requested_tools(state, reply):
    """Run every tool the agent asked for and put the results into the conversation."""
    for call in reply.tool_calls:
        result = run_tool(state, call)
        state["messages"].append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))


def ask_agent(state, step, llm):
    """One answer from the agent in step 2 or 3. If it asks for a tool instead of answering,
    the tool is run and it is asked again (a few times at most)."""
    for extra_turn in range(MAX_EXTRA_TOOL_TURNS + 1):
        reply = call_model(state, step, llm, state["messages"])
        state["messages"].append(reply)
        if len(reply.tool_calls) == 0:
            return reply
        run_requested_tools(state, reply)
    raise RuntimeError("The agent kept calling tools in the {0} step instead of answering.".format(step))


def gather_evidence(state, llm):
    """Step 1: the tool-calling loop, with a budget and an evidence gate."""
    state["messages"] = [
        SystemMessage(content=AGENT_PROMPT),
        HumanMessage(content=CASE_MESSAGE.format(applicant_id=state["SK_ID_CURR"])),
    ]
    state["stop_reason"] = "budget_exhausted"
    for turn in range(MAX_EVIDENCE_TURNS):
        reply = call_model(state, "evidence", llm, state["messages"])
        state["messages"].append(reply)
        run_requested_tools(state, reply)

        if len(missing_sources(state)) == 0:   # gate passed: every source has an answer
            state["stop_reason"] = "goal_met"
            break
        if len(reply.tool_calls) == 0:         # the agent wants to stop early: the gate sends it back
            state["gate_nudges"] += 1
            state["messages"].append(HumanMessage(content=GATE_MESSAGE.format(
                missing=missing_sources(state), applicant_id=state["SK_ID_CURR"])))
    # Coverage gaps are recorded by code, so they cannot disappear from a confident report.
    state["not_checked"] = missing_sources(state)
    state["no_record_sources"] = [name for name, output in state["evidence"].items() if output.get("has_record") is False]


# ----------------------------------------------------------------------------------------------
# Step 2: findings
# ----------------------------------------------------------------------------------------------
def same_value(cited, actual):
    """True if a cited value equals the tool's value (numbers within 1%, text ignoring case)."""
    if actual is None:
        return cited is None
    if cited is None:
        return False
    try:
        return math.isclose(float(cited), float(actual), rel_tol=0.01, abs_tol=0.005)
    except (TypeError, ValueError):
        return str(cited).strip().lower() == str(actual).strip().lower()


def check_citation(citation, evidence):
    """'match', 'mismatch' (wrong value) or 'not_found' (no such tool or field in the evidence)."""
    fields = evidence.get(citation["tool"], {})
    if citation["field"] not in fields:
        return "not_found"
    if same_value(citation["value"], fields[citation["field"]]):
        return "match"
    return "mismatch"


def parse_findings(text):
    """Turn the model's reply into validated Findings. Raises ValueError with a readable reason."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The reply contains no JSON object.")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as problem:
        raise ValueError("The JSON could not be read: {0}".format(problem))
    try:
        return Findings.model_validate(data)
    except ValidationError as problem:
        raise ValueError("The JSON does not follow the required format: {0}".format(problem))


def evidence_value(policy_line, evidence):
    """What the tools returned for the field a policy line names (None if it is not on file)."""
    return evidence.get(policy_line["tool"], {}).get(policy_line["field"])


def find_problems(findings, evidence):
    """Everything the automatic check rejects in a draft. It checks the form and the copied values.
    It does NOT check whether a rule really applies: that is what the evaluation measures."""
    problems = []
    seen = []
    for check in findings["checks"]:
        rule = check["rule"]
        if rule not in POLICY_BY_ID:
            problems.append("check '{0}': there is no policy line with this name".format(rule))
            continue
        if rule in seen:
            problems.append("check '{0}' is given twice".format(rule))
            continue
        seen.append(rule)
        policy_line = POLICY_BY_ID[rule]
        actual = evidence_value(policy_line, evidence)
        if not same_value(check["value"], actual):
            problems.append("check '{0}': {1} is {2} in the evidence, you wrote {3}".format(
                rule, policy_line["field"], json.dumps(actual), json.dumps(check["value"])))
        elif actual is None and check["applies"]:
            problems.append("check '{0}': {1} is null, so the line cannot apply".format(rule, policy_line["field"]))
    missing = [policy_line["id"] for policy_line in POLICY if policy_line["id"] not in seen]
    if len(missing) > 0:
        problems.append("these policy lines have no check: {0}".format(missing))
    for number, note in enumerate(findings["notes"], start=1):
        check = check_citation(note, evidence)
        if check == "not_found":
            problems.append("note {0}: {1}.{2} is not in the evidence".format(number, note["tool"], note["field"]))
        if check == "mismatch":
            actual = evidence[note["tool"]][note["field"]]
            problems.append("note {0}: {1}.{2} is {3} in the evidence, you wrote {4}".format(
                number, note["tool"], note["field"], json.dumps(actual), json.dumps(note["value"])))
    return problems


SIGNAL_FOR_KIND = {"red_flag": "risk", "raises_risk": "risk", "lowers_risk": "mitigating", "context": "context"}


def build_citations(findings, evidence):
    """Citations for the underwriter: one for every policy line the model says applies, plus the notes.
    The tool and the field come from the policy, so a citation always points to the right place."""
    citations = []
    for check in findings["checks"]:
        if not check["applies"] or check["rule"] not in POLICY_BY_ID:
            continue
        policy_line = POLICY_BY_ID[check["rule"]]
        citations.append({
            "tool": policy_line["tool"],
            "field": policy_line["field"],
            "value": check["value"],
            "basis": policy_line["id"],
            "kind": policy_line["kind"],
            "finding": "Policy line {0}: {1} {2}.".format(policy_line["id"], policy_line["field"],
                                                         policy_line["rule"].format(**THRESHOLDS)),
        })
    # The underwriter always sees the external score, also when it is not a red flag.
    cited_fields = [citation["field"] for citation in citations]
    profile = evidence.get("get_applicant_profile", {})
    if "external_score_mean" not in cited_fields and "external_score_mean" in profile:
        citations.append({
            "tool": "get_applicant_profile", "field": "external_score_mean", "value": profile["external_score_mean"],
            "basis": "CONTEXT", "kind": "context",
            "finding": "External credit score (0 to 1, higher is safer). It is at percentile {0} of past applicants.".format(
                profile.get("external_score_percentile")),
        })
    for note in findings["notes"]:
        if note["field"] in cited_fields or note["field"] in ["has_record", "applicant_id", "truncated", "note"]:
            continue                                   # a note must add something new
        citations.append({"tool": note["tool"], "field": note["field"], "value": note["value"],
                          "basis": "CONTEXT", "kind": "context", "finding": note["finding"]})
    for citation in citations:
        citation["signal"] = SIGNAL_FOR_KIND[citation["kind"]]
        citation["check"] = check_citation(citation, evidence)
    return citations


def write_findings(state, llm):
    """Step 2: the agent checks the evidence against every policy line. Code turns its checks into citations."""
    state["messages"].append(HumanMessage(content=FINDINGS_MESSAGE.format(
        no_record=state["no_record_sources"] if state["no_record_sources"] else "none",
        not_checked=state["not_checked"] if state["not_checked"] else "none")))
    findings = None
    problems = ["no findings yet"]
    for attempt in range(1, MAX_FINDINGS_ATTEMPTS + 1):
        reply = ask_agent(state, "findings", llm)
        state["findings_attempts"] = attempt
        try:
            findings = parse_findings(str(reply.content)).model_dump()
            problems = find_problems(findings, state["evidence"])
        except ValueError as problem:
            problems = [str(problem)]
        if attempt == 1:
            # Kept so that the evaluation can see what the agent wrote BEFORE any repair.
            state["citations_first_draft"] = build_citations(findings, state["evidence"]) if findings is not None else []
            state["problems_first_draft"] = problems
        if len(problems) == 0:
            break
        if attempt < MAX_FINDINGS_ATTEMPTS:
            state["messages"].append(HumanMessage(content=REPAIR_MESSAGE.format(problems="\n- ".join(problems))))
    if findings is None:
        raise RuntimeError("No valid findings after {0} attempts: {1}".format(MAX_FINDINGS_ATTEMPTS, problems))

    state["checks"] = findings["checks"]
    state["llm_own_estimate"] = findings["own_estimate"]      # the model's own guess, kept only for comparison
    state["citations"] = build_citations(findings, state["evidence"])
    state["open_problems"] = problems          # empty if the findings passed the check


# ----------------------------------------------------------------------------------------------
# Score (code only)
# ----------------------------------------------------------------------------------------------
def rating_for(risk_score):
    if risk_score >= HIGH_FROM:
        return "high"
    if risk_score >= MEDIUM_FROM:
        return "medium"
    return "low"


def table_row(external_score_percentile):
    """Row of the scoring table for an external-score percentile."""
    for row_number, upper_edge in enumerate(PERCENTILE_BANDS):
        if external_score_percentile <= upper_edge:
            return row_number
    return len(PERCENTILE_BANDS) - 1


def table_column(n_other_flags):
    """Column of the scoring table for a number of red flags (LOW_EXT_SCORE not counted)."""
    if n_other_flags <= 1:
        return 0
    if n_other_flags <= 3:
        return 1
    return 2


def base_rate(external_score_percentile, n_other_flags):
    """Default rate of TRAIN applicants with the same external-score band and number of red flags."""
    if SCORE_TABLE is None:
        raise RuntimeError("data/score_table.json is missing. Build it with: python Exp_3/exp3_checks.py score-table")
    if external_score_percentile is None:
        return SCORE_TABLE["no_external_score_rate"]
    return SCORE_TABLE["default_rates"][table_row(external_score_percentile)][table_column(n_other_flags)]


def score_from_lines(external_score_percentile, lines_that_apply):
    """The scoring rule. Input: the external-score percentile and the names of the policy lines that apply.
    Returns (risk score, how it was calculated)."""
    red_flags = []
    raising = []
    lowering = []
    for line_name in lines_that_apply:
        kind = POLICY_BY_ID[line_name]["kind"]
        if kind == "red_flag" and line_name != "LOW_EXT_SCORE":     # the external score is already the table row
            red_flags.append(line_name)
        if kind == "raises_risk":
            raising.append(line_name)
        if kind == "lowers_risk":
            lowering.append(line_name)

    steps = len(raising) - len(lowering)
    steps = max(-MAX_STEPS, min(MAX_STEPS, steps))
    base = base_rate(external_score_percentile, len(red_flags))
    risk_score = round(min(1.0, base * STEP_FACTOR ** steps), 4)

    basis = {
        "external_score_percentile": external_score_percentile,
        "red_flags_counted": red_flags,
        "base_rate": base,
        "facts_raising_risk": raising,
        "facts_lowering_risk": lowering,
        "adjustment_steps": steps,
        "average_applicant": SCORE_TABLE["average_default_rate"],
    }
    return risk_score, basis


def score_case(state):
    """Turn the model's findings into a risk score. All counting and arithmetic is done in code."""
    lines_that_apply = []
    for citation in state["citations"]:
        if citation["check"] == "match" and citation["basis"] in POLICY_BY_ID:      # an unverified value is never counted
            lines_that_apply.append(citation["basis"])
    profile = state["evidence"].get("get_applicant_profile", {})
    risk_score, basis = score_from_lines(profile.get("external_score_percentile"), lines_that_apply)

    state["red_flags"] = [name for name in lines_that_apply if POLICY_BY_ID[name]["kind"] == "red_flag"]
    state["score_basis"] = basis
    state["risk_score"] = risk_score
    state["rating"] = rating_for(risk_score)


# ----------------------------------------------------------------------------------------------
# Step 3: case note
# ----------------------------------------------------------------------------------------------
def size_of_source(tool_name, output):
    """A few plain facts that say how much history a source holds, from the tool output."""
    label, template, fields = SOURCE_SUMMARY[tool_name]
    values = []
    for field in fields:
        value = output.get(field)
        values.append("not on file" if value is None else value)
    return template.format(*values)


def source_results(state):
    """One line per data source for the case note: what the source holds and the verdict on it.
    The verdict is worked out by code from the verified findings, so a source where nothing
    applies is named as such, and a missing history is never called clean."""
    counts = {tool_name: {"red_flag": 0, "raises_risk": 0, "lowers_risk": 0} for tool_name in SOURCE_SUMMARY}
    for citation in state["citations"]:
        if citation["check"] == "match" and citation["kind"] in counts.get(citation["tool"], {}):
            counts[citation["tool"]][citation["kind"]] += 1

    lines = []
    for tool_name, (label, template, fields) in SOURCE_SUMMARY.items():
        output = state["evidence"].get(tool_name)
        if output is None:
            lines.append("- {0}: not checked.".format(label))
            continue
        if output.get("has_record") is False:
            lines.append("- {0}: no record. This is missing history, not a clean history.".format(label))
            continue
        found = counts[tool_name]
        parts = []
        for kind, one, many in [("red_flag", "red flag", "red flags"),
                                ("raises_risk", "fact that raises the risk", "facts that raise the risk"),
                                ("lowers_risk", "fact that lowers the risk", "facts that lower the risk")]:
            if found[kind] > 0:
                parts.append("{0} {1}".format(found[kind], one if found[kind] == 1 else many))
        if found["red_flag"] + found["raises_risk"] > 0:
            verdict = "Concern: {0} (see the findings).".format(", ".join(parts))
        elif found["lowers_risk"] > 0:
            verdict = "No red flag. {0} (see the findings).".format(", ".join(parts))
        else:
            verdict = "No red flag, and no other policy line applies."
        lines.append("- {0}: {1}. {2}".format(label, size_of_source(tool_name, output), verdict))
    return "\n".join(lines)


def describe_findings(state):
    """The facts the case note may use: verified findings, the result per source, the score calculation and the gaps."""
    findings = []
    for citation in state["citations"]:
        if citation["check"] == "match":            # an unverified value is never offered to the case note
            findings.append({"kind": citation["kind"], "field": citation["field"],
                             "value": citation["value"], "finding": citation["finding"]})
    basis = state["score_basis"]
    calculation = ("Past applicants with a similar external score and {0} other red flags defaulted at a rate of {1}. "
                   "Adjustment: {2} steps ({3} facts raise the risk, {4} lower it; one step changes the rate by a quarter). "
                   "Estimated risk score: {5} (the average applicant is at {6}). Rating: {7}.".format(
                       len(basis["red_flags_counted"]), basis["base_rate"], basis["adjustment_steps"],
                       len(basis["facts_raising_risk"]), len(basis["facts_lowering_risk"]),
                       state["risk_score"], basis["average_applicant"], state["rating"]))
    return ("Applicant ID: {0}\n\nFindings:\n{1}\n\nResult per source:\n{2}\n\nScore calculation: {3}\n\n"
            "Sources with no record: {4}\nSources not checked: {5}".format(
                state["SK_ID_CURR"], json.dumps(findings, indent=1), source_results(state), calculation,
                state["no_record_sources"] if state["no_record_sources"] else "none",
                state["not_checked"] if state["not_checked"] else "none"))


def write_report(state, llm):
    """Step 3: the agent writes the case note, still in the same conversation."""
    state["report_input"] = describe_findings(state)       # saved so the judge can grade the note later
    state["messages"].append(HumanMessage(content=REPORT_MESSAGE.format(facts=state["report_input"])))
    reply = ask_agent(state, "report", llm)
    report = str(reply.content).strip()
    if report == "":
        raise RuntimeError("The agent returned an empty case note.")
    state["report"] = report


# ----------------------------------------------------------------------------------------------
# Decision (code only)
# ----------------------------------------------------------------------------------------------
def failed_tool_calls(state):
    """The tool calls of a case that did not complete (wrong ID, unknown tool, bad argument, broken tool)."""
    failed = []
    for entry in state["audit_log"]:
        if entry["type"] == "tool_call" and entry["status"].startswith("error"):
            failed.append(entry)
    return failed


def must_escalate(state):
    """The escalation gate, as the team's evaluation defines it: a case must go to a human for a closer
    look when it is rated high, when the evidence is incomplete or unverified, or when a tool call failed."""
    incomplete = len(state["not_checked"]) > 0 or len(state["open_problems"]) > 0
    return state["rating"] == "high" or incomplete or len(failed_tool_calls(state)) > 0


def decide(state):
    """The model supplies the findings. Code decides what follows from them."""
    incomplete = len(state["not_checked"]) > 0 or len(state["open_problems"]) > 0
    if state["stop_reason"] != "goal_met":
        state["status"] = "incomplete"         # never reported as complete
    elif len(state["open_problems"]) > 0:
        state["status"] = "needs_review"       # a citation could not be verified
    else:
        state["status"] = "complete"
    action = RECOMMENDED_ACTION[state["rating"]]
    if incomplete:
        action = action + " Evidence is incomplete or unverified: human review required."
    elif len(failed_tool_calls(state)) > 0:
        action = action + " A tool call failed during the investigation: human review required."
    state["recommended_action"] = action
    state["escalated"] = must_escalate(state)  # set by code, never by the model
    state["human_decision_required"] = True    # the agent recommends; it has no tool that acts
    state["action_executed"] = False


# ----------------------------------------------------------------------------------------------
# The fields the team's evaluation file reads
# ----------------------------------------------------------------------------------------------
def team_tool_calls(run):
    """The tool calls of a run, in the team's format: {"tool", "args": {"SK_ID_CURR"}, "status"}.
    status is ok, no_history (the tool ran and found no records) or error (the check did not complete)."""
    calls = []
    for entry in run["audit_log"]:
        if entry["type"] != "tool_call":
            continue
        status = entry["status"]
        if status == "cached":                 # a repeated call got the same answer as the first one
            has_record = run["tool_outputs"].get(entry["tool"], {}).get("has_record", True)
            status = "ok" if has_record else "no_record"
        if status == "no_record":
            status = "no_history"
        elif status != "ok":
            status = "error"

        # The tools call the ID applicant_id. The team's file looks for it under the name SK_ID_CURR.
        args = dict(entry["args"])
        if "applicant_id" in args:
            given_id = args.pop("applicant_id")
            try:
                args["SK_ID_CURR"] = int(given_id)
            except (TypeError, ValueError):
                args["SK_ID_CURR"] = 0         # not a number: no applicant has ID 0, so it counts as a wrong ID
        calls.append({"tool": entry["tool"], "args": args, "status": status})
    return calls


def team_source_findings(run):
    """One verdict per data source, in the team's words: concern, clean, no_history or unknown.
    It is worked out by code from the agent's verified findings, so a missing history can never be called clean."""
    sources_with_a_risk = []
    for citation in run["citations"]:
        if citation.get("signal") == "risk" and citation.get("check") == "match":
            sources_with_a_risk.append(citation["tool"])

    verdicts = {}
    for tool_name, source in SOURCE_OF_TOOL.items():
        output = run["tool_outputs"].get(tool_name)
        if output is None:
            verdicts[source] = "unknown"       # the source was not checked
        elif output.get("has_record") is False:
            verdicts[source] = "no_history"    # missing history, not a clean one
        elif tool_name in sources_with_a_risk:
            verdicts[source] = "concern"       # a red flag or a fact that raises the risk was found here
        else:
            verdicts[source] = "clean"
    return verdicts


def add_team_fields(run):
    """Add the fields the team's evaluation file reads: escalated, tool_calls, source_findings and usage.
    They are worked out from what the run already recorded, so no model is called. Nothing else is changed."""
    if "escalated" not in run:
        run["escalated"] = must_escalate(run)
    run["tool_calls"] = team_tool_calls(run)
    run["source_findings"] = team_source_findings(run)
    run["usage"] = {"input_tokens": run["tokens_in"], "output_tokens": run["tokens_out"], "latency_s": run["seconds"]}
    return run


# ----------------------------------------------------------------------------------------------
# One investigation
# ----------------------------------------------------------------------------------------------
def investigate(applicant_id, llm, run=1):
    """Investigate one applicant. Returns the case state as a dict (one line of agent_runs.jsonl)."""
    state = {
        "trace_id": uuid.uuid4().hex[:12],
        "SK_ID_CURR": int(applicant_id),
        "run": run,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_requested": llm.model_name,
        "model_served": None,
        "temperature": llm.temperature,
        "prompt_version": PROMPT_VERSION,
        "code_version": CODE_VERSION,
        "audit_log": [],
        "cache": {},
        "evidence": {},
        "gate_nudges": 0,
    }
    start = time.perf_counter()
    agent_llm = llm.bind_tools(LC_TOOLS)       # one agent: the same model, with the same tools, for every step
    gather_evidence(state, agent_llm)
    write_findings(state, agent_llm)
    score_case(state)
    write_report(state, agent_llm)
    decide(state)

    model_calls = [entry for entry in state["audit_log"] if entry["type"] == "model_call"]
    tool_calls = [entry for entry in state["audit_log"] if entry["type"] == "tool_call"]
    state["n_model_calls"] = len(model_calls)
    state["n_tool_calls"] = len(tool_calls)
    state["tokens_in"] = sum(entry["tokens_in"] or 0 for entry in model_calls)
    state["tokens_out"] = sum(entry["tokens_out"] or 0 for entry in model_calls)
    state["seconds"] = round(time.perf_counter() - start, 2)
    state["tool_outputs"] = state.pop("evidence")     # the name the evaluation file expects
    add_team_fields(state)                             # escalated, tool_calls, source_findings, usage
    state["n_messages"] = len(state["messages"])
    del state["messages"]                              # the audit log and the saved outputs hold what is needed
    del state["cache"]
    return state


# ----------------------------------------------------------------------------------------------
# Many investigations
# ----------------------------------------------------------------------------------------------
def read_runs(path):
    """Finished runs from a .jsonl file (an empty list if the file does not exist yet)."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# Errors that mean the service, not the model, is the problem (server error, rate limit, no connection, timeout).
# langchain re-raises them as its own subclasses (OpenAIAPIError, ...), which are caught by the same check.
# APITimeoutError is a kind of APIConnectionError.
SERVICE_ERROR_TYPES = (openai.InternalServerError, openai.RateLimitError, openai.APIConnectionError)
# Their names, for failure files saved before the "service_error" field was added (langchain's names, then openai's)
SERVICE_ERROR_NAMES = {"OpenAIAPIError", "OpenAIRateLimitError", "OpenAIConnectionError", "OpenAITimeoutError",
                       "InternalServerError", "RateLimitError", "APIConnectionError", "APITimeoutError"}


def server_error(failure):
    """True when a saved failed run failed because of the LLM service (busy, timed out, unreachable),
    not because of the model. failure is one line of a *_failed.jsonl file."""
    if "service_error" in failure:
        return failure["service_error"]
    return failure["error"].split(":")[0].strip() in SERVICE_ERROR_NAMES


def run_batch(applicant_ids, llm, runs_per_applicant, output_file, workers=4, retry_model_failures=True):
    """Investigate every applicant several times. Each finished run is appended to output_file at once,
    so the batch can be stopped and started again: finished (applicant, run) pairs are skipped.
    With retry_model_failures=False, a run that failed because of the model itself (for example it kept
    calling tools instead of answering) is not tried again, so that failure stays in the results.
    A run that failed because the service was busy is always tried again."""
    output_file = Path(output_file)
    failed_file = output_file.with_name(output_file.stem + "_failed.jsonl")
    saved = read_runs(output_file)
    for record in saved:                                # one file = one prompt and one model
        same_version = record["prompt_version"] == PROMPT_VERSION and record["code_version"] == CODE_VERSION
        if not same_version or record["model_requested"] != llm.model_name:
            raise RuntimeError("{0} holds runs made with another prompt, code version or model. "
                               "Use a new file name so that versions are not mixed.".format(output_file.name))
    done = {(record["SK_ID_CURR"], record["run"]) for record in saved}
    n_saved = len(done)
    if not retry_model_failures:
        for failure in read_runs(failed_file):
            if not server_error(failure):
                done.add((failure["SK_ID_CURR"], failure["run"]))
    todo = []
    for run in range(1, runs_per_applicant + 1):        # run 1 for everyone first, then run 2, ...
        for applicant_id in applicant_ids:
            if (int(applicant_id), run) not in done:
                todo.append((int(applicant_id), run))
    message = "{0} runs already saved".format(n_saved)
    if len(done) > n_saved:
        message += ", {0} model failures not tried again".format(len(done) - n_saved)
    print("{0}, {1} to do".format(message, len(todo)))

    lock = threading.Lock()
    counts = {"saved": 0, "failed": 0}
    tools.load_data()                                   # load once, before the threads start

    def one_job(job):
        applicant_id, run = job
        try:
            record = investigate(applicant_id, llm, run)
            target, line, kind = output_file, json.dumps(record), "saved"
        except Exception as problem:                    # a failed run is recorded, never silently dropped
            failure = {"SK_ID_CURR": applicant_id, "run": run, "error": "{0}: {1}".format(type(problem).__name__, problem),
                       "service_error": isinstance(problem, SERVICE_ERROR_TYPES)}
            target, line, kind = failed_file, json.dumps(failure), "failed"
        with lock:
            with open(target, "a") as handle:
                handle.write(line + "\n")
            counts[kind] += 1
            finished = counts["saved"] + counts["failed"]
            if finished % 20 == 0 or finished == len(todo):
                print("  {0}/{1} finished ({2} failed)".format(finished, len(todo), counts["failed"]))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one_job, todo))
    if counts["failed"] > 0:
        print("{0} runs failed (see {1}). Run the batch again to retry them.".format(counts["failed"], failed_file.name))
    return counts

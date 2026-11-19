"""Experiment 3: everything that checks and grades the agent. Nothing in this file is part of the agent.

The team's eval_exp3.py gives the shared metrics: tool recall, argument accuracy, gate adherence,
faithfulness, completeness, uncertainty honesty, consistency, the judge summary, discrimination,
calibration, fairness and cost. This file does not repeat them. It holds what the team's file
needs from this agent, and the checks the team's file does not have. It has five parts:

  1. The policy in code     which of the 33 policy lines really apply to an applicant
  2. Tool check             do the six tools show every red flag in the team's answer key?
  3. Extra evaluation       repairs, policy verdicts, the LLM's own estimate, cut-off table
  4. The judge              a second LLM grades the case note with the TEAM'S rubric; code checks the numbers in it
  5. The scoring table      built once from TRAIN applicants (data/score_table.json)

Parts 3 and 5 read TARGET. That is fine here: this is grading and set-up.
The agent (exp3_agent.py) and the tools (exp3_tools.py) never read it.

Command line:
    python Exp_3/exp3_checks.py tool-check 3000      tools against the answer key on 3,000 random applicants
    python Exp_3/exp3_checks.py score-table          rebuild data/score_table.json
    python Exp_3/exp3_checks.py team-fields          add the team's fields to runs saved before they were asked for
"""
import copy
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics import roc_auc_score

import exp3_agent as agent
import exp3_tools as tools

HERE = Path(__file__).resolve().parent


# ==============================================================================================
# 1. THE POLICY IN CODE
# ==============================================================================================
SEVERE_BUREAU_STATUS = ["31-60 days late", "61-90 days late", "91-120 days late", "over 120 days late or written off"]


def at_least(value, threshold):
    return value is not None and value >= threshold


def at_most(value, threshold):
    return value is not None and value <= threshold


def more_than(value, threshold):
    return value is not None and value > threshold


def less_than(value, threshold):
    return value is not None and value < threshold


def load_thresholds(shared_folder=None):
    """The cut-offs of the team's rules that come from TRAIN percentiles (red_flag_rules.json)."""
    if shared_folder is None:
        shared_folder = tools.find_shared_folder()
    rules = json.loads((agent.HERE / "data" / "red_flag_rules.json").read_text())
    return {rule["id"]: rule["threshold"] for rule in rules}


def flags_from_tools(outputs, thresholds):
    """Apply the team's 21 red-flag rules to the tool outputs of one applicant. Returns {flag: True/False}."""
    profile = outputs["get_applicant_profile"]
    bureau = outputs["get_credit_bureau_history"]
    previous = outputs["get_previous_applications"]
    instalments = outputs["get_installment_history"]
    card = outputs["get_credit_card_history"]
    pos = outputs["get_pos_cash_history"]
    return {
        "HIGH_ANNUITY_TO_INCOME": at_least(profile.get("annuity_to_income"), thresholds["HIGH_ANNUITY_TO_INCOME"]),
        "LOW_EXT_SCORE": at_most(profile.get("external_score_mean"), thresholds["LOW_EXT_SCORE"]),
        "RECENT_ID_CHANGE": at_most(profile.get("years_since_id_change"), 1.0),
        "RECENT_PHONE_CHANGE": at_most(profile.get("days_since_phone_change"), 30),
        "SHORT_EMPLOYMENT": less_than(profile.get("years_employed"), 1.0),
        "MANY_ENQUIRIES": at_least(profile.get("bureau_enquiries_last_year"), thresholds["MANY_ENQUIRIES"]),
        "BUREAU_CURRENT_OVERDUE": more_than(bureau.get("bureau_overdue_amount"), 0),
        "BUREAU_SEVERE_DELINQUENCY": bureau.get("bureau_worst_monthly_status") in SEVERE_BUREAU_STATUS,
        "BUREAU_HIGH_UTILISATION": at_least(bureau.get("bureau_active_debt_to_credit"), 0.8),
        "PREV_HIGH_REFUSAL": at_least(previous.get("prev_refusal_rate"), 0.5),
        "PREV_LAST_REFUSED": previous.get("prev_last_status") == "Refused",
        "PREV_HIGH_FREQUENCY": at_least(previous.get("prev_n_applications_last_year"), thresholds["PREV_HIGH_FREQUENCY"]),
        "INST_FREQUENT_LATE": at_least(instalments.get("inst_late_rate"), 0.10),
        "INST_SEVERE_LATE": at_least(instalments.get("inst_max_days_late"), 30),
        "INST_UNDERPAYMENT": at_least(instalments.get("inst_underpaid_rate"), 0.05),
        "INST_WORSENING": more_than(instalments.get("inst_late_rate_change"), 0.05),
        "CC_HIGH_UTILISATION": at_least(card.get("card_latest_utilisation"), 0.9),
        "CC_BELOW_MIN_PAYMENT": at_least(card.get("card_months_below_min_payment"), 3),
        "CC_CASH_RELIANCE": at_least(card.get("card_cash_share_of_drawings"), 0.5),
        "POS_DELINQUENT": at_least(pos.get("pos_max_days_past_due"), 30),
        "POS_DEBT_DEMAND": more_than(pos.get("pos_months_in_debt_demand"), 0),
    }


def facts_from_tools(outputs):
    """Apply the 12 other policy lines (facts that raise or lower the risk) to the tool outputs."""
    profile = outputs["get_applicant_profile"]
    bureau = outputs["get_credit_bureau_history"]
    previous = outputs["get_previous_applications"]
    instalments = outputs["get_installment_history"]
    card = outputs["get_credit_card_history"]
    clean_instalments = instalments.get("inst_late_rate") == 0 and at_least(instalments.get("inst_n_installments"), 20)
    return {
        "RAISE_MOSTLY_REFUSED": at_least(previous.get("prev_refusal_rate"), 0.75),
        "RAISE_HIGH_INTEREST_LOANS": at_least(previous.get("prev_high_interest_share"), 0.75),
        "RAISE_CARD_CASH_WITHDRAWALS": at_least(card.get("card_atm_withdrawals_per_month"), 1.0),
        "RAISE_LOAN_ABOVE_GOODS_PRICE": at_least(profile.get("credit_to_goods_price"), 1.20),
        "RAISE_YOUNG_BUREAU_LOANS": less_than(bureau.get("bureau_mean_loan_age_years"), 1.0),
        "RAISE_MANY_NEW_BUREAU_LOANS": at_least(bureau.get("bureau_n_loans_opened_last_year"), 4),
        "LOWER_HIGHER_EDUCATION": profile.get("education") in ["Higher education", "Academic degree"],
        "LOWER_LONG_EMPLOYMENT": at_least(profile.get("years_employed"), 10.0),
        "LOWER_OLD_BUREAU_LOANS": at_least(bureau.get("bureau_mean_loan_age_years"), 4.0),
        "LOWER_LOW_CARD_USE": less_than(card.get("card_latest_utilisation"), 0.30),
        "LOWER_CLEAN_INSTALMENTS": clean_instalments,
        "LOWER_LOAN_NEAR_GOODS_PRICE": at_most(profile.get("credit_to_goods_price"), 1.10),
    }


def policy_truth(outputs, thresholds):
    """Which of the 33 policy lines really apply, worked out in code from the tool outputs."""
    truth = flags_from_tools(outputs, thresholds)
    truth.update(facts_from_tools(outputs))
    return truth


def call_all_tools(applicant_id):
    """The six tool outputs for one applicant, by tool name."""
    outputs = {}
    for tool_function in tools.TOOLS:
        outputs[tool_function.__name__] = tool_function(int(applicant_id))
    return outputs


def flag_map():
    """Which tool fields count as evidence for each red flag, in the form eval_exp3.py expects.
    It is read from prompts/policy.json, so the agent and the evaluation use the same list."""
    mapping = {}
    for policy_line in agent.POLICY:
        if policy_line["kind"] == "red_flag":
            mapping[policy_line["id"]] = {"fields": [policy_line["field"]] + policy_line.get("other_fields", [])}
    return mapping


def tool_map():
    """Which tool reads each data source, in the form eval_exp3.py expects: {source: [tool names]}."""
    mapping = {}
    for tool_name, source in agent.SOURCE_OF_TOOL.items():
        mapping[source] = [tool_name]
    return mapping


def update_saved_runs(runs_file):
    """Add the team's fields (escalated, tool_calls, source_findings, usage) to runs that were saved before
    the team asked for them. They are worked out from each run's own audit log by agent.add_team_fields,
    the same function new runs go through. No model is called and no saved field is changed.
    Returns the number of runs that were updated."""
    runs_file = Path(runs_file)
    runs = agent.read_runs(runs_file)
    updated = 0
    for run in runs:
        if "tool_calls" not in run:
            agent.add_team_fields(run)
            updated += 1
    if updated > 0:
        lines = [json.dumps(run) for run in runs]
        runs_file.write_text("\n".join(lines) + "\n")
    return updated


# ==============================================================================================
# 2. TOOL CHECK: do the tools show every red flag in the answer key?
# ==============================================================================================
def tool_check(applicant_ids, shared_folder=None):
    """The answer key (red_flags.parquet) was built straight from the cleaned tables. The agent only
    sees what the tools return. This applies each rule to the tool outputs and compares the result
    with the answer key. Returns one row per red flag."""
    if shared_folder is None:
        shared_folder = tools.find_shared_folder()
    shared_folder = Path(shared_folder)
    tools.load_data(shared_folder)
    thresholds = load_thresholds(shared_folder)
    answer_key = pd.read_parquet(agent.HERE / "data" / "red_flags.parquet").set_index("SK_ID_CURR")

    from_tools = {}
    for applicant_id in applicant_ids:
        from_tools[applicant_id] = flags_from_tools(call_all_tools(applicant_id), thresholds)
    from_tools = pd.DataFrame.from_dict(from_tools, orient="index").astype(int)

    rows = []
    for flag in from_tools.columns:
        key = answer_key.loc[from_tools.index, flag]
        seen = from_tools[flag]
        rows.append({
            "red_flag": flag,
            "in_answer_key": int(key.sum()),
            "seen_by_tool": int(seen.sum()),
            "key_flags_tool_missed": int(((key == 1) & (seen == 0)).sum()),
            "tool_flags_not_in_key": int(((key == 0) & (seen == 1)).sum()),
            "agreement": round(float((key == seen).mean()), 4),
        })
    return pd.DataFrame(rows).set_index("red_flag")


# ==============================================================================================
# 3. EXTRA EVALUATION
# ==============================================================================================
def tool_outputs_of(run):
    """The six tool outputs of a saved run (an empty dict for a source that was not checked)."""
    outputs = {}
    for tool_name in tools.TOOL_NAMES:
        outputs[tool_name] = run["tool_outputs"].get(tool_name, {})
    return outputs


def auc_or_nan(labels, scores):
    """ROC-AUC, or NaN when the group has only defaulters or only non-defaulters."""
    labels = np.asarray(labels)
    if len(set(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, np.asarray(scores, dtype=float))


def load_targets(applicant_ids, shared_folder=None):
    """TARGET for these applicants. For grading only."""
    if shared_folder is None:
        shared_folder = tools.find_shared_folder()
    labels = pd.read_parquet(Path(shared_folder) / "clean_data" / "application.parquet", columns=["SK_ID_CURR", "TARGET"])
    return labels[labels["SK_ID_CURR"].isin(list(applicant_ids))]


def process_table(runs):
    """How the agent worked: one row per run, taken from the audit log (Week 12: check the path).
    Tool recall and argument accuracy are not here: the team's file measures them."""
    rows = []
    for run in runs:
        tool_calls = [entry for entry in run["audit_log"] if entry["type"] == "tool_call"]
        rows.append({
            "tool_calls": len(tool_calls),
            "failed_tool_calls": len(agent.failed_tool_calls(run)),
            "gate_nudges": run["gate_nudges"],
            "first_draft_rejected": len(run["problems_first_draft"]) > 0,
            "status": run["status"],
            "model_calls": run["n_model_calls"],
            "tokens_in": run["tokens_in"],
            "tokens_out": run["tokens_out"],
            "seconds": run["seconds"],
        })
    return pd.DataFrame(rows)


def first_draft_faithfulness(runs, faithfulness_function):
    """Faithfulness of the agent's FIRST draft, before the harness asked for any repair.
    faithfulness_function is eval_exp3.faithfulness, so the number is measured the team's way."""
    scores = []
    for run in runs:
        result = faithfulness_function(run["citations_first_draft"], run["tool_outputs"])
        scores.append(result["faithfulness"])
    return float(np.nanmean(scores))


def policy_verdict_table(runs, shared_folder=None):
    """Compare the agent's verdict on every policy line with the rule worked out in code.

    truly_applies   runs where the line really applies (policy_truth on the tool outputs)
    found_share     of those, the share where the agent said it applies (recall)
    model_said      runs where the agent said the line applies
    correct_share   of those, the share where it really applies (precision)
    """
    thresholds = load_thresholds(shared_folder)
    counts = {}
    for policy_line in agent.POLICY:
        counts[policy_line["id"]] = {"kind": policy_line["kind"], "truly_applies": 0, "model_said": 0, "both": 0}

    for run in runs:
        truth = policy_truth(tool_outputs_of(run), thresholds)
        said = {}
        for check in run["checks"]:
            said[check["rule"]] = check["applies"]
        for line_name in counts:
            if truth[line_name]:
                counts[line_name]["truly_applies"] += 1
            if said.get(line_name, False):
                counts[line_name]["model_said"] += 1
            if truth[line_name] and said.get(line_name, False):
                counts[line_name]["both"] += 1

    rows = []
    for line_name, count in counts.items():
        rows.append({
            "policy_line": line_name,
            "kind": count["kind"],
            "truly_applies": count["truly_applies"],
            "found_share": round(count["both"] / count["truly_applies"], 3) if count["truly_applies"] > 0 else None,
            "model_said": count["model_said"],
            "correct_share": round(count["both"] / count["model_said"], 3) if count["model_said"] > 0 else None,
        })
    table = pd.DataFrame(rows).set_index("policy_line")

    total_true = int(table["truly_applies"].sum())
    total_said = int(table["model_said"].sum())
    total_both = sum(count["both"] for count in counts.values())
    overall = {
        "verdicts": len(runs) * len(agent.POLICY),
        "recall": total_both / total_true if total_true > 0 else np.nan,
        "precision": total_both / total_said if total_said > 0 else np.nan,
    }
    return table, overall


def claimed_flags(run, red_flag_fields):
    """The red flags a run claims. Runs of this agent name the rule. Runs of version 1 only cited fields
    as 'risk', so for them a flag counts as claimed when one of its fields was cited as a risk."""
    names_the_rule = any("kind" in citation for citation in run["citations"])
    if names_the_rule:
        return set(citation["basis"] for citation in run["citations"] if citation["kind"] == "red_flag")
    risk_fields = set(citation["field"] for citation in run["citations"] if citation["signal"] == "risk")
    claimed = set()
    for flag_name, flag in red_flag_fields.items():
        if len(risk_fields & set(flag["fields"])) > 0:
            claimed.add(flag_name)
    return claimed


def flag_precision_recall(runs, red_flag_fields, answer_key):
    """Red flags claimed by the agent against the team's answer key (red_flags.parquet).
    recall = share of the key's flags that were claimed, precision = share of claimed flags that are in the key."""
    in_key = 0
    claimed_total = 0
    both = 0
    for run in runs:
        claimed = claimed_flags(run, red_flag_fields)
        for flag_name in red_flag_fields:
            present = answer_key.at[run["SK_ID_CURR"], flag_name] == 1
            if present:
                in_key += 1
            if flag_name in claimed:
                claimed_total += 1
            if present and flag_name in claimed:
                both += 1
    return {"flags_in_key": in_key, "flags_claimed": claimed_total,
            "recall": both / in_key if in_key > 0 else np.nan,
            "precision": both / claimed_total if claimed_total > 0 else np.nan}


def own_estimates(runs):
    """The model's own risk guess per applicant, averaged over the runs (never used by the agent)."""
    rows = []
    for run in runs:
        rows.append({"SK_ID_CURR": run["SK_ID_CURR"], "llm_own_estimate": run.get("llm_own_estimate")})
    table = pd.DataFrame(rows)
    table["llm_own_estimate"] = table["llm_own_estimate"].astype(float)
    return table.groupby("SK_ID_CURR", as_index=False)["llm_own_estimate"].mean()


def discrimination_table(scored):
    """ROC-AUC of three scores on the same applicants, for everyone and inside each group.
    scored needs the columns TARGET, group, risk_score, llm_own_estimate and exp1_score."""
    rows = []
    for label in ["all", "flagged", "control"]:
        part = scored if label == "all" else scored[scored["group"] == label]
        own = part.dropna(subset=["llm_own_estimate"])
        rows.append({
            "applicants": label,
            "n": len(part),
            "defaulters": int(part["TARGET"].sum()),
            "agent_score": round(auc_or_nan(part["TARGET"], part["risk_score"]), 3),
            "llm_own_estimate": round(auc_or_nan(own["TARGET"], own["llm_own_estimate"]), 3),
            "exp1_model": round(auc_or_nan(part["TARGET"], part["exp1_score"]), 3),
        })
    return pd.DataFrame(rows).set_index("applicants")


def threshold_table(scored, thresholds):
    """Precision and recall when everyone at or above a cut-off is flagged (Week 12: report them separately)."""
    rows = []
    defaulters = int(scored["TARGET"].sum())
    for threshold in thresholds:
        flagged = scored[scored["risk_score"] >= threshold]
        caught = int(flagged["TARGET"].sum())
        rows.append({
            "flag_if_score_at_least": threshold,
            "flagged": len(flagged),
            "defaulters_caught": caught,
            "false_alarms": len(flagged) - caught,
            "defaulters_missed": defaulters - caught,
            "precision": round(caught / len(flagged), 3) if len(flagged) > 0 else None,
            "recall": round(caught / defaulters, 3) if defaulters > 0 else None,
        })
    return pd.DataFrame(rows).set_index("flag_if_score_at_least")


def model_failures(runs_file):
    """Runs that failed because of the model itself, for example it kept calling tools instead of answering.
    Runs that failed because the service was busy were run again, so they are not counted."""
    runs_file = Path(runs_file)
    failed = agent.read_runs(runs_file.with_name(runs_file.stem + "_failed.jsonl"))
    return len(set((f["SK_ID_CURR"], f["run"]) for f in failed if not agent.server_error(f["error"])))


def with_interval(numbers):
    """'0.536 (0.418 to 0.645)' from one entry of the team's summary."""
    return "{0:.3f} ({1:.3f} to {2:.3f})".format(numbers["value"], numbers["ci_low"], numbers["ci_high"])


def model_comparison(models, evaluate, cohort, flag_map, tool_map, answer_key, runs_per_applicant, n_boot=1000):
    """The same agent with different LLMs, measured the same way. models is {label: runs file}.
    evaluate is eval_exp3.evaluate_agent_runs, so the main numbers are the team's.
    Only runs 1 to runs_per_applicant are used, so every model is measured on the same runs."""
    columns = {}
    for label, runs_file in models.items():
        runs = [run for run in agent.read_runs(runs_file) if run["run"] <= runs_per_applicant]
        expected = len(cohort) * runs_per_applicant
        table, summary = evaluate(runs, flag_map=flag_map, tool_map=tool_map, n_boot=n_boot, cohort=cohort)
        _, verdicts = policy_verdict_table(runs)
        flags = flag_precision_recall(runs, flag_map, answer_key)
        words = [len(run["report"].split()) for run in runs]
        process = summary["process"]
        flagged = summary["outcome"]["flagged group (main)"]
        whole = summary["outcome"]["whole cohort (secondary)"]
        column = {
            "runs finished (of {0})".format(expected): len(runs),
            "runs that failed because of the model": model_failures(runs_file),
            "tool recall": round(process["tool_recall"], 3),
            "faithfulness": round(process["faithfulness"], 3),
            "completeness (red flags cited)": round(process["completeness"], 3),
            "policy verdicts: recall": round(verdicts["recall"], 3),
            "policy verdicts: precision": round(verdicts["precision"], 3),
            "red flags: precision against the key": round(flags["precision"], 3),
            "first drafts rejected": sum(len(run["problems_first_draft"]) > 0 for run in runs),
            "notes that name all six sources": sum(len(sources_named(run["report"])) == 6 for run in runs),
            "notes with a number not in the facts": sum(len(unsupported_numbers(run["report"], run["report_input"])) > 0
                                                        for run in runs),
            "words per note (mean)": round(float(np.mean(words))),
            "ROC-AUC, flagged group (main)": with_interval(flagged["roc_auc"]),
            "PR-AUC, flagged group": with_interval(flagged["pr_auc"]),
            "ROC-AUC, whole cohort": with_interval(whole["roc_auc"]),
            "precision of 'high', whole cohort": round(whole["precision of 'high'"]["value"], 3),
            "recall of 'high', whole cohort": round(whole["recall of 'high'"]["value"], 3),
            "seconds per run (median)": round(float(np.median([run["seconds"] for run in runs])), 1),
            "input tokens per run": round(float(np.mean([run["tokens_in"] for run in runs]))),
            "output tokens per run": round(float(np.mean([run["tokens_out"] for run in runs]))),
        }
        if runs_per_applicant > 1:
            column["same rating in every run"] = round(process["same_rating_all_runs_share"], 3)
        columns[label] = column
    return pd.DataFrame(columns)


def score_rule_table(applicant_ids, shared_folder=None):
    """Apply the policy and the scoring rule in code to many applicants. No model is called.
    This shows how well the score would rank applicants if the agent judged every policy line correctly."""
    if shared_folder is None:
        shared_folder = tools.find_shared_folder()
    tools.load_data(shared_folder)
    thresholds = load_thresholds(shared_folder)
    rows = []
    for applicant_id in applicant_ids:
        outputs = call_all_tools(applicant_id)
        truth = policy_truth(outputs, thresholds)
        lines_that_apply = [line_name for line_name, applies in truth.items() if applies]
        percentile = outputs["get_applicant_profile"].get("external_score_percentile")
        risk_score, basis = agent.score_from_lines(percentile, lines_that_apply)
        rows.append({"SK_ID_CURR": int(applicant_id), "rule_score": risk_score, "rating": agent.rating_for(risk_score)})
    return pd.DataFrame(rows)


# ==============================================================================================
# 4. THE JUDGE: a second LLM grades the case note (LLM-as-judge, Week 12 lecture)
# ==============================================================================================
# The harness checks the citations. It cannot read prose. The judge reads the case note and grades it
# with the TEAM'S rubric (eval_exp3.JUDGE_RUBRIC), so every experiment is graded the same way:
#     GROUNDING      every claim traces to a value in the tool outputs (a 0 fails the note)
#     HEDGING        missing or failed data is said to be missing, never clean
#     COMPLETENESS   every data source has a verdict, including the ones with nothing to report
# Each gets 0, 1 or 2. The judge runs afterwards on saved runs. It never takes part in an investigation.
JUDGE_MODEL = "nemotron3.5:30b"     # a different model family from the one that wrote the note (agent.MODEL)
MAX_JUDGE_ATTEMPTS = 2
CRITERIA = ["grounding", "hedging", "completeness"]


class Judgement(BaseModel):
    grounding: int = Field(ge=0, le=2)
    hedging: int = Field(ge=0, le=2)
    completeness: int = Field(ge=0, le=2)
    evidence: dict = Field(default_factory=dict)


def make_judge(model=JUDGE_MODEL):
    """The judge model, on the same school service as the agent."""
    return agent.make_llm(model=model, temperature=0)


def parse_judgement(text):
    """Turn the judge's reply into a validated Judgement. Raises ValueError with a readable reason."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The reply contains no JSON object.")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # A long quote sometimes breaks the JSON. The three scores are at the top and can still be read.
        data = {"evidence": {}}
        for criterion in CRITERIA:
            found = re.search('"{0}"\\s*:\\s*(\\d+)'.format(criterion), text)
            if found is None:
                raise ValueError("The JSON could not be read and the score for {0} was not found.".format(criterion))
            data[criterion] = int(found.group(1))
            data["evidence"][criterion] = "(the judge's quote could not be read)"
    try:
        return Judgement.model_validate(data)
    except ValidationError as problem:
        raise ValueError("The JSON does not follow the required format: {0}".format(problem))


def score_sentences(run):
    """The score calculation in the words the system gave to the writer of the note (taken from report_input)."""
    after = run["report_input"].split("Score calculation: ")[1]
    return after.split("\n\nSources with no record")[0]


def judge_view(run):
    """The run as the judge sees it: the six tool outputs and the case note.
    The note is asked to state the risk score, and the score is calculated by code, not returned by a tool.
    So what the system told the writer about the score is added as one more entry: the calculation in words,
    the policy lines that apply, and the sources with no record. Without it the judge would mark the score
    in every note as a value that does not exist."""
    policy_lines = []
    for citation in run["citations"]:
        if citation["check"] == "match" and citation["basis"] in agent.POLICY_BY_ID:
            policy_lines.append(citation["finding"])
    outputs = dict(run["tool_outputs"])
    outputs["given_to_the_writer_by_the_system"] = {
        "risk_score": run["risk_score"],
        "rating": run["rating"],
        "score_calculation": score_sentences(run),
        "policy_lines_that_apply": policy_lines,
        "sources_with_no_record": run["no_record_sources"],
        "sources_not_checked": run["not_checked"],
    }
    return {"tool_outputs": outputs, "report": run["report"]}


def judge_run(run, judge_llm, prompt_function):
    """Grade the case note of one saved agent run. prompt_function is eval_exp3.judge_prompt (the team's rubric)."""
    messages = [HumanMessage(content=prompt_function(judge_view(run)))]
    judgement = None
    problem_text = ""
    for attempt in range(MAX_JUDGE_ATTEMPTS):
        reply = judge_llm.invoke(messages)
        try:
            judgement = parse_judgement(str(reply.content)).model_dump()
            break
        except ValueError as problem:
            problem_text = str(problem)
            messages.append(reply)
            messages.append(HumanMessage(content="That could not be read: {0}\nReturn the JSON object only.".format(problem_text)))
    if judgement is None:
        raise RuntimeError("The judge gave no valid grade: {0}".format(problem_text))

    result = {"SK_ID_CURR": run["SK_ID_CURR"], "run": run["run"], "judge_model": judge_llm.model_name}
    for criterion in CRITERIA:
        result[criterion] = judgement[criterion]
    for criterion in CRITERIA:
        result[criterion + "_evidence"] = str(judgement["evidence"].get(criterion, ""))
    result["passed"] = result["grounding"] > 0       # grounding is the blocking criterion
    return result


def judge_batch(runs, judge_llm, prompt_function, output_file, workers=4):
    """Grade many runs. Each grade is appended to output_file at once, so the work can be continued later."""
    output_file = Path(output_file)
    done = {(grade["SK_ID_CURR"], grade["run"]) for grade in agent.read_runs(output_file)}
    todo = [run for run in runs if (run["SK_ID_CURR"], run["run"]) not in done]
    print("{0} notes already graded, {1} to do".format(len(done), len(todo)))
    lock = threading.Lock()
    failed = []

    def one_job(run):
        try:
            grade = judge_run(run, judge_llm, prompt_function)
        except Exception as problem:                 # a failed grade is reported, never silently dropped
            with lock:
                failed.append((run["SK_ID_CURR"], run["run"], str(problem)))
            return
        with lock:
            with open(output_file, "a") as handle:
                handle.write(json.dumps(grade) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one_job, todo))
    if len(failed) > 0:
        print("{0} notes could not be graded. Run this again to retry them.".format(len(failed)))
    return agent.read_runs(output_file)


def human_grading_sheet(runs, rubric, notes_file, grades_file, n_notes=20, seed=42):
    """A judge is a tool, not the truth, so a person grades some of the same notes (Week 12).
    This picks n_notes runs at random and writes two files for the person who grades:
        notes_file    the rubric, then the tool outputs and the case note of each run
        grades_file   a CSV with one row per note and empty score columns to fill in (0, 1 or 2)
    The judge's scores are in neither file, so the grading is blind. Existing files are never overwritten."""
    notes_file = Path(notes_file)
    grades_file = Path(grades_file)
    if grades_file.exists():
        return
    ordered = sorted(runs, key=lambda run: (run["SK_ID_CURR"], run["run"]))
    chosen = random.Random(seed).sample(ordered, min(n_notes, len(ordered)))

    parts = [rubric, "", "Grade each note below and type the three scores into {0}.".format(grades_file.name)]
    rows = []
    for number, run in enumerate(chosen, start=1):
        view = judge_view(run)
        parts.append("")
        parts.append("=" * 100)
        parts.append("NOTE {0} of {1}     SK_ID_CURR {2}, run {3}".format(number, len(chosen), run["SK_ID_CURR"], run["run"]))
        parts.append("=" * 100)
        parts.append("TOOL OUTPUTS")
        parts.append(json.dumps(view["tool_outputs"], indent=2))
        parts.append("")
        parts.append("REPORT")
        parts.append(view["report"])
        rows.append({"SK_ID_CURR": run["SK_ID_CURR"], "run": run["run"], "grounding": "", "hedging": "", "completeness": ""})
    notes_file.write_text("\n".join(parts) + "\n")
    pd.DataFrame(rows).to_csv(grades_file, index=False)


def read_human_grades(grades_file):
    """The human grades, or None while the sheet is missing or not completely filled in."""
    grades_file = Path(grades_file)
    if not grades_file.exists():
        return None
    grades = pd.read_csv(grades_file)
    if grades[CRITERIA].isna().any().any():
        return None
    return grades


def numbers_in(text):
    """Every number written in digits in a text, as floats ("1,200" and "1.20" are read as numbers)."""
    numbers = []
    for found in re.findall(r"\d[\d,]*\.?\d*", text):
        cleaned = found.replace(",", "").rstrip(".")
        if cleaned != "":
            numbers.append(float(cleaned))
    return numbers


def unsupported_numbers(note, facts):
    """Numbers in the case note that appear nowhere in the facts its writer was given.
    Numbers do not need an LLM: this check cannot be talked round."""
    allowed = numbers_in(facts)
    missing = []
    for number in numbers_in(note):
        supported = any(abs(number - other) < 1e-9 for other in allowed)
        if not supported and number not in missing:
            missing.append(number)
    return missing


# Words that show a case note talks about a data source. A rough check: it shows that a source is
# named, not that what is said about it is right (that is the judge's and the person's job).
SOURCE_WORDS = {
    "application": ["external score"],
    "bureau": ["bureau"],
    "previous_application": ["previous", "prior application"],
    "installments": ["instalment", "installment"],
    "credit_card": ["card"],
    "pos_cash": ["pos ", "pos-", "point-of-sale", "point of sale", "cash loan"],
}


def sources_named(note):
    """The data sources a case note talks about (the team's rubric asks for a verdict on all six)."""
    text = note.lower()
    return [source for source, words in SOURCE_WORDS.items() if any(word in text for word in words)]


def plant_wrong_number(note):
    """Change the first decimal number in a note, so that it no longer matches the facts.
    Returns (changed note, old number, new number), or None if the note has no decimal number."""
    match = re.search(r"\d+\.\d+", note)
    if match is None:
        return None
    old_text = match.group(0)
    old_value = float(old_text)
    if old_value < 0.5:
        new_value = old_value + 0.4
    else:
        new_value = old_value / 3.0
    new_text = "{0:.{1}f}".format(new_value, len(old_text.split(".")[1]))
    changed = note[:match.start()] + new_text + note[match.end():]
    return changed, old_text, new_text


def check_judge(runs, judge_llm, prompt_function, n_notes=10):
    """A judge is a tool, not the truth, so it is tested: plant a wrong number in n_notes case notes and
    let the judge grade them. A good judge gives grounding 0 to every one. Returns one row per note."""
    rows = []
    for run in runs:
        if len(rows) == n_notes:
            break
        planted = plant_wrong_number(run["report"])
        if planted is None:
            continue
        changed_note, old_text, new_text = planted
        changed_run = copy.deepcopy(run)
        changed_run["report"] = changed_note
        grade = judge_run(changed_run, judge_llm, prompt_function)
        rows.append({"SK_ID_CURR": run["SK_ID_CURR"], "number_in_note": old_text, "changed_to": new_text,
                     "grounding": grade["grounding"], "caught_by_judge": grade["grounding"] == 0,
                     "caught_by_code": len(unsupported_numbers(changed_note, run["report_input"])) > 0,
                     "judge_evidence": grade["grounding_evidence"]})
    return rows


# ==============================================================================================
# 5. THE SCORING TABLE (set-up, run once)
# ==============================================================================================
# The agent does not guess a risk score. It looks the score up in a table of past applicants:
#     row    = the applicant's external-score percentile (7 bands)
#     column = how many red flags the applicant has, not counting LOW_EXT_SCORE (0-1, 2-3, 4 or more)
#     cell   = the share of TRAIN applicants in that row and column who defaulted
# Only the TRAIN split is used, so the validation and test applicants stay unseen.
def build_score_table(shared_folder=None):
    """Build data/score_table.json from the TRAIN split."""
    if shared_folder is None:
        shared_folder = tools.find_shared_folder()
    shared_folder = Path(shared_folder)
    tools.load_data(shared_folder)

    splits = pd.read_csv(shared_folder / "clean_data" / "splits.csv", usecols=["SK_ID_CURR", "split"])
    train_ids = splits.loc[splits["split"] == "train", "SK_ID_CURR"]
    application = pd.read_parquet(shared_folder / "clean_data" / "application.parquet",
                                  columns=["SK_ID_CURR", "TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
    train = application[application["SK_ID_CURR"].isin(train_ids)].set_index("SK_ID_CURR")

    # The red flags of the TRAIN applicants (the team's 21 rules)
    red_flags = pd.read_parquet(agent.HERE / "data" / "red_flags.parquet").set_index("SK_ID_CURR")
    flag_columns = [column for column in red_flags.columns if column not in ["N_RED_FLAGS", "LOW_EXT_SCORE"]]
    train["n_other_flags"] = red_flags.loc[train.index, flag_columns].sum(axis=1)

    # The same percentile the profile tool returns
    external_score = train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    train["percentile"] = [tools._percentile(value, "external_score_mean") for value in external_score]

    has_score = train[train["percentile"].notna()].copy()
    has_score["row"] = [agent.table_row(value) for value in has_score["percentile"]]
    has_score["column"] = [agent.table_column(value) for value in has_score["n_other_flags"]]

    rates = []
    counts = []
    for row_number in range(len(agent.PERCENTILE_BANDS)):
        rate_row = []
        count_row = []
        for column_number in range(len(agent.FLAG_BAND_LABELS)):
            cell = has_score[(has_score["row"] == row_number) & (has_score["column"] == column_number)]
            rate_row.append(round(float(cell["TARGET"].mean()), 4))
            count_row.append(int(len(cell)))
        rates.append(rate_row)
        counts.append(count_row)

    table = {
        "built_from": "TRAIN split only ({0} applicants)".format(len(train)),
        "average_default_rate": round(float(train["TARGET"].mean()), 4),
        "percentile_bands": agent.PERCENTILE_BANDS,
        "flag_band_labels": agent.FLAG_BAND_LABELS,
        "default_rates": rates,
        "applicants_per_cell": counts,
        # Too few applicants have no external score for a separate row, so they get the average rate.
        "no_external_score_rate": round(float(train["TARGET"].mean()), 4),
        "applicants_with_no_external_score": int(train["percentile"].isna().sum()),
    }
    agent.SCORE_TABLE_FILE.parent.mkdir(exist_ok=True)
    agent.SCORE_TABLE_FILE.write_text(json.dumps(table, indent=2))
    agent.SCORE_TABLE = table
    return table


def score_table_frame(table):
    """The scoring table as a DataFrame, for display."""
    row_labels = []
    lower_edge = 0
    for upper_edge in table["percentile_bands"]:
        row_labels.append("percentile {0}-{1}".format(lower_edge, upper_edge))
        lower_edge = upper_edge + 1
    column_labels = ["{0} other red flags".format(label) for label in table["flag_band_labels"]]
    return pd.DataFrame(table["default_rates"], index=row_labels, columns=column_labels)


# ==============================================================================================
# COMMAND LINE
# ==============================================================================================
if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    pd.set_option("display.width", 200)
    if command == "tool-check":
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
        shared = tools.find_shared_folder()
        all_ids = pd.read_csv(shared / "clean_data" / "splits.csv", usecols=["SK_ID_CURR"])
        sample = all_ids["SK_ID_CURR"].sample(sample_size, random_state=42).tolist()
        result = tool_check(sample, shared)
        print("Tools vs answer key on {0:,} random applicants".format(sample_size))
        print(result.to_string())
        print("\nRed flags in the answer key that no tool field shows: {0}".format(int(result["key_flags_tool_missed"].sum())))
    elif command == "score-table":
        finished_table = build_score_table()
        print("Average default rate in TRAIN: {0}".format(finished_table["average_default_rate"]))
        print(score_table_frame(finished_table).to_string())
        print("Saved {0}".format(agent.SCORE_TABLE_FILE.name))
    elif command == "team-fields":
        for file_name in ["agent_runs.jsonl", "pilot_runs.jsonl"]:
            runs_file = HERE / "runs" / file_name
            if runs_file.exists():
                print("{0}: the team's fields were added to {1} runs".format(file_name, update_saved_runs(runs_file)))
    else:
        print("Usage: python exp3_checks.py tool-check [sample size] | score-table | team-fields")

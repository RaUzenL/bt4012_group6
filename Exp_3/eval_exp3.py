"""Exp3 evaluation: LLM-agent investigations, scored on the process (trajectory, faithfulness, completeness,
uncertainty, consistency), the output (LLM-as-judge rubric) and the outcome (discrimination, calibration, fairness).

Usage
-----
1. Pick who to investigate (test applicants only): the riskiest by the Exp1 model plus a random control group.
       python Exp_3/eval_exp3.py --make-cohort exp1_test_scores.csv      # writes data/cohort.csv
2. Map each red flag to the field names, and each data source to the tool names, YOUR agent uses (done once).
       python Exp_3/eval_exp3.py --flag-map-template --tool-map-template  # writes data/flag_to_fields.json, data/tool_map.json
3. Run the agent (e.g. 3 runs per applicant) and save one JSON object per line (agent_runs.jsonl):
       {"SK_ID_CURR": 100054, "run": 1, "risk_score": 0.82, "rating": "high", "escalated": true,
        "citations": [{"tool": "get_installment_history", "field": "late_rate", "value": 0.21}, ...],
        "tool_outputs": {"get_installment_history": {"late_rate": 0.214, ...}, ...},
        "tool_calls": [{"tool": "get_installment_history", "args": {"SK_ID_CURR": 100054}, "status": "ok"}, ...],
        "source_findings": {"installments": "concern", "credit_card": "no_history", ...},
        "usage": {"input_tokens": 5200, "output_tokens": 640, "latency_s": 14.2},
        "report": "Applicant paid 21% of instalments late ..."}
   tool_calls status: ok | no_history | error.  source_findings: concern | clean | no_history | unknown.
   Fields after tool_outputs are optional; the metrics that need them are skipped when they are missing.
   Cite values exactly as the tools return them (fractions such as 0.214, not "21%"), or faithfulness marks them wrong.
4. (Optional) Grade each report with an LLM judge using JUDGE_RUBRIC / judge_prompt(), have a person grade ~20 of
   the same reports, and save both as CSV: SK_ID_CURR, run, grounding, hedging, completeness (0-2 each).
5. Evaluate:
       python Exp_3/eval_exp3.py agent_runs.jsonl --flag-map flag_to_fields.json --tool-map tool_map.json \
           --judge judge_scores.csv --human human_scores.csv
   data/cohort.csv from step 1 is picked up automatically: it gives the flagged/control groups and the Exp1 baseline score.

Metrics (lecture 12: process correct != result correct, so both are measured)
  Process
    tool recall          data sources the agent checked successfully / all sources (a source with no records still
                         has to be checked: that is how the agent learns there is no history)
    argument accuracy    tool calls made with the applicant's own SK_ID_CURR / calls that pass an SK_ID_CURR
    gate adherence       runs escalated to a human / runs that had to be (rating 'high' or a failed tool call); target 100%
    faithfulness         share of cited values that match what the tools actually returned
    completeness         share of the applicant's rule-based red flags (red_flags.parquet) that the agent cited.
                         The answer key is rule-based rather than the Exp1 model's SHAP values on purpose: it does
                         not depend on the model, so the agent cannot score well just by repeating the model
    uncertainty honesty  sources with no records reported as no_history/unknown (not 'clean'), and failed checks
                         reported as unknown
    consistency          agreement of rating / risk score / cited evidence across repeated runs
  Output
    judge rubric         grounding (blocking), hedging, completeness on 0-2; pass rate; agreement with human grades
  Outcome
    flagged group (MAIN) among the applicants the Exp1 model flagged (top-k): PR-AUC / ROC-AUC of the agent's risk
                         score and precision / recall of its 'high' rating vs the hidden TARGET, compared with the
                         group's default rate (= precision of the Exp1 flag) and, paired, with the Exp1 score
    control group        share of random applicants rated 'high' and false-positive rate (over-flagging check)
    whole cohort         same discrimination metrics on flagged + control (secondary: the cohort was chosen with the
                         Exp1 score, which gives Exp1 a head start, and it is enriched with defaulters)
    calibration          Brier score and a reliability table: is a risk score of 0.8 right 80% of the time?
                         (descriptive only with ~200 applicants)
    fairness             flag rate and false-positive rate per gender / age band (audit only; the agent must not
                         receive these)
    cost                 tokens and latency per run
"""
# ============================================================================================
# COMMON METRICS (identical in eval_exp1.py, eval_exp2.py and eval_exp3.py, so each file runs on its own)
# ============================================================================================
import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
CLEAN_DIR = HERE.parent / "outputs" / "outputs" / "clean_data"   # the team's shared cleaned data
SHOW = ["pr_auc", "roc_auc", "precision", "recall", "f1"]


def load_splits(clean_dir=CLEAN_DIR):
    """The shared split: SK_ID_CURR, TARGET, N_PREV_APPS, PREV_BUCKET, split (train/val/test), is_dev."""
    return pd.read_csv(Path(clean_dir) / "splits.csv")


def attach_split(df, id_col="SK_ID_CURR", clean_dir=CLEAN_DIR):
    """Add TARGET, split and is_dev columns from the shared splits.csv."""
    s = load_splits(clean_dir)[["SK_ID_CURR", "TARGET", "split", "is_dev"]]
    df = df.drop(columns=[c for c in ["TARGET", "split", "is_dev"] if c in df.columns])
    return df.merge(s, left_on=id_col, right_on="SK_ID_CURR", how="left")


def best_f1_threshold(y_val, s_val):
    """Decision threshold that maximises F1 on VALIDATION data (never choose it on test)."""
    p, r, t = precision_recall_curve(y_val, s_val)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.nanargmax(f1[:-1])])


def _metrics(y, s, thr):
    yhat = (s >= thr).astype(int)
    return {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "f1": f1_score(y, yhat)}


def evaluate(y_test, s_test, y_val=None, s_val=None, threshold=None, n_boot=1000, seed=42):
    """Test metrics with bootstrap 95% CIs: PR-AUC (primary), ROC-AUC, and precision/recall/F1 at the threshold.
    Threshold = max-F1 on validation, or a fixed `threshold`."""
    if threshold is None:
        if y_val is None:
            raise ValueError("give validation labels/scores (y_val, s_val) or a fixed threshold")
        threshold = best_f1_threshold(y_val, s_val)
    y, s = np.asarray(y_test).astype(int), np.asarray(s_test, dtype=float)
    point = _metrics(y, s, threshold)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(_metrics(y[i], s[i], threshold))
    boots = pd.DataFrame(boots)
    res = pd.DataFrame({"value": point, "ci_low": boots.quantile(0.025), "ci_high": boots.quantile(0.975)})
    res.attrs["threshold"] = threshold
    return res


def delong_test(y, s_a, s_b):
    """DeLong test for the difference between two ROC-AUCs measured on the same rows."""
    y = np.asarray(y).astype(int)
    preds = np.vstack([s_a, s_b]).astype(float)
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tz = stats.rankdata(np.concatenate([pos, neg], axis=1), axis=1)
    tx, ty = stats.rankdata(pos, axis=1), stats.rankdata(neg, axis=1)
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    cov = np.cov((tz[:, :m] - tx) / n) / m + np.cov(1 - (tz[:, m:] - ty) / m) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / math.sqrt(var) if var > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": float(aucs[0] - aucs[1]),
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


def compare_models(y, s_a, s_b, metric="pr_auc", n_boot=1000, seed=42):
    """Is model A better than model B on the SAME rows? Paired bootstrap of metric(A) - metric(B), plus DeLong."""
    f = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}[metric]
    y, s_a, s_b = np.asarray(y).astype(int), np.asarray(s_a, float), np.asarray(s_b, float)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(f(y[i], s_a[i]) - f(y[i], s_b[i]))
    d = np.array(d)
    return {"metric": metric, "delta": float(f(y, s_a) - f(y, s_b)), "ci_low": float(np.quantile(d, 0.025)),
            "ci_high": float(np.quantile(d, 0.975)),
            "p_bootstrap": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            "delong_roc_auc": delong_test(y, s_a, s_b)}


def holm(p):
    """Holm-adjusted p-values: keeps the chance of ANY false 'significant' result across a family of tests at 5%."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty(len(p))
    out[order] = adj
    return out


def _add_p_values(comps):
    """p_value = DeLong for ROC-AUC, paired bootstrap for PR-AUC; p_holm adjusts across every row of comps."""
    if len(comps):
        comps["p_value"] = np.where(comps.metric == "roc_auc", comps.p_delong, comps.p_bootstrap)
        comps["p_holm"] = holm(comps.p_value)
    return comps


def load_predictions(pred, clean_dir=CLEAN_DIR):
    """Predictions as a DataFrame or a CSV path, with columns SK_ID_CURR, score (higher = riskier).
    Returns them with TARGET and split attached."""
    name = "predictions DataFrame" if isinstance(pred, pd.DataFrame) else str(pred)
    p = pred if isinstance(pred, pd.DataFrame) else pd.read_csv(pred)
    missing = {"SK_ID_CURR", "score"} - set(p.columns)
    if missing:
        raise ValueError(f"{name}: missing column(s) {sorted(missing)}; expected SK_ID_CURR, score")
    df = attach_split(p[["SK_ID_CURR", "score"]].drop_duplicates("SK_ID_CURR"), clean_dir=clean_dir)
    if df["split"].isna().any():
        raise ValueError(f"{name}: {df['split'].isna().sum()} SK_ID_CURR are not in the shared sample (splits.csv)")
    return df


def compare_prediction_files(paths, labels=None, n_boot=1000, clean_dir=CLEAN_DIR):
    """Evaluate several models on the SAME test applicants. Each entry (DataFrame or CSV path) needs val rows
    (to pick the threshold) and every test applicant. The first entry is the baseline the others are compared with.
    Returns (metrics table, comparison table, per-model predictions)."""
    labels = labels or [f"model_{k + 1}" if isinstance(p, pd.DataFrame) else Path(p).stem for k, p in enumerate(paths)]
    test_ids = set(load_splits(clean_dir).query("split == 'test'")["SK_ID_CURR"])
    preds, rows = {}, []
    for lab, path in zip(labels, paths):
        df = load_predictions(path, clean_dir)
        va, te = df[df.split == "val"], df[df.split == "test"].sort_values("SK_ID_CURR")
        if va.empty:
            raise ValueError(f"{lab}: no validation rows; they are needed to choose the decision threshold")
        if set(te["SK_ID_CURR"]) != test_ids:
            raise ValueError(f"{lab}: must score all {len(test_ids):,} test applicants (found {len(te):,})")
        res = evaluate(te.TARGET, te.score, y_val=va.TARGET, s_val=va.score, n_boot=n_boot)
        preds[lab] = {"val": va, "test": te, "threshold": res.attrs["threshold"]}
        row = {"model": lab, "threshold": res.attrs["threshold"]}
        for m, r in res.iterrows():
            row[m], row[f"{m}_ci_low"], row[f"{m}_ci_high"] = r["value"], r["ci_low"], r["ci_high"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    comps = []
    base = labels[0]
    for lab in labels[1:]:
        y = preds[base]["test"]["TARGET"].to_numpy()
        for metric in ("pr_auc", "roc_auc"):
            c = compare_models(y, preds[lab]["test"]["score"], preds[base]["test"]["score"], metric, n_boot)
            comps.append({"model": lab, "vs_baseline": base, "metric": metric, "delta": c["delta"],
                          "ci_low": c["ci_low"], "ci_high": c["ci_high"], "p_bootstrap": c["p_bootstrap"],
                          "p_delong": c["delong_roc_auc"]["p_value"] if metric == "roc_auc" else np.nan})
    return table, _add_p_values(pd.DataFrame(comps)), preds


def pretty_metrics(table, metrics=SHOW):
    """Readable metrics table: 'value (95% CI low-high)' per metric, one row per model."""
    out = pd.DataFrame({"threshold": table["threshold"].round(3)}, index=table.index)
    for m in metrics:
        out[m] = [f"{v:.3f} ({lo:.3f}-{hi:.3f})"
                  for v, lo, hi in zip(table[m], table[f"{m}_ci_low"], table[f"{m}_ci_high"])]
    return out


def pretty_comparisons(comps):
    """Readable comparison table with a plain-language verdict based on the Holm-adjusted p-value (< 0.05)."""
    if comps.empty:
        return comps
    c = comps.copy()
    c["difference (95% CI)"] = [f"{d:+.4f} ({lo:+.4f} to {hi:+.4f})" for d, lo, hi in zip(c.delta, c.ci_low, c.ci_high)]
    c["verdict"] = np.where(c.p_holm >= 0.05, "no significant difference",
                            np.where(c.delta > 0, "significantly better", "significantly worse"))
    c["p_value"], c["p_holm"] = c.p_value.map("{:.2g}".format), c.p_holm.map("{:.2g}".format)
    keys = [k for k in ["grouping", "segment"] if k in c]
    return c[keys + ["model", "vs_baseline", "metric", "difference (95% CI)", "p_value", "p_holm", "verdict"]]


def _print_model_table(table, comps):
    pd.set_option("display.width", 200)
    print("\nTest-set metrics, value (95% CI); threshold chosen on validation")
    print(pretty_metrics(table).to_string())
    if len(comps):
        print("\nPaired comparison vs baseline (same test applicants)")
        print(pretty_comparisons(comps).to_string(index=False))


def _predictions_cli(description, epilog):
    ap = argparse.ArgumentParser(description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+",
                    help="CSV file(s) with columns SK_ID_CURR, score for val + test applicants. First file = baseline.")
    ap.add_argument("--labels", nargs="+", help="model names (default: file names)")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for CIs (default 1000)")
    return ap


# ============================================================================================
# EXP3: RED-FLAG ANSWER KEY (for completeness)
# ============================================================================================
RED_FLAGS_FILE, RULES_FILE = HERE / "data" / "red_flags.parquet", HERE / "data" / "red_flag_rules.json"
RED_FLAG_RULES = [
    # id, source table, concept the agent should mention, rule
    ("HIGH_ANNUITY_TO_INCOME", "application", "repayment burden", "annuity / income in top 5% (train)"),
    ("LOW_EXT_SCORE", "application", "external credit score", "mean EXT_SOURCE in bottom 10% (train)"),
    ("RECENT_ID_CHANGE", "application", "identity change", "ID document changed within 1 year"),
    ("RECENT_PHONE_CHANGE", "application", "identity change", "phone changed within 30 days"),
    ("SHORT_EMPLOYMENT", "application", "employment", "employed less than 1 year"),
    ("MANY_ENQUIRIES", "application", "credit enquiries", "bureau enquiries last year in top 5% (train)"),
    ("BUREAU_CURRENT_OVERDUE", "bureau", "current overdue debt", "overdue amount at another lender > 0"),
    ("BUREAU_SEVERE_DELINQUENCY", "bureau_balance", "past delinquency", "worst monthly status 31+ days past due"),
    ("BUREAU_HIGH_UTILISATION", "bureau", "debt burden", "active debt >= 80% of active credit"),
    ("PREV_HIGH_REFUSAL", "previous_application", "rejection history", "half or more of past applications refused"),
    ("PREV_LAST_REFUSED", "previous_application", "rejection history", "most recent application refused"),
    ("PREV_HIGH_FREQUENCY", "previous_application", "application frequency", "applications in last year in top 5% (train)"),
    ("INST_FREQUENT_LATE", "installments", "late payments", "10%+ of instalments paid late"),
    ("INST_SEVERE_LATE", "installments", "late payments", "an instalment paid 30+ days late"),
    ("INST_UNDERPAYMENT", "installments", "underpayment", "5%+ of instalments underpaid"),
    ("INST_WORSENING", "installments", "worsening trend", "late rate in last year > overall late rate + 5pp"),
    ("CC_HIGH_UTILISATION", "credit_card", "card utilisation", "latest month balance >= 90% of limit"),
    ("CC_BELOW_MIN_PAYMENT", "credit_card", "minimum payments", "3+ months paid below the minimum"),
    ("CC_CASH_RELIANCE", "credit_card", "cash withdrawals", "half or more of card drawings are cash"),
    ("POS_DELINQUENT", "pos_cash", "POS/cash delinquency", "POS/cash loan 30+ days past due"),
    ("POS_DEBT_DEMAND", "pos_cash", "debt collection", "a contract escalated to debt demand"),
]


def compute_red_flags(clean_dir=CLEAN_DIR):
    """Rule-based answer key: one row per applicant, one 0/1 column per rule (no record -> 0).
    Percentile thresholds are fitted on TRAIN applicants only. Returns (flags, thresholds)."""
    d = Path(clean_dir)
    sp = load_splits(d)
    ids = pd.Index(sp["SK_ID_CURR"], name="SK_ID_CURR")
    train = set(sp.loc[sp.split == "train", "SK_ID_CURR"])
    F, thr = pd.DataFrame(index=ids), {}

    def pct(series, q, floor=None):
        v = float(series[series.index.isin(train)].quantile(q))
        return max(v, floor) if floor is not None else v

    a = pd.read_parquet(d / "application.parquet").set_index("SK_ID_CURR")
    burden = a.AMT_ANNUITY / a.AMT_INCOME_TOTAL
    ext = a[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    thr["HIGH_ANNUITY_TO_INCOME"] = pct(burden, 0.95)
    thr["LOW_EXT_SCORE"] = pct(ext, 0.10)
    thr["MANY_ENQUIRIES"] = pct(a.AMT_REQ_CREDIT_BUREAU_YEAR, 0.95, floor=1)
    F["HIGH_ANNUITY_TO_INCOME"] = burden >= thr["HIGH_ANNUITY_TO_INCOME"]
    F["LOW_EXT_SCORE"] = ext <= thr["LOW_EXT_SCORE"]
    F["RECENT_ID_CHANGE"] = a.DAYS_ID_PUBLISH >= -365
    F["RECENT_PHONE_CHANGE"] = a.DAYS_LAST_PHONE_CHANGE >= -30
    F["SHORT_EMPLOYMENT"] = a.DAYS_EMPLOYED > -365
    F["MANY_ENQUIRIES"] = a.AMT_REQ_CREDIT_BUREAU_YEAR >= thr["MANY_ENQUIRIES"]

    b = pd.read_parquet(d / "bureau.parquet", columns=["SK_ID_CURR", "CREDIT_ACTIVE", "AMT_CREDIT_SUM",
                                                        "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE"])
    F["BUREAU_CURRENT_OVERDUE"] = b.groupby("SK_ID_CURR").AMT_CREDIT_SUM_OVERDUE.sum() > 0
    act = b[b.CREDIT_ACTIVE.astype(str) == "Active"].groupby("SK_ID_CURR")
    F["BUREAU_HIGH_UTILISATION"] = act.AMT_CREDIT_SUM_DEBT.sum() / act.AMT_CREDIT_SUM.sum().replace(0, np.nan) >= 0.8
    bb = pd.read_parquet(d / "bureau_balance.parquet", columns=["SK_ID_CURR", "STATUS_DPD"])
    F["BUREAU_SEVERE_DELINQUENCY"] = bb.groupby("SK_ID_CURR").STATUS_DPD.max() >= 2

    p = pd.read_parquet(d / "previous_application.parquet",
                        columns=["SK_ID_CURR", "NAME_CONTRACT_STATUS", "DAYS_DECISION", "IS_DUPLICATE_SUBMISSION"])
    p = p[p.IS_DUPLICATE_SUBMISSION == 0].assign(REFUSED=lambda x: x.NAME_CONTRACT_STATUS.astype(str) == "Refused",
                                                 LAST_YEAR=lambda x: x.DAYS_DECISION >= -365)
    g = p.groupby("SK_ID_CURR")
    apps_1y = g.LAST_YEAR.sum()
    thr["PREV_HIGH_FREQUENCY"] = pct(apps_1y, 0.95, floor=1)
    F["PREV_HIGH_REFUSAL"] = g.REFUSED.mean() >= 0.5
    F["PREV_LAST_REFUSED"] = p.sort_values("DAYS_DECISION").groupby("SK_ID_CURR").REFUSED.last()
    F["PREV_HIGH_FREQUENCY"] = apps_1y >= thr["PREV_HIGH_FREQUENCY"]

    i = pd.read_parquet(d / "installments.parquet")    # already one row per instalment (split payments merged)
    i = i.rename(columns={"DAYS_INSTALMENT": "due_day", "AMT_INSTALMENT": "due", "DAYS_ENTRY_PAYMENT": "paid_day"})
    i["paid"] = i["AMT_PAYMENT"].fillna(0)
    i["LATE"] = (i.paid_day - i.due_day) > 0
    i["DPD"] = (i.paid_day - i.due_day).clip(lower=0)
    i["UNDERPAID"] = (i.due - i.paid) > 1.0
    i["LATE_RECENT"] = i.LATE.where(i.due_day >= -365)
    g = i.groupby("SK_ID_CURR")
    late_rate, late_recent = g.LATE.mean(), g.LATE_RECENT.mean()
    F["INST_FREQUENT_LATE"] = late_rate >= 0.10
    F["INST_SEVERE_LATE"] = g.DPD.max() >= 30
    F["INST_UNDERPAYMENT"] = g.UNDERPAID.mean() >= 0.05
    F["INST_WORSENING"] = (late_recent - late_rate) > 0.05
    del i

    c = pd.read_parquet(d / "credit_card.parquet",
                        columns=["SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
                                 "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT", "AMT_INST_MIN_REGULARITY",
                                 "AMT_PAYMENT_TOTAL_CURRENT"])
    m = c.groupby(["SK_ID_CURR", "MONTHS_BALANCE"], as_index=False)[["AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]].sum()
    latest = m.sort_values("MONTHS_BALANCE").groupby("SK_ID_CURR").tail(1).set_index("SK_ID_CURR")
    F["CC_HIGH_UTILISATION"] = latest.AMT_BALANCE / latest.AMT_CREDIT_LIMIT_ACTUAL.where(latest.AMT_CREDIT_LIMIT_ACTUAL > 0) >= 0.9
    below = (c.AMT_INST_MIN_REGULARITY > 0) & (c.AMT_PAYMENT_TOTAL_CURRENT.fillna(0) < c.AMT_INST_MIN_REGULARITY)
    F["CC_BELOW_MIN_PAYMENT"] = below.groupby(c.SK_ID_CURR).sum() >= 3
    g = c.groupby("SK_ID_CURR")
    F["CC_CASH_RELIANCE"] = g.AMT_DRAWINGS_ATM_CURRENT.sum() / g.AMT_DRAWINGS_CURRENT.sum().replace(0, np.nan) >= 0.5

    q = pd.read_parquet(d / "pos_cash.parquet", columns=["SK_ID_CURR", "SK_DPD", "NAME_CONTRACT_STATUS"])
    F["POS_DELINQUENT"] = q.groupby("SK_ID_CURR").SK_DPD.max() >= 30
    F["POS_DEBT_DEMAND"] = (q.NAME_CONTRACT_STATUS.astype(str) == "Demand").groupby(q.SK_ID_CURR).any()

    F = F[[r[0] for r in RED_FLAG_RULES]].fillna(False).astype("int8")
    F["N_RED_FLAGS"] = F.sum(axis=1)
    return F.reset_index(), thr


def load_red_flags():
    """red_flags.parquet (built if missing), indexed by SK_ID_CURR."""
    if not RED_FLAGS_FILE.exists():
        flags, thr = compute_red_flags()
        flags.to_parquet(RED_FLAGS_FILE, index=False)
        RULES_FILE.write_text(json.dumps([{"id": r, "source_table": s, "concept": c, "rule": d,
                                           "threshold": thr.get(r)} for r, s, c, d in RED_FLAG_RULES], indent=2))
    return pd.read_parquet(RED_FLAGS_FILE).set_index("SK_ID_CURR")


# ============================================================================================
# EXP3: AGENT METRICS
# ============================================================================================
import warnings

from sklearn.metrics import brier_score_loss, cohen_kappa_score


def _mean(xs):
    xs = np.array(xs, dtype=float)
    return float(np.nanmean(xs)) if np.isfinite(xs).any() else np.nan


def _same(cited, actual, rel_tol):
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        return cited is None
    try:
        return math.isclose(float(cited), float(actual), rel_tol=rel_tol, abs_tol=0.005)
    except (TypeError, ValueError):
        return str(cited).strip().lower() == str(actual).strip().lower()


def faithfulness(citations, tool_outputs, rel_tol=0.01):
    """Share of citations [{tool, field, value}] whose value matches what that tool returned
    (numbers within 1%). A citation of a tool/field that was never returned counts as 'not_found'."""
    rows = []
    for cit in citations:
        fields = tool_outputs.get(cit.get("tool"), {})
        if cit.get("field") not in fields:
            status = "not_found"
        else:
            status = "match" if _same(cit.get("value"), fields[cit["field"]], rel_tol) else "mismatch"
        rows.append({**cit, "status": status, "actual": fields.get(cit.get("field"))})
    details = pd.DataFrame(rows, columns=["tool", "field", "value", "status", "actual"])
    score = float((details["status"] == "match").mean()) if len(details) else np.nan
    return {"faithfulness": score, "n_citations": len(details), "details": details}


def _normalise_flag_map(flag_map):
    return {k: (v["fields"] if isinstance(v, dict) else v) for k, v in (flag_map or {}).items()}


def completeness(cited_fields, present_flags, flag_to_fields):
    """Share of the applicant's red flags that the agent cited.
    flag_to_fields maps each red-flag id to the agent tool field names that count as evidence for it."""
    flag_to_fields = _normalise_flag_map(flag_to_fields)
    cited = {str(f) for f in cited_fields}
    found = [f for f in present_flags if cited & set(flag_to_fields.get(f, []))]
    missed = [f for f in present_flags if f not in found]
    score = len(found) / len(present_flags) if present_flags else np.nan
    return {"completeness": score, "n_present": len(present_flags), "found": found, "missed": missed}


def consistency(runs):
    """Repeated runs for ONE applicant, each {rating, risk_score, cited_fields or citations}."""
    ratings = pd.Series([r.get("rating") for r in runs])
    scores = np.array([r.get("risk_score", np.nan) for r in runs], dtype=float)
    sets = [set(r.get("cited_fields") or [c.get("field") for c in r.get("citations", [])]) for r in runs]
    jac = [len(x & y) / len(x | y) if (x | y) else 1.0 for x, y in combinations(sets, 2)]
    return {"n_runs": len(runs), "same_rating_all_runs": bool(ratings.nunique() == 1),
            "rating_agreement": float(ratings.value_counts(normalize=True).iloc[0]),
            "risk_score_std": float(np.nanstd(scores)), "risk_score_range": float(np.nanmax(scores) - np.nanmin(scores)),
            "evidence_jaccard": float(np.mean(jac)) if jac else 1.0}


# ============================================================================================
# EXP3: TRAJECTORY, ESCALATION GATE AND UNCERTAINTY (lecture 12: check the process, not just the final answer)
# ============================================================================================
SOURCES = ["application", "bureau", "previous_application", "installments", "pos_cash", "credit_card"]
TOOL_MAP_TEMPLATE = {"application": ["get_applicant_profile"], "bureau": ["get_credit_bureau_history"],
                     "previous_application": ["get_previous_applications"], "installments": ["get_installment_history"],
                     "pos_cash": ["get_pos_cash_history"], "credit_card": ["get_credit_card_history"]}
FLAG_RATINGS = ("high",)       # ratings that count as 'flagged' and must be escalated to a human


def source_coverage(ids, clean_dir=CLEAN_DIR):
    """True where the applicant has at least one record in that data source (application always does)."""
    d, ids = Path(clean_dir), pd.Index(ids, name="SK_ID_CURR")
    cov = pd.DataFrame({"application": True}, index=ids)
    for s in SOURCES[1:]:
        cov[s] = ids.isin(pd.read_parquet(d / f"{s}.parquet", columns=["SK_ID_CURR"])["SK_ID_CURR"])
    return cov


def _tool_sources(tool_map):
    return {t: s for s, tools in (tool_map or {}).items() for t in tools}


def trajectory(run, sid, tool_map=None):
    """One run's tool calls. tool_recall: data sources checked without error / all sources in tool_map
    (None skips it). argument_accuracy: calls passing this applicant's SK_ID_CURR / calls passing any SK_ID_CURR."""
    calls = run.get("tool_calls")
    if calls is None:
        return {"tool_recall": np.nan, "argument_accuracy": np.nan}
    src = _tool_sources(tool_map)
    done = {src.get(c.get("tool")) for c in calls if c.get("status", "ok") != "error"}
    with_id = [c for c in calls if "SK_ID_CURR" in (c.get("args") or {})]
    right = [c for c in with_id if int(c["args"]["SK_ID_CURR"]) == sid]
    return {"tool_recall": len(done & set(tool_map)) / len(tool_map) if tool_map else np.nan,
            "argument_accuracy": len(right) / len(with_id) if with_id else np.nan}


def gate_required(run, flag_ratings=FLAG_RATINGS):
    """A run must go to human review when it rates the applicant high risk or any tool check failed."""
    return run.get("rating") in flag_ratings or any(c.get("status") == "error" for c in run.get("tool_calls") or [])


def uncertainty_honesty(run, has_records, tool_map=None):
    """Share of 'nothing to see' sources reported honestly: a source with no records must be reported as no_history
    or unknown (never 'clean'), and a source whose tool call failed as unknown (needs tool_map). A source with no
    verdict counts as not honest. NaN when the run has no source_findings or nothing to check."""
    findings = run.get("source_findings")
    if findings is None:
        return np.nan
    src = _tool_sources(tool_map)
    failed = {src[c["tool"]] for c in run.get("tool_calls") or [] if c.get("status") == "error" and c.get("tool") in src}
    checks = [(s, {"unknown"}) for s in failed] + \
             [(s, {"no_history", "unknown"}) for s in SOURCES if not has_records.get(s, True) and s not in failed]
    return float(np.mean([findings.get(s) in ok for s, ok in checks])) if checks else np.nan


# ============================================================================================
# EXP3: CALIBRATION AND FAIRNESS (outcome checks)
# ============================================================================================
def calibration(y, s, n_bins=5):
    """Is the risk score a probability? Brier score, expected calibration error, and a reliability table
    (equal-size score bins: mean risk score vs observed default rate). If not calibrated, read it as a ranking."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    df = pd.DataFrame({"y": y, "s": s, "bin": pd.qcut(s, n_bins, duplicates="drop")})
    tab = df.groupby("bin", observed=True).agg(n=("y", "size"), mean_risk_score=("s", "mean"), default_rate=("y", "mean"))
    ece = float((tab.n * (tab.mean_risk_score - tab.default_rate).abs()).sum() / tab.n.sum())
    return {"brier": float(brier_score_loss(y, s)), "ece": ece}, tab


FAIRNESS_GROUPS = ["CODE_GENDER", "AGE_BAND"]   # family status left out: its groups have only ~5-25 of 200 applicants


def fairness_table(per_app, flag_col="flagged", clean_dir=CLEAN_DIR):
    """Audit only (the agent must not receive these attributes). Per group: flag rate and false-positive rate
    (non-defaulters flagged). Compare false-positive rates across groups, not just overall. With ~200 applicants
    this is descriptive: report n with every rate."""
    a = pd.read_parquet(Path(clean_dir) / "application.parquet", columns=["SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH"])
    a["AGE_BAND"] = pd.cut(-a.DAYS_BIRTH / 365.25, [0, 30, 45, 60, 200], right=False,
                           labels=["<30", "30-44", "45-59", "60+"])
    df = per_app[["SK_ID_CURR", "TARGET", flag_col]].merge(a.drop(columns="DAYS_BIRTH"), on="SK_ID_CURR", how="left")
    rows = []
    for g in FAIRNESS_GROUPS:
        for val, part in df.groupby(df[g].astype(str)):
            neg = part[part.TARGET == 0]
            rows.append({"grouping": g, "group": val, "n": len(part), "default_rate": part.TARGET.mean(),
                         "flag_rate": part[flag_col].mean(),
                         "false_positive_rate": neg[flag_col].mean() if len(neg) else np.nan})
    return pd.DataFrame(rows)


# ============================================================================================
# EXP3: LLM-AS-JUDGE RUBRIC (output quality; the judge is a tool, not ground truth)
# ============================================================================================
JUDGE_CRITERIA = ["grounding", "hedging", "completeness"]
JUDGE_RUBRIC = """You are reviewing a credit-risk investigation report written by an AI agent.
Score the report on each criterion below, using ONLY the tool outputs provided. Quote the evidence for each score.
Reply as JSON: {"grounding": 0-2, "hedging": 0-2, "completeness": 0-2, "evidence": {"grounding": "...", ...}}

GROUNDING (blocking)
  2 = every factual claim traces to a value in the tool outputs
  1 = one claim is unsourced but plausible
  0 = any claim contradicts the tool outputs, or cites a tool or value that does not exist
HEDGING
  2 = missing or failed data is stated as missing/unknown (e.g. "no credit-card history"), never as a clean record
  1 = hedged vaguely ("may have issues") without saying what is missing
  0 = states something as fact that the tool outputs do not show
COMPLETENESS
  2 = every data source has a verdict, including negative results
  1 = one data source has no verdict
  0 = two or more data sources, or a clear red flag in the tool outputs, are not addressed
Any 0 on GROUNDING fails the report regardless of the other scores."""


def judge_prompt(run):
    """Prompt for the LLM judge: the rubric, the tool outputs the agent saw, and its report."""
    return (f"{JUDGE_RUBRIC}\n\nTOOL OUTPUTS\n{json.dumps(run.get('tool_outputs', {}), indent=2, default=str)}"
            f"\n\nREPORT\n{run.get('report', '')}")


def evaluate_judge(judge, human=None):
    """judge / human: DataFrame or CSV with SK_ID_CURR, run, grounding, hedging, completeness (0-2).
    Mean score per criterion, pass rate (grounding > 0) and, on reports graded by both, agreement with the human
    grades (exact agreement and quadratic-weighted Cohen's kappa)."""
    j = judge if isinstance(judge, pd.DataFrame) else pd.read_csv(judge)
    out = {f"{c}_mean": float(j[c].mean()) for c in JUDGE_CRITERIA}
    out["pass_rate"] = float((j.grounding > 0).mean())
    if human is not None:
        h = human if isinstance(human, pd.DataFrame) else pd.read_csv(human)
        m = j.merge(h, on=["SK_ID_CURR", "run"], suffixes=("_judge", "_human"))
        out["vs_human_n"] = len(m)
        for c in JUDGE_CRITERIA:
            a, b = m[f"{c}_judge"].astype(int), m[f"{c}_human"].astype(int)
            out[f"vs_human_{c}_agreement"] = float((a == b).mean())
            with warnings.catch_warnings():     # kappa is undefined (NaN) when both graders give one score only
                warnings.simplefilter("ignore")
                out[f"vs_human_{c}_kappa"] = float(cohen_kappa_score(a, b, labels=[0, 1, 2], weights="quadratic"))
    return out


def select_cohort(scores_csv, k_flagged=100, k_control=100, seed=42, clean_dir=CLEAN_DIR):
    """Investigation cohort from TEST applicants: top-k by model score ('flagged') + random controls.
    Controls are needed so discrimination can be measured. TARGET is deliberately not included."""
    sc = load_predictions(scores_csv, clean_dir)
    sc = sc[sc.split == "test"].sort_values("score", ascending=False)
    flagged = sc.head(k_flagged).assign(group="flagged")
    control = sc.iloc[k_flagged:].sample(k_control, random_state=seed).assign(group="control")
    return pd.concat([flagged, control])[["SK_ID_CURR", "score", "group"]]


def read_runs(path):
    """Agent runs from a .jsonl file (one JSON object per line) or a .json list."""
    text = Path(path).read_text().strip()
    return json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]


def _ci(res, metrics):
    return {m: {"value": float(r.value), "ci_low": float(r.ci_low), "ci_high": float(r.ci_high)}
            for m, r in res.loc[metrics].iterrows()}


def _discrimination(df, n_boot):
    """PR-AUC / ROC-AUC of the risk score, and precision / recall / F1 of the flagged ('high') rating."""
    auc = evaluate(df.TARGET, df.risk_score, threshold=0.5, n_boot=n_boot)
    hi = evaluate(df.TARGET, df.flagged, threshold=1, n_boot=n_boot)
    return {**_ci(auc, ["pr_auc", "roc_auc"]), **{f"{m} of 'high'": v for m, v in _ci(hi, ["precision", "recall", "f1"]).items()}}


def _vs_baseline(df, n_boot):
    """Paired difference agent risk score - baseline score on the same applicants."""
    out = {}
    for metric in ("pr_auc", "roc_auc"):
        c = compare_models(df.TARGET, df.risk_score, df.baseline_score, metric=metric, n_boot=n_boot)
        out[metric] = {**{k: c[k] for k in ["delta", "ci_low", "ci_high", "p_bootstrap"]}, "n": len(df)}
    return out


def evaluate_agent_runs(runs, flag_map=None, tool_map=None, flag_ratings=FLAG_RATINGS, baseline_csv=None,
                        judge=None, human=None, n_boot=1000, cohort=None, clean_dir=CLEAN_DIR):
    """Score every investigated applicant and summarise. Returns (per_applicant table, summary dict).
    runs: list of run dicts or a .jsonl/.json path.  flag_map / tool_map: dict or JSON path (None skips completeness /
    tool recall).  flag_ratings: ratings that count as flagged.  judge / human: rubric scores (see evaluate_judge).
    cohort: select_cohort output (DataFrame or CSV; default cohort.csv next to this file if it exists). It gives each
    applicant's group (flagged / control) and the Exp1 score used as the baseline (baseline_csv overrides the score).
    Main outcome = the FLAGGED group: among applicants the model already flagged, does the agent separate real
    defaulters from false alarms? Whole-cohort numbers are secondary: the cohort was chosen with the Exp1 score, which
    gives Exp1 a built-in head start there. Metrics whose run fields are missing are left out of the summary."""
    if isinstance(runs, (str, Path)):
        runs = read_runs(runs)
    if cohort is None and (HERE / "data" / "cohort.csv").exists():
        cohort = HERE / "data" / "cohort.csv"
    if cohort is not None and not isinstance(cohort, pd.DataFrame):
        cohort = pd.read_csv(cohort)
    if isinstance(flag_map, (str, Path)):
        flag_map = json.loads(Path(flag_map).read_text())
    if isinstance(tool_map, (str, Path)):
        tool_map = json.loads(Path(tool_map).read_text())
    flags = load_red_flags()
    rule_ids = [r[0] for r in RED_FLAG_RULES]
    by_app = {}
    for r in runs:
        by_app.setdefault(int(r["SK_ID_CURR"]), []).append(r)
    cov = source_coverage(list(by_app), clean_dir)
    rows, all_cits = [], []
    for sid, rs in by_app.items():
        f = [faithfulness(r.get("citations", []), r.get("tool_outputs", {})) for r in rs]
        all_cits += [d for x in f for d in x["details"].to_dict("records")]
        present = [k for k in rule_ids if sid in flags.index and flags.at[sid, k] == 1]
        rating = pd.Series([r.get("rating") for r in rs]).mode().iloc[0]
        row = {"SK_ID_CURR": sid, "n_runs": len(rs), "risk_score": _mean([r.get("risk_score", np.nan) for r in rs]),
               "rating": rating, "flagged": int(rating in flag_ratings),
               "faithfulness": _mean([x["faithfulness"] for x in f]), "n_red_flags": len(present)}
        if flag_map is not None:
            comp = [completeness([c.get("field") for c in r.get("citations", [])], present, flag_map) for r in rs]
            row["completeness"] = _mean([c["completeness"] for c in comp]) if present else np.nan
            row["missed_flags"] = ", ".join(sorted(set.intersection(*[set(c["missed"]) for c in comp]))) if present else ""
        traj = [trajectory(r, sid, tool_map) for r in rs]
        row["tool_recall"] = _mean([t["tool_recall"] for t in traj])
        row["argument_accuracy"] = _mean([t["argument_accuracy"] for t in traj])
        esc = [bool(r["escalated"]) for r in rs if gate_required(r, flag_ratings) and "escalated" in r]
        row["gate_required_runs"], row["gate_escalated_runs"] = len(esc), sum(esc)
        row["uncertainty_honesty"] = _mean([uncertainty_honesty(r, cov.loc[sid].to_dict(), tool_map) for r in rs])
        for k in ["input_tokens", "output_tokens", "latency_s"]:
            row[k] = _mean([(r.get("usage") or {}).get(k, np.nan) for r in rs])
        if len(rs) > 1:
            row.update(consistency(rs))
        rows.append(row)
    per_app = attach_split(pd.DataFrame(rows), clean_dir=clean_dir)
    if cohort is not None:
        per_app = per_app.merge(cohort[["SK_ID_CURR", "group", "score"]].rename(columns={"score": "baseline_score"}),
                                on="SK_ID_CURR", how="left")
    if baseline_csv:
        base = load_predictions(baseline_csv, clean_dir)[["SK_ID_CURR", "score"]].rename(columns={"score": "baseline_score"})
        per_app = per_app.drop(columns="baseline_score", errors="ignore").merge(base, on="SK_ID_CURR", how="left")
    cits = pd.DataFrame(all_cits)

    def means(cols):
        return {c: float(per_app[c].mean()) for c in cols if c in per_app and per_app[c].notna().any()}

    process = means(["tool_recall", "argument_accuracy"])
    if per_app.gate_required_runs.sum():
        process["gate_adherence"] = float(per_app.gate_escalated_runs.sum() / per_app.gate_required_runs.sum())
    process.update(means(["faithfulness"]))
    if len(cits):
        process["citations_not_found_share"] = float((cits.status == "not_found").mean())
        process["citations_mismatch_share"] = float((cits.status == "mismatch").mean())
    process.update(means(["completeness", "uncertainty_honesty"]))
    if "same_rating_all_runs" in per_app:
        process["same_rating_all_runs_share"] = float(per_app.same_rating_all_runs.mean())
        process.update(means(["evidence_jaccard", "risk_score_std"]))
    summary = {"cohort": {"applicants": len(per_app), "runs": len(runs),
                          "default_rate": float(per_app.TARGET.mean()), "flagged_share": float(per_app.flagged.mean())},
               "process": process}
    if judge is not None:
        summary["output (judge)"] = evaluate_judge(judge, human)
    outcome = {}
    has_base = "baseline_score" in per_app and per_app.baseline_score.notna().any()
    flagged = per_app[per_app.group == "flagged"] if "group" in per_app else per_app.iloc[:0]
    if flagged.TARGET.nunique() == 2:
        outcome["flagged group (main)"] = {"applicants": len(flagged),
                                           "default_rate": float(flagged.TARGET.mean()),
                                           **_discrimination(flagged, n_boot)}
        if has_base:
            outcome["flagged group (main)"]["agent vs Exp1"] = _vs_baseline(flagged.dropna(subset=["baseline_score"]), n_boot)
    if "group" in per_app and (per_app.group == "control").any():
        ctrl = per_app[per_app.group == "control"]
        outcome["control group"] = {
            "applicants": len(ctrl), "default_rate": float(ctrl.TARGET.mean()), "rated_high_share": float(ctrl.flagged.mean()),
            "false_positive_rate": float(ctrl.loc[ctrl.TARGET == 0, "flagged"].mean())}
    if per_app.TARGET.nunique() == 2:
        whole = _discrimination(per_app, n_boot)
        if has_base and not outcome.get("flagged group (main)"):     # no groups known: whole cohort is all we have
            whole["agent vs baseline"] = _vs_baseline(per_app.dropna(subset=["baseline_score"]), n_boot)
        outcome["whole cohort (secondary)"] = whole
        if per_app.risk_score.between(0, 1).all():
            outcome["calibration"] = calibration(per_app.TARGET, per_app.risk_score)[0]
    outcome["cost per run"] = means(["input_tokens", "output_tokens", "latency_s"])
    summary["outcome"] = {k: v for k, v in outcome.items() if v}
    return per_app, summary


def summary_table(summary):
    """The evaluate_agent_runs summary as a two-column table (metric, value with 95% CI where available)."""
    rows = []

    def walk(name, v):
        if isinstance(v, dict) and "value" in v:
            rows.append((name, f"{v['value']:.3f} ({v['ci_low']:.3f}-{v['ci_high']:.3f})"))
        elif isinstance(v, dict) and "delta" in v:
            rows.append((name, f"{v['delta']:+.3f} ({v['ci_low']:+.3f} to {v['ci_high']:+.3f}), "
                               f"p={v['p_bootstrap']:.2g}, n={v['n']}"))
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(f"{name}: {k}" if name else k, x)
        else:
            rows.append((name, f"{v:.3f}" if isinstance(v, float) else v))

    walk("", summary)
    return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")


# ============================================================================================
# EXP3 COMMAND LINE
# ============================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="?", help="agent runs: .jsonl (one JSON per line) or .json list")
    ap.add_argument("--flag-map", help="JSON mapping red-flag id -> agent field names (needed for completeness)")
    ap.add_argument("--tool-map", help="JSON mapping data source -> agent tool names (needed for tool recall)")
    ap.add_argument("--flag-ratings", nargs="+", default=list(FLAG_RATINGS),
                    help="ratings that count as flagged and must be escalated (default: high)")
    ap.add_argument("--cohort", help="cohort.csv from --make-cohort (default: cohort.csv next to this file if present)")
    ap.add_argument("--baseline", help="CSV SK_ID_CURR, score of a model to compare the agent's risk score with "
                                       "(default: the Exp1 score stored in the cohort)")
    ap.add_argument("--judge", help="CSV of LLM-judge rubric scores: SK_ID_CURR, run, grounding, hedging, completeness")
    ap.add_argument("--human", help="CSV of human rubric scores for some of the same reports (judge agreement)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--make-cohort", metavar="SCORES_CSV", help="write cohort.csv from a model's test scores")
    ap.add_argument("--k-flagged", type=int, default=100)
    ap.add_argument("--k-control", type=int, default=100)
    ap.add_argument("--flag-map-template", action="store_true", help="write flag_to_fields.json to fill in")
    ap.add_argument("--tool-map-template", action="store_true", help="write tool_map.json to fill in")
    ap.add_argument("--out", default=str(HERE / "results" / "exp3"), help="output file prefix (default Exp_3/results/exp3)")
    a = ap.parse_args(argv)

    if a.make_cohort:
        cohort = select_cohort(a.make_cohort, a.k_flagged, a.k_control)
        cohort.to_csv(HERE / "data" / "cohort.csv", index=False)
        print(f"Wrote {HERE / 'data' / 'cohort.csv'}: {len(cohort)} test applicants ({a.k_flagged} flagged + {a.k_control} control)")
    if a.flag_map_template:
        tpl = {r: {"concept": c, "rule": d, "fields": []} for r, _, c, d in RED_FLAG_RULES}
        (HERE / "data" / "flag_to_fields.json").write_text(json.dumps(tpl, indent=2))
        print(f"Wrote {HERE / 'data' / 'flag_to_fields.json'}: fill each 'fields' list with your tool field names")
    if a.tool_map_template:
        (HERE / "data" / "tool_map.json").write_text(json.dumps(TOOL_MAP_TEMPLATE, indent=2))
        print(f"Wrote {HERE / 'data' / 'tool_map.json'}: replace the example tool names with your agent's tool names")
    if a.runs:
        flag_map = json.loads(Path(a.flag_map).read_text()) if a.flag_map else None
        tool_map = json.loads(Path(a.tool_map).read_text()) if a.tool_map else None
        if flag_map is None:
            print("No --flag-map given: completeness is skipped.")
        if tool_map is None:
            print("No --tool-map given: tool recall is skipped.")
        per_app, summary = evaluate_agent_runs(read_runs(a.runs), flag_map, tool_map, tuple(a.flag_ratings), a.baseline,
                                               a.judge, a.human, a.n_boot, a.cohort)
        print(summary_table(summary).to_string())
        per_app.to_csv(f"{a.out}_per_applicant.csv", index=False)
        Path(f"{a.out}_summary.json").write_text(json.dumps(summary, indent=2))
        saved = [f"{a.out}_per_applicant.csv", f"{a.out}_summary.json"]
        if per_app.TARGET.nunique() == 2:
            if per_app.risk_score.between(0, 1).all():
                calibration(per_app.TARGET, per_app.risk_score)[1].to_csv(f"{a.out}_calibration.csv")
                saved.append(f"{a.out}_calibration.csv")
            fairness_table(per_app).to_csv(f"{a.out}_fairness.csv", index=False)
            saved.append(f"{a.out}_fairness.csv")
        print("\nSaved " + ", ".join(saved))
    if not (a.make_cohort or a.flag_map_template or a.tool_map_template or a.runs):
        ap.print_help()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")

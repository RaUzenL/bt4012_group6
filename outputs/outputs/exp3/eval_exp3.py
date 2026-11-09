"""Exp3 evaluation: LLM-agent investigations, scored on faithfulness, completeness, consistency and discrimination.

Usage
-----
1. Pick who to investigate (test applicants only): the riskiest by the Exp1 model plus a random control group.
       python outputs/exp3/eval_exp3.py --make-cohort exp1_test_scores.csv      # writes cohort.csv
2. Map each red flag to the field names YOUR agent's tools return (done once).
       python outputs/exp3/eval_exp3.py --flag-map-template                      # writes flag_to_fields.json
3. Run the agent (e.g. 3 runs per applicant) and save one JSON object per line (agent_runs.jsonl):
       {"SK_ID_CURR": 100054, "run": 1, "risk_score": 0.82, "rating": "high",
        "citations": [{"tool": "get_installment_history", "field": "late_rate", "value": 0.21}, ...],
        "tool_outputs": {"get_installment_history": {"late_rate": 0.214, ...}, ...}}
4. Evaluate:
       python outputs/exp3/eval_exp3.py agent_runs.jsonl --flag-map flag_to_fields.json --baseline exp1_test_scores.csv

Metrics
  faithfulness   share of cited values that match what the tools actually returned
  completeness   share of the applicant's rule-based red flags (red_flags.parquet) that the agent cited
  consistency    agreement of rating / risk score / cited evidence across repeated runs
  discrimination ROC-AUC / PR-AUC of the agent's risk score vs the hidden TARGET (optionally vs a baseline model)
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
CLEAN_DIR = HERE.parent / "clean_data"
SHOW = ["roc_auc", "pr_auc", "precision", "recall", "f1", "precision@top10%", "recall@top10%"]


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


def _top_k(y, s, frac):
    k = max(1, int(round(len(s) * frac)))
    hits = y[np.argsort(-s)[:k]].sum()
    return hits / k, hits / max(1, y.sum())


def _metrics(y, s, thr):
    yhat = (s >= thr).astype(int)
    out = {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
           "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
           "f1": f1_score(y, yhat)}
    for frac in (0.05, 0.10):
        out[f"precision@top{int(frac * 100)}%"], out[f"recall@top{int(frac * 100)}%"] = _top_k(y, s, frac)
    return out


def evaluate(y_test, s_test, y_val=None, s_val=None, threshold=None, n_boot=1000, seed=42):
    """Test metrics with bootstrap 95% CIs: ROC-AUC, PR-AUC, precision/recall/F1 at the threshold, and
    precision/recall among the riskiest 5% and 10%. Threshold = max-F1 on validation, or a fixed `threshold`."""
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
        for metric in ("roc_auc", "pr_auc"):
            c = compare_models(y, preds[lab]["test"]["score"], preds[base]["test"]["score"], metric, n_boot)
            comps.append({"model": lab, "vs_baseline": base, "metric": metric, "delta": c["delta"],
                          "ci_low": c["ci_low"], "ci_high": c["ci_high"], "p_bootstrap": c["p_bootstrap"],
                          "p_delong": c["delong_roc_auc"]["p_value"] if metric == "roc_auc" else np.nan})
    return table, pd.DataFrame(comps), preds


def pretty_metrics(table, metrics=SHOW):
    """Readable metrics table: 'value (95% CI low-high)' per metric, one row per model."""
    out = pd.DataFrame({"threshold": table["threshold"].round(3)}, index=table.index)
    for m in metrics:
        out[m] = [f"{v:.3f} ({lo:.3f}-{hi:.3f})"
                  for v, lo, hi in zip(table[m], table[f"{m}_ci_low"], table[f"{m}_ci_high"])]
    return out


def pretty_comparisons(comps):
    """Readable comparison table with a plain-language verdict (95% CI of the difference excludes 0)."""
    if comps.empty:
        return comps
    c = comps.copy()
    c["difference (95% CI)"] = [f"{d:+.4f} ({lo:+.4f} to {hi:+.4f})" for d, lo, hi in zip(c.delta, c.ci_low, c.ci_high)]
    c["p_value"] = [f"{pd_:.2g}" if metric == "roc_auc" else f"{pb:.2g}"
                    for pd_, pb, metric in zip(c.p_delong, c.p_bootstrap, c.metric)]
    c["verdict"] = np.where(c.ci_low > 0, "significantly better", np.where(c.ci_high < 0, "significantly worse",
                                                                           "no significant difference"))
    return c[["model", "vs_baseline", "metric", "difference (95% CI)", "p_value", "verdict"]]


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
RED_FLAGS_FILE, RULES_FILE = HERE / "red_flags.parquet", HERE / "red_flag_rules.json"
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


def evaluate_agent_runs(runs, flag_map=None, threshold=0.5, baseline_csv=None, n_boot=1000, clean_dir=CLEAN_DIR):
    """Score every investigated applicant and summarise. Returns (per_applicant table, summary dict).
    runs: list of run dicts or a .jsonl/.json path.  flag_map: dict or JSON path (None skips completeness)."""
    if isinstance(runs, (str, Path)):
        runs = read_runs(runs)
    if isinstance(flag_map, (str, Path)):
        flag_map = json.loads(Path(flag_map).read_text())
    flags = load_red_flags()
    rule_ids = [r[0] for r in RED_FLAG_RULES]
    by_app = {}
    for r in runs:
        by_app.setdefault(int(r["SK_ID_CURR"]), []).append(r)
    rows, all_cits = [], []
    for sid, rs in by_app.items():
        f = [faithfulness(r.get("citations", []), r.get("tool_outputs", {})) for r in rs]
        all_cits += [d for x in f for d in x["details"].to_dict("records")]
        present = [k for k in rule_ids if sid in flags.index and flags.at[sid, k] == 1]
        row = {"SK_ID_CURR": sid, "n_runs": len(rs), "risk_score": float(np.mean([r.get("risk_score", np.nan) for r in rs])),
               "rating": pd.Series([r.get("rating") for r in rs]).mode().iloc[0],
               "faithfulness": float(np.nanmean([x["faithfulness"] for x in f])), "n_red_flags": len(present)}
        if flag_map is not None:
            comp = [completeness([c.get("field") for c in r.get("citations", [])], present, flag_map) for r in rs]
            row["completeness"] = float(np.nanmean([c["completeness"] for c in comp])) if present else np.nan
            row["missed_flags"] = ", ".join(sorted(set.intersection(*[set(c["missed"]) for c in comp]))) if present else ""
        if len(rs) > 1:
            row.update(consistency(rs))
        rows.append(row)
    per_app = attach_split(pd.DataFrame(rows), clean_dir=clean_dir)
    cits = pd.DataFrame(all_cits)
    summary = {"applicants": len(per_app), "runs": len(runs),
               "faithfulness_mean": float(per_app.faithfulness.mean()),
               "citations_not_found_share": float((cits.status == "not_found").mean()) if len(cits) else np.nan,
               "citations_mismatch_share": float((cits.status == "mismatch").mean()) if len(cits) else np.nan}
    if "completeness" in per_app:
        summary["completeness_mean"] = float(per_app.completeness.mean())
    if "same_rating_all_runs" in per_app:
        summary["same_rating_all_runs_share"] = float(per_app.same_rating_all_runs.mean())
        summary["evidence_jaccard_mean"] = float(per_app.evidence_jaccard.mean())
        summary["risk_score_std_mean"] = float(per_app.risk_score_std.mean())
    if per_app.TARGET.nunique() == 2:
        res = evaluate(per_app.TARGET, per_app.risk_score, threshold=threshold, n_boot=n_boot)
        summary["discrimination"] = {m: {"value": float(r.value), "ci_low": float(r.ci_low), "ci_high": float(r.ci_high)}
                                     for m, r in res.loc[["roc_auc", "pr_auc", "precision", "recall"]].iterrows()}
        if baseline_csv:
            base = load_predictions(baseline_csv, clean_dir)[["SK_ID_CURR", "score"]]
            m = per_app.merge(base, on="SK_ID_CURR", how="inner")
            c = compare_models(m.TARGET, m.risk_score, m.score, metric="roc_auc", n_boot=n_boot)
            summary["agent_vs_baseline_roc_auc"] = {k: c[k] for k in ["delta", "ci_low", "ci_high", "p_bootstrap"]}
            summary["agent_vs_baseline_roc_auc"]["n"] = len(m)
    return per_app, summary


def summary_table(summary):
    """The evaluate_agent_runs summary as a two-column table (metric, value with 95% CI where available)."""
    rows = []
    for k, v in summary.items():
        if isinstance(v, dict) and "value" not in v and all(isinstance(x, dict) for x in v.values()):
            for m, r in v.items():
                rows.append((f"{k}: {m}", f"{r['value']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})"))
        elif isinstance(v, dict):
            rows.append((k, f"{v['delta']:+.3f} ({v['ci_low']:+.3f} to {v['ci_high']:+.3f}), p={v['p_bootstrap']:.2g}, n={v['n']}"))
        else:
            rows.append((k, f"{v:.3f}" if isinstance(v, float) else v))
    return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")


# ============================================================================================
# EXP3 COMMAND LINE
# ============================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="?", help="agent runs: .jsonl (one JSON per line) or .json list")
    ap.add_argument("--flag-map", help="JSON mapping red-flag id -> agent field names (needed for completeness)")
    ap.add_argument("--baseline", help="CSV SK_ID_CURR, score of a model to compare the agent's risk score with")
    ap.add_argument("--threshold", type=float, default=0.5, help="risk score above which the agent 'flags' (default 0.5)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--make-cohort", metavar="SCORES_CSV", help="write cohort.csv from a model's test scores")
    ap.add_argument("--k-flagged", type=int, default=100)
    ap.add_argument("--k-control", type=int, default=100)
    ap.add_argument("--flag-map-template", action="store_true", help="write flag_to_fields.json to fill in")
    ap.add_argument("--out", default=str(HERE / "exp3"), help="output file prefix (default outputs/exp3/exp3)")
    a = ap.parse_args(argv)

    if a.make_cohort:
        cohort = select_cohort(a.make_cohort, a.k_flagged, a.k_control)
        cohort.to_csv(HERE / "cohort.csv", index=False)
        print(f"Wrote {HERE / 'cohort.csv'}: {len(cohort)} test applicants ({a.k_flagged} flagged + {a.k_control} control)")
    if a.flag_map_template:
        tpl = {r: {"concept": c, "rule": d, "fields": []} for r, _, c, d in RED_FLAG_RULES}
        (HERE / "flag_to_fields.json").write_text(json.dumps(tpl, indent=2))
        print(f"Wrote {HERE / 'flag_to_fields.json'}: fill each 'fields' list with your tool field names")
    if a.runs:
        flag_map = json.loads(Path(a.flag_map).read_text()) if a.flag_map else None
        if flag_map is None:
            print("No --flag-map given: completeness is skipped.")
        per_app, summary = evaluate_agent_runs(read_runs(a.runs), flag_map, a.threshold, a.baseline, a.n_boot)
        print(json.dumps(summary, indent=2))
        per_app.to_csv(f"{a.out}_per_applicant.csv", index=False)
        Path(f"{a.out}_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSaved {a.out}_per_applicant.csv and {a.out}_summary.json")
    if not (a.make_cohort or a.flag_map_template or a.runs):
        ap.print_help()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")

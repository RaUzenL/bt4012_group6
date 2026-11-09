"""Exp1 evaluation: application-only vs multi-source models (ablation), all on the shared test set.

Usage
-----
Save one predictions CSV per model with columns SK_ID_CURR, score (val + test applicants; higher = riskier):

    python outputs/exp1/eval_exp1.py app_only.csv app_bureau.csv app_prev.csv app_repay.csv all_sources.csv \
        --labels App +Bureau +Prev +Repayment All

The first file is the baseline. Prints test metrics for every model and paired significance tests vs the baseline,
and saves exp1_metrics.csv and exp1_comparisons.csv next to this file.
In Python:  import eval_exp1 as ev;  ev.evaluate(y_test, p_test, y_val=y_val, s_val=p_val)
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
# EXP1 COMMAND LINE
# ============================================================================================
def main(argv=None):
    ap = _predictions_cli("Exp1: compare application-only vs multi-source models on the shared test set.",
                          "Example:\n  python eval_exp1.py app_only.csv all_sources.csv --labels App All")
    ap.add_argument("--out", default=str(HERE / "exp1"), help="output file prefix (default outputs/exp1/exp1)")
    a = ap.parse_args(argv)
    table, comps, _ = compare_prediction_files(a.predictions, a.labels, a.n_boot)
    _print_model_table(table, comps)
    table.to_csv(f"{a.out}_metrics.csv")
    comps.to_csv(f"{a.out}_comparisons.csv", index=False)
    print(f"\nSaved {a.out}_metrics.csv and {a.out}_comparisons.csv")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")

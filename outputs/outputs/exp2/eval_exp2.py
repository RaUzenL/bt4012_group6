"""Exp2 evaluation: aggregated vs temporal (sequence) models, overall and by history segment (RQ2.1 + RQ2.2).

Usage
-----
Save one predictions CSV per model with columns SK_ID_CURR, score (val + test applicants; higher = riskier):

    python outputs/exp2/eval_exp2.py aggregated_mlp.csv sequence_gru.csv --labels Aggregated Sequence

The first file is the baseline. Prints overall test metrics, paired significance tests vs the baseline, the same
metrics for each history segment (default groups in history_segments.parquet), and paired tests vs the baseline
inside each segment. To use your own groups, pass --segments my_groups.csv with columns SK_ID_CURR, segment.
Saves exp2_metrics.csv, exp2_comparisons.csv, exp2_by_segment.csv and exp2_segment_comparisons.csv next to this file.
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
# EXP2: HISTORY SEGMENTS (RQ2.2)
# ============================================================================================
SEGMENTS_FILE = HERE / "history_segments.parquet"


def history_segments(clean_dir=CLEAN_DIR):
    """Default RQ2.2 groups per applicant.
    HISTORY_SEGMENT: internal (Home Credit records) and/or external (credit bureau) history, or none.
    INTERNAL_RICHNESS: quartile of the number of internal records (cut-points from TRAIN applicants), or 'none'."""
    d = Path(clean_dir)
    sp = load_splits(d)
    ids = sp["SK_ID_CURR"]
    internal = sum(pd.read_parquet(d / f"{t}.parquet", columns=["SK_ID_CURR"])["SK_ID_CURR"].value_counts()
                   .reindex(ids, fill_value=0).to_numpy()
                   for t in ["previous_application", "installments", "pos_cash", "credit_card"])
    has_bureau = ids.isin(pd.read_parquet(d / "bureau.parquet", columns=["SK_ID_CURR"])["SK_ID_CURR"]).to_numpy()
    seg = pd.DataFrame({"SK_ID_CURR": ids, "split": sp["split"], "N_INTERNAL_RECORDS": internal,
                        "HAS_INTERNAL": internal > 0, "HAS_EXTERNAL": has_bureau})
    seg["HISTORY_SEGMENT"] = np.select(
        [seg.HAS_INTERNAL & seg.HAS_EXTERNAL, seg.HAS_INTERNAL, seg.HAS_EXTERNAL],
        ["internal+external", "internal_only", "external_only"], default="no_history")
    cuts = seg.loc[(seg.split == "train") & seg.HAS_INTERNAL, "N_INTERNAL_RECORDS"].quantile([.25, .5, .75]).to_numpy()
    q = np.searchsorted(cuts, seg["N_INTERNAL_RECORDS"], side="right") + 1
    seg["INTERNAL_RICHNESS"] = np.where(seg.HAS_INTERNAL, np.char.add("Q", q.astype(str)), "none")
    return seg.drop(columns="split")


def load_segments(custom_csv=None):
    """Default groups from history_segments.parquet (built if missing), or your own CSV (SK_ID_CURR, segment)."""
    if custom_csv:
        s = pd.read_csv(custom_csv)
        return s[["SK_ID_CURR", "segment"]].rename(columns={"segment": "CUSTOM_SEGMENT"})
    if not SEGMENTS_FILE.exists():
        history_segments().to_parquet(SEGMENTS_FILE, index=False)
    return pd.read_parquet(SEGMENTS_FILE)


def evaluate_by_segment(y, s, segment, threshold, min_positives=20):
    """Metrics per segment at one fixed threshold (chosen on validation). Segments with < min_positives defaulters
    get only n and default rate, because AUC on a handful of positives is meaningless."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "s": np.asarray(s, float), "seg": np.asarray(segment)})
    rows = {}
    for g, part in df.groupby("seg"):
        rows[g] = {"n": len(part), "defaulters": int(part.y.sum()), "default_rate": part.y.mean()}
        if min_positives <= part.y.sum() < len(part):
            rows[g].update(_metrics(part.y.to_numpy(), part.s.to_numpy(), threshold))
    return pd.DataFrame(rows).T


def segment_table(preds, segments, columns):
    """Per model x segment metrics on the test set, each model at its own validation threshold."""
    out = []
    for model, p in preds.items():
        te = p["test"].merge(segments, on="SK_ID_CURR", how="left")
        for col in columns:
            t = evaluate_by_segment(te.TARGET, te.score, te[col].fillna("unknown"), p["threshold"])
            out.append(t.assign(model=model, grouping=col).rename_axis("segment").reset_index())
    return pd.concat(out, ignore_index=True)


def segment_pivot(by_seg, metric="roc_auc"):
    """One row per segment, one column per model, for a chosen metric (plus segment size and default rate)."""
    base = by_seg.drop_duplicates(["grouping", "segment"]).set_index(["grouping", "segment"])[["n", "defaulters", "default_rate"]]
    piv = by_seg.pivot_table(index=["grouping", "segment"], columns="model", values=metric, sort=False)
    return base.join(piv.add_prefix(f"{metric} | "))


def segment_comparisons(preds, segments, columns, n_boot=1000, min_positives=20):
    """RQ2.2: inside each segment, paired difference of each model vs the baseline (first model) in PR-AUC and
    ROC-AUC, with bootstrap 95% CI. p_holm adjusts across ALL segment tests, because many are run at once.
    Segments with < min_positives defaulters are skipped."""
    labels = list(preds)
    te = preds[labels[0]]["test"].merge(segments, on="SK_ID_CURR", how="left")   # every model's test rows share this order
    y = te.TARGET.to_numpy()
    rows = []
    for col in columns:
        seg = te[col].fillna("unknown").to_numpy()
        for g in sorted(set(seg)):
            mask = seg == g
            if not min_positives <= y[mask].sum() < mask.sum():
                continue
            for lab in labels[1:]:
                for metric in ("pr_auc", "roc_auc"):
                    c = compare_models(y[mask], preds[lab]["test"]["score"].to_numpy()[mask],
                                       preds[labels[0]]["test"]["score"].to_numpy()[mask], metric, n_boot)
                    rows.append({"grouping": col, "segment": g, "n": int(mask.sum()), "defaulters": int(y[mask].sum()),
                                 "model": lab, "vs_baseline": labels[0], "metric": metric, "delta": c["delta"],
                                 "ci_low": c["ci_low"], "ci_high": c["ci_high"], "p_bootstrap": c["p_bootstrap"],
                                 "p_delong": c["delong_roc_auc"]["p_value"] if metric == "roc_auc" else np.nan})
    return _add_p_values(pd.DataFrame(rows))


# ============================================================================================
# EXP2 COMMAND LINE
# ============================================================================================
def main(argv=None):
    ap = _predictions_cli("Exp2: aggregated vs sequence models, overall and per history segment.",
                          "Example:\n  python eval_exp2.py aggregated.csv sequence.csv --labels Aggregated Sequence")
    ap.add_argument("--segments", help="optional CSV with your own groups: SK_ID_CURR, segment")
    ap.add_argument("--out", default=str(HERE / "exp2"), help="output file prefix (default outputs/exp2/exp2)")
    a = ap.parse_args(argv)
    table, comps, preds = compare_prediction_files(a.predictions, a.labels, a.n_boot)
    _print_model_table(table, comps)
    seg = load_segments(a.segments)
    cols = ["CUSTOM_SEGMENT"] if a.segments else ["HISTORY_SEGMENT", "INTERNAL_RICHNESS"]
    by_seg = segment_table(preds, seg, cols)
    show = ["grouping", "segment", "model", "n", "defaulters", "default_rate", "roc_auc", "pr_auc", "recall"]
    print("\nTest-set metrics by segment")
    print(by_seg[[c for c in show if c in by_seg]].round(4).to_string(index=False))
    seg_comps = segment_comparisons(preds, seg, cols, a.n_boot) if len(preds) > 1 else pd.DataFrame()
    if len(seg_comps):
        print("\nPaired comparison vs baseline inside each segment (Holm-adjusted across all segment tests)")
        print(pretty_comparisons(seg_comps).to_string(index=False))
    table.to_csv(f"{a.out}_metrics.csv")
    comps.to_csv(f"{a.out}_comparisons.csv", index=False)
    by_seg.to_csv(f"{a.out}_by_segment.csv", index=False)
    seg_comps.to_csv(f"{a.out}_segment_comparisons.csv", index=False)
    print(f"\nSaved {a.out}_metrics.csv, {a.out}_comparisons.csv, {a.out}_by_segment.csv and "
          f"{a.out}_segment_comparisons.csv")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")

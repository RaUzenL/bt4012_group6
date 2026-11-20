# bt4012_group6

## Background

This project investigates whether historical credit and repayment behaviour improves the identification of high-risk loan applications beyond application information alone, and whether those records can support a traceable investigation for a human underwriter.

We use the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset, with a shared sample of **100,000 applicants** and fixed training, validation and test splits. The target records **repayment difficulties**, so the findings concern credit risk rather than confirmed fraud. Because only about 8% of applicants have repayment difficulties, PR-AUC is the primary ranking metric, with ROC-AUC and other measures reported alongside it.

The project contains three experiments:

- **Experiment 1 — Multi-source feature engineering:** compare logistic regression and histogram gradient boosting, identify the contribution of historical sources, and examine feature selection and demographic inclusion.
- **Experiment 2 — Temporal credit-risk modelling:** test whether handcrafted temporal features and LSTM-derived risk scores add information beyond aggregated historical features, including robustness and history-richness analyses.
- **Experiment 3 — Agent-assisted investigation:** use a tool-calling LLM to retrieve applicant evidence and produce cited investigation notes; evaluate evidence quality, consistency and risk discrimination.

**[`data_cleaning.ipynb`](data_cleaning.ipynb) contains the shared data-preprocessing steps—applicant sampling, data cleaning and train/validation/test splitting. It writes the cleaned tables to `clean_data/`.**

## Project layout

```
bt4012_group6/
├── requirements.txt      shared dependencies (all experiments)
├── data_cleaning.ipynb   sampling, cleaning, splitting -> clean_data/
├── clean_data/           shared cleaned data (all experiments)
├── Exp_1/                exp1.ipynb, evaluate_exp1.ipynb (+ eval_exp1.py evaluation file)
├── Exp_2/                training notebook, evaluate_exp2 copy.ipynb (+ eval_exp2.py), history_segments.parquet
└── Exp_3/                exp3.ipynb, evaluate_exp3.ipynb (+ eval_exp3.py), data/red_flags.parquet, red_flag_rules.json
```

Each `eval_expN.py` finds `clean_data/` one level up from its own folder, so keep this layout.

## Shared data (`clean_data/`)

**Everyone uses these files; do not re-sample or re-split.**

| File | Contents |
|---|---|
| `splits.csv` | the 100,000 sampled applicants and their phase: `SK_ID_CURR, TARGET, N_PREV_APPS, PREV_BUCKET, split, is_dev` |
| `application.parquet` | application_train, sampled + cleaned (has TARGET) |
| `bureau.parquet`, `bureau_balance.parquet` | loans at other lenders + their monthly status (`STATUS_DPD` added) |
| `previous_application.parquet` | past Home Credit applications (`IS_DUPLICATE_SUBMISSION` added) |
| `installments.parquet` | one row per scheduled instalment: amount due vs **total** paid, due date vs last payment date, `N_PAYMENTS` (split payments merged) |
| `pos_cash.parquet`, `credit_card.parquet` | monthly POS/cash-loan and credit-card snapshots |
| `cleaning_report.csv` | every cleaning rule and how many rows it changed |

Sample: stratified on TARGET x number of previous applications; every table filtered to the sampled applicants.
Split: train 70% / val 15% / test 15% (same stratification). `is_dev` = 20% of each split for quick trial runs.

Rules for all experiments:
1. Fit anything learned from data (imputation, scaling, encoders) on **train** only.
2. Tune on **val**; report final numbers on **test** using your experiment's evaluation file.
3. Missing values are NaN on purpose. A missing history table means the applicant has no such history.

**Protected attributes.** `application.parquet` still contains `CODE_GENDER`, `NAME_FAMILY_STATUS` and `DAYS_BIRTH`
(age). Exp3 does not pass gender or marital status to the agent's tools. Exp1/Exp2 state whether their models use
them; Exp1 reports the best model with and without them.

## Evaluation metrics

**Exp1** (`Exp_1/eval_exp1.py`)
| Metric | Role | Why |
|---|---|---|
| PR-AUC | **primary** | only ~8% of applicants default; PR-AUC shows how many flagged applicants really default (8% = random) |
| ROC-AUC | secondary | ranking quality; unaffected by the default rate; standard in credit scoring |
| Precision / recall / F1 | operating point | at one threshold per model, chosen on **val** by max F1. Depends on where each threshold lands, so do not use it to compare models |
| Precision / recall @ top 5% and 10% | review queue | how many defaulters are caught if only the riskiest 5% / 10% are reviewed (added by `evaluate_exp1.ipynb`); descriptive, not tested |
| Paired tests vs the baseline | significance | bootstrap 95% CI of the difference; DeLong p-value for ROC-AUC, bootstrap p-value for PR-AUC. The verdict uses the **Holm-adjusted** p-value (`p_holm`) across the 16 planned tests |

**Exp2** (`Exp_2/eval_exp2.py`, same test applicants as Exp1)
| Metric | Role | Why |
|---|---|---|
| PR-AUC | **primary** | only ~8% of applicants default; PR-AUC shows how many flagged applicants really default (8% = random) |
| ROC-AUC | secondary | ranking quality; unaffected by the default rate, so it compares fairly across history segments; standard in credit scoring |
| Precision / recall / F1 | operating point | at one threshold per model, chosen on **val** by max F1. Depends on where each threshold lands, so do not use it to compare models |
| Precision / recall @ top 5% and 10% | review queue | how many defaulters are caught if only the riskiest 5% / 10% of applicants are reviewed; descriptive, not tested |
| Paired tests | significance | bootstrap 95% CI of the difference; DeLong p-value for ROC-AUC, bootstrap p-value for PR-AUC. The verdict uses the **Holm-adjusted** p-value (`p_holm`) across the 18 planned tests (9 contrasts x 2 metrics) |

Exp2 also reports every metric per history segment (RQ2.2). Segments use the model's single threshold; do not tune
one per segment. Segment-level paired tests vs M0 (segments with at least 20 defaulters) are Holm-adjusted across all
segment tests.

**Exp3** (LLM agent; lecture 12: a correct process does not guarantee a correct result, so both are measured)
| Level | Metrics |
|---|---|
| Process | tool recall, argument accuracy, gate adherence (must be 100%), faithfulness, completeness vs the rule-based red flags, uncertainty honesty, consistency across runs |
| Output | LLM-as-judge rubric (grounding is blocking, hedging, completeness), checked against ~20 human grades |
| Outcome | **main:** PR-AUC / ROC-AUC of the risk score and precision / recall of the `high` rating **within the flagged group** (top 100 by Exp1), vs its default rate and paired vs the Exp1 score; over-flagging in the control group; whole cohort (secondary); calibration and fairness (descriptive); cost per run |

The cohort was chosen with the Exp1 score and is enriched with defaulters, so whole-cohort AUCs are not comparable
with Exp1/Exp2 and give Exp1 a head start; that is why the flagged group is the main result. Completeness uses the
rule-based red flags (`Exp_3/data/red_flag_rules.json`), not SHAP, so the agent is not rewarded for repeating the model.

Each `eval_expN.py` also runs from the terminal (`python Exp_N/eval_expN.py --help`).

## How to Reproduce Results

### Experiment 1: Multi-Source Feature Engineering

#### Setup

1. Clone the repository and use **Python 3.13.1**.
2. Install the Exp1 dependencies from the repository root:

   ```bash
   pip install -r Exp_1/requirements.txt
   ```

3. Select the Python environment containing these packages as the notebook kernel.

Keep the supplied files in `clean_data/` and `Exp_1/exp1_outputs/` in their existing locations.

#### Reproduce the final results

Open [Exp_1/evaluate_exp1.ipynb](Exp_1/evaluate_exp1.ipynb) and **Run All** with these default settings:

```python
RUN_FINAL_COMPARISON = True
N_BOOT = 1000
SAVE = True
```

The notebook evaluates **nine model configurations** and displays performance tables, paired significance tests and ROC/precision–recall curves. It retrains seven configurations and loads the two saved all-sources variants; it does not repeat tuning or feature selection.

| Output | Contents |
| --- | --- |
| [exp1_metrics.csv](Exp_1/exp1_outputs/exp1_metrics.csv) | Nine models: performance metrics, confidence intervals and review-queue metrics |
| [exp1_comparisons.csv](Exp_1/exp1_outputs/exp1_comparisons.csv) | Eight paired model contrasts across two metrics, with Holm-adjusted p-values |

#### Inspect or reproduce model tuning

Open [Exp_1/exp1.ipynb](Exp_1/exp1.ipynb) to inspect the original tuning code and saved results. To rerun the complete workflow, execute the sections in this order:

| Order | Sections | Purpose |
| --- | --- | --- |
| 1 | **0–5** | Feature construction, gradient-boosting grid search, logistic-regression tuning, source ablation and feature selection |
| 2 | **7** | Generate demographic sensitivity scores and settings |
| 3 | **6** | Generate the complete final comparison |

**Run Section 7 before Section 6:** the final comparison reads the scores and settings generated by the sensitivity analysis.

### Experiment 2: Temporal Credit-Risk Modelling

#### Setup

1. Use **Python 3.13** and install the shared dependencies plus PyTorch from the repository root:

   ```bash
   pip install -r requirements.txt
   pip install torch
   ```

2. Select the Python environment containing these packages as the notebook kernel.

Keep the supplied files in `clean_data/` and `Exp_2/` in their existing locations.

#### Reproduce the final results

Open [Exp_2/evaluate_exp2 copy.ipynb](Exp_2/evaluate_exp2%20copy.ipynb) and **Run All** with these default settings:

```python
N_BOOT = 1000
SAVE = True
```

The notebook reads the six saved prediction CSVs from `Exp_2/experiment2_outputs/`, builds three fusion models from the M0–M3 scores (validation-AUC weighted average, M0+M2 blend, meta-learner), and computes test-set metrics, the nine planned paired contrasts with Holm-adjusted p-values, and a segment breakdown by history availability and richness. It calls no LSTM/GBDT training; the segment comparisons take a few minutes.

| Output | Contents |
| --- | --- |
| [exp2_metrics.csv](Exp_2/exp2_metrics.csv) | Nine models (six GBDT variants + three fusions): performance metrics with 95% bootstrap confidence intervals |
| [exp2_comparisons.csv](Exp_2/exp2_comparisons.csv) | Nine planned contrasts across PR-AUC and ROC-AUC, with Holm-adjusted p-values |
| [exp2_by_segment.csv](Exp_2/exp2_by_segment.csv) | Metrics broken down by history segment and internal richness quartile |
| [exp2_segment_comparisons.csv](Exp_2/exp2_segment_comparisons.csv) | Paired comparisons vs M0 inside each segment, with Holm-adjusted p-values |

#### Inspect or reproduce model training

Open [Exp_2/BT4012_Experiment2_V6_Application_LSTM_Controls copy.ipynb](Exp_2/BT4012_Experiment2_V6_Application_LSTM_Controls%20copy.ipynb) to inspect the full workflow: feature engineering, LSTM architecture, out-of-fold scoring, GBDT training and all model variants. Execute the sections in this order:

| Order | Sections | Purpose |
| --- | --- | --- |
| 1 | **1–4** | Load clean data, rebuild the Exp1 feature matrix, build multi-source monthly sequences, normalise |
| 2 | **5–6** | Define the masked-attention LSTM and validate standalone vs ordinary LSTM |
| 3 | **7** | Generate out-of-fold LSTM risk scores |
| 4 | **8** | Train six GBDT model variants (M_app, M_app+LSTM, M0–M3) and export prediction CSVs |
| 5 | **9–12** | Diagnostics: time-order ablation, richness analysis, richness-aware fusion, bootstrap uncertainty |

The notebook uses a GPU/MPS if available and falls back to CPU. A full run takes roughly 15–30 minutes on CPU. The six prediction CSVs (`v6_pred_*.csv`) are written to `Exp_2/experiment2_outputs/` and are read by `evaluate_exp2 copy.ipynb`.

### Experiment 3: Agent-Assisted Investigation

#### Setup

1. Use **Python 3.13** and install the shared and Exp3 dependencies from the repository root:

   ```bash
   pip install -r requirements.txt -r Exp_3/requirements.txt
   ```

2. Select the Python environment containing these packages as the notebook kernel.

Keep the supplied files in `clean_data/`, `Exp_1/exp1_outputs/` and `Exp_3/` (`data/`, `prompts/`, `runs/`) in their existing locations. [Exp_3/runs/agent_runs.jsonl](Exp_3/runs/agent_runs.jsonl) holds the 600 saved investigations (200 test applicants, 3 runs each) that all results are computed from.

#### Reproduce the final results (no API key needed)

Open [Exp_3/evaluate_exp3.ipynb](Exp_3/evaluate_exp3.ipynb) and **Run All** with these default settings:

```python
N_BOOT = 1000
SAVE = True
```

The notebook scores the saved investigations with the team's evaluation file ([Exp_3/eval_exp3.py](Exp_3/eval_exp3.py)): process metrics (tool recall, faithfulness, red-flag completeness, consistency), the LLM judge's and a person's grades of the case notes, and risk discrimination in the flagged group (main result), the control group and the whole cohort, with calibration and a fairness audit. It calls no LLM and takes about a minute.

| Output | Contents |
| --- | --- |
| [exp3_summary.json](Exp_3/results/exp3_summary.json) | All metrics, with 95% bootstrap confidence intervals and the paired comparison with the Exp1 score |
| [exp3_per_applicant.csv](Exp_3/results/exp3_per_applicant.csv) | One row per applicant, averaged over the three runs |
| [exp3_calibration.csv](Exp_3/results/exp3_calibration.csv) | Reliability table of the agent's risk score |
| [exp3_fairness.csv](Exp_3/results/exp3_fairness.csv) | Flag rate and false-positive rate by gender and age band |

#### Inspect or re-run the agent

Open [Exp_3/exp3.ipynb](Exp_3/exp3.ipynb) to inspect the six tools, the written red-flag policy, the agent, one full investigation with its audit log, the extra checks, the earlier agent versions and the choice of LLM. Running it needs a key for the NUS SoC LLM service (SoCLaaS), saved in `project/.env` as `SOCLAAS_API_KEY=...` (this file is git-ignored).

- **Run All** reuses the saved runs and grades: finished investigations are skipped, so the only LLM call is one demonstration investigation. It takes about 5 minutes, mostly a check of the scoring rule on 3,000 applicants.
- To repeat the investigations from scratch, move the `agent_*` files out of `Exp_3/runs/` first. The full run (600 investigations) takes about an hour when the service is quiet and several hours when it is busy, the judge grades the new notes again, and the 20 human grades must be redone. LLM output is not fully repeatable, even at temperature 0.
- Run `exp3.ipynb` before `evaluate_exp3.ipynb`: its last section writes the judge's grades and the two maps the evaluation reads.

The agent's safety checks are tested without any API call (66 tests, about 10 seconds):

```bash
python Exp_3/test_agent.py
```

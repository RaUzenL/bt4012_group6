# bt4012_group6

## Background

This project investigates whether historical credit and repayment behaviour improves the identification of high-risk loan applications beyond application information alone, and whether those records can support a traceable investigation for a human underwriter.

We use the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset, with a shared sample of **100,000 applicants** and fixed training, validation and test splits. The target records **repayment difficulties**, so the findings concern credit risk rather than confirmed fraud. Because only about 8% of applicants have repayment difficulties, PR-AUC is the primary ranking metric, with ROC-AUC and other measures reported alongside it.

The project contains three experiments:

- **Experiment 1 — Multi-source feature engineering:** compare logistic regression and histogram gradient boosting, identify the contribution of historical sources, and examine feature selection and demographic inclusion.
- **Experiment 2 — Temporal credit-risk modelling:** test whether handcrafted temporal features and LSTM-derived risk scores add information beyond aggregated historical features, including robustness and history-richness analyses.
- **Experiment 3 — Agent-assisted investigation:** use a tool-calling LLM to retrieve applicant evidence and produce cited investigation notes; evaluate evidence quality, consistency and risk discrimination.

## Repository Structure

```text
bt4012_group6/
├── README.md
├── eda.ipynb                         # Initial exploratory data analysis
├── outputs/
│   ├── data_cleaning.ipynb           # Shared sampling, cleaning and evaluation preparation
│   └── outputs/
│       ├── README.md                 # Shared data and evaluation handover
│       ├── requirements.txt          # Shared framework dependencies
│       ├── clean_data/               # Cleaned tables, fixed splits and cleaning log
│       ├── exp1/                     # Original shared Exp1 evaluation notebook and engine
│       ├── exp2/                     # Original shared Exp2 evaluator and history segments
│       └── exp3/                     # Original shared Exp3 evaluator and red-flag references
├── Exp_1/                            # Tabular models, feature engineering and source ablation
│   ├── exp1.ipynb                    # Full workflow, tuning and feature selection
│   ├── exp1_features.ipynb           # Focused historical-feature construction and source ablation
│   ├── evaluate_exp1.ipynb           # Complete nine-configuration evaluation
│   ├── requirements.txt             # Dependencies pinned for the reported Exp1 results
│   ├── exp1_outputs/                 # Derived features, metadata, saved scores and result tables
│   └── exp1_test_scores.csv           # Existing baseline score file used by Exp3
├── Exp_2/                            # Temporal modelling, fusion and robustness experiments
│   ├── BT4012_Experiment2_Robustness_v8 copy.ipynb
│   ├── evaluate_exp2.ipynb           # Overall and history-segment evaluation
│   ├── v6_pred_*.csv                 # Saved model predictions
│   ├── history_segments.parquet      # History-availability and richness groups
│   └── exp2_*.csv                    # Metrics, paired comparisons and segment results
└── Exp_3/                            # LLM investigation workflow
    ├── exp3.ipynb                    # Agent workflow, model checks and evaluation
    ├── exp3_agent.py                 # Agent orchestration, evidence checks and scoring
    ├── exp3_tools.py                 # Six structured evidence-retrieval tools
    ├── exp3_eval.py                  # Agent-specific evaluation helpers and additional checks
    ├── requirements.txt             # Additional agent dependencies
    ├── prompts/                     # Agent prompt and written investigation policy
    ├── data/                        # Investigation cohorts, scoring table and field mappings
    ├── runs/                        # Saved investigations, audit logs and grading records
    └── results/                     # Agent metrics and applicant-level result tables
```

### Shared data and evaluation: `outputs/`

The outer `outputs/` folder holds the shared data-cleaning notebook. Its nested `outputs/outputs/` folder supplies the cleaned inputs and evaluation framework used by all three experiments. In particular, `clean_data/splits.csv` fixes the applicants assigned to each split, and the experiment-specific evaluation folders implement the shared metrics and tests. **Keep this nested structure and reuse the supplied splits** so that the evaluators can locate their data and comparisons use the same applicant partitions. See the [shared handover notes](outputs/outputs/README.md) for details.

### Experiment code and results

| Experiment | Main implementation | Evaluation and saved results |
| --- | --- | --- |
| **Exp1** | [exp1.ipynb](Exp_1/exp1.ipynb): feature engineering, tuning, source ablation and feature selection | [evaluate_exp1.ipynb](Exp_1/evaluate_exp1.ipynb); tables and reusable inputs in `Exp_1/exp1_outputs/` |
| **Exp2** | [Temporal modelling notebook](<Exp_2/BT4012_Experiment2_Robustness_v8 copy.ipynb>): monthly sequences, LSTM fusion and robustness checks | [evaluate_exp2.ipynb](Exp_2/evaluate_exp2.ipynb); `exp2_metrics.csv`, `exp2_comparisons.csv` and `exp2_by_segment.csv` in `Exp_2/` |
| **Exp3** | [exp3.ipynb](Exp_3/exp3.ipynb): evidence retrieval, investigation notes and agent evaluation | Saved investigations in `Exp_3/runs/`; evaluation summaries, calibration, fairness and applicant-level results in `Exp_3/results/` |

Exp3 selects its investigation cohort using `Exp_1/exp1_test_scores.csv`. This is a separate existing score file from the updated `Exp_1/exp1_outputs/best_model_scores.csv`; changing the latter does not automatically update the saved Exp3 cohort. Exp3 is evaluated on its investigation cohort, so its aggregate metrics should not be directly compared with full-test-set Exp1/Exp2 metrics without accounting for the different applicants.

## How to Reproduce Results

### Experiment 1: Multi-Source Feature Engineering

#### Setup

1. Clone the repository and use **Python 3.13.1**.
2. Install the Exp1 dependencies from the repository root:

   ```bash
   pip install -r Exp_1/requirements.txt
   ```

3. Select the Python environment containing these packages as the notebook kernel.

Keep the supplied files in `outputs/outputs/clean_data/` and `Exp_1/exp1_outputs/` in their existing locations.

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

### Experiment 2

### Experiment 3

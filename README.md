# bt4012_group6

## Background

## Repo Structure

## How to Reproduce Results

### Experiment 1

1. Clone the repository and use Python 3.13.1.

2. Install dependencies from the repository root:

`pip install -r Exp_1/requirements.txt`

3. Open Exp_1/evaluate_exp1.ipynb and Run All with the default settings. Performance tables, significance tests and plots appear in the notebook.

Results are saved to Exp_1/exp1_outputs/exp1_metrics.csv and exp1_comparisons.csv. Keep the supplied data and output files in their existing folders.

*Inspection on the Model Tuning:*

Open Exp_1/exp1.ipynb to inspect the original tuning code and saved results.

To rerun the complete workflow:
1. Run Sections 0–5: feature construction, gradient-boosting grid search, logistic-regression tuning, source ablation and feature selection.

2. Run Section 7 to generate the demographic sensitivity scores.

3. Run Section 6 to generate the complete final comparison.

Section 7 precedes Section 6 because the final comparison reads its generated scores.


### Experiment 2

### Experiment 3

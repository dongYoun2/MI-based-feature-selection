# MI-based Feature Selection

An empirical study of mutual information-based feature selection methods (e.g. mRMR, CMIM, JMI, PID) against standard baselines (e.g.correlation
filtering, L1, RFE, SHAP) on tabular benchmarks (Communities & Crime, Lending Club, NHANES).

## Basic Structure

```
.
├── data/                   # raw + optional processed CSV cache
├── src/
│   ├── data/               # dataset loaders & cleaning
│   ├── feature_selection/  # mi.py, baselines.py
│   ├── cli.py
│   ├── experiment.py       # CV pipeline
│   ├── models.py
│   └── evaluate.py
├── preprocess.py           # writes processed cache used by main
├── main.py
├── notebooks/              # includes EDA to draw insights for preprocessing decisions
├── results/                # saved metrics, plots
├── requirements.txt
```

## Setup

### With Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### With Conda
```bash
conda env create -f environment.yaml
conda activate mi-feature
```


## Dataset Configuration

Raw datasets live under `data/raw/<dataset_dir>/` and are loaded from `src/data/`.
Download each dataset from the link below and arrange the files exactly as
shown under "Expected layout".

### Communities & Crime

- **Download link:** https://archive.ics.uci.edu/dataset/183/communities+and+crime
- **Files to keep:** `communities.data`, `communities.names`
- **Target directory:** `data/raw/communities/`
- **Notes:** target column is `ViolentCrimesPerPop` (regression); missing values are encoded as `?`.

### Lending Club

- **Download link:** https://www.kaggle.com/datasets/wordsforthewise/lending-club (accepted loans, 2007-2018Q4).
- **Files to keep:** `accepted_2007_to_2018Q4.csv`
- **Target directory:** `data/raw/lending_club/`
- **Notes:** target is `loan_status`, reduced to `Charged Off`/`Default` (1) vs. `Fully Paid` (0); ~2.2M rows; the loader randomly samples `nrows` (default 5k) across the full 2007-2018 window — seeded by `random_state` — since the CSV is sorted by `issue_d` and head-N would only see 2015 issues; performed offline preprocessing by running `python preprocess.py --dataset lending_club --nrows 5000`; default rate ~21%.

### NHANES 2013-2014

- **Download link:** https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey (use the 2013-2014 cycle).
- **Files to keep:** `demographic.csv`, `examination.csv`, `questionnaire.csv`
- **Target directory:** `data/raw/nhanes/`
- **Notes:** target column is `DIQ010` (diabetes diagnosis) from `questionnaire.csv`, joined on `SEQN`; classes 1/2 are mapped to 1/0, others are dropped; positive rate ~8%.


### Sample and feature counts

`(samples, features)` — **Raw** = stage 1 load (raw data loading); **Preprocessed** = stages 1+2 (after dataset-level cleaning; before per-fold model-level preprocessing). Lending Club raw is the full resolved-loan pool; experiments default to `nrows=5_000`.

|                  | Communities  | Lending Club |    NHANES    |
| :--------------: | :----------: | :----------: | :----------: |
|     **Raw**      | (1,994, 127) | (~2.2M, 150) | (9,236, 270) |
| **Preprocessed** | (1,994, 100) | (5,000, 65)  | (9,236, 112) |


## Run Experiments

```bash
python main.py \
    --dataset <dataset> \
    --selectors correlation l1 rfe shap mi mrmr_heuristic pid cmim jmi \
    --models logreg gradient_boosting svm \
    --ks 5 10 20 40 \
    --cv-folds 5
```

Each run writes to `results/<YYYY-MM-DD_HH-MM-SS>/`:

- `metrics_<dataset>.csv` — aggregated summary (mean/std of eval metrics across folds).
- `metrics_<dataset>_per_fold.csv` — one row per (fold, selector, k, model), including the list of selected feature names.
- `<dataset>.png` — metric vs. k plots (mean ± std).

## Preprocessed cache

Caches load + dataset cleaning to CSV; fold preprocessing still runs per CV fold. Example:

```bash
python preprocess.py --dataset lending_club --nrows 5000
```

Writes `data/processed/lending_club/lending_club.csv`. `main.py` picks it up automatically.

## SVM Grid Search

SVM hyperparameter grid search; full grid of `(C, gamma)` pairs for each selector and `k`. Example (**3 × 3 = 9** SVMs):

```bash
python svm_param_search.py \
  --dataset communities \
  --selectors correlation l1 rfe shap mi mrmr_heuristic pid cmim jmi \
  --models svm \
  --ks 5 10 20 40 \
  --cv-folds 5 \
  --svm-c-grid 0.1 1 10 \
  --svm-gamma-grid scale 0.01 0.1 \
  --out-dir results/_svm_grid
```
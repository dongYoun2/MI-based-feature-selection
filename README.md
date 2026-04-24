# MI-based Feature Selection

An empirical study of mutual information-based feature selection methods
(mRMR, PID-based, CMI-based) against standard baselines (correlation
filtering, L1 regularization, RFE, SHAP) on tabular benchmarks
(Communities & Crime, Lending Club, NHANES).

## Basic Structure

```
.
├── data/
├── src/
│   ├── data.py             # load + preprocess
│   ├── feature_selection/
│   │   ├── mi.py           # mRMR, CMI, PID, ...
│   │   └── baselines.py    # correlation, L1, RFE, SHAP, ...
│   ├── models.py           # train models
│   ├── evaluate.py         # metrics (accuracy, AUC, ...)
│   └── experiment.py       # main pipeline
├── notebooks/
├── results/                # saved metrics, plots
├── main.py                 # run experiments
└── requirements.txt
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

Raw datasets live under `data/raw/<dataset_dir>/` and are loaded by `src/data.py`.
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
- **Notes:** target is `loan_status`, reduced to `Charged Off`/`Default` (1) vs. `Fully Paid` (0); ~2.2M rows, the loader samples 100k by default.

### NHANES 2013-2014

- **Download link:** https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey (use the 2013-2014 cycle).
- **Files to keep:** `demographic.csv`, `examination.csv`, `questionnaire.csv`
- **Target directory:** `data/raw/nhanes/`
- **Notes:** target column is `DIQ010` (diabetes diagnosis) from `questionnaire.csv`, joined on `SEQN`; classes 1/2 are mapped to 1/0, others are dropped.

## Run

```bash
python main.py --datasets communities --selectors mrmr correlation l1 rfe \
    --models logreg random_forest gradient_boosting --ks 5 10 20 40
```

Each run writes to `results/<YYYY-MM-DD_HH-MM-SS>/metrics.csv`.

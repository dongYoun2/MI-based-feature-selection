# MI-based Feature Selection

An empirical study of mutual information-based feature selection methods
(mRMR, PID-based, CMI-based) against standard baselines (correlation
filtering, L1 regularization, RFE, SHAP) on tabular benchmarks
(Communities & Crime, Lending Club, NHANES).

## Structure

```
.
├── data/                   # raw + processed datasets
├── src/
│   ├── data.py             # load + preprocess
│   ├── feature_selection/
│   │   ├── mi.py           # mRMR, CMI, PID
│   │   └── baselines.py    # correlation, L1, RFE, SHAP
│   ├── models.py           # train models
│   ├── evaluate.py         # metrics (accuracy, AUC, ...)
│   └── experiment.py       # main pipeline
├── notebooks/
│   ├── exploration.ipynb
│   └── results.ipynb
├── results/                # saved metrics, plots
├── main.py                 # run experiments
└── requirements.txt
```

## Setup

### With virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### With conda
```bash
conda env create -f environment.yaml
conda activate mi-feature
```


## Run

```bash
python main.py --datasets communities --selectors mrmr correlation l1 rfe \
    --models logreg random_forest gradient_boosting --ks 5 10 20 40
```

Results are written to `results/metrics.csv`.

# INSE 6450 - Credit Card Fraud Detection

Milestone 1 repository for the Credit Card Fraud Detection project.

## Raw data

Download the **Credit Card Transactions Fraud Detection** dataset and place:

- `fraudTrain.csv`
- `fraudTest.csv`

inside `data/raw/`.

The raw CSV files are intentionally not committed to the repository because of their size.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run Milestone 1 analysis

```bash
python src/eda.py
python src/feature_engineering.py
```

Generated figures and summary tables are written to `outputs/`. Engineered datasets are written to `data/processed/`.

## Dataset handling decisions

- Raw files are never edited in place.
- Validation/test data are not resampled.
- Direct identifiers are not used as ordinary predictive features.
- Time-derived and geographic features are calculated from information available at transaction time.
- Feature engineering is kept reproducible so the same transformations can be reused in Milestone 2.

# DemandPilot — Hourly Mobility Demand Forecasting

**Role skills:** Time-series cleaning, temporal validation, lag engineering, forecasting, baselines, uncertainty quantification, operations planning, regressors, ML tests.

## Business challenge
Estimate rental demand one hour at a time to support fleet allocation and operations planning. The system forecasts a point estimate and an **uncertainty interval** rather than an unjustifiably precise single number.

## Source data
[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset) — 17,379 hourly Capital Bikeshare records (2011–2012) with weather and calendar variables. Data and benchmark are historical, not real-time streams.

## Distinctive approach
- Validates timestamps, removes duplicate hours, **reindexes the hourly time axis** and calculates lag features on the complete timeline to prevent missing-hour misalignment.
- Excludes `casual` and `registered` because these **sum to target `cnt`**, preventing leakage.
- Uses strictly historical lag 1/24/168-hour demand and 24-hour rolling mean shifted by one hour.
- Splits chronologically: **65% train → 10% model selection → 10% interval calibration → 15% test**.
- Benchmarks seasonal-naive previous-day demand, Ridge regression and histogram gradient boosting.
- Calibrates 90% split-conformal intervals on held-out calibration data and checks actual coverage on the untouched test set.
- Exports prediction CSV, metrics, plots and reusable estimator.

**Forecasting caveat:** This is a **one-step-ahead nowcast**, conditioned on weather *observed during the predicted hour*. For an advance forecast, replace same-hour realized weather with weather forecasts or remove those predictors, then re-evaluate.

## Run
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.download_data
python -m src.train --data data/raw/hour.csv --out reports/real_experiment
python -m pytest -q
```

The outputs include `test_predictions.csv` with point estimates and interval bounds, `metrics.json`, and two figures.

## Inference
`python -m src.predict --model reports/real_experiment/model.joblib --json '<JSON object with weather, calendar and lag feature values>'`

## Validation status
Automated tests and an end-to-end **synthetic-only** smoke run completed. `reports/synthetic_smoke/` contains illustrative output, not validated results from UCI Bike Sharing. Official dataset experimentation remains pending source retrieval.

## Suggested next experiment
Backtest a forecast-only-weather variant and quantify the fleet allocation penalty of underforecast/overforecast decisions. Do not claim operational savings from the historic dataset alone.

### Example chart from the **synthetic software smoke test** (not a real dataset result)

![Synthetic smoke-test visualization](reports/synthetic_smoke/figures/forecast.png)

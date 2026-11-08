# creditlens

**An end-to-end credit risk scoring platform for UK personal loans, from raw application data to a monitored scoring API.**

![CI](https://github.com/waqastechie/creditlens/actions/workflows/ci.yml/badge.svg)

## The problem

A lender approving personal loans has to estimate each applicant's **probability of default (PD)**. If the estimate is too strict, the lender turns away good customers. If it is too loose, losses pile up. In the UK the lender must also be able to explain declines and show fair outcomes under the FCA Consumer Duty.

creditlens builds the whole system a lending data science team would own:

- data storage and validation
- feature engineering and models
- calibrated probabilities and a points-based scorecard
- explainability and a fairness audit
- an expected-loss decision policy
- an API, drift monitoring and a dashboard

## Why synthetic data?

Real loan books are confidential. creditlens generates realistic UK applications (GBP incomes, Experian-style 0–999 scores, UK regions) from a **known ground-truth risk model**. That makes it possible to measure how close each trained model gets to the true default probability, which can't be done with real data. The generator can also inject missing values, inconsistent labels and outliers to exercise the cleaning pipeline.

## Status

This project is built module by module. See [ROADMAP.md](ROADMAP.md) for the full plan and progress.

| Phase | Focus | Progress |
|---|---|---|
| 1 | Data foundation | 1 / 5 |
| 2 | Modelling | 0 / 7 |
| 3 | Responsible AI & business value | 0 / 3 |
| 4 | MLOps & deployment | 0 / 7 |
| 5 | Presentation | 0 / 3 |

## Quickstart

```bash
git clone https://github.com/waqastechie/creditlens.git
cd creditlens
pip install -e ".[dev]"

# Generate 50,000 clean applications
creditlens generate --rows 50000 --out data/raw/applications.csv

# Generate a messy batch with 5% missing values, label variants and outliers
creditlens generate --rows 50000 --missing-rate 0.05 --messy --out data/raw/messy.csv

pytest
```

## Project structure

```
src/creditlens/
  data/synthetic.py   # UK loan-application generator with a ground-truth risk model
  cli.py              # command-line entry point
tests/                # pytest suite, run in CI on every push
```

## Tech stack

Python · pandas · NumPy · scikit-learn · SQL (SQLite) · pytest · GitHub Actions. Later modules add MLflow, FastAPI, Docker and Streamlit.

## Author

**Waqas Afzal**, Data Scientist, London. MSc Data Science & Analytics (University of Hertfordshire), MSc Big Data Technologies (GCU London).

## Licence

MIT

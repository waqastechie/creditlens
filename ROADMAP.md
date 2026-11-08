# creditlens roadmap

One module per day. Each module is small, tested, and leaves the repo in a working state.
A module is ticked only when its code, tests and docs are merged to `main` and CI passes.

## Phase 1 — Data foundation
- [x] **01. Project scaffold & synthetic data generator** — reproducible UK loan-application generator with a known ground-truth risk model, CLI, tests, CI.
- [ ] **02. SQL storage layer** — SQLite schema for applications, a loader, and SQL views for analyst queries (default rate by region, cohort by month). Uses only the standard library `sqlite3`.
- [ ] **03. Data validation** — schema and business-rule checks (types, ranges, allowed categories, uniqueness) that produce a data-quality report and fail fast on bad batches.
- [ ] **04. Cleaning pipeline** — normalise category labels, fix impossible values, cap outliers, impute missing values; reversible and logged.
- [ ] **05. EDA report** — a script that writes a Markdown report with charts: target balance, distributions, default rate by segment, correlations.

## Phase 2 — Modelling
- [ ] **06. Feature engineering** — affordability features (loan-to-income, payment-to-income), a scikit-learn `ColumnTransformer` pipeline.
- [ ] **07. Time-based splitting & leakage guards** — out-of-time train/validation/test split and checks that no target or future information leaks into features.
- [ ] **08. Baselines** — majority class, a simple rule-based policy, and logistic regression.
- [ ] **09. Gradient boosting model** — `HistGradientBoostingClassifier` with cross-validated hyperparameter search.
- [ ] **10. Evaluation suite** — ROC-AUC, PR-AUC, Gini, KS statistic, Brier score, calibration plots, and comparison against the oracle `true_pd`.
- [ ] **11. Probability calibration** — Platt and isotonic calibration, with before-and-after reliability curves.
- [ ] **12. Points-based scorecard** — Weight-of-Evidence binning, Information Value feature selection, and a classic credit scorecard (industry standard in UK lending).

## Phase 3 — Responsible AI & business value
- [ ] **13. Explainability** — permutation importance, partial dependence, and per-applicant reason codes for declined applications.
- [ ] **14. Fairness audit** — performance and approval-rate parity across age bands and regions, framed around the FCA Consumer Duty.
- [ ] **15. Decision policy & expected loss** — PD × LGD × EAD, cost-sensitive cut-off selection, and the approval-rate vs bad-rate trade-off curve.

## Phase 4 — MLOps & deployment
- [ ] **16. Experiment tracking** — MLflow logging of parameters, metrics and artefacts for every training run.
- [ ] **17. Model packaging & model card** — versioned model artefacts, metadata, and a model card documenting data, metrics, limitations and intended use.
- [ ] **18. Scoring API** — FastAPI service with `/health`, `/score` and `/score/batch`, Pydantic input validation.
- [ ] **19. API test suite** — endpoint tests, invalid-input tests, and contract tests on the response schema.
- [ ] **20. Containerisation** — Dockerfile and docker-compose for the API and MLflow.
- [ ] **21. Drift monitoring** — Population Stability Index and characteristic stability reports comparing live traffic to training data.
- [ ] **22. Production simulation** — replay monthly batches with injected drift, fire alerts, and trigger a retraining job.

## Phase 5 — Presentation
- [ ] **23. Analyst dashboard** — Streamlit app to score an applicant, see reason codes, and explore portfolio risk.
- [ ] **24. Performance & load testing** — Locust load test, p50/p95/p99 latency, and one optimisation with before-and-after numbers.
- [ ] **25. Final write-up** — architecture diagram, results table, and a polished README for recruiters.

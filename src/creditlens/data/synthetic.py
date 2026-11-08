"""Synthetic UK consumer-loan application generator.

Real lending data is confidential, so creditlens ships with a reproducible
generator that produces applications shaped like a UK personal-loan book:
incomes in GBP, Experian-style credit scores (0-999), UK regions, and a
default label driven by a known logistic risk model plus noise.

Because the true risk function is known, later modules can measure how close
a trained model gets to the "oracle" — something impossible with real data.

The generator can also inject realistic data-quality issues (missing values,
inconsistent category labels, outliers) so the cleaning and validation
modules have something real to work on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

UK_REGIONS = [
    "London",
    "South East",
    "South West",
    "East of England",
    "West Midlands",
    "East Midlands",
    "Yorkshire and the Humber",
    "North West",
    "North East",
    "Wales",
    "Scotland",
    "Northern Ireland",
]
# Rough share of the adult population by region (sums to 1).
REGION_WEIGHTS = [0.135, 0.137, 0.085, 0.095, 0.088, 0.073, 0.082, 0.110, 0.040, 0.047, 0.081, 0.027]

EMPLOYMENT_STATUS = ["employed", "self_employed", "part_time", "retired", "unemployed", "student"]
EMPLOYMENT_WEIGHTS = [0.62, 0.13, 0.11, 0.08, 0.03, 0.03]

HOME_OWNERSHIP = ["mortgage", "own_outright", "rent", "living_with_family"]
HOME_WEIGHTS = [0.38, 0.17, 0.33, 0.12]

LOAN_PURPOSE = [
    "debt_consolidation",
    "home_improvement",
    "car",
    "wedding",
    "holiday",
    "business",
    "medical",
    "other",
]
PURPOSE_WEIGHTS = [0.36, 0.20, 0.17, 0.05, 0.06, 0.06, 0.03, 0.07]

# Log-odds contribution of each category to default risk (the "true" model).
_EMPLOYMENT_RISK = {
    "employed": 0.0,
    "self_employed": 0.35,
    "part_time": 0.30,
    "retired": -0.10,
    "unemployed": 1.20,
    "student": 0.70,
}
_HOME_RISK = {"mortgage": -0.15, "own_outright": -0.35, "rent": 0.20, "living_with_family": 0.10}
_PURPOSE_RISK = {
    "debt_consolidation": 0.30,
    "home_improvement": -0.20,
    "car": -0.10,
    "wedding": 0.05,
    "holiday": 0.25,
    "business": 0.45,
    "medical": 0.20,
    "other": 0.15,
}


@dataclass(frozen=True)
class GeneratorConfig:
    """Settings for :func:`generate_applications`.

    Attributes:
        n_rows: Number of loan applications to generate.
        seed: Random seed; the same seed always produces the same data.
        target_default_rate: Approximate share of loans that default.
        start_date / end_date: Range of application dates (ISO strings).
        missing_rate: Share of values blanked out in nullable columns.
        messy: If True, inject inconsistent labels and outliers as well.
    """

    n_rows: int = 10_000
    seed: int = 42
    target_default_rate: float = 0.08
    start_date: str = "2022-01-01"
    end_date: str = "2025-12-31"
    missing_rate: float = 0.0
    messy: bool = False


def _sample_features(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_rows

    age = np.clip(rng.normal(41, 12, n), 18, 80).round().astype(int)
    employment_status = rng.choice(EMPLOYMENT_STATUS, n, p=EMPLOYMENT_WEIGHTS)
    # Students are young, retirees are older.
    age = np.where(employment_status == "student", rng.integers(18, 26, n), age)
    age = np.where(employment_status == "retired", rng.integers(60, 81, n), age)

    base_income = rng.lognormal(mean=np.log(34_000), sigma=0.5, size=n)
    income_multiplier = pd.Series(employment_status).map(
        {
            "employed": 1.0,
            "self_employed": 1.1,
            "part_time": 0.5,
            "retired": 0.6,
            "unemployed": 0.25,
            "student": 0.3,
        }
    ).to_numpy()
    annual_income = np.round(base_income * income_multiplier, -2)

    employment_length_years = np.where(
        np.isin(employment_status, ["unemployed", "student"]),
        0,
        np.clip(rng.gamma(2.0, 3.0, n), 0, age - 16),
    ).round(1)

    home_ownership = rng.choice(HOME_OWNERSHIP, n, p=HOME_WEIGHTS)
    region = rng.choice(UK_REGIONS, n, p=REGION_WEIGHTS)
    loan_purpose = rng.choice(LOAN_PURPOSE, n, p=PURPOSE_WEIGHTS)

    # Experian-style 0-999 score, correlated with age and income.
    credit_score = (
        650
        + 2.0 * (age - 40)
        + 40 * np.log(annual_income / 30_000 + 0.1)
        + rng.normal(0, 110, n)
    )
    credit_score = np.clip(credit_score, 0, 999).round().astype(int)

    loan_amount = np.clip(rng.lognormal(np.log(9_000), 0.7, n), 1_000, 50_000).round(-2)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n, p=[0.08, 0.17, 0.35, 0.15, 0.25])

    # Lenders price risk: lower score -> higher APR.
    interest_rate = np.clip(
        24.0 - 0.018 * credit_score + rng.normal(0, 1.5, n), 3.0, 39.9
    ).round(2)

    existing_monthly_debt = np.clip(rng.lognormal(np.log(350), 0.8, n), 0, None).round(2)
    monthly_income = annual_income / 12
    monthly_rate = interest_rate / 100 / 12
    monthly_payment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** -loan_term_months)
    debt_to_income = np.round((existing_monthly_debt + monthly_payment) / np.maximum(monthly_income, 1), 3)

    credit_utilisation = np.clip(rng.beta(2, 3, n) + (700 - credit_score) / 2_000, 0, 1).round(3)
    num_open_accounts = rng.poisson(4, n)
    num_delinquencies_2y = rng.poisson(np.clip((750 - credit_score) / 400, 0.02, None))

    start = pd.Timestamp(cfg.start_date).value // 10**9
    end = pd.Timestamp(cfg.end_date).value // 10**9
    application_date = pd.to_datetime(rng.integers(start, end, n), unit="s").normalize()

    return pd.DataFrame(
        {
            "application_id": [f"APP{idx:08d}" for idx in range(1, n + 1)],
            "application_date": application_date,
            "age": age,
            "region": region,
            "employment_status": employment_status,
            "employment_length_years": employment_length_years,
            "annual_income": annual_income,
            "home_ownership": home_ownership,
            "credit_score": credit_score,
            "credit_utilisation": credit_utilisation,
            "num_open_accounts": num_open_accounts,
            "num_delinquencies_2y": num_delinquencies_2y,
            "existing_monthly_debt": existing_monthly_debt,
            "loan_purpose": loan_purpose,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate,
            "monthly_payment": monthly_payment.round(2),
            "debt_to_income": debt_to_income,
        }
    )


def true_default_logit(df: pd.DataFrame) -> np.ndarray:
    """The ground-truth risk model (log-odds of default, before the intercept).

    Exposed so later modules can benchmark trained models against the oracle.
    """
    return (
        -0.012 * (df["credit_score"].to_numpy() - 650)
        + 2.2 * np.clip(df["debt_to_income"].to_numpy(), 0, 2)
        + 1.4 * df["credit_utilisation"].to_numpy()
        + 0.35 * df["num_delinquencies_2y"].to_numpy()
        + 0.08 * df["interest_rate"].to_numpy()
        - 0.04 * np.minimum(df["employment_length_years"].to_numpy(), 10)
        + 0.25 * (df["loan_term_months"].to_numpy() >= 48)
        - 0.012 * (df["age"].to_numpy() - 40)
        + df["employment_status"].map(_EMPLOYMENT_RISK).to_numpy()
        + df["home_ownership"].map(_HOME_RISK).to_numpy()
        + df["loan_purpose"].map(_PURPOSE_RISK).to_numpy()
    )


def _calibrate_intercept(logit: np.ndarray, target_rate: float) -> float:
    """Find the intercept that makes the mean default probability hit the target."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(logit + mid)))).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _inject_mess(df: pd.DataFrame, cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    if cfg.missing_rate > 0:
        for col in ["employment_length_years", "annual_income", "credit_utilisation", "home_ownership"]:
            mask = rng.random(n) < cfg.missing_rate
            df.loc[mask, col] = np.nan if df[col].dtype.kind in "fi" else None

    if cfg.messy:
        # Inconsistent category spellings, as seen in hand-keyed application data.
        variants = {"self_employed": ["Self Employed", "self-employed"], "employed": ["Employed", "EMPLOYED"]}
        for canonical, alternatives in variants.items():
            idx = df.index[(df["employment_status"] == canonical) & (rng.random(n) < 0.05)]
            df.loc[idx, "employment_status"] = rng.choice(alternatives, len(idx))
        # A few implausible incomes (typos with an extra zero).
        idx = df.index[rng.random(n) < 0.002]
        df.loc[idx, "annual_income"] = df.loc[idx, "annual_income"] * 10
        # A few negative ages from a broken upstream form.
        idx = df.index[rng.random(n) < 0.001]
        df.loc[idx, "age"] = -df.loc[idx, "age"]

    return df


def generate_applications(cfg: GeneratorConfig | None = None) -> pd.DataFrame:
    """Generate a reproducible table of UK personal-loan applications.

    Returns one row per application with a binary ``defaulted`` target and the
    true default probability ``true_pd`` (useful for evaluation only — never
    use it as a feature).
    """
    cfg = cfg or GeneratorConfig()
    if cfg.n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if not 0 < cfg.target_default_rate < 1:
        raise ValueError("target_default_rate must be between 0 and 1")
    if not 0 <= cfg.missing_rate < 1:
        raise ValueError("missing_rate must be in [0, 1)")

    rng = np.random.default_rng(cfg.seed)
    df = _sample_features(cfg, rng)

    logit = true_default_logit(df) + rng.normal(0, 0.3, len(df))
    intercept = _calibrate_intercept(logit, cfg.target_default_rate)
    true_pd = 1 / (1 + np.exp(-(logit + intercept)))

    df["true_pd"] = true_pd.round(5)
    df["defaulted"] = (rng.random(len(df)) < true_pd).astype(int)

    if cfg.missing_rate > 0 or cfg.messy:
        df = _inject_mess(df, cfg, rng)

    return df.sort_values("application_date", kind="stable").reset_index(drop=True)

import numpy as np
import pandas as pd
import pytest

from creditlens.cli import main
from creditlens.data.synthetic import GeneratorConfig, generate_applications, true_default_logit


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    return generate_applications(GeneratorConfig(n_rows=20_000, seed=7))


def test_same_seed_is_reproducible():
    a = generate_applications(GeneratorConfig(n_rows=500, seed=1))
    b = generate_applications(GeneratorConfig(n_rows=500, seed=1))
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data():
    a = generate_applications(GeneratorConfig(n_rows=500, seed=1))
    b = generate_applications(GeneratorConfig(n_rows=500, seed=2))
    assert not a["annual_income"].equals(b["annual_income"])


def test_default_rate_close_to_target(clean_df):
    assert abs(clean_df["defaulted"].mean() - 0.08) < 0.01


def test_value_ranges(clean_df):
    assert clean_df["age"].between(18, 80).all()
    assert clean_df["credit_score"].between(0, 999).all()
    assert clean_df["credit_utilisation"].between(0, 1).all()
    assert clean_df["interest_rate"].between(3.0, 39.9).all()
    assert clean_df["loan_amount"].between(1_000, 50_000).all()
    assert clean_df["application_id"].is_unique
    assert clean_df["application_date"].is_monotonic_increasing


def test_clean_data_has_no_missing_values(clean_df):
    assert not clean_df.isna().any().any()


def test_risk_signal_is_learnable(clean_df):
    # Applicants in the worst credit-score decile should default far more often.
    low = clean_df["credit_score"] <= clean_df["credit_score"].quantile(0.1)
    high = clean_df["credit_score"] >= clean_df["credit_score"].quantile(0.9)
    assert clean_df.loc[low, "defaulted"].mean() > 3 * clean_df.loc[high, "defaulted"].mean()


def test_true_pd_ranks_defaults(clean_df):
    # The oracle probability should separate defaulters from non-defaulters.
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(clean_df["defaulted"], clean_df["true_pd"]) > 0.75
    assert np.isfinite(true_default_logit(clean_df)).all()


def test_missing_rate_injects_nulls():
    df = generate_applications(GeneratorConfig(n_rows=5_000, missing_rate=0.1, seed=3))
    share = df["annual_income"].isna().mean()
    assert 0.07 < share < 0.13
    assert df["defaulted"].notna().all()


def test_messy_mode_adds_label_variants_and_outliers():
    df = generate_applications(GeneratorConfig(n_rows=20_000, messy=True, seed=4))
    assert {"Self Employed", "self-employed"} & set(df["employment_status"])
    assert (df["age"] < 0).any()


@pytest.mark.parametrize(
    "kwargs",
    [{"n_rows": 0}, {"target_default_rate": 0}, {"target_default_rate": 1.5}, {"missing_rate": 1.0}],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        generate_applications(GeneratorConfig(**kwargs))


def test_cli_writes_csv(tmp_path):
    out = tmp_path / "apps.csv"
    main(["generate", "--rows", "300", "--out", str(out)])
    df = pd.read_csv(out)
    assert len(df) == 300
    assert "defaulted" in df.columns

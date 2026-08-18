"""Unit and integration tests for Data Cleaning & Quality Agent."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from amea.data_cleaning.agent import DataCleaningAgent
from amea.data_cleaning.models import CleaningAction, CleaningActionType, CleaningPlan
from amea.data_cleaning.pipeline_builder import CleaningPipelineBuilder
from amea.data_cleaning.transformers import (
    ColumnDropperTransformer,
    OutlierClipperTransformer,
    RareCategoryGrouperTransformer,
    AdaptiveImputerTransformer,
)
from amea.data_cleaning.validator import PostCleaningValidator
from amea.data_intelligence.models import DataTreatmentCandidate


def test_column_dropper():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "id_col": [101, 102]})
    dropper = ColumnDropperTransformer(columns_to_drop=["id_col", "non_existent"])
    transformed = dropper.fit_transform(df)
    assert "id_col" not in transformed.columns
    assert "a" in transformed.columns
    assert "b" in transformed.columns


def test_outlier_clipper_leakage_free():
    # Train data: values from 0 to 100
    train_df = pd.DataFrame({"val": list(range(101))})
    clipper = OutlierClipperTransformer(lower_quantile=0.05, upper_quantile=0.95)
    clipper.fit(train_df)

    # Test data: contains unseen extreme values
    test_df = pd.DataFrame({"val": [-50.0, 50.0, 200.0]})
    transformed_test = clipper.transform(test_df)

    # Learned bounds from train should clip test
    assert transformed_test["val"].iloc[0] == 5.0  # 5th percentile of 0..100
    assert transformed_test["val"].iloc[2] == 95.0  # 95th percentile of 0..100


def test_rare_category_grouper():
    train_df = pd.DataFrame({
        "city": ["New York"] * 90 + ["London"] * 9 + ["TinyVillage"] * 1  # 1% frequency
    })
    grouper = RareCategoryGrouperTransformer(min_frequency=0.05)
    grouper.fit(train_df)

    test_df = pd.DataFrame({
        "city": ["New York", "London", "TinyVillage", "UnknownNewCity"]
    })
    transformed = grouper.transform(test_df)
    assert transformed["city"].iloc[0] == "New York"
    assert transformed["city"].iloc[1] == "London"
    assert transformed["city"].iloc[2] == "__OTHER__"
    assert transformed["city"].iloc[3] == "__OTHER__"


def test_adaptive_imputer():
    train_df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, np.nan],  # median = 20.0
        "cat": ["A", "B", "A", np.nan],     # mode = "A"
    })
    imputer = AdaptiveImputerTransformer()
    imputer.fit(train_df)

    test_df = pd.DataFrame({
        "num": [np.nan, 50.0],
        "cat": [np.nan, "B"],
    })
    transformed = imputer.transform(test_df)
    assert transformed["num"].iloc[0] == 20.0
    assert transformed["cat"].iloc[0] == "A"
    assert transformed["num"].isnull().sum() == 0
    assert transformed["cat"].isnull().sum() == 0


def test_post_cleaning_validator():
    raw_df = pd.DataFrame({"feat": [1.0, np.nan, 3.0], "target": [0, 1, 0]})
    clean_df = pd.DataFrame({"feat": [1.0, 2.0, 3.0], "target": [0, 1, 0]})

    report = PostCleaningValidator.validate(
        initial_df=raw_df,
        cleaned_df=clean_df,
        target_column="target",
    )
    assert report.is_valid
    assert report.remaining_null_count == 0
    assert report.target_column_preserved


def test_data_cleaning_agent_end_to_end(tmp_path):
    # Create messy dataset
    np.random.seed(42)
    df = pd.DataFrame({
        "user_id": list(range(100)),
        "income": [1000.0 if i % 10 != 0 else np.nan for i in range(100)],
        "city": ["Metropolis"] * 95 + [f"Village_{i}" for i in range(5)],
        "churn": [0, 1] * 50,
    })
    data_path = tmp_path / "messy_churn.csv"
    df.to_csv(data_path, index=False)

    candidates = [
        DataTreatmentCandidate(
            strategy_id="strat_drop_id",
            target_columns=["user_id"],
            treatment_type="drop_feature",
            proposed_transformer="ColumnDropper",
            rationale="Drop ID column",
        ),
        DataTreatmentCandidate(
            strategy_id="strat_rare_city",
            target_columns=["city"],
            treatment_type="encoding",
            proposed_transformer="RareCategoryGrouper",
            rationale="Group rare cities",
        ),
    ]

    agent = DataCleaningAgent(persistence_dir=tmp_path / "persistence")
    artifact = agent.clean_dataset(
        raw_dataset_path=data_path,
        treatment_candidates=candidates,
        target_column="churn",
    )

    assert artifact.validation_report.is_valid
    assert artifact.validation_report.remaining_null_count == 0
    assert "user_id" not in pd.read_csv(artifact.cleaned_dataset_path).columns
    assert Path(artifact.cleaned_dataset_path).exists()

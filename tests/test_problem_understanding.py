"""Unit and integration tests for Problem Understanding Agent."""

import numpy as np
import pandas as pd
import pytest

from amea.core.state import TaskType
from amea.problem_understanding.agent import ProblemUnderstandingAgent
from amea.problem_understanding.intent_parser import IntentParser
from amea.problem_understanding.metric_recommender import MetricRecommender
from amea.problem_understanding.models import IntentCategory
from amea.problem_understanding.task_arbitrator import TaskArbitrator
from amea.problem_understanding.validation_recommender import ValidationRecommender


def test_intent_parser():
    # Classification query
    res1 = IntentParser.parse("Please build a classifier to predict churn on the customer dataset")
    assert res1.primary_intent == IntentCategory.CLASSIFICATION
    assert res1.mentioned_target_candidate == "churn"

    # Regression query
    res2 = IntentParser.parse("Estimate the continuous house price for real estate")
    assert res2.primary_intent == IntentCategory.REGRESSION
    assert res2.mentioned_target_candidate == "price"

    # Forecasting query
    res3 = IntentParser.parse("Forecast hourly energy demand for the next month")
    assert res3.primary_intent == IntentCategory.FORECASTING


def test_task_arbitrator_binary_classification():
    intent = IntentParser.parse("Predict customer churn")
    df = pd.DataFrame({"age": [20, 30, 40], "churn": [0, 1, 0]})

    task_type, target, conflicts, assumptions = TaskArbitrator.arbitrate(intent=intent, df=df, target_column="churn")
    assert task_type == TaskType.BINARY_CLASSIFICATION
    assert target == "churn"
    assert len(conflicts) == 0


def test_task_arbitrator_conflict_resolution():
    # User asked for regression, but column is binary 0/1
    intent = IntentParser.parse("Fit regression model for default")
    df = pd.DataFrame({"credit_score": [600, 700, 800], "default": [0, 1, 0]})

    task_type, target, conflicts, assumptions = TaskArbitrator.arbitrate(intent=intent, df=df, target_column="default")
    assert task_type == TaskType.BINARY_CLASSIFICATION
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "task_type_mismatch"


def test_task_arbitrator_multiclass():
    intent = IntentParser.parse("Classify ticket category")
    df = pd.DataFrame({"text_len": [10, 20, 30], "category": ["Billing", "Tech", "Account"]})

    task_type, target, conflicts, assumptions = TaskArbitrator.arbitrate(intent=intent, df=df, target_column="category")
    assert task_type == TaskType.MULTICLASS_CLASSIFICATION
    assert target == "category"


def test_metric_and_validation_recommender():
    # Imbalanced classification -> PR-AUC + StratifiedKFold
    df_imbalanced = pd.DataFrame({"target": [0] * 95 + [1] * 5})
    intent = IntentParser.parse("Predict fraud")
    primary_metric, secondaries, direction = MetricRecommender.recommend(
        task_type=TaskType.BINARY_CLASSIFICATION,
        intent=intent,
        df=df_imbalanced,
        target_column="target",
    )
    assert primary_metric == "pr_auc"
    assert direction == "maximize"

    cv_strat = ValidationRecommender.recommend(
        task_type=TaskType.BINARY_CLASSIFICATION,
        df=df_imbalanced,
        target_column="target",
    )
    assert cv_strat == "StratifiedKFold"


def test_problem_understanding_agent_end_to_end():
    agent = ProblemUnderstandingAgent()
    df = pd.DataFrame({
        "tenure": [12, 24, 36],
        "monthly_charges": [50.0, 70.0, 90.0],
        "churn": [0, 1, 0],
    })

    report = agent.formulate_problem(
        user_request="Predict customer churn with high accuracy",
        df=df,
        target_column_hint="churn",
    )

    assert report.is_feasible
    assert report.task_spec.task_type == TaskType.BINARY_CLASSIFICATION
    assert report.task_spec.target_column == "churn"
    assert report.task_spec.primary_metric in ["roc_auc", "pr_auc"]
    assert report.validation_strategy == "StratifiedKFold"

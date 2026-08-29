"""Test Suite for Supabase Authentication and User Database Integration."""

import uuid
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from amea.server import app
from amea.auth.supabase_service import SupabaseService

client = TestClient(app)


@pytest.fixture
def auth_service(tmp_path):
    """Isolated Supabase Service with dedicated SQLite test database."""
    db_path = tmp_path / "test_supabase.db"
    return SupabaseService(db_fallback_path=db_path, jwt_secret="test-secret-key-12345")


def test_supabase_signup_and_login_lifecycle(auth_service):
    """Verify signup, login, and JWT token issuance."""
    email = f"user_{uuid.uuid4().hex[:6]}@example.com"
    password = "StrongPassword123!"

    # 1. Sign Up
    signup_res = auth_service.sign_up(email=email, password=password, full_name="Test Engineer")
    assert "access_token" in signup_res
    assert signup_res["user"]["email"] == email
    assert signup_res["user"]["full_name"] == "Test Engineer"

    # 2. Verify issued token
    user_info = auth_service.verify_token(signup_res["access_token"])
    assert user_info is not None
    assert user_info["email"] == email
    assert user_info["id"] == signup_res["user"]["id"]

    # 3. Sign In
    signin_res = auth_service.sign_in(email=email, password=password)
    assert "access_token" in signin_res
    assert signin_res["user"]["email"] == email


def test_invalid_credentials_rejected(auth_service):
    """Verify invalid password or non-existent user raises ValueError."""
    email = f"user_{uuid.uuid4().hex[:6]}@example.com"
    auth_service.sign_up(email=email, password="correct_password")

    with pytest.raises(ValueError, match="Invalid email or password"):
        auth_service.sign_in(email=email, password="wrong_password")

    with pytest.raises(ValueError, match="Invalid email or password"):
        auth_service.sign_in(email="nobody@example.com", password="any_password")


def test_duplicate_signup_rejected(auth_service):
    """Verify duplicate email registration is rejected."""
    email = "duplicate@example.com"
    auth_service.sign_up(email=email, password="pwd1")

    with pytest.raises(ValueError, match="already exists"):
        auth_service.sign_up(email=email, password="pwd2")


def test_public_chat_allows_unauthenticated_greetings():
    """Verify simple greetings ('hello', 'hi') work without authentication."""
    res = client.post("/api/public/chat", json={"message": "hello!"})
    assert res.status_code == 200
    data = res.json()
    assert data["requires_auth"] is False
    assert "Hello!" in data["message"]


def test_public_chat_prompts_auth_for_protected_actions():
    """Verify asking to train or execute pipeline prompts for Supabase authentication."""
    res = client.post("/api/public/chat", json={"message": "Train a Random Forest model on my dataset"})
    assert res.status_code == 200
    data = res.json()
    assert data["requires_auth"] is True
    assert "sign in with Supabase Auth" in data["message"]


def test_api_auth_endpoints_lifecycle():
    """Verify /api/auth/signup, /api/auth/login, and /api/auth/me endpoints."""
    email = f"api_user_{uuid.uuid4().hex[:6]}@example.com"
    password = "SafePassword999!"

    # 1. API Sign Up
    signup_res = client.post("/api/auth/signup", json={"email": email, "password": password, "full_name": "API User"})
    assert signup_res.status_code == 200
    data = signup_res.json()
    token = data["access_token"]
    assert token is not None

    # 2. API /auth/me with Authorization Header
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["user"]["email"] == email

    # 3. API Sign In
    login_res = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login_res.status_code == 200
    assert "access_token" in login_res.json()


def test_user_project_isolation_and_persistence(auth_service):
    """Verify Row-Level Security rule: User A cannot see User B's projects."""
    user_a = auth_service.sign_up("user_a@test.com", "pwd123")["user"]["id"]
    user_b = auth_service.sign_up("user_b@test.com", "pwd123")["user"]["id"]

    # User A creates a project
    auth_service.save_project(user_id=user_a, project_id="proj_a1", name="Vision Model A")
    
    # User B creates a project
    auth_service.save_project(user_id=user_b, project_id="proj_b1", name="NLP Model B")

    # List User A's projects
    projs_a = auth_service.list_user_projects(user_a)
    assert len(projs_a) == 1
    assert projs_a[0]["id"] == "proj_a1"

    # List User B's projects
    projs_b = auth_service.list_user_projects(user_b)
    assert len(projs_b) == 1
    assert projs_b[0]["id"] == "proj_b1"

    # Verify ownership check
    assert auth_service.verify_project_ownership(user_a, "proj_a1") is True
    assert auth_service.verify_project_ownership(user_a, "proj_b1") is False


def test_database_persistence_experiments_and_artifacts(auth_service):
    """Verify storing experiments and artifact metadata in user database."""
    user = auth_service.sign_up("ml_eng@test.com", "pwd123")["user"]["id"]
    proj_id = "churn_pipeline_proj"
    auth_service.save_project(user_id=user, project_id=proj_id, name="Churn Pipeline")

    # Save Experiment
    exp_res = auth_service.save_experiment(
        user_id=user,
        project_id=proj_id,
        experiment_id="exp_rf_01",
        model_family="RandomForest",
        cv_metrics={"roc_auc": 0.942, "accuracy": 0.88},
        hyperparameters={"n_estimators": 100, "max_depth": 10},
        duration_sec=4.2,
        exit_code=0,
    )
    assert exp_res["status"] == "persisted"

    # Save Artifact
    art_res = auth_service.save_artifact(
        user_id=user,
        project_id=proj_id,
        artifact_id="art_model_pkl",
        artifact_type="model",
        name="model.joblib",
        path="/models/model.joblib",
        metadata={"size_mb": 1.2},
    )
    assert art_res["status"] == "persisted"
